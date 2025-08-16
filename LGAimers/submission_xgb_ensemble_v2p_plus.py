# -*- coding: utf-8 -*-
"""
Ensemble XGBoost v2p_plus (Feature-Boosted, No Calibration, Rule-Compliant)
- Base: your best (v2_fix2) behavior maintained (RMSE objective, 7-step recursive inference)
- ONLY safer features added:
    * richer calendar (doy, doy_sin/cos, weekofyear, is_month_start/end)
    * rolling median 7, rolling min/max 28 (past-only)
    * zero counts in last 7/14/28 days (past-only)
    * days since last nonzero (past-only, capped)
    * mean28-based ratio features
- NO post-processing calibration, NO SMAPE early stopping
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# ========== Paths ==========
BASE_DIR = Path(r"C:\Users\lw105\OneDrive\바탕 화면\open")
TRAIN_FP = BASE_DIR / "train" / "train.csv"
TEST_DIR = BASE_DIR / "test"
SAMPLE_FP = BASE_DIR / "sample_submission.csv"
OUT_FP = BASE_DIR / "submission_xgb_ensemble_v2p_plus.csv"

assert TRAIN_FP.exists(), f"Train file not found: {TRAIN_FP}"
assert SAMPLE_FP.exists(), f"Sample file not found: {SAMPLE_FP}"

# ========== Feature helpers ==========
def make_date_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["영업일자"])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["weekday"] = dt.dt.weekday
    out["is_weekend"] = out["weekday"].isin([5,6]).astype(int)
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12.0)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12.0)
    out["wday_sin"] = np.sin(2*np.pi*out["weekday"]/7.0)
    out["wday_cos"] = np.cos(2*np.pi*out["weekday"]/7.0)
    out["doy"] = dt.dt.dayofyear
    out["doy_sin"] = np.sin(2*np.pi*out["doy"]/365.25)
    out["doy_cos"] = np.cos(2*np.pi*out["doy"]/365.25)
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)
    return out

def add_train_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id","영업일자"]).copy()
    g = df.groupby("item_id")["매출수량"]

    # standard lags
    for lag in [1,7,14,28]:
        df[f"lag{lag}"] = g.shift(lag)

    # standard rollings (past-only)
    df["roll7_mean"]  = g.shift(1).rolling(7).mean()
    df["roll14_mean"] = g.shift(1).rolling(14).mean()
    df["roll7_std"]   = g.shift(1).rolling(7).std()
    df["roll7_median"] = g.shift(1).rolling(7).median()

    # min/max 28 (past-only)
    df["min28"] = g.shift(1).rolling(28).min()
    df["max28"] = g.shift(1).rolling(28).max()

    # zero counts in last N days (past-only)
    z = (df["매출수량"]==0).astype(int)
    gz = df.assign(zflag=z).groupby("item_id")["zflag"]
    df["zeros7"]  = gz.shift(1).rolling(7).sum()
    df["zeros14"] = gz.shift(1).rolling(14).sum()
    df["zeros28"] = gz.shift(1).rolling(28).sum()

    # days since last nonzero (past-only, capped at 60)
    def _dsls(series: pd.Series) -> pd.Series:
        prev = series.shift(1).fillna(0).values
        out = np.zeros_like(prev, dtype=float)
        cnt = 0
        for i, v in enumerate(prev):
            if v > 0:
                cnt = 0
            else:
                cnt += 1
            out[i] = cnt
        out = np.clip(out, 0, 60)
        return pd.Series(out, index=series.index)
    df["days_since_nz"] = df.groupby("item_id")["매출수량"].transform(_dsls)

    # simple 7-day trend
    def _trend_7(x: pd.Series) -> float:
        if x.isna().sum() > 0:
            return np.nan
        y = x.values.astype(float)
        x_idx = np.arange(len(y))
        return np.polyfit(x_idx, y, 1)[0]
    df["trend7"] = g.shift(1).rolling(7).apply(lambda s: _trend_7(s), raw=False)

    # same-weekday rolling mean (past-only)
    df["weekday"] = pd.to_datetime(df["영업일자"]).dt.weekday
    grp = df.groupby(["item_id","weekday"])["매출수량"]
    df["weekday_roll4_mean"] = grp.shift(1).rolling(4).mean()

    # ratios & volatility
    df["lag1_div_lag7"] = df["lag1"] / (df["lag7"] + 1e-6)
    df["lag1_minus_lag7"] = df["lag1"] - df["lag7"]
    df["vol7"] = df["roll7_std"] / (df["roll7_mean"] + 1e-6)

    # mean28-based ratios, past-only
    df["mean28"] = g.shift(1).rolling(28).mean()
    df["lag1_div_mean28"] = df["lag1"] / (df["mean28"] + 1e-6)
    df["lag7_div_mean28"] = df["lag7"] / (df["mean28"] + 1e-6)
    df["roll7_div_mean28"] = df["roll7_mean"] / (df["mean28"] + 1e-6)

    return df

FEATURE_COLS = [
    # calendar
    "year","month","day","weekday","is_weekend",
    "month_sin","month_cos","wday_sin","wday_cos",
    "doy","doy_sin","doy_cos","weekofyear","is_month_start","is_month_end",
    # id
    "item_id",
    # lags
    "lag1","lag7","lag14","lag28",
    # roll stats
    "roll7_mean","roll14_mean","roll7_std","roll7_median",
    "min28","max28",
    # zeros & dsls
    "zeros7","zeros14","zeros28","days_since_nz",
    # trend & weekday mean
    "trend7","weekday_roll4_mean",
    # ratios & vol
    "lag1_div_lag7","lag1_minus_lag7","vol7",
    "lag1_div_mean28","lag7_div_mean28","roll7_div_mean28",
]

def safe_fillna_by_item(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    if "item_id" not in out.columns:
        raise KeyError("safe_fillna_by_item requires an 'item_id' column")
    out = out.loc[:, ~out.columns.duplicated()].copy()
    for c in cols:
        item_means = out.groupby("item_id")[c].transform(lambda s: s.fillna(s.mean()))
        out[c] = out[c].fillna(item_means)
        out[c] = out[c].fillna(out[c].mean()).fillna(0.0)
    return out

# ========== Load train ==========
print("Loading train...")
train = pd.read_csv(TRAIN_FP)
train["영업일자"] = pd.to_datetime(train["영업일자"])

le = LabelEncoder()
train["item_id"] = le.fit_transform(train["영업장명_메뉴명"])
train = make_date_feats(train)

# robust but rule-compliant outlier clamp per item
def handle_outliers_iqr(df_group):
    non_zero = df_group[df_group["매출수량"] > 0]["매출수량"]
    if len(non_zero) < 5:
        return df_group
    q1, q3 = non_zero.quantile(0.25), non_zero.quantile(0.75)
    iqr = q3 - q1
    lower, upper = max(0, q1 - 1.5*iqr), q3 + 1.5*iqr
    df_group["매출수량"] = np.clip(df_group["매출수량"], lower, upper)
    return df_group
train = train.groupby("영업장명_메뉴명", group_keys=False).apply(handle_outliers_iqr)

# add features
train = add_train_lag_roll_features(train)

X_all = train[FEATURE_COLS].copy()
y_all = train["매출수량"].astype(float)
mask = X_all.notna().all(axis=1)
X_all = X_all[mask]; y_all = y_all[mask]
print(f"Train matrix: {X_all.shape}, target: {y_all.shape}")

# ========== Train Ensemble (same as v2_fix2) ==========
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

tscv = TimeSeriesSplit(n_splits=5)
boosters, best_iters = [], []

for seed, max_depth in [(42,6),(13,8),(77,10),(101,6)]:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": device,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "max_depth": max_depth,
        "gamma": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": seed,
    }
    last_tr_idx, last_va_idx = list(tscv.split(X_all))[-1]
    dtr = xgb.DMatrix(X_all.iloc[last_tr_idx], label=y_all.iloc[last_tr_idx])
    dva = xgb.DMatrix(X_all.iloc[last_va_idx], label=y_all.iloc[last_va_idx])
    booster = xgb.train(params, dtr, num_boost_round=5000, evals=[(dva,"val")],
                        early_stopping_rounds=100, verbose_eval=False)
    best_iter = booster.best_iteration
    dall = xgb.DMatrix(X_all, label=y_all)
    booster_full = xgb.train(params, dall, num_boost_round=best_iter, verbose_eval=False)
    boosters.append(booster_full)
    best_iters.append(best_iter)

print("Ensemble trained. Best iterations per model:", best_iters)

def predict_ensemble(boosters, X_pred: pd.DataFrame) -> np.ndarray:
    dpred = xgb.DMatrix(X_pred)
    preds = [bst.predict(dpred) for bst in boosters]
    return np.mean(preds, axis=0)

# ========== Inference (7-step recursive, rule-compliant) ==========
print("Loading sample & tests...")
sample = pd.read_csv(SAMPLE_FP)

tests = {}
for i in range(10):
    name = f"TEST_{i:02d}"
    df = pd.read_csv(TEST_DIR / f"{name}.csv")
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    tests[name] = df

def build_step_features(history: pd.DataFrame, target_date: pd.Timestamp):
    items = history["영업장명_메뉴명"].unique()
    frame = pd.DataFrame({"영업일자": np.repeat(target_date, len(items)), "영업장명_메뉴명": items})
    frame["item_id"] = history.drop_duplicates("영업장명_메뉴명").set_index("영업장명_메뉴명")["item_id"].reindex(items).values
    frame = make_date_feats(frame)
    temp_hist = history.copy()

    # Lags
    for lag in [1,7,14,28]:
        lagged = temp_hist[["영업일자","item_id","매출수량"]].copy()
        lagged["영업일자"] = lagged["영업일자"] + pd.Timedelta(days=lag)
        frame = frame.merge(lagged.rename(columns={"매출수량": f"lag{lag}"}), on=["영업일자","item_id"], how="left")

    # Rolling & aggregates from past (align via +1 day)
    roll_base = temp_hist.sort_values(["item_id","영업일자"]).copy()
    gb = roll_base.groupby("item_id")["매출수량"]
    roll_base["roll7_mean"]  = gb.rolling(7).mean().reset_index(0, drop=True)
    roll_base["roll14_mean"] = gb.rolling(14).mean().reset_index(0, drop=True)
    roll_base["roll7_std"]   = gb.rolling(7).std().reset_index(0, drop=True)
    roll_base["roll7_median"]= gb.rolling(7).median().reset_index(0, drop=True)
    roll_base["min28"]       = gb.rolling(28).min().reset_index(0, drop=True)
    roll_base["max28"]       = gb.rolling(28).max().reset_index(0, drop=True)
    roll_base["mean28"]      = gb.rolling(28).mean().reset_index(0, drop=True)

    # zero counts
    roll_base["zflag"] = (roll_base["매출수량"]==0).astype(int)
    gzz = roll_base.groupby("item_id")["zflag"]
    roll_base["zeros7"]  = gzz.rolling(7).sum().reset_index(0, drop=True)
    roll_base["zeros14"] = gzz.rolling(14).sum().reset_index(0, drop=True)
    roll_base["zeros28"] = gzz.rolling(28).sum().reset_index(0, drop=True)

    roll_base["영업일자"] = roll_base["영업일자"] + pd.Timedelta(days=1)
    frame = frame.merge(roll_base[[
        "영업일자","item_id",
        "roll7_mean","roll14_mean","roll7_std","roll7_median",
        "min28","max28","mean28",
        "zeros7","zeros14","zeros28"
    ]], on=["영업일자","item_id"], how="left")

    # days since last nonzero (from 28-day history)
    def dsls_for_item(iid: int) -> float:
        h = temp_hist[temp_hist["item_id"]==iid].sort_values("영업일자")
        h = h.tail(28)
        h_nz = h[h["매출수량"]>0]
        if len(h_nz)==0:
            return 60.0
        last_nz = h_nz["영업일자"].max()
        return float(min(60, (pd.to_datetime(target_date - pd.Timedelta(days=1)) - last_nz).days))
    frame["days_since_nz"] = frame["item_id"].map(dsls_for_item)

    # trend7
    def compute_trend7_per_item(item_id):
        h = temp_hist[temp_hist["item_id"]==item_id].sort_values("영업일자")["매출수량"].values[-7:]
        if len(h) < 7 or np.isnan(h).any():
            return np.nan
        x = np.arange(7)
        return np.polyfit(x, h.astype(float), 1)[0]
    frame["trend7"] = frame["item_id"].map(lambda iid: compute_trend7_per_item(iid))

    # same weekday last4
    temp_hist["weekday"] = pd.to_datetime(temp_hist["영업일자"]).dt.weekday
    target_wday = pd.to_datetime(target_date).weekday()
    def same_weekday_last4_mean(iid):
        h = temp_hist[(temp_hist["item_id"]==iid) & (temp_hist["weekday"]==target_wday)].sort_values("영업일자")["매출수량"].tail(4)
        if len(h)==0:
            return np.nan
        return float(h.mean())
    frame["weekday_roll4_mean"] = frame["item_id"].map(same_weekday_last4_mean)

    # ratios & vol
    frame["lag1_div_lag7"] = frame["lag1"] / (frame["lag7"] + 1e-6)
    frame["lag1_minus_lag7"] = frame["lag1"] - frame["lag7"]
    frame["vol7"] = frame["roll7_std"] / (frame["roll7_mean"] + 1e-6)
    frame["lag1_div_mean28"] = frame["lag1"] / (frame["mean28"] + 1e-6)
    frame["lag7_div_mean28"] = frame["lag7"] / (frame["mean28"] + 1e-6)
    frame["roll7_div_mean28"] = frame["roll7_mean"] / (frame["mean28"] + 1e-6)

    X_pred_full = frame[FEATURE_COLS].copy()
    X_pred_full = X_pred_full.loc[:, ~X_pred_full.columns.duplicated()].copy()
    X_pred_full = safe_fillna_by_item(X_pred_full, cols=[c for c in FEATURE_COLS if c != "item_id"])
    X_pred = X_pred_full[FEATURE_COLS]
    return X_pred, frame

all_preds = []
for test_name, test_df in tests.items():
    test_df = test_df.copy()
    test_df["item_id"] = le.transform(test_df["영업장명_메뉴명"])
    test_df = make_date_feats(test_df)
    history = test_df.sort_values(["item_id","영업일자"]).copy()

    last_date = history["영업일자"].max()
    items = history["영업장명_메뉴명"].unique()
    preds_rows = []
    current_date = last_date

    for step in range(1, 8):
        target_date = current_date + pd.Timedelta(days=1)
        X_pred, frame = build_step_features(history, target_date)
        yhat = predict_ensemble(boosters, X_pred)
        yhat = np.clip(yhat, 0, None)

        add_hist = frame[["영업일자","item_id","영업장명_메뉴명"]].copy()
        add_hist["매출수량"] = yhat
        history = pd.concat([history, add_hist], ignore_index=True)

        out_row = frame[["영업일자","영업장명_메뉴명"]].copy()
        out_row["pred"] = yhat
        out_row["영업일자"] = f"{test_name}+{step}일"
        preds_rows.append(out_row)

        current_date = target_date

    test_pred = pd.concat(preds_rows, ignore_index=True)
    wide = test_pred.pivot(index="영업일자", columns="영업장명_메뉴명", values="pred")
    all_preds.append(wide)

submission = pd.concat(all_preds)
submission = submission.reset_index().rename(columns={"index": "영업일자"})
sample = pd.read_csv(SAMPLE_FP)
submission = submission[sample.columns]

submission.to_csv(OUT_FP, index=False, encoding="utf-8-sig")
print(f"✅ Saved: {OUT_FP}")
