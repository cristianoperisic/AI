# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# ===== 경로 설정 =====
BASE_DIR = Path(r"C:\Users\lw105\OneDrive\바탕 화면\open")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
SAMPLE_FP = BASE_DIR / "sample_submission.csv"

# ===== 데이터 로드 =====
train = pd.read_csv(TRAIN_DIR / "train.csv")
train["영업일자"] = pd.to_datetime(train["영업일자"], format="%Y-%m-%d")

sample = pd.read_csv(SAMPLE_FP)

tests = {}
for i in range(10):
    name = f"TEST_{i:02d}"
    df = pd.read_csv(TEST_DIR / f"{name}.csv")
    df["영업일자"] = pd.to_datetime(df["영업일자"], format="%Y-%m-%d")
    tests[name] = df

# ===== 라벨 인코딩 =====
le = LabelEncoder()
train["item_id"] = le.fit_transform(train["영업장명_메뉴명"])


# ===== 날짜/주기 특징 =====
def make_date_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["영업일자"].dt.year
    df["month"] = df["영업일자"].dt.month
    df["day"] = df["영업일자"].dt.day
    df["weekday"] = df["영업일자"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    # 주기성(월/요일) 사인/코사인
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["wday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["wday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    return df


# ===== Train용 Lag & Rolling =====
train = make_date_feats(train)
train = train.sort_values(["item_id", "영업일자"])

for lag in [1, 7, 14]:
    train[f"lag{lag}"] = train.groupby("item_id")["매출수량"].shift(lag)

# rolling은 누수 방지 위해 shift(1) 이후 rolling
train["roll7_mean"] = (
    train.groupby("item_id")["매출수량"]
    .shift(1)
    .rolling(7)
    .mean()
    .reset_index(0, drop=True)
)
train["roll14_mean"] = (
    train.groupby("item_id")["매출수량"]
    .shift(1)
    .rolling(14)
    .mean()
    .reset_index(0, drop=True)
)
train["roll7_std"] = (
    train.groupby("item_id")["매출수량"]
    .shift(1)
    .rolling(7)
    .std()
    .reset_index(0, drop=True)
)
train = train.dropna()

feature_cols = [
    "year",
    "month",
    "day",
    "weekday",
    "is_weekend",
    "month_sin",
    "month_cos",
    "wday_sin",
    "wday_cos",
    "item_id",
    "lag1",
    "lag7",
    "lag14",
    "roll7_mean",
    "roll14_mean",
    "roll7_std",
]
X = train[feature_cols]
y = train["매출수량"].astype(float)

# ===== XGBoost 파라미터 =====
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",  # GPU가 없으면 아래 try/except에서 cpu_hist로 변경
    "predictor": "gpu_predictor",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# GPU 미지원 환경 대비
try:
    _ = xgb.DeviceQuantileDMatrix(X, y)  # 빠르게 체크
except Exception:
    params["tree_method"] = "hist"
    params["predictor"] = "auto"

# ===== TimeSeriesSplit CV로 best_iteration 찾기 =====
tscv = TimeSeriesSplit(n_splits=5)
best_iters = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=5000,
        evals=[(dva, "val")],
        early_stopping_rounds=100,
        verbose_eval=200,
    )
    best_iters.append(booster.best_iteration)

# 안전하게 중앙값 사용(너무 큰/작은 fold 방지)
best_iter = int(np.median(best_iters)) if len(best_iters) else 1000

# ===== 전체 데이터로 재학습 =====
dall = xgb.DMatrix(X, label=y)
final_model = xgb.train(params, dall, num_boost_round=best_iter, verbose_eval=False)

# ===== 7일 재귀 예측 =====
all_preds = []

for test_name, test_df in tests.items():
    test_df = test_df.copy()
    test_df["item_id"] = le.transform(test_df["영업장명_메뉴명"])
    test_df = make_date_feats(test_df)
    test_df = test_df.sort_values(["item_id", "영업일자"])

    # 예측 대상: 마지막 날짜 이후 7일
    last_date = test_df["영업일자"].max()
    items = test_df["영업장명_메뉴명"].unique()

    # history: 테스트 파일에 포함된 과거 값 사용
    history = test_df[["영업일자", "item_id", "영업장명_메뉴명", "매출수량"]].copy()

    # 하루씩 앞으로 진행하며 재귀 예측
    preds_rows = []
    current_date = last_date
    for step in range(1, 8):
        target_date = current_date + pd.Timedelta(days=1)

        frame = pd.DataFrame(
            {"영업일자": np.repeat(target_date, len(items)), "영업장명_메뉴명": items}
        )
        frame["item_id"] = le.transform(frame["영업장명_메뉴명"])
        frame = make_date_feats(frame)

        # Lag merge (history 기준으로 미래 날짜에 맞춰 join)
        temp = history.copy()
        for lag in [1, 7, 14]:
            lagged = temp[["영업일자", "item_id", "매출수량"]].copy()
            lagged["영업일자"] = lagged["영업일자"] + pd.Timedelta(days=lag)
            frame = frame.merge(
                lagged.rename(columns={"매출수량": f"lag{lag}"}),
                on=["영업일자", "item_id"],
                how="left",
            )

        # rolling mean/std: (영업일자, item_id) 맞춰 당일의 roll을 미리 precompute
        # 간단히 최근 14일만 써서 계산(속도/메모리 절충)
        # history에는 step-1까지의 예측이 누적되므로 점점 풍부해짐
        roll_base = history.sort_values(["item_id", "영업일자"]).copy()
        roll_base["roll7_mean"] = (
            roll_base.groupby("item_id")["매출수량"]
            .rolling(7)
            .mean()
            .reset_index(0, drop=True)
        )
        roll_base["roll14_mean"] = (
            roll_base.groupby("item_id")["매출수량"]
            .rolling(14)
            .mean()
            .reset_index(0, drop=True)
        )
        roll_base["roll7_std"] = (
            roll_base.groupby("item_id")["매출수량"]
            .rolling(7)
            .std()
            .reset_index(0, drop=True)
        )
        # 하루 뒤에 쓰일 값이므로 날짜+1 셔프트
        roll_base["영업일자"] = roll_base["영업일자"] + pd.Timedelta(days=1)

        frame = frame.merge(
            roll_base[
                ["영업일자", "item_id", "roll7_mean", "roll14_mean", "roll7_std"]
            ],
            on=["영업일자", "item_id"],
            how="left",
        )

        # 결측 보정
        frame[["lag1", "lag7", "lag14", "roll7_mean", "roll14_mean", "roll7_std"]] = (
            frame[
                ["lag1", "lag7", "lag14", "roll7_mean", "roll14_mean", "roll7_std"]
            ].fillna(0)
        )

        X_pred = frame[feature_cols]
        dpred = xgb.DMatrix(X_pred)
        yhat = final_model.predict(dpred)

        # 음수 방지
        yhat = np.clip(yhat, 0, None)

        frame["pred"] = yhat
        # 예측을 history에 누적(다음 step의 lag/rolling 계산에 사용)
        add_hist = frame[["영업일자", "item_id", "영업장명_메뉴명", "pred"]].rename(
            columns={"pred": "매출수량"}
        )
        history = pd.concat([history, add_hist], ignore_index=True)

        # 제출 포맷용 저장
        frame_out = frame[["영업일자", "영업장명_메뉴명", "pred"]].copy()
        frame_out["영업일자"] = f"{test_name}+{step}일"
        preds_rows.append(frame_out)

        current_date = target_date

    test_pred = pd.concat(preds_rows, ignore_index=True)
    wide = test_pred.pivot(index="영업일자", columns="영업장명_메뉴명", values="pred")
    all_preds.append(wide)

# ===== 제출 파일 생성 =====
submission = pd.concat(all_preds)
submission = submission.reset_index().rename(columns={"index": "영업일자"})
submission = submission[sample.columns]  # 컬럼 순서/이름 맞추기
out_path = BASE_DIR / "submission_xgboost_recursive.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ XGBoost 재귀예측 제출 파일 저장 완료: {out_path}")
