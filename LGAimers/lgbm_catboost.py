# -*- coding: utf-8 -*-
# =================================================================================================
# LightGBM + CatBoost 앙상블 모델
# -------------------------------------------------------------------------------------------------
# [핵심 전략]
#  1. 데이터 처리: 이상치 처리 및 특징 공학은 기존 XGBoost 코드와 동일한 파이프라인 사용
#  2. 개별 모델 학습: 속도의 LightGBM과 안정성의 CatBoost를 각각 최적의 상태로 학습
#  3. 실시간 앙상블: 재귀 예측 시, 각 날짜마다 두 모델의 예측 결과를 평균내어 최종 예측값으로 사용
# =================================================================================================

# ===== 1. 라이브러리 불러오기 =====
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import catboost as cb

print("라이브러리 로드 완료.")

# ===== 2. 경로 설정 =====
BASE_DIR = Path(r"C:\Users\lw105\OneDrive\바탕 화면\open")
TRAIN_FP = BASE_DIR / "train" / "train.csv"
TEST_DIR = BASE_DIR / "test"
SAMPLE_FP = BASE_DIR / "sample_submission.csv"

if not TRAIN_FP.exists():
    print(f"🚨 경로 오류: {TRAIN_FP} 파일을 찾을 수 없습니다.")
    exit()
else:
    print("파일 경로 설정 완료.")

# ===== 3. 데이터 로드 및 전처리 =====
print("데이터 로딩 및 전처리 시작...")
train = pd.read_csv(TRAIN_FP)
train["영업일자"] = pd.to_datetime(train["영업일자"])


# 3.1. 이상치 처리
def handle_outliers_iqr(df_group):
    non_zero_sales = df_group[df_group["매출수량"] > 0]["매출수량"]
    if len(non_zero_sales) < 5:
        return df_group
    q1, q3 = non_zero_sales.quantile(0.25), non_zero_sales.quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    df_group["매출수량"] = np.clip(df_group["매출수량"], lower_bound, upper_bound)
    return df_group


train = train.groupby("영업장명_메뉴명", group_keys=False).apply(handle_outliers_iqr)
print("이상치 처리 완료.")

# 3.2. 테스트 데이터 로드
sample = pd.read_csv(SAMPLE_FP)
tests = {}
for i in range(10):
    name = f"TEST_{i:02d}"
    df = pd.read_csv(TEST_DIR / f"{name}.csv")
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    tests[name] = df

# ===== 4. 특징 공학 (Feature Engineering) =====
print("특징 공학 시작...")
le = LabelEncoder()
train["item_id"] = le.fit_transform(train["영업장명_메뉴명"])


def make_date_feats(df):
    out = df.copy()
    out["year"], out["month"], out["day"], out["weekday"] = (
        out["영업일자"].dt.year,
        out["영업일자"].dt.month,
        out["영업일자"].dt.day,
        out["영업일자"].dt.weekday,
    )
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["month_sin"], out["month_cos"] = np.sin(
        2 * np.pi * out["month"] / 12.0
    ), np.cos(2 * np.pi * out["month"] / 12.0)
    out["wday_sin"], out["wday_cos"] = np.sin(2 * np.pi * out["weekday"] / 7.0), np.cos(
        2 * np.pi * out["weekday"] / 7.0
    )
    return out


train = make_date_feats(train)
train = train.sort_values(["item_id", "영업일자"])

for lag in [1, 7, 14, 28]:
    train[f"lag{lag}"] = train.groupby("item_id")["매출수량"].shift(lag)

g = train.groupby("item_id")["매출수량"]
train["roll7_mean"], train["roll14_mean"], train["roll7_std"] = (
    g.shift(1).rolling(7).mean(),
    g.shift(1).rolling(14).mean(),
    g.shift(1).rolling(7).std(),
)
train = train.dropna()
print("특징 공학 완료.")

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
    "lag28",
    "roll7_mean",
    "roll14_mean",
    "roll7_std",
]
categorical_feature = ["item_id"]
X, y = train[feature_cols], train["매출수량"].astype(float)

# ===== 5. 모델 학습 =====
tscv = TimeSeriesSplit(n_splits=5)
# 교차 검증의 마지막 fold를 사용하여 학습 횟수를 결정하고 최종 모델을 학습합니다.
tr_idx, va_idx = list(tscv.split(X))[-1]
X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

# --- 5.1. LightGBM 모델 학습 ---
print("LightGBM 모델 학습 시작...")
lgbm = lgb.LGBMRegressor(
    objective="regression_l1",  # MAE를 손실 함수로 사용하여 이상치에 좀 더 강건하게 만듭니다.
    metric="rmse",
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    seed=42,
    n_jobs=-1,
    verbose=-1,
    colsample_bytree=0.8,
    subsample=0.8,
)
lgbm.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
print("LightGBM 모델 학습 완료.")

# --- 5.2. CatBoost 모델 학습 ---
print("CatBoost 모델 학습 시작...")
cat = cb.CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=5000,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=0,
    cat_features=categorical_feature,  # 'item_id'가 범주형 특징임을 명시
)
cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False)
print("CatBoost 모델 학습 완료.")

# ===== 6. 재귀 예측 및 앙상블 =====
print("재귀 예측 및 앙상블 시작...")
all_preds = []
full_history = train.copy()

for test_name, test_df in tests.items():
    test_df = test_df.copy()
    test_df["item_id"] = le.transform(test_df["영업장명_메뉴명"])
    test_df = make_date_feats(test_df)

    history = pd.concat([full_history, test_df], ignore_index=True)
    history = history.sort_values(["item_id", "영업일자"])

    last_date = test_df["영업일자"].max()
    items = test_df["영업장명_메뉴명"].unique()

    preds_rows = []
    current_date = last_date
    for step in range(1, 8):
        target_date = current_date + pd.Timedelta(days=1)
        frame = pd.DataFrame(
            {"영업일자": np.repeat(target_date, len(items)), "영업장명_메뉴명": items}
        )
        frame["item_id"] = le.transform(frame["영업장명_메뉴명"])
        frame = make_date_feats(frame)

        temp_hist = history.copy()
        for lag in [1, 7, 14, 28]:
            lagged = temp_hist[["영업일자", "item_id", "매출수량"]].copy()
            lagged["영업일자"] = lagged["영업일자"] + pd.Timedelta(days=lag)
            frame = frame.merge(
                lagged.rename(columns={"매출수량": f"lag{lag}"}),
                on=["영업일자", "item_id"],
                how="left",
            )

        roll_base = temp_hist.sort_values(["item_id", "영업일자"]).copy()
        gb = roll_base.groupby("item_id")["매출수량"]
        roll_base["roll7_mean"] = gb.rolling(7).mean().reset_index(0, drop=True)
        roll_base["roll14_mean"] = gb.rolling(14).mean().reset_index(0, drop=True)
        roll_base["roll7_std"] = gb.rolling(7).std().reset_index(0, drop=True)
        roll_base["영업일자"] = roll_base["영업일자"] + pd.Timedelta(days=1)
        frame = frame.merge(
            roll_base[
                ["영업일자", "item_id", "roll7_mean", "roll14_mean", "roll7_std"]
            ],
            on=["영업일자", "item_id"],
            how="left",
        )

        frame[feature_cols] = frame[feature_cols].fillna(0)
        X_pred = frame[feature_cols]

        # --- 두 모델로 각각 예측 ---
        pred_lgbm = lgbm.predict(X_pred)
        pred_cat = cat.predict(X_pred)

        # --- 예측 결과 앙상블 (단순 평균) ---
        yhat = (pred_lgbm + pred_cat) / 2.0
        yhat = np.clip(yhat, 0, None)  # 음수 예측 방지
        frame["pred"] = yhat

        # 앙상블된 예측값을 history에 추가하여 다음 날 예측에 사용
        add_hist = frame[["영업일자", "item_id", "영업장명_메뉴명", "pred"]].rename(
            columns={"pred": "매출수량"}
        )
        history = pd.concat([history, add_hist], ignore_index=True)

        frame_out = frame[["영업일자", "영업장명_메뉴명", "pred"]].copy()
        frame_out["영업일자"] = f"{test_name}+{step}일"
        preds_rows.append(frame_out)

        current_date = target_date

    test_pred = pd.concat(preds_rows, ignore_index=True)
    wide = test_pred.pivot(index="영업일자", columns="영업장명_메뉴명", values="pred")
    all_preds.append(wide)

# ===== 7. 최종 제출 파일 생성 =====
submission = pd.concat(all_preds)
submission = submission.reset_index().rename(columns={"index": "영업일자"})
# sample.columns에 없는 컬럼이 submission에 있을 경우를 대비하여, sample에 있는 컬럼만 선택
submission = submission[sample.columns]

out_path = BASE_DIR / "submission_lgbm_cat_ensemble.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ 최종 앙상블 제출 파일 저장 완료: {out_path}")
