# -*- coding: utf-8 -*-
# =================================================================================================
# 최종 버전 v5: 상세 주석 추가 버전
# -------------------------------------------------------------------------------------------------
# [프로젝트 목표]
#  - 과거 매출 데이터를 기반으로, 각 상점의 메뉴별 '미래 7일간의 매출 수량'을 예측합니다.
#
# [핵심 전략]
#  1. 데이터 정제: '이상치 처리'를 통해 모델 학습을 방해하는 비정상적인 데이터를 안정화시킵니다.
#  2. 특징 공학: 날짜, Lag(과거 값), Rolling(이동 통계) 등 다양한 특징을 생성하여 모델에게 유용한 정보를 제공합니다.
#  3. 모델 최적화: 'Optuna'를 사용하여 데이터에 가장 적합한 XGBoost 하이퍼파라미터 조합을 자동으로 찾습니다.
#  4. 미래 예측: '재귀 예측' 기법을 사용하여, 예측한 값을 다시 입력으로 활용하며 7일 후까지 순차적으로 예측합니다.
# =================================================================================================

# ===== 1. 라이브러리 불러오기 =====
# 데이터 분석과 모델링에 필요한 도구(라이브러리)들을 불러옵니다.

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # 불필요한 경고 메시지를 숨겨서 실행 결과를 깔끔하게 유지합니다.

import pandas as pd  # 데이터를 표(DataFrame) 형태로 다루는 데이터 분석의 핵심 도구
import numpy as np  # 복잡한 수치 및 배열 계산을 위한 도구
from pathlib import (
    Path,
)  # 운영체제(OS)에 상관없이 파일 경로를 안정적으로 다루기 위한 도구
from sklearn.preprocessing import (
    LabelEncoder,
)  # 문자열 데이터를 숫자 ID로 변환하는 도구
from sklearn.model_selection import (
    TimeSeriesSplit,
)  # 시계열 데이터에 특화된 교차 검증 도구
import xgboost as xgb  # 이 프로젝트의 핵심 예측 모델
import optuna  # 하이퍼파라미터 자동 튜닝 라이브러리

print("라이브러리 로드 완료.")

# ===== 2. 경로 설정 =====
# 분석에 필요한 데이터 파일들의 위치를 지정합니다.

# 사용자의 원래 코드 경로 구조를 그대로 사용합니다.
BASE_DIR = Path(r"C:\Users\lw105\OneDrive\바탕 화면\open")
TRAIN_FP = (
    BASE_DIR / "train" / "train.csv"
)  # 'open' 폴더 안의 'train' 폴더에 있는 파일을 직접 지정
TEST_DIR = BASE_DIR / "test"
SAMPLE_FP = BASE_DIR / "sample_submission.csv"

# 경로가 올바른지 확인하여 오류를 사전에 방지합니다.
if not TRAIN_FP.exists() or not TEST_DIR.exists() or not SAMPLE_FP.exists():
    print(
        "🚨 경로 오류: 'open/train', 'test' 폴더 및 'sample_submission.csv' 파일이 지정된 경로에 있는지 확인해주세요."
    )
    print(f"예상 학습 데이터 경로: {TRAIN_FP}")
    exit()  # 파일이 없으면 여기서 실행을 중단합니다.
else:
    print("파일 경로 설정 완료.")

# ===== 3. 데이터 로드 및 전처리 =====
print("데이터 로딩 및 전처리 시작...")
# 'Long' 포맷(세로로 긴 형태)의 train.csv를 직접 로드합니다.
train = pd.read_csv(TRAIN_FP)
train["영업일자"] = pd.to_datetime(
    train["영업일자"]
)  # 날짜 관련 특징을 추출하기 위해 데이터 타입을 변환합니다.

# 3.1. 이상치 처리 (IQR 기반)
# "상식적인 범위를 벗어나는 데이터(이상치)는 모델 학습에 방해가 되므로, 안정적인 값으로 바꿔주자"
print("이상치 처리 시작...")


def handle_outliers_iqr(df_group):
    # 매출이 0인 경우가 많을 수 있으므로, 0을 제외하고 분위수를 계산하여 더 현실적인 이상치 범위를 설정합니다.
    non_zero_sales = df_group[df_group["매출수량"] > 0]["매출수량"]
    if len(non_zero_sales) < 5:  # 데이터가 너무 적으면 이상치 처리를 건너뜁니다.
        return df_group

    # 데이터의 분포를 파악하기 위해 1사분위수(Q1)와 3사분위수(Q3)를 계산합니다.
    q1, q3 = non_zero_sales.quantile(0.25), non_zero_sales.quantile(0.75)
    iqr = q3 - q1  # Q1과 Q3 사이의 범위(IQR)

    # 이 범위를 벗어나면 이상치로 간주합니다. (통계적으로 널리 사용되는 1.5배수 기준)
    lower_bound = max(
        0, q1 - 1.5 * iqr
    )  # 하한선 (매출이 음수일 수는 없으므로 0보다 작아지지 않게 함)
    upper_bound = q3 + 1.5 * iqr  # 상한선

    # 이상치를 정상 범위의 최대/최소값으로 대체(Clipping)합니다.
    df_group["매출수량"] = np.clip(df_group["매출수량"], lower_bound, upper_bound)
    return df_group


# 각 메뉴별로 그룹을 지어 이상치 처리 함수를 적용합니다.
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
# "모델이 더 똑똑하게 예측할 수 있도록, 원본 데이터로부터 유용한 힌트(특징)들을 만들어주자"
print("특징 공학 시작...")

# 4.1. 라벨 인코딩
le = LabelEncoder()
train["item_id"] = le.fit_transform(
    train["영업장명_메뉴명"]
)  # 메뉴 이름을 모델이 이해할 수 있는 고유 숫자 ID로 변환


# 4.2. 날짜 특징 생성 함수
def make_date_feats(df):
    out = df.copy()
    # 기본 날짜 정보
    out["year"], out["month"], out["day"], out["weekday"] = (
        out["영업일자"].dt.year,
        out["영업일자"].dt.month,
        out["영업일자"].dt.day,
        out["영업일자"].dt.weekday,
    )
    out["is_weekend"] = (
        out["weekday"].isin([5, 6]).astype(int)
    )  # 주말 여부 (토=5, 일=6)

    # 주기성 특징: 12월과 1월이 가깝다는 것을 모델에게 알려주기 위해 시계처럼 원형으로 변환
    out["month_sin"], out["month_cos"] = np.sin(
        2 * np.pi * out["month"] / 12.0
    ), np.cos(2 * np.pi * out["month"] / 12.0)
    out["wday_sin"], out["wday_cos"] = np.sin(2 * np.pi * out["weekday"] / 7.0), np.cos(
        2 * np.pi * out["weekday"] / 7.0
    )
    return out


train = make_date_feats(train)
train = train.sort_values(
    ["item_id", "영업일자"]
)  # Lag, Rolling 계산을 위해 아이템별, 날짜순으로 정렬

# 4.3. Lag & Rolling 특징 생성
# Lag: "어제는 몇 개 팔렸나?", "지난주 같은 요일에는 몇 개 팔렸나?"
for lag in [1, 7, 14, 28]:
    train[f"lag{lag}"] = train.groupby("item_id")["매출수량"].shift(lag)

# Rolling: "지난 7일간의 평균 매출은?", "매출 변동성은 어땠나?" (데이터 누수 방지를 위해 shift(1) 적용)
g = train.groupby("item_id")["매출수량"]
train["roll7_mean"], train["roll14_mean"], train["roll7_std"] = (
    g.shift(1).rolling(7).mean(),
    g.shift(1).rolling(14).mean(),
    g.shift(1).rolling(7).std(),
)

# 특징 생성 과정에서 생긴 결측치(NaN)가 있는 행은 학습에 사용할 수 없으므로 제거
train = train.dropna()
print("특징 공학 완료.")

# 4.4. 최종 학습 데이터 준비
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
X, y = train[feature_cols], train["매출수량"].astype(
    float
)  # X: 문제지(특징), y: 정답지(매출수량)

# ===== 5. Optuna를 이용한 하이퍼파라미터 튜닝 =====
# "모델의 성능을 최대로 끌어올리기 위해, 최적의 설정값을 자동으로 찾아보자"
print("Optuna 하이퍼파라미터 튜닝 시작...")
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except:
    HAS_CUDA = False

# TimeSeriesSplit 객체를 미리 생성하여 objective 함수와 최종 학습에서 공유
tscv = TimeSeriesSplit(n_splits=5)


# Optuna가 최적화할 목표(Objective) 함수를 정의합니다.
# 이 함수는 특정 하이퍼파라미터 조합으로 모델을 학습하고, 그 성능(RMSE)을 반환합니다.
def objective(trial):
    # 탐색할 하이퍼파라미터의 범위(Search Space)를 지정합니다.
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda" if HAS_CUDA else "cpu",
        "seed": 42,
        # trial.suggest_... : Optuna가 이 범위 내에서 다음 시도해볼 값을 '제안'합니다.
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }
    rmses = []
    # 교차 검증을 통해 파라미터 조합의 성능을 안정적으로 평가합니다.
    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        dtr, dva = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_va, label=y_va)

        # Pruning Callback: 성능이 나쁠 것으로 예상되는 시도를 조기에 중단시켜 탐색 시간을 절약합니다.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "val-rmse")
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=1000,
            evals=[(dva, "val")],
            early_stopping_rounds=50,
            callbacks=[pruning_callback],
            verbose_eval=False,
        )
        rmses.append(booster.best_score)

    # 교차 검증 결과의 평균 RMSE를 반환합니다. Optuna는 이 값을 '최소화'하는 방향으로 탐색합니다.
    return np.mean(rmses)


# Optuna 스터디(탐색 과정)를 생성하고 실행합니다.
study = optuna.create_study(
    direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
study.optimize(
    objective, n_trials=50
)  # 50번의 다른 파라미터 조합으로 최적화를 시도합니다.

print("튜닝 완료!")
print(f"최적의 하이퍼파라미터: {study.best_params}")
print(f"최적 RMSE: {study.best_value}")

# ===== 6. 최적 파라미터로 최종 모델 학습 =====
print("최종 모델 학습 시작...")
best_params = study.best_params
best_params.update(
    {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda" if HAS_CUDA else "cpu",
    }
)

# 최적 파라미터로 최적 학습 횟수(best_iteration)를 다시 찾습니다.
last_tr_idx, last_va_idx = list(tscv.split(X))[-1]
X_tr, X_va = X.iloc[last_tr_idx], X.iloc[last_va_idx]
y_tr, y_va = y.iloc[last_tr_idx], y.iloc[last_va_idx]
dtr, dva = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_va, label=y_va)
booster = xgb.train(
    best_params,
    dtr,
    num_boost_round=5000,
    evals=[(dva, "val")],
    early_stopping_rounds=100,
    verbose_eval=False,
)
best_iter = booster.best_iteration
print(f"최적 파라미터로 찾은 학습 횟수: {best_iter}")

# 모든 학습 데이터를 사용하여 최종 모델을 만듭니다.
dall = xgb.DMatrix(X, label=y)
final_model = xgb.train(
    best_params, dall, num_boost_round=best_iter, verbose_eval=False
)
print("최종 모델 학습 완료.")

# ===== 7. 재귀 예측 및 제출 파일 생성 =====
print("재귀 예측 및 제출 파일 생성 시작...")
all_preds = []
full_history = train.copy()  # 재귀 예측에 사용할 전체 과거 데이터를 미리 복사해 둡니다.

for test_name, test_df in tests.items():
    test_df = test_df.copy()
    test_df["item_id"] = le.transform(test_df["영업장명_메뉴명"])
    test_df = make_date_feats(test_df)

    # 예측의 기반이 될 과거 데이터를 매번 새로 만듭니다. (학습 데이터 + 해당 테스트 데이터)
    history = pd.concat([full_history, test_df], ignore_index=True)
    history = history.sort_values(["item_id", "영업일자"])

    last_date = test_df["영업일자"].max()
    items = test_df["영업장명_메뉴명"].unique()

    preds_rows = []
    current_date = last_date
    for step in range(1, 8):  # 7일간 하루씩 예측을 반복합니다.
        target_date = current_date + pd.Timedelta(days=1)

        # 1. 예측할 날짜의 기본 프레임(뼈대) 생성
        frame = pd.DataFrame(
            {"영업일자": np.repeat(target_date, len(items)), "영업장명_메뉴명": items}
        )
        frame["item_id"] = le.transform(frame["영업장명_메뉴명"])
        frame = make_date_feats(frame)

        # 2. 업데이트된 'history'를 사용하여 Lag & Rolling 특징 계산
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

        frame[feature_cols] = frame[feature_cols].fillna(0)  # 결측치는 0으로 채움

        # 3. 모델로 예측 수행
        X_pred = frame[feature_cols]
        dpred = xgb.DMatrix(X_pred)
        yhat = final_model.predict(dpred)
        yhat = np.clip(yhat, 0, None)  # 매출이 음수가 나오지 않도록 0으로 조정
        frame["pred"] = yhat

        # 4. 예측값을 history에 추가하여 다음 날 예측에 사용 (재귀의 핵심)
        add_hist = frame[["영업일자", "item_id", "영업장명_메뉴명", "pred"]].rename(
            columns={"pred": "매출수량"}
        )
        history = pd.concat([history, add_hist], ignore_index=True)

        # 5. 최종 제출용으로 결과 저장
        frame_out = frame[["영업일자", "영업장명_메뉴명", "pred"]].copy()
        frame_out["영업일자"] = f"{test_name}+{step}일"
        preds_rows.append(frame_out)

        current_date = target_date  # 기준 날짜를 하루 뒤로 업데이트

    test_pred = pd.concat(preds_rows, ignore_index=True)
    wide = test_pred.pivot(index="영업일자", columns="영업장명_메뉴명", values="pred")
    all_preds.append(wide)

# ===== 8. 최종 제출 파일 생성 =====
submission = pd.concat(all_preds)
submission = submission.reset_index().rename(columns={"index": "영업일자"})
submission = submission[sample.columns]  # 제출 샘플과 열 순서/이름을 정확히 일치시킴
out_path = BASE_DIR / "submission_final_tuned_v5_detailed_comments.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ 최종 제출 파일 저장 완료: {out_path}")
