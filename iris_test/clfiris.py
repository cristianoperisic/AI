# scikit_learn의 붓꽃 데이터 셋을 가져오기

import numpy as np
from sklearn.datasets import load_iris

irisData = load_iris()

# load_iris() 함수가 리턴하는 객체는 Bunch 클래스 객체
# Bunch 클래스 객체는 파이썬의 Dictionary와 유사한 객체로
# 키와 값으로 구성되어 있다

# 훈련데이터와 테스트데이터를 나누기 위한 함수
# train_test_split모듈에 있는 train_test_split()
# train_test_split모듈은 sklearn.model_selection
from sklearn.model_selection import train_test_split

# scikit-learn에서 데이터는 대문자 X로 표시하고 레이블은 소문자y
# 로 표시한다.

X_train, X_test, y_train, y_test = train_test_split(
    irisData["data"], irisData["target"], random_state=0
)

# train_test_split()의 리턴 타입은 모두 numpy 배열이다

# KNN(K Nearest Neighbors): K-최근접 이웃 알고리즘
# 사용하기 쉬운 분류 알고리즘(분류기) 중의 하나이다.

# K의 의미는 가장 가까운 이웃 하나를 의미하는 것이 아니라
# 훈련데이터에서 새로운 데이터에 가장 가까운 K개의 이웃을 찾는다는 의미

# KNN을 사용하기 위해서는 neighbors모듈에 KNeighborsClassifier함수를 사용
# KNeighborsClassifier()함수 중에 중요한 매개변수는 n_neighbors
# 이 매개 변수는 이웃의 개수를 지정하는 매개변수이다.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

# 훈련 데이터셋을 가지고 모델을 만들려면 fit 메서드 사용한다
# fit 메서드의 리턴 값은 knn의 객체를 리턴한다.
knn.fit(X_train, y_train)

# 채집한 붓꽃의 새로운 데이터(샘플)라고 가정하고 특성값을 만든다.
# scikit-learn에서는 항상 데이터가 2차원 배열일 것으로 예측해야 한다.

X_newData = np.array([[5.1, 2.9, 1, 0.3]])
prediction = knn.predict(X_newData)
print(f"예측: {prediction}")
print(f"예측 품종의 이름: {irisData['target_names'][prediction]}")

y_predict = knn.predict(X_test)

# 정확도를 계산하기 위해서 numpy의 mean()메서드를 이용
# knn객체의 score()매서드를 사용해도 된다.

x = np.array([1, 2, 3, 2])
print(f"정확도: {np.mean(y_predict==y_test):.2f}")

# 머신 러닝의 용어 정리

# iris 분류 문제 있어서 각 품종을 클래스라고 한다.
# 개별 붓꽃의 품종은 레이블 이라고 한다.

# 붓꽃의 데이터 셋은 두개의 Numpy 배열로 이루어져 있다.
# 하나는 데이터, 다른 하나는 출력을 가지고 있다.
# scikit-learn에서는 데이터 X로 표기하고, 출력은 y로 표기한다.

# 이때 배열 X는 2차원 배열이고 각 행은 데이터포인트(샘플)에 해당된다
# 각 컬럼(열)은 특성이라고 한다.
# 배열 y는 1차원 배열이고, 각 샘플의 클래스 레이블에 해당된다.
