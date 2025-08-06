import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


# 데이터 읽어오기(pandas 이용하기)
csv = pd.read_csv("C:\\Users\\lw105\\OneDrive\\바탕 화면\\iris.csv")

# 데이터와 레이블 분리하기
csv_data = csv[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
csv_label = csv["variety"]

# 훈련 데이터와 테스트 데이터로 분리하기
X_train, X_test, y_train, y_test = train_test_split(
    csv_data, csv_label, test_size=0.3, random_state=42, stratify=csv_label
)


clf = svm.SVC()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_predict)
print(f"정확도: {ac_score}")
