from sklearn import svm

xor_data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

# 주어진 데이터를 분리한다 (학습데이터와 레이블을 분리)
training_data = []
label = []

for row in xor_data:
    p = row[0]
    q = row[1]
    res = row[2]

    training_data.append([p, q])
    label.append(res)

# SVM 알고리즘을 사용하는 머신러닝 객체를 만든다.
# SVM은 분류, 회귀 알고리즘 가짐 회귀는 SVR
clf = svm.SVC()

# fit() 메서드 : 학습기계에 데이터를 학습시킨다.
clf.fit(training_data, label)

# predic() 메서드 : 학습데이터를 이용하여 예측한다.
pre = clf.predict(training_data)
print("예측결과: ", pre)

ok = 0
total = 0

for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer:
        ok += 1
    total += 1

print("정확도: ", ok, "/", total, "=", ok / total)
