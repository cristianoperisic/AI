from sklearn import svm, metrics
import random, re

csv = []
with open(
    "C:\\Users\\lw105\\OneDrive\\바탕 화면\\iris.csv", "r", encoding="utf-8"
) as fp:
    # 한 줄씩 읽어 오기
    for line in fp:
        line = line.strip()  # 줄바꿈 제거
        cols = line.split(",")  # 쉼표로 컬럼을 구분
        # 문자열 데이터를 숫자로 변환하기
        fn = lambda n: float(n) if re.match(r"^[0-9\.]+$", n) else n
        cols = list(map(fn, cols))
        csv.append(cols)

# 헤더 제거(컬럼명 제거)
del csv[0]

# 데이터를 섞어주기
random.shuffle(csv)

# 훈련(학습) 데이터와 테스트 데이터로 분리하기
total_len = len(csv)
train_len = int(total_len * 2 / 3)

train_data = []
train_label = []

test_data = []
test_label = []

for i in range(total_len):
    data = csv[i][:4]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)


clf = svm.SVC()

# 모델 학습
clf.fit(train_data, train_label)

# 테스트
pre_label = clf.predict(test_data)

# 정확도 구하기
ac_score = metrics.accuracy_score(test_label, pre_label)

print(f"정확도: {ac_score}")
