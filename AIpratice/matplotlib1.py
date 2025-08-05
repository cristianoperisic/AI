# matplotlib : 과학계산용 그래프 라이브러리
# ( 선그래프, 히스토그램, 산점도 등을 지원)

# 그래프를 그리기 위해서는 matplotlib의 pyplot 모듈을 이용한다.
import numpy as np
import matplotlib.pyplot as plt

# 데이터를 준비
x = np.arange(0, 6.4, 0.1)
y1 = np.cos(x)
y2 = np.sin(x)

# 그래프 그리기
plt.plot(x, y1, label="cos")
plt.plot(x, y2, label="sin")
plt.legend()
plt.title("sin/cos graph")
plt.xlabel("x")
plt.ylabel("y")
plt.show()  # 사인,코사인함수 나옴
