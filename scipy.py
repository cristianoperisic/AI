import numpy as np

# numpy에서 특수행렬을 만드는 함수
# eye(N,M=,k= 얼마나 오른쪽으로 밀건지,dtype) 항등행렬을 생성하는 함수
# N은 행의 수, M은 열의 수, k는 대각의 위치

print(np.eye(3, k=1, dtype=int))

print(np.eye(4, M=5, k=2, dtype=int))

# diag() 함수는 정방행렬에서 대각요소만 추출하여 벡터를 만든다.
# diag(v, k=),
# diag()함수는 벡터요소를 대각요소로 하는 정방행렬로 만들기도 함
print(np.diag(np.arange(1, 10).reshape(3, 3)))  # 1,5,9
print(np.diag(np.arange(1, 10).reshape(3, 3), k=1))  # 2,6
