# scipy에서 scikit-learn 알고리즘을 구현 할 때
# 많이 사용하는 모듈은 scipy.sparse 모듈
# 이때 희소 행렬기능은 주요 기능 중의 하나이다
# 희소 행렬(sparse matrix)
import numpy as np
from scipy import sparse

b1 = np.eye(4, dtype=int)
print(f"Numpy 배열: \n {b1}")

# sparse.csr_matrix() 메소드: 0이 아닌 원소만 저장
sparse_matrix = sparse.csr_matrix(b1)
print(f"Scipy의 CSR 행렬: \n {sparse_matrix}")

# CSR(Compressed Row Storage)

b2 = np.eye(5, k=-1, dtype=int)
sparse_matrix = sparse.csr_matrix(b2)
print(f"SciPy의 CSR행렬: \n{sparse_matrix}")

b3 = np.arange(16).reshape(4, 4)
sparse_matrix2 = sparse.csr_matrix(b3)
print(sparse_matrix2)
x = np.diag(b3)
print(x)
y = np.diag(x)
print(f"-------------\n{y}")

#  희소행렬을 직접 만들 때 사용하는 포맷
# COO 포맷 (Coordinate 포맷)

data = np.ones(4)

row_indices = np.arange(4)
column_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, column_indices)))
print(f"COO 표현: \n{eye_coo}")
