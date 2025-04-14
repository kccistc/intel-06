items = [1, 2, 3, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print('---------------------')
for item in items:
    print(item)
print('---------------------')
items = [[1,2], [3,4], [5,6]]
for item in items:
    print(item[0], item[1])
print('---------------------')
for item1, item2 in items:
    print(item1, item2)
print('---------------------')
info = {'A': 1, 'B': 2, 'C': 3}
for key in info:
    print(key, info[key])
print('---------------------')
for key, value in info.items():
    print(key, value)


items1 = [[1, 2], [3, 4], [5, 6]]
items2 = [['A', 'B'], ['C', 'D'], ['E', 'F']]
print(items1)
print(items2)
print("=======")

for digits, characters in zip(items1, items2):
    print(digits, characters)



 # 리스트 생성
a = []
for k in range(0, 5):
    a.append(k)
print(a)
print("-------")

# 리스트 컴프리헨션
a = [k for k in range(0, 5)]
print(a)
print("-------")

# 조건부 리스트 컴프리헨션
a = [k if (k+1) % 2 else k+5 for k in range(0, 5)]
print(a)
print("-------")

# 필터링 리스트 컴프리헨션
a = [k for k in range(0, 5) if k % 2 == 0]
print(a)
print("-------")

# 딕셔너리 컴프리헨션
a = {k: k*10 for k in range(0, 5)}
print(a)
print("-------")

# 리스트 내 요소 합치기
a = [1, 3, 4]
c = [a[i] + a[i] for i in range(len(a))]
print(c)


import numpy as np

# 배열 생성 및 덧셈
a = np.array([1, 2, 3, 4])
print(a)
print(a + a)


import numpy as np

# 일반 리스트 덧셈
b = [1, 2, 3, 4]
print(b + b)

# 2차원 배열 생성
a = np.array([[1, 2], [3, 4]])
print(a)

# 3차원 배열 생성
a = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
print(a)


# 배열의 모양 확인
a = np.array([1, 2, 3, 4])
b = np.array([[1], [2], [3], [4]])
print(a)
print(a.shape)
print(b)
print(b.shape)


import numpy as np
from numpy import linalg as LA

# 배열 생성 및 놈 계산
c = np.array([[1, 2, 3], [-1, 1, 4]])
print(LA.norm(c, axis=0))  # 각 열의 놈
print(LA.norm(c, axis=1))  # 각 행의 놈
print(LA.norm(c, ord=1, axis=1))  # L1 놈
print(LA.norm(c, ord=2, axis=1))  # L2 놈

A = np.array([[1,4,2,0], [9,5,0,0], [4,0,2,4], [6,1,8,3]])
x = np.array([1,2,3,4])
b = np.array([0,0,0,0])
n = 4
for i in range(0, n):
  val = 0.0
  for j in range(0,n):
    # TODO 2
    val += A[i,j] * x[j]

  b[i] = val

print("calculate=",b)

b = np.dot(A,x)
print("dot=",b)

b = np.matmul(A,x)
print("matmul=",b)

b = A@x 
print("A@x=",b)

b= A*x
print("A*x=",b)
