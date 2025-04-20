#for loop
items = [1, 2, 3,4, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print('=====================')
for item in items:


    print(item)
print('=====================')
items = [[1,2], [3,4], [5,6]]
for item in items:
    print(item[0], item[1])
print('=====================')
for item1, item2 in items:
    print(item1, item2)
print('=====================')
info = {'A' : 1, 'B': 2, 'C': 3}
for key in info:
    print(key, info[key])
print('=====================')
for key, value in info.items():
    print(key, value)

#numpy
#numpy 배열
import numpy as np
a = np.array([1, 2, 3, 4])
print(a)
print(a + a)

#일반 배열
b = [1, 2, 3, 4]
print(b + b)

#이중 배열(행렬)
a = np.array([[1, 2], [3, 4]])
print(a)

#삼중 배열
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a)

#배열의 모양(shape)
a = np.array([1, 2, 3, 4])  #list를 np.array로 변환
b = np.array([[1], [2], [3], [4]])
print(a)
print(a.shape)
print(b)
print(b.shape)

#Norm
import numpy as np
from numpy import linalg as LA
c = np.array([[1, 2, 3], [-1, 1, 4]])
print(LA.norm(c, axis = 0))
print(LA.norm(c, axis = 1))
print(LA.norm(c, ord = 1, axis = 1))
print(LA.norm(c, ord = 2, axis = 1))

#Transpose
a = np.array([[1,], [2], [3], [4]])
print(a)
print(a.T)
print(a.T.reshape(-1, 4))
print(a.shape)
print(a.T.reshape(-1, 4).T.shape)

a = np.array([1, 2, 3,4 ])
b = a.reshape(4, -1)
print(a)
print(a.reshape(2, -1))
print(a.shape, ",", b.shape, ",", np.array([[1, 2, 3, 4]]).shape)

#Reshape
a = np.array([1, 2, 3, 4, 5, 6])
print(a.reshape(3, 2))
print(a.shape)
b = a.reshape(3, -1)
print(b)
print(b.shape)
c = a.reshape(-1, 2)
print(c)
print(c.shape)

a = np.array([1, 2, 3, 4])
print(a)
print(a, T)
b = a.reshape(4, -1)
print(b.shape)
print(b)
print(b.T.shape)

#배열 인덱싱
a = np.array([10, 20, 30, 40, 50, 60])
print(a)
b = a[[4, 2, 0]]
print(b)
idx = np.arange(0, len(a))
print(idx)
np.random.shuffle(idx)
print(idx)
print(a[idx])

import numpy as np
c = np.array([1, 2, 3, 4, 5, 6])
print(c[0])
print(c[5])
print(c[-1])
print(c[-2])
print(c[-6])
print(c[0:2])
print(c[2:5])
print(c[:2])
print(c[4:])
print(c[-2:])
print(c[:2], c[2:])
print(c[:4], c[4:])
print(np.arange(5, 10))

#matrix
import numpy as np

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
x = np.array([1, 2, 3, 4])
b = np.array([0, 0, 0, 0])
n = 4
for i in range(0, n):
    val = 0.0
    for j in range(0, n):
        val += A[i, j] * x[j]
    b[i] = val

print("calculate = ", b)

b = np.dot(A, x)
print("dot = ", b)

b = np.matmul(A, x)
print("matmul = ", b)

b = A @ x
print("A @ x = ", b)

b = A * x
print(" A * x = ", b)

import numpy as np
A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
b = np.array([14, 19, 26, 44])

print("det = ", np.linalg.det(A))

x = np.linalg.solve(A, b)
print("solve = ", x)

x = np.dot(np.linalg.inv(A), b)
print("inverse1 = ", x)

tmp_b = np.dot(A.T, b)
tmp_T = np.dot(A.T, A)
tmp_inv = np.linalg.inv(np.dot(A.T, A))
x = np.dot(tmp_inv, tmp_b)
print("inverse2 = ", x)