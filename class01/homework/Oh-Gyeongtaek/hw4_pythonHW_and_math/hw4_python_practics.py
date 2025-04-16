thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
}
print(thisdict)
print(type(thisdict))
print(thisdict["brand"])

thisdict = dict(name = "John", age = 36, country = "Norway")
print(thisdict)
print(type(thisdict))

info = {'A': 1, 'B': 2, 'C': 3}
for key in info:
    print(key, info[key])

print("====================================")

for key, value in info.items():
    print(key, value)

print("====================================")

import numpy as np
a = np.array([1,2,3,4])
print(a)
print(a+a)

print("====================================")

a = np.array([[[1,2],[3,4]],[[1,2],[3,4]]])
print(a)

print("====================================")

a = np.array([1,2,3,4])
b = np.array([[1],[2],[3],[4]])
print(a)
print(a.shape)
print(b)
print(b.shape)

print("====================================")

import numpy as np
from numpy import linalg as LA
c = np.array([[1,2,3],[-1,1,4]])
print(LA.norm(c, axis=0))
print(LA.norm(c, axis=1))
print(LA.norm(c, ord=1, axis=1))
print(LA.norm(c, ord=2, axis=1))

print("====================================")

# 전치연산(Transpose)
a = np.array([[1], [2], [3], [4]])
print(a)
print(a.T)
print(a.T.reshape(-1, 4))
print(a.shape)
print(a.T.reshape(-1, 4).T.shape)

print("====================================")

a = np.array([1, 2, 3, 4])
b = a.reshape(4, -1)
print(a)
print(a.reshape(2, -1))
print(a.shape, ",", b.shape, ",", np.array([[1, 2, 3, 4]]).shape)

print("====================================")

# Reshape
a = np.array([1, 2, 3, 4, 5, 6])
print(a.reshape(3, 2))
print(a.shape)
b = a.reshape(3, -1)
print(b)
print(b.shape)
c = a.reshape(-1, 2)
print(c)
print(c.shape)

print("====================================")

a = np.array([1, 2, 3, 4])
print(a)
print(a.T)
b = a.reshape(4, -1)
print(b.shape)
print(b)
print(b.T.shape)

print("====================================")

word = 'Python'
print(word[0])     # character P
print(word[5])     # character n
print(word[-1])    # last character n
print(word[-2])    # second-last character o
print(word[-6])    # 'P'
print(word[0:2])   # characters from position 0 (included) to 2 (excluded) - Py
print(word[2:5])   # characters from position 2 (included) to 5 (excluded) - tho
print(word[:2])    # characters from the beginning to position 2 (excluded) - Py
print(word[4:])    # characters from position 4 (included) to the end - on
print(word[-2:])   # characters from the second-last (included) to the end - on
print(word[:2] + word[2:])
print(word[:4] + word[4:])

print("====================================")

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
x = np.array([1, 2, 3, 4])
b = np.array([0, 0, 0, 0])
n = 4
for i in range(0, n):
    val = 0.0
    for j in range(0, n):
        val += A[i, j] * x[j]
    b[i] = val

print("calculate=", b)

b = np.dot(A, x)
print("dot=", b)

b = np.matmul(A, x)
print("matmul=", b)

b = A@x
print("A@x=", b)

b = A*x
print("A*x=", b)