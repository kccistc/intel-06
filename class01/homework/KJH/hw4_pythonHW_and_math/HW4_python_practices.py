# a = [1, 3, 4]
# print(a)
# a[0] = 9
# print(a)

# b = [1, 3, 'string']
# print(b)
# b.append(6.24)
# print(b)

# print(2*a)
# print(b*2)
# c = [a[i] + a[i] for i in range(len(a))]
# print(c)

# a = (1, 2, 3)
# print(a)
# b = (1, 3, 'string')
# print(b)

# a[0] = 2
# a.append(4)

# info = {'A' : 2.3, 'B' : 'C', 5 : 'D'}
# print(info)

# thisdict = {
#     "brand": "Ford",
#     "model": "Mustang",
#     "year": 1964
# }
# print(thisdict["brand"])
# print(type(thisdict))

# thisdict = dict(name = "John", age = 36, \
#                 country = "Norway")
# print(thisdict)
# print(type(thisdict))

# 반복문 (for loop)
items = [1, 2, 3, 4, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print('----------------')
for item in items:
    print(item)
print('----------------')
items = [[1,2], [3,4], [5,6]]
for item in items:
    print(item[0], item[1])
print('----------------')
for item1, item2 in items:
    print(item1, item2)
print('----------------')
info = {'A' : 1, 'B' : 2, 'C' : 3}
for key in info:
    print(key, info[key])
print('----------------')
for key, value in info.items():
    print(key, value)

items1 = [[1,2], [3,4], [5,6]]
items2 = [['A', 'B'], ['C','D'], ['E','F']]
print(items1)
print(items2)
print('----------------')
for digits, charaters in zip(items1, items2):
    print(digits, charaters)

a = []
for k in range(0,5):
    a.append(k)
print(a)
print('----------------')
a = [k for k in range(0,5)]
print(a)
print('----------------')
a = [k if (k+1)%2 else k*5+1 for k in range(0,5)]
print(a)
print('----------------')
a = [k for k in range(0,5) if k % 2 == 0]
print(a)
print('----------------')
a = {k : k*10 for k in range(0,5)}
print(a)
print('----------------')
a = [1, 3, 4]
c = [a[i] + a[i] for i in range(len(a))]
print(c)

# Numpy 기본 실습
import numpy as np
a = np.array([1,2,3,4])
print(a)
print(a + a)
b = [1,2,3,4]
print(b + b)

a = np.array([[1,2], [3,4]])
print(a)

a = np.array([[[1,2], [3,4]], [[1,2], [3,4]]])
print(a)

a = np.array([1,2,3,4])
b = np.array([[1], [2], [3], [4]])
print(a)
print(a.shape)
print(b)
print(b.shape)

import numpy as np
from numpy import linalg as LA
c = np.array([[1,2,3], [-1,1,4]])
print(LA.norm(c, axis=0))
print(LA.norm(c, axis=1))
print(LA.norm(c, ord=1, axis=1))
print(LA.norm(c, ord=2, axis=1))

import numpy as np
a = np.array([[1], [2], [3], [4]])
print(a) #shape = (4,1)
print(a.T) #shape = (1,4)
print(a.T.reshape(-1,4))
print(a.shape)
print(a.T.reshape(-1,4).T.shape)

a = np.array([1,2,3,4])
b = a.reshape(4,-1)
print(a)
print(a.reshape(2, -1))
print(a.shape,",", b.shape, ",", np.array([[1,2,3,4]]).shape)

a = np.array([1,2,3,4,5,6])
print(a.reshape(3,2))
print(a.shape)
b = a.reshape(3,-1)
print(b)
print(b.shape)
c = a.reshape(-1,2)
print(c)
print(c.shape)

a = np.array([1,2,3,4])
print(a)
print(a.T)
b = a.reshape(4, -1)
print(b.shape)
print(b)
print(b.T.shape)



# 배열 인덱싱

a = np.array([10,20,30,40,50,60])
print(a)
b = a[[4,2,0]]
print(b)
idx = np.arrange(0, len(a))
print(idx)
np.random.shuffle(idx)
print(idx)
print(a[idx])

import numpy as np
c = np.array{[1,2,3,4,5,6]}
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
print(np.arange(5,10))

word = 'Python'
print(word[0])
print(word[5])
print(word[-1])
print(word[-2])
print(word[-6])
print(word[0:2])
print(word[2:5])
print(word[:2])
print(word[4:])
print(word[-2:])
print(word[:2] + word[2:])
print(word[:4] + word[4:])

# Matrix 실습

import numpy as np
A = np.array([[1, 4, 2, 0],
              [9, 5, 0, 0],
              [4, 0, 2, 4],
              [6, 1, 8, 3]])

x = np.array([1, 2, 3, 4])
b = np.array([0, 0, 0, 0])
n = 4

for i in range(0, n):
    val = 0.0
    for j in range(n):
        # TODO 2
        val += A[i, j] * x[j]
    b[i] = val

print("calculate=", b)

b = np.dot(A, x)
print("dot=", b)

b = np.matmul(A, x)
print("matmul=", b)

b = A @ x
print("A@x=", b)

b = A * x
print("A*x=", b)

b = A + x
print("A+x=", b)



import numpy as np
A = np.array([[1,4,2,0], [9,5,0,0], [4,0,2,4], [6,1,8,3]])
b = np.array([15,19,26,44])

print("det=",np.linalg.det(A))

x = np.linalg.solve(A,b)
print("solver =", x)

x = np.dot(np.linalg.inv(A),b)
print("inverse1 = ",x)

tmp_b=np.dot(A.T,b)
tmp_T=np.dot(A.T,A)
tmp_inv=np.linalg.inv(np.dot(A.T,A))
x = np.dot(tmp_inv,tmp_b)
print("inverse2 =",x)