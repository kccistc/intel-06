# for loop & Default

# list
a = [1, 3, 4]
print(a)
a[0] = 9
print(a)

b = [1, 3, 'string']
print(b)
b.append(6.24)
print(b)

print(2*a)
print(b*2)
c = [a[i] + a[i] for i in range(len(a))]
print(c)

# tuple 
a = (1, 2, 3)
print(a)
b = (1, 3, 'string')
print(b)

# a[0] = 2      <-- error!
# a.append(4)   <-- error!

info = {'A' : 2.3, 'B' : 'C', 5 : 'D'}
print(info)

info['A'] = 5.2
print(info)

info['Hello'] = [1, 2, 3, 4, 'World']
print(info)

thisdict = {
    "brand" : "Ford",
    "model" : "Mustang",
    "year"  : 1964
}
print(thisdict["brand"])
print(type(thisdict))

items = [1, 2, 3, 4, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print('-----------------------------')
for item in items:
    print(item)
print('-----------------------------')
items = [[1, 2], [3, 4], [5, 6]]
for item in items:
    print(item[0], item[1])
print('-----------------------------')
for item1, item2 in items:
    print(item1, item2)
print('-----------------------------')
info = {
    'A' : 1, 
    'B' : 2,
    'C' : 3
}
for key in info:
    print(key, info[key])
for key, value in info.items():
    print(key, value)

# numpy practice & Indexing & Matrix
import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
print(a + a)

b = [1, 2, 3, 4]
print(b + b)

a = np.array([[1, 2], [3, 4]])
print(a)

a = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
print(a)

a = np.array([1, 2, 3, 4])
b = np.array([[1], [2], [3], [4]])
print(a)
print(a.shape)
print(b)
print(b.shape)

a = np.array([[1], [2], [3], [4]])
print(a.shape) # shape
print(a.T) 
print(a.T.reshape(-1, 4).T.shape)

a = np.array([1, 2, 3, 4])
b = a.reshape(4, -1)
print(a)
print(a.reshape(2, -1))
print(a.shape, ",", b.shape, ",", np.array([1, 2, 3, 4]).shape)
      
# matrix
A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
x = np.array([1, 2, 3, 4])
b = np.array([0, 0, 0, 0])
n = 4
for i in range(0, n):
    val = 0.0
    for j in range(0, n):
        # todo 2
        val += A[i, j] * x[j]
    b[i] = val

print("calculate = ", b)

b = np.dot(A, x)
print("dot : ", b)

b = np.matmul(A, x)
print("matmul = ", b)

b = A@x
print("A@x : ", b)

b = A*x
print("A*x : ", b)

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
B = np.array([15, 19, 26, 44])

print("det = ", np.linalg.det(A))
x = np.linalg.solve(A, B)
print("solver = ", x)

x = np.dot(np.linalg.inv(A), B)
print("inverse1 = ", x)

tmp_b = np.dot(A.T, B)
tmp_T = np.dot(A.T, A)
tmp_inv = np.linalg.inv(np.dot(A.T, A))
x = np.dot(tmp_inv, tmp_b)
print("inverse2 = ", x)
