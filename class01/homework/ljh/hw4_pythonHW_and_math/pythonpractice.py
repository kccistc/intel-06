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

thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": "1964"
}
print(thisdict["brand"])

thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": "1964"
}
print(thisdict["brand"])
print(type(thisdict))

thisdict = dict(name = "John", age = 36, country = "Norway")
print(thisdict)
print(type(thisdict))

items = [1, 2, 3, 4, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print('------------------')
for item in items:
    print(item)
print('------------')
items = [[1,2], [3,4], [5,6]]
for item in items:
    print(item[0], item[1])
print('------------')
for item1, item2 in items:
    print(item1, item2) 
print('-----------')
info = {'A' : 1, 'B' : 2, 'C' : 3}
for key in info:
    print(key, info[key])
print('-----------')
for key, value in info.items():
    print(key,value)   


import numpy as np
a = np.array([1,2,3,4])
print(a)
print(a + a)

b = [1,2,3,4]
print(b + b)


a = np.array([[1,2],[3,4]])
print(a)

print('-----------------')
a = np.array([[[1,2],[3,4]], [[1,2],[3,4]]])
print(a)

print('--------------')

a = np.array([1,2,3,4])
b = np.array([[1],[2],[3],[4]])
print(a)
print(a.shape)
print(b)
print(b.shape)

print('----------')

a = np.array([[1],[2],[3],[4]])
print(a) 
print(a.T)
print(a.T.reshape(-1,4))
print(a.shape)
print(a.T.reshape(-1,4).T.shape)

print('--------')

a = np.array([[1],[2],[3],[4]])
b = a.reshape(4,-1)
print(a)
print(a.reshape(4,-1))
print(a.shape,",",b.shape,",",np.array([[1,2,3,4]]).shape)

print('------------')

a = np.array([1,2,3,4,5,6])
print(a.reshape(3,2))
print(a.shape)
b=a.reshape(3,-1)
print(b)
print(b.shape)
c=a.reshape(-1,2)
print(c)
print(c.shape)

print('---------numpymartix---')

A = np.array([[1,4,2,0], [9,5,0,0], [4,0,2,4], [6, 1, 8, 3]])
x = np.array([1,2,3,4])
b = np.array([0,0,0,0])
n = 4 
for i in range(0,n):
    val = 0.0
    for j in range(0,n):
        val += A[i,j] * x[j]
    b[i] = val

print("calculate=", b)

b = np.dot(A,x)
print("dot=",b)

b = np.matmul(A,x)
print("matmul=",b)

b = A@x
print("A@x=",b)

b = A*x
print("A+x=",b)
