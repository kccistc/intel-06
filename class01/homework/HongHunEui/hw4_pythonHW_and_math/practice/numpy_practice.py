import numpy as np
a = np.array([1, 2, 3, 4])
print(a)
print(a + a)

b = [1, 2, 3, 4]
print(b)
print(b + b)

a = np.array([[1, 2], [3, 4]])
print(a)

a = np.array([[1, 2], [3, 4]], [[1, 2], [3, 4]])
print(a)

a = np.array([[1, 2, 3, 4]])
b = np.array([[1], [2], [3], [4]])
print(a)
print(a.shape)
print(b)
print(b.shape)
