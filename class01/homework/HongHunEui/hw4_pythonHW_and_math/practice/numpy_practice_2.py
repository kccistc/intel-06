import numpy as np
a = np.array([[1], [2], [3], [4]])
print(a)
print(a.T)
print(a.T.reshape(-1,4))
print(a.shape)
print(a.T.reshape(-1,4).shape)


a = np.array([[1, 2, 3, 4]])
b = a.reshape(-1, 4)
print(a)
print(a.reshape(2, -1))
print(a.shape, ",", b.shape, ",", np.array([[1,2,3,4]]).shape)

########################################################################

a = np.array([1, 2, 3, 4, 5, 6])
print(a.reshape(3, 2))
print(a.shape)
b= a.reshape(3, -1)
print(b)
print(b.shape)
c= a.reshape(-1, 2)
print(c)
print(c.shape)


########################################################################

a = np.array([1, 2, 3 ,4])
print(a)
print(a.T)
b = a.reshape(4, -1)
print(b.shape)
print(b)
print(b.T.shape)