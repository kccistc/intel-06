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