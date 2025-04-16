#HW 1
X = ('A', 'B')
X = X + ('C',)
print(X)

#HW 2
import numpy as np

n = int(input())

A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[i][j] = i*n +j + 1
print(A)

#HW 3
import numpy as np

n = int(input())

A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[i][j] = i*n +j + 1
B = A.reshape(n*n)
print(B)

#HW 4
import cv2
import numpy as np

img  = cv2.imread("lena.png")

print("original", img.shape)
img1 = np.expand_dims(img, 0) #모델에 집어넣을때 차원이 안맞으면 못넣기 때문에 제일 앞에 집어넣기 위해 0이라고 쓴 것
print("expand", img1.shape)
img1_1 = np.squeeze(img1) #차원에서 하나를 빼는 경우
print("squeeze", img1_1.shape)
img2 = np.transpose(img1, (0, 3, 1, 2))
print("transpose", img2.shape)