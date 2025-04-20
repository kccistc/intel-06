# example 1
A = ("A", "B")
B = A + ("C", )

print(B, type(B))

# example 2
import numpy as np

input_x = int(input())
A = np.zeros((input_x, input_x))
for _i in range(input_x):
    for _j in range(input_x):
        A[_i][_j] = _i * input_x + _j + 1

print(A)

# example 3
import numpy as np

input_x = int(input())
A = np.zeros((input_x, input_x))
for _i in range(input_x):
    for _j in range(input_x):
        A[_i][_j] = _i * input_x + _j + 1

print(A)

B = A.reshape(-1)
print(B)

# example 4
import cv2
import numpy as np

img = cv2.imread("lena.png")
img1 = np.expand_dims(img, 0)
img1_1 = np.squeeze(img1)
img2 = np.transpose(img1, (0, 3, 1, 2))

print("original : ", img.shape)
print("expand : ", img1.shape)
print("squeeze : ", img1_1.shape)
print("transpose : ", img2.shape)

#example 5
import cv2
import numpy as np

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
print(kernel)

cv2.imshow('before', img)

# 직접 구현해보기 - 외곽선 검출
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)
# ^ not solved yet