
data = ('A', 'B')


data = data + ('C',)

print(data) 

n = int(input("정수 n을 입력하세요: "))


num = 1
for i in range(n):
    for j in range(n):
        print(f"{num:2}", end=' ')
        num += 1
    print()

    import numpy as np


array_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10,11,12],
    [13,14,15,16]
])

# reshape 1차원 배열로
array_1d = array_2d.reshape(-1)

print(array_1d)

import cv2

import numpy as np

img = cv2.imread('Lenna.jpg')

img1 = np.expand_dims(img, 0)
img1_1 = np.squeeze(img1)
img2 = np.transpose(img1, (0, 3, 2, 1))


print("original", img.shape)
print("expand",img1.shape)
print("squeeze",img1_1.shape)
