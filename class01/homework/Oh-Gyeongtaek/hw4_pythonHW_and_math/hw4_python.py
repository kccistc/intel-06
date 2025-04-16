original_tuple = ('A', 'B')
temp_list = list(original_tuple)
temp_list.append('C')
new_tuple = tuple(temp_list)

import numpy as np

# def create_number_square_numpy(n):
#     square = np.arange(1, n*n + 1).reshape(n, n)
    
#     for row in square:
#         print(' '.join(map(str, row)))

# n = int(input("정수 n을 입력하세요: "))
# create_number_square_numpy(n)

nn = int(input())
A = np.zeros((nn, nn))
for i in range(nn):
    for j in range(nn):
        A[i, j] = i*nn + j + 1
print(A)

B = A.reshape(-1)
print(B)

import numpy as np
import cv2

img_path = "class01/homework/Oh-Gyeongtaek/hw4_pythonHW_and_math/image.jpg"

img = cv2.imread(img_path)
img_batch = np.expand_dims(img, 0)
img_transposed = np.transpose(img_batch, (0, 3, 2, 1))

print("\n모든 변환 결과:")
print("원본 이미지:", img.shape)
print("배치 차원 추가:", img_batch.shape)
print("차원 순서 변경:", img_transposed.shape)

# from PIL import Image
# img_path = "class01/homework/Oh-Gyeongtaek/hw4_pythonHW_and_math/image.jpg"
# img = np.array(Image.open(img_path))