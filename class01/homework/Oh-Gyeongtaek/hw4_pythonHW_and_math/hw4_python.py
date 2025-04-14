# original_tuple = ('A', 'B')
# temp_list = list(original_tuple)
# temp_list.append('C')
# new_tuple = tuple(temp_list)

# import numpy as np

# def create_number_square_numpy(n):
#     square = np.arange(1, n*n + 1).reshape(n, n)
    
#     for row in square:
#         print(' '.join(map(str, row)))

# n = int(input("정수 n을 입력하세요: "))
# create_number_square_numpy(n)


# def create_and_reshape_square(n):
#     square = np.arange(1, n*n + 1).reshape(n, n)
    
#     print("n x n 사각형:")
#     for row in square:
#         print(' '.join(map(str, row)))
    
#     flattened = square.reshape(-1)
    
#     print("\n1차원 배열:")
#     print(flattened)
    
#     return flattened

# n = int(input("정수 n을 입력하세요: "))
# create_and_reshape_square(n)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "class01/homework/Oh-Gyeongtaek/image.jpg"
img = np.array(Image.open(img_path))

plt.figure(figsize=(8, 6))
plt.subplot(1, 3, 1)

img_batch = np.expand_dims(img, axis=0)

plt.subplot(1, 3, 2)

img_transposed = np.transpose(img_batch, (0, 3, 2, 1))

img_back = np.transpose(img_transposed, (0, 3, 2, 1))[0]
plt.subplot(1, 3, 3)

print("\n모든 변환 결과:")
print("원본 이미지:", img.shape)
print("배치 차원 추가:", img_batch.shape)
print("차원 순서 변경:", img_transposed.shape)