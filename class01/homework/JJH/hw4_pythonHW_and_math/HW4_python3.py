import numpy as np
import cv2

# 임의의 이미지 파일을 불러온다.
img = cv2.imread('your_image.png')

# 이미지 파일의 차원을 (Batch, Height, Width, Channel)로 확장
img_expanded = np.expand_dims(img, axis=0)
print("Expanded shape:", img_expanded.shape)

# 차원의 순서를 (Batch, Width, Height, Channel)에서 (Batch, Channel, Width, Height)로 변경
img_transposed = np.transpose(img_expanded, (0, 3, 2, 1))
print("Transposed shape:", img_transposed.shape)