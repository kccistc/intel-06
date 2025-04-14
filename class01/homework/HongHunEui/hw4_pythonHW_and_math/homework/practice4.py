import cv2
import numpy as np

img = cv2.imread("image.jpg")

img1 = np.expand_dims(img, 0)
img2 = np.transpose(img1, (0, 3, 1, 2))
print(f"Image shape: {img.shape}")
print(f"Image shape after expand_dims: {img1.shape}")
print(f"Transposed image shape: {img2.shape}")
cv2.imshow("Image", img)