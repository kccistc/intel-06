import numpy as np
import cv2

img = cv2.imread('pxfuel.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])

pad_img = np.pad(img,((1,1),(1,1)),'reflect')
i_x, i_y = pad_img.shape
k_x, k_y = kernel.shape
result = []
for i in range(i_x-k_x+1):
    for j in range(i_y-k_y+1):
        result.append((pad_img[i:i+k_x,j:j+k_y]*kernel).sum())
result=np.array(result).reshape(i_x-k_x+1,i_y-k_y+1)

print(kernel)
output = cv2.filter2D(img, -1, kernel)
print(result)
print(output)
cv2.imshow('edge', output)
cv2.waitKey(0)