import numpy as np
import cv2

img = cv2.imread('pxfuel.jpg')
print(img.shape)
img = np.expand_dims(img, axis=0)
img = np.transpose(img,(0,3,1,2))
print(img.shape)