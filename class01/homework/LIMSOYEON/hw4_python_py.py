# -*- coding: utf-8 -*-
"""hw4_python.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1K77QIl5VuNfGz16I4mSxz97wyEZ9O-wZ
"""

t1=('A','B')
t2=('C',)
t3=t1+t2
print(t3)

import numpy as np
nn= int(input())
A= np.zeros((nn,nn))
for ii in range(nn):
  for jj in range(nn):
    A[ii][jj] = ii*nn+jj+1
print(A)
B = A.reshape(-1,)
print(B)

import cv2
import numpy as np
img = cv2.imread("Lenna.png")

img1 = np.expand_dims(img, 0)
img1_1 = np.squeeze(img1)
img2 = np.transpose(img1, (0,3,1 ,2))

print("original ",img.shape)
print("expand",img1.shape)
print("squeeze",img1_1.shape)
print("transpose",img2.shape)