import numpy as np
import cv2

img = cv2.imread("프사.jpg")

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

print(img.shape)

cv2.imshow("image",img)

cv2.waitKey(0)

cv2.imwrite("output.png", img)

cv2.destroyAllWindows()

color = cv2.imread("berry.jpg", cv2.IMREAD_COLOR)
print(color.shape)

H,W,C = color.shape
cv2.imshow("Original Image", color)

b,g,r = cv2.split(color)
rgb_split = np.concatenate((b,g,r),axis=1)
cv2.imshow("BGR Channels",rgb_split)

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v),axis=1)
cv2.imshow("HSV Channels",hsv_split)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("lena.jpg")

cropped = img[50:450, 100:400]

resized = cv2.resize(cropped,(400,200))

cv2.imshow("Original", img)
cv2.imshow("Cropped", cropped)
cv2.imshow("Resized", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

dst = cv2.bitwise_not(color)

cv2.imshow("color",color)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
src = cv2.imread("berry.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
canny = cv2.Canny(gray, 50,150)
cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()