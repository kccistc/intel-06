import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('lena.png')

# 붉은색 네모 박스 그리기
start_point = (50, 50)
end_point = (200, 200)
color = (0, 0, 255)  # BGR 형식에서 빨간색
thickness = 2
img_with_box = cv2.rectangle(img, start_point, end_point, color, thickness)

# 결과 출력
cv2.imshow('Red Box', img_with_box)
cv2.waitKey(0)
cv2.destroyAllWindows()