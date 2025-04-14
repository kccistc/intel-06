# 튜플 생성
A = 5
B = 10
data_tuple = (A, B)

# 튜플에 요소 추가 (튜플은 불변이므로 리스트로 변환 후 추가)
C = 15
data_list = list(data_tuple)
data_list.append(C)
data_tuple = tuple(data_list)

print(data_tuple)


def print_number_square(n):
    matrix = np.arange(1, n*n+1).reshape(n, n)
    for row in matrix:
        print(" ".join(map(str, row)))

n = int(input("Enter the size of the square: "))
print_number_square(n)



import numpy as np

# 실습 1의 결과를 배열로 변환
data_array = np.array(data_tuple)

# 1차원 형태로 변환
flattened_array = data_array.reshape(-1)
print(flattened_array)


import numpy as np
import cv2

# 임의의 이미지 파일을 불러온다.
img = cv2.imread("your_image.png")

# 이미지 파일의 차원을 (Batch, Height, Width, Channel)로 확장
img_expanded = np.expand_dims(img, axis=0)
print("Expanded shape:", img_expanded.shape)

# 차원의 순서를 (Batch, Width, Height, Channel)에서 (Batch, Channel, Width, Height)로 변경
img_transposed = np.transpose(img_expanded, (0, 3, 2, 1))
print("Transposed shape:", img_transposed.shape)



import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('your_image.png')

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