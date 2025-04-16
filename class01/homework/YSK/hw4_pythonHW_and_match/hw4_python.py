
# #1
# x = ('A','B')
# y = ('C',)   #콜론을 붙여 튜플 ('C') 이렇게만 하면 스트링
# x += y
# print(x,type(x))


# #2
# import numpy as np

# nn = int(input())
# A = np.zeros((nn,nn))  #데이터 공간할당 malloc과 같은 개념
# for ii in range(nn):
#   for jj in range(nn):
#     A[ii][jj] = ii*nn+jj+1
# print(A)

# #3
# B = A.reshape(-1,)
# print(B)

# #4
# import cv2 #이미지 읽는 라이브러리
# import numpy as np

# img = cv2.imread("Lenna.png")
# print("original" ,img.shape)
# img1 = np.expand_dims(img,0)   #batch 부터 추가 차원 동기화를 해주려고 차원추가를 사용함
# img1_1= np.squeeze(img)   #batch 부터 추가 차원 동기화를 차원을 빼줄수 있음
# img2 =np.transpose(img1,(0,3,1,2)) #데이터 값을 어떤순서 넣을ㅣ

# #반대는 스퀴즈 차원 하나를 뺀다 1은 무의미한 값이라
