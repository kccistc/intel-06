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