import numpy as np

# 실습 1의 결과를 배열로 변환
data_array = np.array(data_tuple)

# 1차원 형태로 변환
flattened_array = data_array.reshape(-1)
print(flattened_array)