import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

# 두 데이터를 리스트 형태로 저장
x = dfx.values.tolist() # open, high, log, volume, 데이터
y = dfy.values.tolist() # close 데이터
#ex) 1월 1일 ~ 1월 10일까지의 OHLV 데이터로 1월 11일 종가 (Close) 예측
#ex) 1월 2일 ~ 1월 11일까지의 OHLV 데이터로 1월 12일 종가 (Close) 예측
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]
    # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])

test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])

print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)

model = Sequential()
model.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(val_x, val_y), verbose=1)

# 예측
pred_y = model.predict(test_x)

# 실제 종가와 예측 종가를 그래프로 비교
plt.figure(figsize=(12, 6))
plt.plot(test_y, label='Actual Close Price')         # 실제 종가
plt.plot(pred_y, label='Predicted Close Price')      # 예측 종가
plt.xlabel('Days')
plt.ylabel('Normalized Close Price')
plt.title('Samsung Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.show()