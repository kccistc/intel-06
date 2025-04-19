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
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]


# 두 데이터를 리스트 형태로 저장
x = dfx.values.tolist()  # open, high, log, volume 데이터
y = dfy.values.tolist()  # close 데이터

#ex) 1월 1일 ~ 1월 10일까지의 OHLV 데이터로 1월 11일 종가 (Close) 예측
#ex) 1월 2일 ~ 1월 11일까지의 OHLV 데이터로 1월 12일 종가 (Close) 예측
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size]  # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]      # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

# numpy 배열로 변환
data_x = np.array(data_x)
data_y = np.array(data_y)

# 데이터 차원 확인
print("전체 데이터의 크기: ", data_x.shape, data_y.shape)

# 데이터 분할 (훈련: 70%, 검증: 20%, 테스트: 10%)
train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
test_size = len(data_y) - train_size - val_size

# 훈련 데이터
train_x = data_x[:train_size]
train_y = data_y[:train_size]

# 검증 데이터
val_x = data_x[train_size:train_size+val_size]
val_y = data_y[train_size:train_size+val_size]

# 테스트 데이터
test_x = data_x[train_size+val_size:]
test_y = data_y[train_size+val_size:]

print("훈련 데이터의 크기 :", train_x.shape, train_y.shape)
print("검증 데이터의 크기 :", val_x.shape, val_y.shape)
print("테스트 데이터의 크기 :", test_x.shape, test_y.shape)

# 데이터 형태 변환 (RNN 입력에 맞게)
# RNN 모델은 3D 텐서 입력을 기대합니다: [samples, time steps, features]
# 현재 데이터는 이미 [samples, time steps, features] 형태이므로 추가 변환이 필요 없습니다

# RNN 모델
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=20, 
                   activation='tanh',
                   return_sequences = True,
                   input_shape=(10, 4)))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(units=20, activation='tanh'))
rnn_model.add(Dropout(0.1))
rnn_model.add(Dense(units=1))
rnn_model.summary()

rnn_model.compile(optimizer='adam',
             loss='mean_squared_error')

rnn_history = rnn_model.fit(train_x, train_y,
                   validation_data = (val_x, val_y),
                   epochs=70, batch_size=30)

# GRU 모델
gru_model = Sequential()
gru_model.add(GRU(units=20, activation='tanh',
              return_sequences=True,
              input_shape=(10, 4)))
gru_model.add(Dropout(0.1))
gru_model.add(GRU(units=20, activation='tanh'))
gru_model.add(Dropout(0.1))
gru_model.add(Dense(units=1))
gru_model.summary()

gru_model.compile(optimizer='adam',
             loss='mean_squared_error')
gru_history = gru_model.fit(train_x, train_y,
                   validation_data = (val_x, val_y),
                   epochs=70, batch_size=30)

# LSTM 모델
lstm_model = Sequential()
lstm_model.add(LSTM(units=20, activation='tanh',
               return_sequences=True,
               input_shape=(10, 4)))
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(units=20, activation='tanh'))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(units=1))
lstm_model.summary()

lstm_model.compile(optimizer='adam',
             loss='mean_squared_error')
lstm_history = lstm_model.fit(train_x, train_y,
                   validation_data = (val_x, val_y),
                   epochs=70, batch_size=30)

# 예측
pred_rnn = rnn_model.predict(test_x)
pred_gru = gru_model.predict(test_x)
pred_lstm = lstm_model.predict(test_x)

# 실제 종가와 예측 종가를 그래프로 비교
plt.figure(figsize=(12, 6))
plt.plot(test_y, 'r-', label='Actual')  # 실제 종가 (빨간색)
plt.plot(pred_rnn, 'b-', label='predicted (rnn)')  # RNN 예측 종가 (파란색)
plt.plot(pred_gru, color='orange', label='predicted (gru)')  # GRU 예측 종가 (주황색)
plt.plot(pred_lstm, 'g-', label='predicted (lstm)')  # LSTM 예측 종가 (초록색)

plt.xlabel('time')
plt.ylabel('stock price')
plt.title('SEC stock price prediction')
plt.ylim(0.5, 1.0)  # y축 범위 설정
plt.legend(loc='upper left')  # 범례 위치 설정
plt.grid(True)
plt.show()

# # 손실 함수 그래프 그리기
# plt.figure(figsize=(12, 6))
# plt.plot(rnn_history.history['loss'], 'b-', label='RNN Train Loss')
# plt.plot(rnn_history.history['val_loss'], 'b--', label='RNN Validation Loss')
# plt.plot(gru_history.history['loss'], color='orange', label='GRU Train Loss')
# plt.plot(gru_history.history['val_loss'], '--', color='orange', label='GRU Validation Loss')
# plt.plot(lstm_history.history['loss'], 'g-', label='LSTM Train Loss')
# plt.plot(lstm_history.history['val_loss'], 'g--', label='LSTM Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()

# # 모델 저장하기
# rnn_model.save('rnn_stock_model.h5')
# gru_model.save('gru_stock_model.h5')
# lstm_model.save('lstm_stock_model.h5')