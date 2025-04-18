import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 데이터 로드 및 정규화
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]

x = dfx.values
y = dfy.values

# 시퀀스 데이터 만들기
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)

data_x = np.array(data_x)
data_y = np.array(data_y)

# train/val/test split
train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)

train_x = data_x[:train_size]
train_y = data_y[:train_size]
val_x = data_x[train_size:train_size + val_size]
val_y = data_y[train_size:train_size + val_size]
test_x = data_x[train_size + val_size:]
test_y = data_y[train_size + val_size:]

print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)

# 예측값 저장 변수
predicted_rnn = None
predicted_gru = None
predicted_lstm = None

# RNN 모델
model = Sequential([
    SimpleRNN(20, activation='tanh', return_sequences=True, input_shape=(10, 4)),
    Dropout(0.1),
    SimpleRNN(20, activation='tanh'),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)
predicted_rnn = model.predict(test_x)

# GRU 모델
model = Sequential([
    GRU(20, activation='tanh', return_sequences=True, input_shape=(10, 4)),
    Dropout(0.1),
    GRU(20, activation='tanh'),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)
predicted_gru = model.predict(test_x)

# LSTM 모델
model = Sequential([
    LSTM(20, activation='tanh', return_sequences=True, input_shape=(10, 4)),
    Dropout(0.1),
    LSTM(20, activation='tanh'),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)
predicted_lstm = model.predict(test_x)

# 그래프 시각화
plt.plot(test_y, label='Actual', color='red')
plt.plot(predicted_rnn, label='predicted (rnn)', color='blue')
plt.plot(predicted_gru, label='predicted (gru)', color='yellow')
plt.plot(predicted_lstm, label='predicted (lstm)', color='green')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
