import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 데이터 불러오기
df = fdr.DataReader('005930', '2020-05-04', '2024-04-18')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

# 데이터 준비
x = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10
data_x, data_y = [], []
for i in range(len(y) - window_size):
    data_x.append(x[i:i+window_size])
    data_y.append(y[i+window_size])

train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)

train_x = np.array(data_x[0:train_size])
train_y = np.array(data_y[0:train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])
test_x = np.array(data_x[train_size+val_size:])
test_y = np.array(data_y[train_size+val_size:])

print('훈련 데이터:', train_x.shape, train_y.shape)
print('검증 데이터:', val_x.shape, val_y.shape)
print('테스트 데이터:', test_x.shape, test_y.shape)

# ---------- RNN 모델 ----------
model_rnn = Sequential()
model_rnn.add(SimpleRNN(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False))
model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(1))
model_rnn.compile(loss='mean_squared_error', optimizer='adam')
model_rnn.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(val_x, val_y), verbose=1)

pred_rnn = model_rnn.predict(test_x)

# ---------- GRU 모델 ----------
model_gru = Sequential()
model_gru.add(GRU(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False))
model_gru.add(Dropout(0.2))
model_gru.add(Dense(1))
model_gru.compile(loss='mean_squared_error', optimizer='adam')
model_gru.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(val_x, val_y), verbose=1)

pred_gru = model_gru.predict(test_x)

# ---------- LSTM 모델 ----------
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(val_x, val_y), verbose=1)

pred_lstm = model_lstm.predict(test_x)

# ---------- 그래프 그리기 ----------
plt.figure(figsize=(10,6))
plt.plot(test_y, color='red', label='Actual')                   # 실제값
plt.plot(pred_rnn, color='blue', label='predicted (rnn)')        # RNN 예측
plt.plot(pred_gru, color='orange', label='predicted (gru)')      # GRU 예측
plt.plot(pred_lstm, color='green', label='predicted (lstm)')     # LSTM 예측
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.grid(True)
plt.savefig('SEC stock price prediction.jpg')  
plt.show()
