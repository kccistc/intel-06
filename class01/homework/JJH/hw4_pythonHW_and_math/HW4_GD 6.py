import numpy as np
import matplotlib.pyplot as plt




def loss(w, x, y):
    # 예시: 간단한 이차 손실 함수
    return np.sum((x.dot(w) - y) ** 2) / len(y)

def loss_grad(w, x, y):
    # 예시: 손실 함수의 그라디언트
    return 2 * x.T.dot(x.dot(w) - y) / len(y)

def generate_batches(batch_size, x, y):
    # 배치 생성기
    for i in range(0, len(y), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

# 예시 데이터
x_train = np.random.rand(100, 2)
y_train = x_train @ np.array([2.0, -3.0]) + np.random.randn(100) * 0.1


# 설정
batch_size = 10
lr = 0.01
max_epochs = 51
alpha = 0.9

# GD
w0 = np.array([4.0, -1.0])
paths_gd = []

for epoch in range(max_epochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))

    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        paths_gd.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        w0 = w0 - lr * grad

# Momentum
w0 = np.array([4.0, -1.0])
paths_mm = []
velocity = np.zeros_like(w0)

for epoch in range(max_epochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))

    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        paths_mm.append(w0)
        grad = loss_grad(w0, x_batch, y_batch)
        velocity = alpha * velocity - lr * grad
        w0 = w0 + velocity

# 시각화
w0 = np.linspace(-2, 5, 101)
w1 = np.linspace(-2, 5, 101)
w0, w1 = np.meshgrid(w0, w1)
loss_w = w0 + 0

for i in range(w0.shape[0]):
    for j in range(w0.shape[1]):
        wij = np.array([w0[i, j], w1[i, j]])
        loss_w[i, j] = loss(wij, x_train, y_train)

fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(w0, w1, loss_w, cmap=plt.cm.jet, levels=np.linspace(0, max(loss_w.flatten()), 20))

# GD 경로
paths = np.array(np.matrix(paths_gd).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1], paths[1, 1:] - paths[1, :-1], scale_units='xy', angles='xy', scale=1, color='k')

# Momentum 경로
paths = np.array(np.matrix(paths_mm).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1], paths[1, 1:] - paths[1, :-1], scale_units='xy', angles='xy', scale=1, color='r')

plt.legend(['GD', 'Momentum'])
plt.show()
