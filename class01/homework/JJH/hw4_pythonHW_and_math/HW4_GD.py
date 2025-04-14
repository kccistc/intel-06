import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

def contour(f, x, y, level = np.logspace(0, 5, 35)):
	fig, ax = plt.subplots(figsize=(8, 8))

	ax.contour(x, y, f(x,y), levels=level, norm=LogNorm(), cmap=plt.cm.jet)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()

def contour_with_quiver(f, x, y, grad_x, grad_y, norm=LogNorm(), level = np.logspace(0, 5, 35),
	minima=None):
	dz_dx = grad_x(x,y)
	dz_dy = grad_y(x,y)

	fig, ax = plt.subplots(figsize=(6, 6))

	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	if minima is not None:
		ax.plot(*minima, 'r*', markersize=18)
	ax.quiver(x, y, -dz_dx, -dz_dy, alpha=.5)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()

def surf(f, x, y, norm=LogNorm(), minima=None):
	fig = plt.figure(figsize=(8, 5))
	ax = plt.axes(projection='3d', elev=50, azim=-50)

	ax.plot_surface(x, y, f(x,y), norm=norm, rstride=1, cstride=1,
	                edgecolor='none', alpha=.8, cmap=plt.cm.jet)

	if minima is not None:
		ax.plot(*minima, f(*minima), 'r*', markersize=10)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_zlabel('$z$')

	plt.show()

def contour_with_path(f, x, y, paths, norm=LogNorm(), level=np.logspace(0, 5, 35), minima=None):
	fig, ax = plt.subplots(figsize=(6, 6))

	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	ax.quiver(paths[0,:-1], paths[1,:-1], paths[0,1:]-paths[0,:-1], paths[1,1:]-paths[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
	if minima is not None:
		ax.plot(*minima, 'r*', markersize=18)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()
	

import numpy as np
import matplotlib.pyplot as plt

# 함수 정의
def f(x):
    return x ** 2 - 4 * x + 6

# Gradient Descent 함수
def gradient_descent(learning_rate, initial_x, num_iterations):
    x = initial_x
    for i in range(num_iterations):
        gradient = 2 * x - 4  # f'(x) = 2x - 4
        x = x - learning_rate * gradient
    return x

# 초기값과 학습률 설정
initial_x = 0
learning_rate = 0.1
num_iterations = 100

# 최소값 찾기
min_x = gradient_descent(learning_rate, initial_x, num_iterations)
min_f_x = f(min_x)

print(f"최소값 x: {min_x}, f(x): {min_f_x}")

# 그래프 그리기
NumberOfPoints = 101
x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)

plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x)')
plt.scatter(min_x, min_f_x, color='red')  # 최소값 표시
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 함수 정의
def f(x):
    return x ** 2 - 4 * x + 6

# Gradient Descent 함수
def gradient_descent(learning_rate, initial_x, num_iterations):
    x = initial_x
    for i in range(num_iterations):
        gradient = 2 * x - 4  # f'(x) = 2x - 4
        x = x - learning_rate * gradient
    return x

# 초기값과 학습률 설정
initial_x = 0
learning_rate = 0.1
num_iterations = 100

# 최소값 찾기
min_x = gradient_descent(learning_rate, initial_x, num_iterations)
min_f_x = f(min_x)

print(f"최소값 x: {min_x}, f(x): {min_f_x}")

# 그래프 그리기
NumberOfPoints = 101
x = np.linspace(-5, 5, NumberOfPoints)
fx = f(x)

plt.plot(x, fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x)')
plt.scatter(min_x, min_f_x, color='red')  # 최소값 표시
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 함수 정의
def f(x):
    return x ** 2 - 4 * x + 6

# 함수의 그래디언트 정의
def grad_fx(x):
    return 2 * x - 4

# Steepest Descent 알고리즘
def steepest_descent(func, grad_func, x0, learning_rate=0.01, max_iter=10, verbose=True):
    x = x0
    path = [x]
    for i in range(max_iter):
        x = x - learning_rate * grad_func(x)
        path.append(x)
        if verbose:
            print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {func(x):.4f}")
    return x, func(x), path

# 초기값과 학습률 설정
initial_x_1 = 0.0
learning_rate_1 = 1.2
initial_x_2 = 1.0
learning_rate_2 = 1.0
max_iter = 100

# 최소값 찾기
min_x_1, min_f_x_1, path_1 = steepest_descent(f, grad_fx, initial_x_1, learning_rate_1, max_iter)
min_x_2, min_f_x_2, path_2 = steepest_descent(f, grad_fx, initial_x_2, learning_rate_2, max_iter)

# 결과 출력
print(f"초기값 0, learning rate 1.2: 최소값 x = {min_x_1:.4f}, f(x) = {min_f_x_1:.4f}")
print(f"초기값 1, learning rate 1.0: 최소값 x = {min_x_2:.4f}, f(x) = {min_f_x_2:.4f}")

# 그래프 그리기
x = np.linspace(-1, 4, 400)
fx = f(x)

plt.figure(figsize=(12, 5))

# 첫 번째 그래프
plt.subplot(1, 2, 1)
plt.plot(x, fx, label='f(x)')
plt.plot(path_1, [f(x) for x in path_1], 'o-', label='Path')
plt.title('초기값 0, learning rate 1.2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(x, fx, label='f(x)')
plt.plot(path_2, [f(x) for x in path_2], 'o-', label='Path')
plt.title('초기값 1, learning rate 1.0')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 함수 정의
f = lambda x, y: (x - 2)**2 + (y - 2)**2

# 그래디언트 정의
grad_f_x = lambda x, y: 2 * (x - 2)
grad_f_y = lambda x, y: 2 * (y - 2)

# 2차원 Gradient Descent 알고리즘
def steepest_descent_twod(func, gradx, grady, x0, learning_rate=0.25, max_iter=10, verbose=True):
    x = x0
    path = [x]
    fval_path = [func(x[0], x[1])]
    
    for i in range(max_iter):
        grad = np.array([gradx(x[0], x[1]), grady(x[0], x[1])])
        x = x - learning_rate * grad
        path.append(x)
        fval_path.append(func(x[0], x[1]))
        
        if verbose:
            print(f"Iteration {i+1}: x = {x}, f(x) = {func(x[0], x[1]):.4f}")
    
    return x, func(x[0], x[1]), np.array(path), np.array(fval_path)

# 초기값 설정
x0 = np.array([-2.0, -2.0])

# 최소값 찾기
min_x, min_f_x, path, fval_path = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)

# 결과 출력
print(f"최소값: x = {min_x}, f(x) = {min_f_x:.4f}")

# 그래프 그리기
x = np.arange(-4.0, 4.0, 0.25)
y = np.arange(-4.0, 4.0, 0.25)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(10, 5))

# 등고선 그래프
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 35), norm=LogNorm(), cmap='jet')
plt.plot(path[:, 0], path[:, 1], 'o-', color='red')
plt.plot(2, 2, 'r*', markersize=15)  # 최소값 위치 표시
plt.title('Contour plot with Gradient Descent path')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(320)
x_train = np.linspace(-1, 1, 51)
f = lambda x: 0.5 * x + 1.0
y_train = f(x_train) + 0.4 * np.random.rand(len(x_train))

plt.plot(x_train, y_train, 'o')
plt.grid()
plt.show()

# 데이터 섞기
np.random.seed(303)
shuffled_id = np.arange(0, len(x_train))
np.random.shuffle(shuffled_id)
x_train = x_train[shuffled_id]
y_train = y_train[shuffled_id]



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




import numpy as np

def loss(w, x, y):
    return np.sum((x.dot(w) - y) ** 2) / len(y)

def loss_grad(w, x, y):
    return 2 * x.T.dot(x.dot(w) - y) / len(y)

def generate_batches(batch_size, x, y):
    for i in range(0, len(y), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

# Parameters
batch_size = 10
lr = 1.5
max_epochs = 51
w0 = np.array([4.0, -1.0])

# Gradient Descent
path_gd = []
for epoch in range(max_epochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_gd.append(w0.copy())
        grad = loss_grad(w0, x_batch, y_batch)
        w0 = w0 - lr * grad


# Parameters
epsilon = 1e-6
delta = 1e-6
w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)

# Adagrad
path_adagrad = []
for epoch in range(max_epochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_adagrad.append(w0.copy())
        grad = loss_grad(w0, x_batch, y_batch)
        r = r + grad * grad
        delw = - (epsilon / (delta + np.sqrt(r))) * grad
        w0 = w0 + delw



# Parameters
rho = 0.9
w0 = np.array([4.0, -1.0])
r = np.zeros_like(w0)

# RMSProp
path_rmsprop = []
for epoch in range(max_epochs):
    if epoch % 10 == 0:
        print(epoch, w0, loss(w0, x_train, y_train))
    for x_batch, y_batch in generate_batches(batch_size, x_train, y_train):
        path_rmsprop.append(w0.copy())
        grad = loss_grad(w0, x_batch, y_batch)
        r = rho * r + (1 - rho) * grad * grad
        delw = - (epsilon / np.sqrt(delta + r)) * grad
        w0 = w0 + delw