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