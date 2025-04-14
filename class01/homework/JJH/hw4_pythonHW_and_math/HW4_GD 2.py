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