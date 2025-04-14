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