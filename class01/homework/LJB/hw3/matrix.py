import numpy as np

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
x = np.array([1, 2, 3, 4])
b = np.array([0, 0, 0, 0])
n = 4
for i in range(0, n):
    val = 0.0
    for j in range(0, n):
        val += A[i, j] * x[j]
    b[i] = val

print("calculate = ", b)

b = np.dot(A, x)
print("dot = ", b)

b = np.matmul(A, x)
print("matmul = ", b)

b = A @ x
print("A @ x = ", b)

b = A * x
print(" A * x = ", b)