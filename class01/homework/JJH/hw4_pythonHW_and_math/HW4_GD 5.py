import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def contour_with_quiver(f, x, y, grad_x, grad_y, norm=LogNorm(), level=np.logspace(0, 5, 35), minima=None):
    dz_dx = grad_x(x, y)
    dz_dy = grad_y(x, y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(x, y, f(x, y), levels=level, norm=norm, cmap=plt.cm.jet)
    if minima is not None:
        ax.plot(*minima, 'r+', markersize=10)
    ax.quiver(x, y, -dz_dx, -dz_dy, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.show()

def surf(f, x, y, norm=LogNorm(), minima=None):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=50)
    ax.plot_surface(x, y, f(x, y), norm=norm, rstride=1, cstride=1, edgecolor='none', alpha=0.8, cmap=plt.cm.jet)
    if minima is not None:
        ax.plot(*minima, f(*minima), 'r+', markersize=10)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.show()

def contour_with_path(f, x, y, paths, norm=LogNorm(), level=np.logspace(0, 5, 35), minima=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(x, y, f(x, y), levels=level, norm=norm, cmap=plt.cm.jet)
    ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1], paths[1, 1:] - paths[1, :-1], scale_units='xy', angles='xy', scale=1, color='k')
    if minima is not None:
        ax.plot(*minima, 'r+', markersize=10)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
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