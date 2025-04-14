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