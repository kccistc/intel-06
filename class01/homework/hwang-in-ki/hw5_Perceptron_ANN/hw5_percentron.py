import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numerical_derivative(f, x):
    h = 1e-4
    gradf = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float((tmp_val) + h)
        fx1 = f(x)
        x[idx] = float((tmp_val) - h)
        fx2 = f(x)
        gradf[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
    return gradf


class logicGate:
    def __init__(self, gate, xdata, tdata, learning_rate=0.01, threshold=0.5):
        self.name = gate

        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        self.__w = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        self.__learning_rate = learning_rate
        self.__threshold = threshold

    def __loss_func(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta))

    def err_val(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta))

    def train(self):
        def f(x): return self.__loss_func()
        print("Initial loss: ", self.__loss_func())
        for step in range(20000):
            self.__w -= self.__learning_rate * \
                numerical_derivative(f, self.__w)
            self.__b -= self.__learning_rate * \
                numerical_derivative(f, self.__b)
            if step % 2000 == 0:
                print("step: ", step, " loss: ", self.__loss_func())

    def predict(self, input_data):
        z = np.dot(input_data, self.__w) + self.__b
        y = sigmoid(z)

        if y > self.__threshold:
            result = 1
        else:
            result = 0
        return y, result


# AND
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [0], [0], [1]])
and_gate = logicGate("AND", xdata, tdata)
and_gate.train()
for in_data in xdata:
    y, result = and_gate.predict(in_data)
    print("input: ", in_data, " predict: ", y, " result: ", result)

# OR
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [1], [1], [1]])
or_gate = logicGate("OR", xdata, tdata)
or_gate.train()
for in_data in xdata:
    y, result = or_gate.predict(in_data)
    print("input: ", in_data, " predict: ", y, " result: ", result)

# XOR
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([[0], [1], [1], [0]])
xor_gate = logicGate("XOR", xdata, tdata)
xor_gate.train()
for in_data in xdata:
    y, result = xor_gate.predict(in_data)
    print("input: ", in_data, " predict: ", y, " result: ", result)
