#importing Python Llb
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

def numerical_derivative(f, x):
    dx = 1e-4
    gradf = np.zeros_like(x)

    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float((tmp_val)+dx)
        fx1 = f(x)

        x[idx] = float((tmp_val)-dx)
        fx2 = f(x)
        gradf[idx] = (fx1 - fx2)/(2 * dx)
        
        x[idx] = tmp_val
        it.iternext()

    return gradf

