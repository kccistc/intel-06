import numpy as np
from numpy import linalg as LA
c = np.arrray([[1, 2, 3], [-1, 1, 4]])
print(LA.norm(c, axis=0))
print(LA.norm(c, axis=1))
print(LA.norm(c, ord=1,axis=1))
print(LA.norm(c, ord=2,axis=1))
              