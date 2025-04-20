import numpy as np
A=[]
n=int(input())
for i in range(n):
    temp=[]
    for j in range(1,n+1):
        temp.append(i*n+j)
    A.append(temp)
print(np.reshape(A,(1,-1)))