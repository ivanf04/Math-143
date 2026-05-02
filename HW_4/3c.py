import numpy as np 

A = np.array([
    [-14, -1, 0],
    [-2, -11, -2],
    [0, -1, -13],    
])

x = np.array([
    [-0.5],
    [1],
    [0.5]
])

y = np.linalg.solve(A, x)

print(y)

x = y * (1/-0.09291581)
print(x)

y = np.linalg.solve(A, x)

print(y)
x = y / (-0.09466857)

print(x)

y = np.linalg.solve(A, x)
print(y)
