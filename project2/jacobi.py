# implementation of jacobi method 

import numpy as np 

# function to create hilbert matrix of dimension dim
def hilbert_matrix(dim: int):
    indices = np.arange(dim)
    return 1.0 / (indices[:, None] + indices + 1)

H = hilbert_matrix(20)
b = np.sum(H, axis=1)[:, None]
x_0 = np.zeros(H.shape[0])[:, None]

# decompostion H = D + L + U
D = np.diag(np.diag(H))
L = np.tril(H, -1)
U = np.triu(H, 1)

invD = np.diag(1 / np.diag(H))
T = -invD @ (L + U)
c =  invD @ b

iterations = 75
x_new =  x_0

for i in range(iterations):
    x_new = T @ x_0 + c
    r = H @ x_new - b

    if np.linalg.norm(r, ord=np.inf) < 1e-3:
        iterations = i + 1
        break
    
    x_0 = x_new


print(iterations)
print(x_new)