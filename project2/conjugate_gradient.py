# Math 143M project 2, iteritavely solve hilbert's matrix 

import numpy as np 

# function to create hilbert matrix of dimension dim
def hilbert_matrix(dim: int):
    indices = np.arange(dim)
    return 1.0 / (indices[:, None] + indices + 1)

H = hilbert_matrix(20)
b = np.sum(H, axis=1)[:, None]
x_0 = np.zeros(H.shape[0])[:, None]
r_0 = H @ x_0 - b 
v =  -r_0
t = -(r_0.T @ v) / (v.T @ H @ v)
x_1 = x_0 + t * v

iternations = 1

while (np.linalg.norm(r_0, ord=np.inf) > 1e-3) and iternations < 76:
    x_0 = x_1
    r_0 = H @ x_0 - b
    v =  -r_0
    t = -(r_0.T @ v) / (v.T @ H @ v)
    x_1 = x_0 + t * v
    iternations += 1

print(iternations)
print(np.linalg.norm(r_0, ord=np.inf))
print(x_1)