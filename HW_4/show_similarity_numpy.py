import numpy as np

np.set_printoptions(precision=6, suppress=True)

A = np.array([
    [2.0,  0.0, 0.0],
    [1.0,  1.0, 2.0],
    [1.0, -1.0, 4.0]
])

print("Matrix A:")
print(A)
print()

# Compute eigenvalues/eigenvectors.
# numpy.linalg.eig returns a matrix V whose columns are eigenvectors,
# and A @ V = V @ diag(eigenvalues).
evals, V = np.linalg.eig(A)
D = np.diag(evals)
V_inv = np.linalg.inv(V)

print("Eigenvalues of A:")
print(evals)
print()

print("Eigenvector matrix P = V (columns are eigenvectors):")
print(V)
print()

print("Diagonal matrix D = diag(eigenvalues):")
print(D)
print()

print("Check that P^{-1} A P = D:")
print(V_inv @ A @ V)
print()

print("Check that A = P D P^{-1}:")
print(V @ D @ V_inv)
print()

print("Verification with allclose:", np.allclose(V_inv @ A @ V, D) and np.allclose(V @ D @ V_inv, A))
print()

# The eigensolver gives D = diag(3, 2, 2), which is D1 from the problem.
D1 = np.diag([3.0, 2.0, 2.0])
D2 = np.diag([2.0, 3.0, 2.0])
D3 = np.diag([2.0, 2.0, 3.0])

print("D1:")
print(D1)
print("Matches computed D?", np.allclose(D, D1))
print()

# To get D2 and D3, permute the eigenvectors (columns of P).
# If S is a permutation matrix, then
#   (P S)^{-1} A (P S) = S^{-1} D S,
# which simply reorders the diagonal entries.
S12 = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

S13 = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0]
])

P1 = V
P2 = V @ S12
P3 = V @ S13

print("Using P1 = V, we get:")
print(np.linalg.inv(P1) @ A @ P1)
print()

print("Using P2 = V @ S12, we get:")
print(f"P2 matrix: \n{P2}")
print(np.linalg.inv(P2) @ A @ P2)
print("Matches D2?", np.allclose(np.linalg.inv(P2) @ A @ P2, D2))
print()

print("Using P3 = V @ S13, we get:")
print(f"P3 matrix:\n {P3}")
print(np.linalg.inv(P3) @ A @ P3)
print("Matches D3?", np.allclose(np.linalg.inv(P3) @ A @ P3, D3))
print()

print("Conclusion:")
print("A is similar to D1, D2, and D3.")
