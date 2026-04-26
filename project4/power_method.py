import numpy as np

# Build the 50x50 tridiagonal matrix A
n = 50
A = np.zeros((n, n))
np.fill_diagonal(A, 1)           # main diagonal = 1
np.fill_diagonal(A[:-1, 1:], 8) # super diagonal = 8
np.fill_diagonal(A[1:, :-1], 2) # sub diagonal = 2

tol = 0.01
x0 = np.ones(n)


def power_method(M, x0, tol, max_iter=10000):
    x = x0.copy().astype(float)
    x /= np.max(np.abs(x))
    lam = 0.0
    iterations = 0

    for _ in range(max_iter):
        y = M @ x
        idx = np.argmax(np.abs(y))
        lam_new = y[idx]          # eigenvalue estimate: max(|Ax|) when ||x||_inf = 1
        x_new = y / y[idx]

        # Converge on eigenvector change, not eigenvalue estimate.
        # Eigenvalue-based stopping stalls at 11 for ~25 iterations because
        # interior rows of Ax are identically 11 until boundary effects propagate.
        err = np.max(np.abs(x_new - x))
        lam = lam_new
        x = x_new
        iterations += 1

        if err < tol:
            break

    return lam, x, iterations


# --- Power Method for dominant eigenvalue ---
lam1, v1, iter1 = power_method(A, x0, tol)

print(f"Dominant eigenvalue  λ1 = {lam1:.4f}")
print(f"Iterations for λ1      : {iter1}")

# --- Wielandt's Deflation for second largest eigenvalue ---
# Deflated matrix: B = A - λ1 * v1 * u1^T
# where u1 = A^T v1 / (λ1 * ||v1||^2)  (left eigenvector scaled so u1^T v1 = 1)
# For non-symmetric A we use: u1 = (A^T v1) / (λ1 * (v1 . v1))
# Wielandt deflation: choose row k where |v1[k]| is maximum, then
# B = A - (1/v1[k]) * (A[:,k] outer v1)

k = np.argmax(np.abs(v1))
# B·v1 = A·v1 - outer(v1, A[k,:])·v1 = λ₁v1 - v1·(Av1)[k] = λ₁v1 - λ₁v1 = 0
B = A - np.outer(v1, A[k, :])

lam2, v2, iter2 = power_method(B, x0, tol)

print(f"\nSecond largest eigenvalue  λ2 = {lam2:.4f}")
print(f"Iterations for λ2            : {iter2}")

# Verify with numpy
eigvals = np.sort(np.linalg.eigvals(A).real)[::-1]
print(f"\nNumpy verification (top 2): λ1 = {eigvals[0]:.4f}, λ2 = {eigvals[1]:.4f}")
