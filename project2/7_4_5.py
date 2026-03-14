import numpy as np

def sor(A, b, w=1.2, tol=1e-3, max_iter=1000):
    n = len(b)
    x = np.zeros(n)              # initial guess
    x_old = np.copy(x)

    for k in range(1, max_iter + 1):
        x_old[:] = x

        for i in range(n):
            sigma1 = np.dot(A[i, :i], x[:i])          # updated values
            sigma2 = np.dot(A[i, i+1:], x_old[i+1:])  # old values

            x[i] = (1 - w) * x_old[i] + (w / A[i, i]) * (b[i] - sigma1 - sigma2)

        # L-infinity norm of difference
        err = np.linalg.norm(x - x_old, ord=np.inf)

        print(f"Iteration {k}: x = {x}, error = {err:.6f}")

        if err < tol:
            print("\nConverged successfully.")
            return x, k

    print("\nMaximum iterations reached without convergence.")
    return x, max_iter


# Example 5x5 system
A = np.array([
    [4, 1,  1,  0,  1],
    [-1, -3, 1,  1,  0],
    [2, 1, 5, -1,  -1],
    [-1,  -1, -1,  4, 0],
    [ 0,  2,  -1, 1,  4]
], dtype=float)

b = np.array([6, 6, 6, 6, 6], dtype=float)

solution, iterations = sor(A, b, w=1.2, tol=1e-3)

print("\nApproximate solution:")
print(solution)
print(f"Number of iterations: {iterations}")