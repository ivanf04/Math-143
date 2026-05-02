import numpy as np


def power_method(A, x0, tol=1e-4, max_iter=25):
    """
    Power method for dominant eigenpair.
    Normalization uses the infinity norm / largest-magnitude component.
    """
    x = np.array(x0, dtype=float).reshape(-1)
    x = x / np.linalg.norm(x, ord=np.inf)

    mu_old = None
    history = []

    for k in range(1, max_iter + 1):
        y = A @ x
        idx = np.argmax(np.abs(y))
        mu = y[idx]
        x = y / mu

        history.append((k, mu, x.copy()))

        if mu_old is not None and abs(mu - mu_old) < tol:
            return mu, x, history

        mu_old = mu

    return mu, x, history


def wielandt_deflation(A, dominant_eigenvalue, dominant_eigenvector):
    """
    Form the Wielandt deflated matrix:
        B = A - lambda_1 * x * e_j^T / x_j
    where x_j is a nonzero component of the dominant eigenvector.
    This removes the dominant eigenvalue while preserving the others.
    """
    x = np.array(dominant_eigenvector, dtype=float).reshape(-1)
    j = np.argmax(np.abs(x))
    ej = np.zeros_like(x)
    ej[j] = 1.0
    B = A - dominant_eigenvalue * np.outer(x, ej) / x[j]
    return B, j


if __name__ == "__main__":
    # Exercise 1(c) matrix and initial vector from the previous prompts
    A = np.array([
        [1, -1,  0],
        [-2, 4, -2],
        [0, -1,  2]
    ], dtype=float)

    x0 = np.array([-1, 2, 1], dtype=float)

    tol = 1e-4
    max_iter = 25

    print("Original matrix A:")
    print(A)
    print("\nInitial vector x^(0):", x0)

    # Step 1: Use Exercise 7 result (power method) to approximate dominant eigenpair
    lambda1, x1, hist1 = power_method(A, x0, tol=tol, max_iter=max_iter)

    print("\nDominant eigenvalue approximation from the power method:")
    print(f"lambda_1 ≈ {lambda1:.8f}")
    print("Approximate dominant eigenvector x_1:")
    print(x1)

    print("\nPower method history on A")
    print("-" * 78)
    print(f"{'k':>3} {'lambda^(k)':>15} {'x^(k)':>50}")
    print("-" * 78)
    for k, mu, xk in hist1:
        print(f"{k:>3} {mu:>15.8f} {np.array2string(xk, precision=8, suppress_small=True):>50}")

    # Step 2: Wielandt deflation
    B, j = wielandt_deflation(A, lambda1, x1)

    print("\nWielandt deflated matrix B = A - lambda_1 * x * e_j^T / x_j")
    print(f"Chosen component index j = {j}")
    print(B)

    # Step 3: Apply power method to the deflated matrix
    lambda2, x2, hist2 = power_method(B, x0, tol=tol, max_iter=max_iter)

    print("\nSecond most dominant eigenvalue approximation (via deflation + power method):")
    print(f"lambda_2 ≈ {lambda2:.8f}")
    print("Approximate eigenvector from deflated system:")
    print(x2)

    print("\nPower method history on deflated matrix B")
    print("-" * 78)
    print(f"{'k':>3} {'lambda^(k)':>15} {'x^(k)':>50}")
    print("-" * 78)
    for k, mu, xk in hist2:
        print(f"{k:>3} {mu:>15.8f} {np.array2string(xk, precision=8, suppress_small=True):>50}")

    # Verification with NumPy
    eigvals = np.linalg.eigvals(A)
    eigvals_sorted = eigvals[np.argsort(-np.abs(eigvals))]

    print("\nNumPy eigenvalues of A:")
    print(eigvals)
    print("\nEigenvalues sorted by magnitude:")
    print(eigvals_sorted)
    print(f"\nExact dominant eigenvalue (by magnitude): {eigvals_sorted[0]:.8f}")
    print(f"Exact second most dominant eigenvalue (by magnitude): {eigvals_sorted[1]:.8f}")
