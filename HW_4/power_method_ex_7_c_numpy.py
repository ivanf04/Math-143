import numpy as np


def power_method(A, x0, tol=1e-4, max_iter=25):
    """
    Approximate the dominant eigenvalue of A using the power method.

    Parameters
    ----------
    A : np.ndarray
        Square matrix.
    x0 : np.ndarray
        Initial vector.
    tol : float
        Stopping tolerance based on consecutive eigenvalue estimates.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    eigenvalue : float
        Approximate dominant eigenvalue.
    eigenvector : np.ndarray
        Approximate dominant eigenvector.
    history : list[tuple[int, float, np.ndarray]]
        Iteration history as (k, eigenvalue estimate, normalized vector).
    """
    x = np.array(x0, dtype=float).reshape(-1)
    x = x / np.linalg.norm(x, ord=np.inf)

    lambda_old = None
    history = []

    for k in range(1, max_iter + 1):
        y = A @ x
        idx = np.argmax(np.abs(y))
        mu = y[idx]
        x = y / mu

        history.append((k, mu, x.copy()))

        if lambda_old is not None and abs(mu - lambda_old) < tol:
            return mu, x, history

        lambda_old = mu

    return mu, x, history


if __name__ == "__main__":
    # Matrix and initial vector from the previous problem (Exercise 1c)
    A = np.array([
        [1, -1, 0],
        [-2, 4, -2],
        [0, -1, 2]
    ], dtype=float)

    x0 = np.array([-1, 2, 1], dtype=float)

    eigenvalue, eigenvector, history = power_method(A, x0, tol=1e-4, max_iter=25)

    print("Matrix A:\n", A)
    print("\nInitial vector x^(0):", x0)
    print("\nPower method iteration table")
    print("-" * 72)
    print(f"{'k':>3} {'lambda^(k)':>15} {'x^(k)':>45}")
    print("-" * 72)

    for k, mu, xk in history:
        print(f"{k:>3} {mu:>15.8f} {np.array2string(xk, precision=8, suppress_small=True):>45}")

    print("-" * 72)
    print(f"Approximate dominant eigenvalue: {eigenvalue:.8f}")
    print("Approximate eigenvector:", eigenvector)

    # Verification against NumPy's exact eigenvalues
    eigvals = np.linalg.eigvals(A)
    dominant_exact = eigvals[np.argmax(np.abs(eigvals))]
    print("\nNumPy eigenvalues:", eigvals)
    print(f"Dominant eigenvalue from NumPy: {dominant_exact:.8f}")