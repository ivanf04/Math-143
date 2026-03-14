# implementation of jacobi method, for question 7.3 11D on HW 

import numpy as np


def gauss_seidel(A, b, x0=None, tol=1e-2, max_iter=300):
    """
    Solve Ax = b using the Gauss-Seidel iterative method.

    Parameters
    ----------
    A : array_like
        3x3 coefficient matrix.
    b : array_like
        Right-hand side vector of length 3.
    x0 : array_like, optional
        Initial guess. Defaults to the zero vector.
    tol : float, optional
        Convergence tolerance based on the infinity norm.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        Approximate solution vector.
    iterations : int
        Number of iterations performed.
    converged : bool
        True if the method converged within max_iter iterations.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.shape != (3, 3):
        raise ValueError("A must be a 3x3 matrix.")
    if b.shape != (3,):
        raise ValueError("b must be a vector of length 3.")
    if np.any(np.diag(A) == 0):
        raise ValueError("A has a zero on the diagonal, so Gauss-Seidel cannot proceed.")

    x = np.zeros(3, dtype=float) if x0 is None else np.array(x0, dtype=float)
    if x.shape != (3,):
        raise ValueError("x0 must be a vector of length 3.")

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(3):
            sum_before = np.dot(A[i, :i], x[:i])
            sum_after = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - sum_before - sum_after) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k, True

    return x, max_iter, False


if __name__ == "__main__":
    # Example system:
    # 4x +  y + 2z = 4
    # 3x + 5y +  z = 7
    #  x +  y + 3z = 3
    A = np.array([
        [1, 0, -2],
        [-0.5, 1, -0.25],
        [1, -0.5, 1]
    ], dtype=float)

    b = np.array([0.2, -1.425, 2], dtype=float)
    x0 = np.zeros(3)

    solution, iterations, converged = gauss_seidel(A, b, x0=x0, tol=1e-2, max_iter=300)

    print("A =")
    print(A)
    print("\nb =")
    print(b)
    print("\nApproximate solution:")
    print(solution)
    print(f"\nIterations: {iterations}")
    print(f"Converged: {converged}")