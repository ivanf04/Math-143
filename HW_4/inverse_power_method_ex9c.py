import numpy as np


def inverse_power_method(A, x0, tol=1e-4, max_iter=25):
    """
    Approximate the dominant eigenvalue of A using the inverse power method.

    Since the inverse power method converges to the eigenvalue of smallest
    magnitude of A, this script reports that value directly and also reports
    the dominant eigenvalue of A for comparison using numpy.
    """
    x = np.array(x0, dtype=float)
    x = x / np.linalg.norm(x)

    lambda_old = None
    history = []

    for k in range(1, max_iter + 1):
        # Solve A y = x instead of forming A^{-1}
        y = np.linalg.solve(A, x)
        y_norm = np.linalg.norm(y)
        x = y / y_norm

        # Rayleigh quotient for the current eigenvalue estimate of A
        lambda_new = float((x.T @ A @ x) / (x.T @ x))

        err = np.nan if lambda_old is None else abs(lambda_new - lambda_old)
        history.append((k, x.copy(), lambda_new, err))

        if lambda_old is not None and err < tol:
            return lambda_new, x, history, True

        lambda_old = lambda_new

    return lambda_new, x, history, False


if __name__ == "__main__":
    A = np.array([
        [1.0, -1.0, 0.0],
        [-2.0, 4.0, -2.0],
        [0.0, -1.0, 2.0]
    ])

    x0 = np.array([-1.0, 2.0, 1.0])
    tol = 1e-4
    max_iter = 25

    eigval, eigvec, history, converged = inverse_power_method(A, x0, tol, max_iter)

    print("Matrix A:\n", A)
    print("\nInitial vector x^(0):", x0)
    print(f"Tolerance: {tol}")
    print(f"Maximum iterations: {max_iter}\n")

    print("Inverse power iterations")
    print("-" * 72)
    print(f"{'k':>3}  {'eigenvalue estimate':>22}  {'error':>14}  {'x^(k)'}")
    print("-" * 72)
    for k, xk, lam, err in history:
        err_str = "-" if np.isnan(err) else f"{err:.6e}"
        print(f"{k:>3}  {lam:>22.10f}  {err_str:>14}  {np.array2string(xk, precision=6)}")

    print("\nResult from inverse power method:")
    if converged:
        print(f"Converged in {len(history)} iterations.")
    else:
        print(f"Did not reach tolerance in {max_iter} iterations.")

    print(f"Approximate eigenvalue found: {eigval:.10f}")
    print("Approximate eigenvector:", eigvec)

    # Exact eigenvalues for reference
    exact_eigs = np.linalg.eigvals(A)
    dominant_exact = exact_eigs[np.argmax(np.abs(exact_eigs))]
    smallest_exact = exact_eigs[np.argmin(np.abs(exact_eigs))]

    print("\nEigenvalues from numpy for reference:")
    print(exact_eigs)
    print(f"Smallest-magnitude eigenvalue (inverse power target): {smallest_exact:.10f}")
    print(f"Dominant eigenvalue (largest magnitude): {dominant_exact:.10f}")

    print("\nNote:")
    print("The inverse power method converges to the eigenvalue of smallest magnitude.")
    print("If your instructor intended the dominant eigenvalue, they may have meant the power method instead.")
