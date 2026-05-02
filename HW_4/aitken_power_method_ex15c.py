import numpy as np


def power_method_with_aitken(A, x0, tol=1e-4, max_iter=25):
    """
    Power method for the dominant eigenvalue, accelerated with Aitken's Δ^2 process
    applied to the scalar eigenvalue estimates.

    Normalization uses the infinity norm / largest-magnitude component.
    """
    x = np.array(x0, dtype=float).reshape(-1)
    x = x / np.linalg.norm(x, ord=np.inf)

    mu_history = []
    iter_history = []

    for k in range(1, max_iter + 1):
        y = A @ x
        idx = np.argmax(np.abs(y))
        mu = y[idx]
        x = y / mu

        mu_history.append(mu)

        aitken_mu = None
        if len(mu_history) >= 3:
            mu0, mu1, mu2 = mu_history[-3], mu_history[-2], mu_history[-1]
            delta1 = mu1 - mu0
            delta2 = mu2 - mu1
            denom = delta2 - delta1
            if abs(denom) > 1e-14:
                aitken_mu = mu0 - (delta1 ** 2) / denom

        iter_history.append((k, mu, aitken_mu, x.copy()))

        if len(mu_history) >= 4:
            prev_aitken = iter_history[-2][2]
            curr_aitken = iter_history[-1][2]
            if prev_aitken is not None and curr_aitken is not None:
                if abs(curr_aitken - prev_aitken) < tol:
                    return curr_aitken, x, iter_history

    final_estimate = iter_history[-1][2] if iter_history[-1][2] is not None else mu_history[-1]
    return final_estimate, x, iter_history


if __name__ == "__main__":
    # Exercise 1(c) matrix and initial vector from the previous prompts
    A = np.array([
        [1, -1, 0],
        [-2, 4, -2],
        [0, -1, 2]
    ], dtype=float)

    x0 = np.array([-1, 2, 1], dtype=float)

    tol = 1e-4
    max_iter = 25

    eigenvalue, eigenvector, history = power_method_with_aitken(A, x0, tol=tol, max_iter=max_iter)

    print("Matrix A:")
    print(A)
    print("\nInitial vector x^(0):", x0)

    print("\nPower method with Aitken's Δ^2 acceleration")
    print("-" * 96)
    print(f"{'k':>3} {'mu_k':>15} {'Aitken(mu)_k':>18} {'x^(k)':>50}")
    print("-" * 96)

    for k, mu, aitken_mu, xk in history:
        aitken_str = f"{aitken_mu:.8f}" if aitken_mu is not None else "N/A"
        print(f"{k:>3} {mu:>15.8f} {aitken_str:>18} {np.array2string(xk, precision=8, suppress_small=True):>50}")

    print("-" * 96)
    print(f"Approximate dominant eigenvalue (Aitken accelerated): {eigenvalue:.8f}")
    print("Approximate eigenvector from last power iteration:")
    print(eigenvector)

    # Verification with NumPy
    eigvals = np.linalg.eigvals(A)
    dominant_exact = eigvals[np.argmax(np.abs(eigvals))]
    print("\nNumPy eigenvalues:", eigvals)
    print(f"Dominant eigenvalue from NumPy: {dominant_exact:.8f}")
