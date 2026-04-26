import numpy as np


def build_matrix(n: int = 50) -> np.ndarray:
    """
    Build the n x n tridiagonal matrix A with
      - 1 on the main diagonal,
      - 8 on the superdiagonal,
      - 2 on the subdiagonal.
    """
    A = np.zeros((n, n), dtype=float)
    np.fill_diagonal(A, 1.0)
    idx = np.arange(n - 1)
    A[idx, idx + 1] = 8.0
    A[idx + 1, idx] = 2.0
    return A



def similar_symmetric_form(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a diagonal matrix D so that T = D^{-1} A D is symmetric.

    For this matrix, choosing D = diag(1, 1/2, 1/4, ..., 1/2^(n-1)) makes
    T tridiagonal with 1 on the diagonal and 4 on both off-diagonals.

    A and T are similar, so they have exactly the same eigenvalues.
    """
    n = A.shape[0]
    d = 0.5 ** np.arange(n, dtype=float)  # diagonal entries of D
    # T = D^{-1} A D, formed without explicitly inverting D.
    T = (A * d[np.newaxis, :]) / d[:, np.newaxis]
    return d, T



def power_method(
    A: np.ndarray,
    x0: np.ndarray,
    tol: float = 1.0e-2,
    max_iter: int = 20000,
) -> tuple[float, np.ndarray, int, float]:
    """
    Power method for the dominant eigenpair.

    Uses 2-norm normalization and the Rayleigh quotient to estimate lambda.
    Since we run on the symmetric similar matrix T, this is stable and accurate.

    Returns
    -------
    lam : float
        Estimated dominant eigenvalue.
    x : np.ndarray
        Estimated dominant eigenvector (2-norm normalized).
    k : int
        Iteration count when the residual first satisfies the tolerance.
    res : float
        Final residual norm ||A x - lam x||_2.
    """
    x = np.array(x0, dtype=float).reshape(-1)
    x /= np.linalg.norm(x)

    lam = 0.0
    for k in range(1, max_iter + 1):
        y = A @ x
        y_norm = np.linalg.norm(y)
        if y_norm == 0.0:
            raise ValueError("Power method encountered the zero vector.")

        x = y / y_norm
        lam = float(x @ (A @ x))
        res = float(np.linalg.norm(A @ x - lam * x))

        if res < tol:
            return lam, x, k, res

    raise RuntimeError("Power method did not converge within max_iter iterations.")



def wielandt_deflation(A: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    """
    Wielandt deflation.

    If v is the dominant eigenvector of A, this returns a reduced matrix B
    whose eigenvalues are the remaining eigenvalues of A.

    Steps:
      1) Pick p = index of the largest component of v in magnitude.
      2) Permute row/column p to the front.
      3) Form the reduced matrix B = A22 - w * a12,
         where w = v_rest / v_p.
    """
    n = A.shape[0]
    p = int(np.argmax(np.abs(v)))
    perm = [p] + [i for i in range(n) if i != p]

    Ap = A[np.ix_(perm, perm)]
    vp = v[perm]

    w = vp[1:] / vp[0]
    B = Ap[1:, 1:] - np.outer(w, Ap[0, 1:])
    return B, p, perm



def main() -> None:
    n = 50
    tol = 1.0e-2
    refine_tol = 1.0e-3

    # Matrix A from the project statement.
    A = build_matrix(n)

    # Work on a symmetric matrix T that is similar to A, so T and A have the
    # same eigenvalues. This keeps the power method numerically well behaved.
    d, T = similar_symmetric_form(A)

    # The project gives x0 = [1, 1, ..., 1]^T for A.
    # Under z = D^{-1} x, the matching starting vector for T is z0 = D^{-1} x0.
    x0_A = np.ones(n, dtype=float)
    z0 = x0_A / d

    # First pass: stop exactly at the assignment tolerance (0.01) and record
    # the iteration count.
    lam1_tol, z1_tol, it1, res1 = power_method(T, z0, tol=tol)

    # Short refinement pass from the converged vector. This does not change the
    # iteration count reported for tol = 0.01; it only stabilizes the vector
    # before Wielandt deflation and sharpens the 4-decimal eigenvalue printout.
    lam1, z1, _, _ = power_method(T, z1_tol, tol=refine_tol)

    # Wielandt deflation on the similar matrix.
    B, p, perm = wielandt_deflation(T, z1)

    # Starting vector for the reduced problem: delete the p-th entry from z0.
    q0 = np.delete(z0, p)

    lam2_tol, y2_tol, it2, res2 = power_method(B, q0, tol=tol)
    lam2, y2, _, _ = power_method(B, y2_tol, tol=refine_tol)

    print("Matrix size:", n)
    print(f"Largest eigenvalue        ~= {lam1:.4f}")
    print(f"Iterations for lambda_1  =  {it1}")
    print(f"Second largest eigenvalue~= {lam2:.4f}")
    print(f"Iterations for lambda_2  =  {it2}")

    # Optional diagnostic output.
    print("\nDiagnostics:")
    print(f"Residual for first pass  = {res1:.6e}")
    print(f"Residual for second pass = {res2:.6e}")
    print(f"Deflation index p        = {p}")


if __name__ == "__main__":
    main()
