"""Solve Part I of Project 3 using NumPy.

This script fits the 20 given data points with a least-squares
4th-degree polynomial

    P4(x) = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

and prints the coefficients rounded to 4 decimal places.
It also generates a labeled plot showing the data points and the fitted curve.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Data from the assignment PDF
X_DATA = np.array([
    -3.14159265358979,
    -2.81089869005402,
    -2.48020472651826,
    -2.14951076298249,
    -1.81881679944672,
    -1.48812283591095,
    -1.15742887237519,
    -0.82673490883942,
    -0.49604094530365,
    -0.16534698176788,
     0.16534698176788,
     0.49604094530365,
     0.82673490883942,
     1.15742887237519,
     1.48812283591095,
     1.81881679944672,
     2.14951076298249,
     2.48020472651826,
     2.81089869005402,
     3.14159265358979,
], dtype=float)

Y_DATA = np.array([
     0.00000000000000,
    -0.43702230525987,
    -0.78170987707147,
    -0.85207605602719,
    -0.52335935447908,
     0.22121102681686,
     1.27130734031775,
     2.41089249371528,
     3.37539264122665,
     3.92749194128624,
     3.92749194128624,
     3.37539264122665,
     2.41089249371528,
     1.27130734031775,
     0.22121102681686,
    -0.52335935447908,
    -0.85207605602719,
    -0.78170987707147,
    -0.43702230525987,
     0.00000000000000,
], dtype=float)


def fit_part1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return least-squares coefficients [a0, a1, a2, a3, a4]."""
    # Vandermonde/design matrix with columns: 1, x, x^2, x^3, x^4
    A = np.vander(x, 5, increasing=True)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs



def evaluate_polynomial(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial in increasing-power coefficient form."""
    powers = np.vstack([x**k for k in range(len(coeffs))])
    return coeffs @ powers



def main() -> None:
    coeffs = fit_part1(X_DATA, Y_DATA)
    a0, a1, a2, a3, a4 = coeffs

    print("Least-squares coefficients for Part I:")
    print(f"a0 = {a0:.4f}")
    print(f"a1 = {a1:.4f}")
    print(f"a2 = {a2:.4f}")
    print(f"a3 = {a3:.4f}")
    print(f"a4 = {a4:.4f}")

    # Create a smooth curve for plotting
    x_plot = np.linspace(-np.pi, np.pi, 500)
    y_plot = evaluate_polynomial(coeffs, x_plot)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_DATA, Y_DATA, label="Data points")
    plt.plot(x_plot, y_plot, label="Least-squares quartic fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Part I: 4th-Degree Least-Squares Polynomial Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("part1_plot.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
