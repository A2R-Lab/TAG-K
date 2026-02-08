"""
Basic online estimation example.

Solves an overdetermined linear system A x = b using several estimators
and compares their convergence.
"""

from __future__ import annotations

import numpy as np

from online_estimators.estimators import GRK, RK, RLS, TAGK, TARK, TAGK_Tikh


def main() -> None:
    # Generate a random linear system  A x = b
    rng = np.random.default_rng(0)
    n = 5  # number of parameters
    m = 30  # number of measurements
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    b = A @ x_true

    estimators = {
        "RLS": RLS(n),
        "RK": RK(n),
        "TARK": TARK(n),
        "GRK": GRK(n),
        "TAGK": TAGK(n),
        "TAGK_Tikh": TAGK_Tikh(n),
    }

    n_iters = 500

    print(f"True parameters: {x_true.round(3)}")
    print(f"System size: {m} x {n}")
    print(f"Running {n_iters} iterations per estimator...\n")
    print(f"{'Estimator':<15} {'Final error':>12} {'Converged?':>12}")
    print("-" * 42)

    for name, est in estimators.items():
        for _ in range(n_iters):
            x_hat = est.iterate(A, b)
        err = np.linalg.norm(x_hat - x_true)
        converged = "yes" if err < 0.5 else "no"
        print(f"{name:<15} {err:>12.6f} {converged:>12}")


if __name__ == "__main__":
    main()
