"""
Compare all Kaczmarz-family estimators on a noisy system.

Generates a noisy overdetermined system and runs every Kaczmarz variant
for a fixed number of iterations, then prints a sorted leaderboard.
"""

from __future__ import annotations

import numpy as np

from online_estimators.estimators import (
    FDBK,
    GRK,
    IGRK,
    MGRK,
    REK,
    RK,
    TAGK,
    TARK,
    GRK_Tikh,
    RK_ColScaled,
    RK_Equi,
    RK_EquiTikh,
    RK_Tikh,
    TAGK_Tikh,
)


def main() -> None:
    rng = np.random.default_rng(42)

    # Overdetermined noisy system
    n = 10
    m = 50
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    noise_std = 0.1
    b = A @ x_true + noise_std * rng.standard_normal(m)

    estimators = {
        "RK": RK(n),
        "TARK": TARK(n),
        "GRK": GRK(n),
        "TAGK": TAGK(n),
        "TAGK_Tikh": TAGK_Tikh(n),
        "GRK_Tikh": GRK_Tikh(n),
        "IGRK": IGRK(n),
        "MGRK": MGRK(n),
        "REK": REK(n),
        "FDBK": FDBK(n),
        "RK_ColScaled": RK_ColScaled(n),
        "RK_Equi": RK_Equi(n),
        "RK_Tikh": RK_Tikh(n),
        "RK_EquiTikh": RK_EquiTikh(n),
    }

    n_iters = 1000
    results: list[tuple[str, float]] = []

    print(f"System: {m} x {n}, noise std = {noise_std}")
    print(f"Running {n_iters} iterations per estimator...\n")

    for name, est in estimators.items():
        for _ in range(n_iters):
            x_hat = est.iterate(A, b)
        err = float(np.linalg.norm(x_hat - x_true))
        results.append((name, err))

    # Sort by error (ascending)
    results.sort(key=lambda r: r[1])

    print(f"{'Rank':<6} {'Estimator':<18} {'Error':>10}")
    print("-" * 36)
    for rank, (name, err) in enumerate(results, 1):
        print(f"{rank:<6} {name:<18} {err:>10.6f}")


if __name__ == "__main__":
    main()
