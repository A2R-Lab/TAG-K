"""
Parameter drift tracking example.

Demonstrates how different estimators handle a slowly drifting true
parameter vector -- a scenario common in robotics when physical
properties change over time (e.g., fuel consumption, wear, temperature).
"""

from __future__ import annotations

import numpy as np

from online_estimators.estimators import KF, RLS, TAGK


def main() -> None:
    rng = np.random.default_rng(123)

    n = 3
    m = 10  # measurements per batch
    n_steps = 300

    # Initial true parameters
    x_true_0 = np.array([2.0, -1.0, 0.5])

    # Drift rate per step
    drift = np.array([0.005, -0.003, 0.002])

    # Set up estimators
    est_rls = RLS(n, forgetting_factor=0.95, c=100)
    est_kf = KF(
        n,
        process_noise=1e-3 * np.eye(n),
        measurement_noise=1e-4 * np.eye(m),
        c=100,
    )
    est_tagk = TAGK(n)

    errors = {"RLS": [], "KF": [], "TAGK": []}

    for t in range(n_steps):
        # Parameters drift linearly
        x_true = x_true_0 + drift * t

        # Generate noisy measurements
        A = rng.standard_normal((m, n))
        b = A @ x_true + 0.01 * rng.standard_normal(m)

        # Update each estimator
        x_rls = est_rls.iterate(A, b)
        x_kf = est_kf.iterate(A, b)
        x_tagk = est_tagk.iterate(A, b)

        errors["RLS"].append(np.linalg.norm(x_rls - x_true))
        errors["KF"].append(np.linalg.norm(x_kf - x_true))
        errors["TAGK"].append(np.linalg.norm(x_tagk - x_true))

    # Print results at key timesteps
    print("Tracking drifting parameters over 300 steps")
    print(f"Drift rate: {drift}")
    print(f"\n{'Step':<8} {'RLS':>10} {'KF':>10} {'TAGK':>10}")
    print("-" * 40)
    for t in [0, 49, 99, 199, 299]:
        print(
            f"{t + 1:<8} "
            f"{errors['RLS'][t]:>10.4f} "
            f"{errors['KF'][t]:>10.4f} "
            f"{errors['TAGK'][t]:>10.4f}"
        )

    # Average error over last 100 steps
    print("\nMean error (last 100 steps):")
    for name in ["RLS", "KF", "TAGK"]:
        mean_err = np.mean(errors[name][-100:])
        print(f"  {name:<8} {mean_err:.6f}")

    # Optional: plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for name, errs in errors.items():
            ax.semilogy(errs, label=name, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Parameter error (log)")
        ax.set_title("Drifting parameter tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig("drift_tracking.png", dpi=150)
        print("\nPlot saved to drift_tracking.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
