"""
Quadrotor closed-loop simulation with online parameter estimation.

Runs a quadrotor tracking a circle trajectory while using several
estimators to identify inertial parameters online.  A payload is
attached and later detached mid-flight to test adaptation speed.
"""

from __future__ import annotations

import numpy as np

from online_estimators.simulation.trial import run_single_trial

SEED = 9
EST_FREQ = 10  # estimator updates every 10 sim steps
WINDOW = 3  # stack 3 measurement snapshots per update
REF_TYPE = "circle"
T_SIM = 15.0
NOISE = "medium"


def main() -> None:
    print("Running quadrotor simulation with TAG-K estimator...")
    print(f"  Trajectory: {REF_TYPE}")
    print(f"  Duration: {T_SIM:.0f} s")
    print(f"  Noise level: {NOISE}")
    print(f"  Seed: {SEED}\n")

    # Run TAG-K first and show summary
    result = run_single_trial(
        estimator_name="tagk",
        base_seed=SEED,
        seed_offset=0,
        est_freq=EST_FREQ,
        window=WINDOW,
        ref_type=REF_TYPE,
        T_sim=T_SIM,
        noise=NOISE,
    )

    if result["failed"]:
        print(f"Trial FAILED: {result.get('fail_reason', 'unknown')}")
        return

    t = np.asarray(result["t"])
    theta_est = np.array(result["theta_est_traj"])
    pos_err = np.array(result["abs_mean_pos_err_t"])

    print("Trial completed successfully!")
    print(f"  Duration: {t[-1]:.1f} s ({len(t)} timesteps)")
    print(f"  Mean position error: {np.nanmean(pos_err):.4f} m")
    print(f"  Max position error:  {np.nanmax(pos_err):.4f} m")
    if theta_est.shape[0] > 0:
        print(f"  Final mass estimate: {theta_est[-1, 0]:.4f} kg")

    # Compare estimators
    print("\n--- Comparing estimators ---\n")
    estimators_to_test = ["tagk", "rls_0.8", "rk", "grk", "tark"]

    print(f"{'Estimator':<15} {'Mean pos err':>14} {'Est error':>12} {'Status':>8}")
    print("-" * 52)

    for est_name in estimators_to_test:
        r = run_single_trial(
            estimator_name=est_name,
            base_seed=SEED,
            seed_offset=0,
            est_freq=EST_FREQ,
            window=WINDOW,
            ref_type=REF_TYPE,
            T_sim=T_SIM,
            noise=NOISE,
        )
        if r["failed"]:
            print(f"{est_name:<15} {'---':>14} {'---':>12} {'FAILED':>8}")
        else:
            pe = np.array(r["abs_mean_pos_err_t"])
            ee = np.array(r["est_err_t"])
            print(f"{est_name:<15} {np.nanmean(pe):>14.4f} {ee[-1]:>12.4f} {'ok':>8}")


if __name__ == "__main__":
    main()
