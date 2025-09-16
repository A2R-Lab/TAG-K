#!/usr/bin/env python3
"""
DEKA grid search on quadrotor online parameter estimation.

This script runs a figure-8 tracking task with an LQR around hover, injects a payload
(add/drop) event, performs online inertial parameter estimation with the DEKA
(Deterministic Kaczmarz with Smoothing & Damping) algorithm, and saves results
for a Cartesian grid of DEKA hyperparameters.

It depends on your existing project modules:
- python.quadrotor.quadrotor.QuadrotorDynamics
- python.common.LQR_controller.LQRController
- python.common.noise_models.apply_noise
- python.common.estimation_methods.DEKA

Outputs:
- NPZ file containing per-config (labelled) arrays of pos/vel/ori error vs time,
  estimator residual vs estimator steps, and (optionally) trajectories.

Example:
    python deka_grid_search.py \\
        --ntrials 3 --est_freq 20 --window 20 \\
        --deka_gamma 0.5 1.0 \\
        --deka_reg 1e-8 1e-6 \\
        --deka_alpha 0.3 0.6 \\
        --deka_steps 1 3 \\
        --deka_tol 0.0
"""

import argparse
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# --- Project imports (assumed available in your environment) ---
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import apply_noise
from python.common.estimation_methods import DEKA

# ----------------------- Exceptions -----------------------
class TrialNumericalError(RuntimeError):
    """Raised when a single trial experiences numerical issues and should be retried."""


def is_valid_state(x: np.ndarray) -> bool:
    if not np.all(np.isfinite(x)):
        return False
    # guard absurd magnitudes (helps overflow before it propagates)
    if np.any(np.abs(x) > 1e6):
        return False
    # quat norm check
    q = x[3:7]
    n = float(np.linalg.norm(q))
    if not (0.5 <= n <= 2.0):  # loose bounds; we'll renormalize anyway
        return False
    return True

def normalize_quat_wxyz(q_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < eps:
        # fallback to identity
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def safe_apply_noise_state(x_state: np.ndarray) -> np.ndarray:
    """Call apply_noise but ensure quaternion is normalized and catch SciPy errors."""
    x = x_state.copy()
    x[3:7] = normalize_quat_wxyz(x[3:7])
    try:
        return apply_noise(x)
    except ValueError as e:
        # SciPy: "Found zero norm quaternions in `quat`."
        raise TrialNumericalError(str(e)) from e

# ----------------------- Helpers -----------------------
def rel_residual(A, x, b, eps=1e-12):
    """Relative residual ||Ax-b|| / max(||b||, eps)."""
    r = A @ x - b
    den = max(float(np.linalg.norm(b)), eps)
    return float(np.linalg.norm(r) / den)



def closest_spd(theta, epsilon=1e-6):
    """
    Project the symmetric inertia matrix (I) implied by theta onto SPD by clipping
    eigenvalues, then return theta with projected inertia components.
    theta = [m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    """
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals = np.clip(eigvals, epsilon, None)
    A_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Ixx_spd, Iyy_spd, Izz_spd = A_spd[0, 0], A_spd[1, 1], A_spd[2, 2]
    Ixy_spd, Ixz_spd, Iyz_spd = A_spd[0, 1], A_spd[0, 2], A_spd[1, 2]
    return np.array([m, cx, cy, cz, Ixx_spd, Iyy_spd, Izz_spd, Ixy_spd, Ixz_spd, Iyz_spd])


def generate_figure8(T, dt, rng, A=None, B=None, z0=None, omega=None):
    """
    Parametric figure-8 in XY with constant Z.
    Returns:
        t        (Tsteps,)
        pos_ref  (Tsteps,3)
        vel_ref  (Tsteps,3)
    """
    A = A if A is not None else rng.uniform(0.4, 0.8)
    B = B if B is not None else rng.uniform(0.2, 0.4)
    z0 = z0 if z0 is not None else rng.uniform(-0.05, 0.05)
    omega_base = 2 * np.pi / 20.0
    omega = omega if omega is not None else omega_base * rng.uniform(0.8, 1.25)

    t = np.arange(0, T, dt)
    x_ref = A * np.sin(omega * t)
    y_ref = B * np.sin(2 * omega * t)
    z_ref = z0 * np.ones_like(t)
    vx_ref = A * omega * np.cos(omega * t)
    vy_ref = 2 * B * omega * np.cos(2 * omega * t)
    vz_ref = np.zeros_like(t)
    pos_ref = np.vstack([x_ref, y_ref, z_ref]).T
    vel_ref = np.vstack([vx_ref, vy_ref, vz_ref]).T
    return t, pos_ref, vel_ref


def get_error_state(quad: QuadrotorDynamics, x: np.ndarray, xg: np.ndarray):
    """
    Build 12D LQR error about goal state xg.
    State layout in x: [pos(3), quat(wxyz)(4), vel(3), omega(3)]
    """
    pos_err = x[0:3] - xg[0:3]
    phi = quad.qtorp(x[3:7])        # small-angle rotation param from quaternion
    vel_err = x[7:10] - xg[7:10]
    om_err = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])


# ----------------------- Single Trial (DEKA only) -----------------------
def run_single_trial_deka(
    trial_idx: int,
    base_seed: int,
    est_freq: int,
    window: int,
    deka_params: dict,
) -> dict:
    rng = np.random.default_rng(base_seed + trial_idx)

    quad = QuadrotorDynamics()
    ug = quad.hover_thrust_true.copy()

    # LQR around hover
    xg0 = np.hstack([np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(6)])
    A_lin, B_lin = quad.get_linearized_true(xg0, ug)
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])
    Q = np.diag(1.0 / max_dev_x**2)
    Rmat = np.diag(1.0 / max_dev_u**2)
    lqr = LQRController(A_lin, B_lin, Q, Rmat)

    # Trajectory
    dt = quad.dt
    T = 20.0
    t, pos_ref, vel_ref = generate_figure8(T, dt, rng)

    # Initial state
    x_true = np.zeros(13)
    x_true[0:3] = pos_ref[0] + 0.1 * rng.standard_normal(3)
    phi0 = 0.02 * rng.standard_normal(3)
    x_true[3:7] = normalize_quat_wxyz(quad.rptoq(phi0))  # ensure normalized
    x_true[7:10] = vel_ref[0] + 0.01 * rng.standard_normal(3)
    x_true[10:13] = 0.01 * rng.standard_normal(3)

    if not is_valid_state(x_true):
        raise TrialNumericalError("Invalid initial state")

    x_meas = safe_apply_noise_state(x_true)

    # Randomized payload event
    add_payload_step = rng.integers(200, 301)
    drop_payload_step = rng.integers(500, 701)
    payload_m = rng.uniform(0.01, 0.02)
    payload_dr = rng.uniform(-0.002, 0.002, size=3)

    # DEKA estimator
    theta_0 = quad.get_true_inertial_params()
    estimator = DEKA(num_params=theta_0.shape[0], x0=theta_0, **deka_params)

    # Rolling buffers
    A_buf = deque(maxlen=window)
    B_buf = deque(maxlen=window)

    # Storage
    x_meas_traj = []
    x_ref_traj = []
    est_err_traj = []

    u = ug.copy()
    for k in range(len(t)):
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1, 0, 0, 0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        x_ref_traj.append(xg.copy())
        x_meas_traj.append(x_meas.copy())

        # control
        x_err12 = get_error_state(quad, x_meas, xg)
        u = ug + lqr.control(x_err12)

        # propagate
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=quad.dt)

        # guard & normalize quaternion
        if not np.all(np.isfinite(x_true_next)):
            raise TrialNumericalError("NaN/Inf in x_true_next")
        x_true_next[3:7] = normalize_quat_wxyz(x_true_next[3:7])

        # optional magnitude guard (prevents runaway angular velocities, etc.)
        if np.any(np.abs(x_true_next) > 1e6):
            raise TrialNumericalError("State magnitude exploded")

        # measure (safe)
        x_meas_next = safe_apply_noise_state(x_true_next)

        # build regression (A, b)
        dx_meas = (x_meas_next - x_meas) / quad.dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)
        b_vec = quad.get_force_vector(x_meas, dx_meas, u)
        A_buf.append(A_mat)
        B_buf.append(b_vec)

        # estimation update (skip k=0)
        if (k % est_freq == 0) and (k > 0):
            A_stack = np.vstack(list(A_buf))
            b_stack = np.concatenate(list(B_buf), axis=0)

            theta_est = estimator.iterate(A_stack, b_stack)
            theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
            theta_est = closest_spd(theta_est)

            quad.set_estimated_inertial_params(theta_est)
            ug = quad.get_hover_thrust_est()
            A_new, B_new = quad.get_linearized_est(xg, ug)
            lqr.update_linearized_dynamics(A_new, B_new)

            est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))

        # event
        if k == add_payload_step:
            quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
        elif k == drop_payload_step:
            quad.detach_payload()

        x_true = x_true_next
        x_meas = x_meas_next

    x_meas_traj = np.array(x_meas_traj)  # (T,13)
    x_ref_traj = np.array(x_ref_traj)    # (T,13)

    # Errors by component
    pos_err_comp = x_meas_traj[:, 0:3] - x_ref_traj[:, 0:3]
    vel_err_comp = x_meas_traj[:, 7:10] - x_ref_traj[:, 7:10]

    # quat in state is (w,x,y,z); SciPy expects (x,y,z,w)
    qu_meas = np.roll(x_meas_traj[:, 3:7], -1, axis=1)
    qu_ref = np.roll(x_ref_traj[:, 3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler('xyz', degrees=True)
    eul_ref = R.from_quat(qu_ref).as_euler('xyz', degrees=True)
    orient_err_comp = eul_meas - eul_ref

    abs_mean_pos_err_t = np.mean(np.abs(pos_err_comp), axis=1)
    abs_mean_vel_err_t = np.mean(np.abs(vel_err_comp), axis=1)
    abs_mean_ori_err_t = np.mean(np.abs(orient_err_comp), axis=1)

    return {
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "est_err_t": np.array(est_err_traj),
        "x_meas_traj": x_meas_traj,
        "x_ref_traj": x_ref_traj,
        "t": t,
        "event_idx": (int(add_payload_step), int(drop_payload_step)),
    }

# ----------------------- Runner -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results_deka_grid",
                    help="Directory to save NPZ results")
    ap.add_argument("--ntrials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--est_freq", type=int, default=20,
                    help="Estimator cadence (steps)")
    ap.add_argument("--window", type=int, default=20,
                    help="Rolling window length (steps)")
    ap.add_argument("--save_traj", action="store_true",
                    help="Save (N,T,13) trajectories for 3D viz")

    # --- DEKA grid flags ---
    ap.add_argument("--deka_gamma", type=float, nargs="*", default=[0.5, 1.0],
                    help="DEKA gamma (damping) values")
    ap.add_argument("--deka_alpha", type=float, nargs="*", default=[0.3, 0.6, 0.9],
                    help="DEKA alpha (EMA smoothing) values")
    ap.add_argument("--deka_tol", type=float, nargs="*", default=[1e-4, 1e-6, 1e-8],
                    help="DEKA residual^2 early-stop tolerance")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N_TRIALS = int(args.ntrials)
    BASE_SEED = int(args.seed)
    est_freq = int(args.est_freq)
    window = int(args.window)

    # Probe time axes (for saving)
    probe = QuadrotorDynamics()
    T = 20.0
    t_full = np.arange(0, T, probe.dt)
    t_est = t_full[::est_freq][1:]  # skip t=0 in updates

    # Build DEKA grid (Cartesian product)
    grid = [
        {"gamma": g, "alpha": a, "tol": tol}
        for g, a, tol in product(
            args.deka_gamma, args.deka_alpha, args.deka_tol
        )
    ]

    def label(p):
        # label strings safe for NPZ keys
        return (f"deka_g{p['gamma']:.3g}"
                f"_a{p['alpha']:.2f}"
                f"_t{p['tol']:.1e}").replace("+", "")

    labels = [label(p) for p in grid]

    # Storage per grid label
    store = {
        lab: {"pos": [], "vel": [], "ori": [], "est": [],
              "xm": [] if args.save_traj else None,
              "xr": [] if args.save_traj else None}
        for lab in labels
    }
    event_idx_trials = np.full((N_TRIALS, 2), -1, dtype=int)

    for p, lab in zip(grid, labels):
        print(f"\nRunning {lab} trials...")
        pbar = tqdm(total=N_TRIALS, desc=lab, leave=False)
        successes = 0
        # we keep trying until we accumulate N_TRIALS successes or exceed retry budget
        attempts = 0
        while successes < N_TRIALS:
            try:
                out = run_single_trial_deka(
                    trial_idx=successes,  # keep seed progression per success
                    base_seed=BASE_SEED,
                    est_freq=est_freq,
                    window=window,
                    deka_params=p,
                )
                store[lab]["pos"].append(out["abs_mean_pos_err_t"])
                store[lab]["vel"].append(out["abs_mean_vel_err_t"])
                store[lab]["ori"].append(out["abs_mean_ori_err_t"])
                store[lab]["est"].append(out["est_err_t"])
                if args.save_traj:
                    store[lab]["xm"].append(out["x_meas_traj"])
                    store[lab]["xr"].append(out["x_ref_traj"])

                if event_idx_trials[successes, 0] == -1:
                    event_idx_trials[successes] = out["event_idx"]

                successes += 1
                pbar.update(1)
                attempts = 0  # reset attempts after a success
            except TrialNumericalError as e:
                attempts += 1
                if attempts > 5: # max retries
                    # log and move on to the next trial index to avoid stalling
                    print(f"[WARN] {lab}: trial {successes} failed {attempts-1} times; skipping.")
                    # advance to next trial index but DON'T count as success; try again
                    attempts = 0
                    # Optionally, bump the base seed for the next attempt to decorrelate
                    BASE_SEED += 17
                else:
                    # just retry the same success index with a different seed offset
                    BASE_SEED += 1
                    continue
        pbar.close()

    # Pack & save
    save_kwargs = {
        "labels": np.array(labels),
        "ntrials": np.array([N_TRIALS]),
        "base_seed": np.array([BASE_SEED]),
        "est_freq": np.array([est_freq]),
        "window": np.array([window]),
        "t": t_full,
        "t_est": t_est,
        "event_idx": event_idx_trials,           # (N_TRIALS, 2)
        "event_time": t_full[event_idx_trials],  # (N_TRIALS, 2)
        "event_labels": np.array(["add", "drop"]),
        # store the grid explicitly (aligned with 'labels')
        "deka_gamma": np.array([p["gamma"] for p in grid]),
        "deka_alpha": np.array([p["alpha"] for p in grid]),
        "deka_tol": np.array([p["tol"] for p in grid]),
    }

    for lab in labels:
        save_kwargs[f"{lab}__pos"] = np.stack(store[lab]["pos"], axis=0)  # (N_TRIALS, Tsteps)
        save_kwargs[f"{lab}__vel"] = np.stack(store[lab]["vel"], axis=0)
        save_kwargs[f"{lab}__ori"] = np.stack(store[lab]["ori"], axis=0)
        save_kwargs[f"{lab}__est"] = np.stack(store[lab]["est"], axis=0)  # (N_TRIALS, Uupdates)
        if args.save_traj:
            save_kwargs[f"{lab}__xm"] = np.stack(store[lab]["xm"], axis=0)  # (N_TRIALS, Tsteps, 13)
            save_kwargs[f"{lab}__xr"] = np.stack(store[lab]["xr"], axis=0)

    out_npz = (Path(args.outdir) / "results_deka_grid.npz").resolve()
    np.savez_compressed(out_npz, **save_kwargs)
    print(f"\nSaved results to: {out_npz}")


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True)
    main()
