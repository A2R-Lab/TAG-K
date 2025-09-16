#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from collections import deque

from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import *   # apply_noise
from python.common.estimation_methods import *  # RLS, KF, RK, TARK

# ----------------------- Config -----------------------
BASE_SEED = 12345
N_TRIALS  = 100
ALGOS     = ["kf", "none", "lstsq", "rk", "tark", "rls", "gt"]  # includes "none" and "gt"
EVENT = "drift"  # "payload" or "wind" or "drift"
np.set_printoptions(precision=7, suppress=True)

# ----------------------- Helpers -----------------------
def mean_relative_error(x_hat, x_true, eps=1e-18):
    return np.mean(np.abs(x_hat - x_true) / (np.abs(x_true) + eps))

def closest_spd(theta, epsilon=1e-6):
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    A_sym = 0.5*(A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals = np.clip(eigvals, epsilon, None)
    A_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Ixx_spd, Iyy_spd, Izz_spd = A_spd[0,0], A_spd[1,1], A_spd[2,2]
    Ixy_spd, Ixz_spd, Iyz_spd = A_spd[0,1], A_spd[0,2], A_spd[1,2]
    return np.array([m, cx, cy, cz, Ixx_spd, Iyy_spd, Izz_spd, Ixy_spd, Ixz_spd, Iyz_spd])

def generate_figure8(T, dt, rng, A=None, B=None, z0=None, omega=None):
    A = A if A is not None else rng.uniform(0.4, 0.8)
    B = B if B is not None else rng.uniform(0.2, 0.4)
    z0 = z0 if z0 is not None else rng.uniform(-0.05, 0.05)
    omega_base = 2*np.pi/20.0
    omega = omega if omega is not None else omega_base * rng.uniform(0.8, 1.25)

    t = np.arange(0, T, dt)
    x_ref = A * np.sin(omega * t)
    y_ref = B * np.sin(2*omega * t)
    z_ref = z0 * np.ones_like(t)
    vx_ref = A * omega * np.cos(omega * t)
    vy_ref = 2 * B * omega * np.cos(2*omega * t)
    vz_ref = np.zeros_like(t)
    pos_ref = np.vstack([x_ref, y_ref, z_ref]).T
    vel_ref = np.vstack([vx_ref, vy_ref, vz_ref]).T
    return t, pos_ref, vel_ref

def get_error_state(quad, x, xg):
    pos_err = x[0:3] - xg[0:3]
    phi     = quad.qtorp(x[3:7])
    vel_err = x[7:10] - xg[7:10]
    om_err  = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])

def moving_average(x, window_size=20):
    if window_size <= 1:
        return x
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

# ----------------------- Single Trial -----------------------
def run_single_trial(estimator_option: str, trial_idx: int):
    """
    Returns per-time arrays:
      abs_mean_pos_err_t, abs_mean_vel_err_t, abs_mean_ori_err_t,
      gt_force_err_t, est_force_err_t
    """
    rng = np.random.default_rng(BASE_SEED + trial_idx)

    quad = QuadrotorDynamics()
    ug   = quad.hover_thrust_true.copy()

    # LQR around hover
    xg0 = np.hstack([np.zeros(3), np.array([1,0,0,0]), np.zeros(6)])
    A_lin, B_lin = quad.get_linearized_true(xg0, ug)
    max_dev_x = np.array([0.1,0.1,0.1,  0.5,0.5,0.05,  0.5,0.5,0.5,  0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0/max_dev_x**2)
    Rmat = np.diag(1.0/max_dev_u**2)
    lqr = LQRController(A_lin, B_lin, Q, Rmat)

    # Trajectory (randomized)
    dt = quad.dt
    T  = 20.0
    t, pos_ref, vel_ref = generate_figure8(T, dt, rng)

    # Initial state
    x_true = np.zeros(13)
    x_true[0:3]   = pos_ref[0] + 0.1*rng.standard_normal(3)
    phi0          = 0.02*rng.standard_normal(3)
    x_true[3:7]   = quad.rptoq(phi0)
    x_true[7:10]  = vel_ref[0] + 0.01*rng.standard_normal(3)
    x_true[10:13] = 0.01*rng.standard_normal(3)
    x_meas = apply_noise(x_true)

    # Randomized event
    event_step = rng.integers(200, 501)
       

    # Estimator setup
    theta_true = quad.get_true_inertial_params()
    theta_est  = None
    estimator  = None
    if estimator_option == "rls":
        estimator = RLS(num_params=theta_true.shape[0], theta_hat=theta_true, forgetting_factor=0.7, c=1000)
    elif estimator_option == "kf":
        estimator = KF(num_params=theta_true.shape[0], process_noise=1e-4*np.eye(theta_true.shape[0]),
                       measurement_noise=1e-4*np.eye(20*6), theta_hat=theta_true, c=1000)
    elif estimator_option == "rk":
        estimator = RK(num_params=theta_true.shape[0], x0=theta_true)
    elif estimator_option == "tark":
        estimator = TARK(num_params=theta_true.shape[0], x0=theta_true)
    # "lstsq", "gt", "none" handled below

    # ---- NEW: rolling buffers (window = 20) ----
    A_buf = deque(maxlen=20)   # each element shape: (m, p)
    B_buf = deque(maxlen=20)   # each element shape: (m,)

    # Storage
    x_meas_traj = []
    x_ref_traj  = []
    gt_err_traj  = []
    est_err_traj = []

    u = ug.copy()
    for k in range(len(t)):
        # goal
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1,0,0,0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        x_ref_traj.append(xg.copy())
        x_meas_traj.append(x_meas.copy())

        # control on measured
        x_err12 = get_error_state(quad, x_meas, xg)
        last_u = u.copy()
        u = ug + lqr.control(x_err12)
        u = 0.5*u + 0.5*last_u  # LPF

        # propagate
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)
        x_meas_next = apply_noise(x_true_next)

        # residual on measured
        dx_meas = (x_meas_next - x_meas) / dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)
        A_norm = np.linalg.norm(A_mat)
        if A_norm == 0.0: A_norm = 1.0
        A_mat /= A_norm
        b_vec  = quad.get_force_vector(x_meas, dx_meas, u) / A_norm

        # ---- NEW: push into buffers ----
        A_buf.append(A_mat)
        B_buf.append(b_vec)

        # GT residual (for logging/plots)
        theta_gt = quad.get_true_inertial_params()
        gt_err_traj.append(mean_relative_error(A_mat @ theta_gt, b_vec))

        # Update estimates every 20 steps (or as you prefer)
        if k % 20 == 0 and k > 0:
            # ---- NEW: stack the window ----
            A_stack = np.vstack(list(A_buf))            # (sum m_i, p)
            b_stack = np.concatenate(list(B_buf), axis=0)  # (sum m_i,)

            if estimator_option == "gt":
                theta_est = theta_gt.copy()
                ug = quad.get_hover_thrust_true()
                A_new, B_new = quad.get_linearized_true(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

            elif estimator_option == "lstsq" and len(A_buf) > 0:
                theta_est = np.linalg.lstsq(A_stack, b_stack, rcond=None)[0]
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

            elif estimator_option in ("rk", "tark", "rls", "kf") and len(A_buf) > 0:
                # Assumes estimator.iterate can take (M,p) and (M,) just like before
                theta_est = estimator.iterate(A_stack, b_stack)
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)
                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

            elif estimator_option == "none":
                theta_est = None  # no updates

        # record est residual (NaN if none)
        if theta_est is None:
            est_err_traj.append(np.nan)
        else:
            est_err_traj.append(mean_relative_error(A_mat @ theta_est, b_vec))

        # payload event
        if k == event_step:
            if EVENT =="payload":
                payload_m  = rng.uniform(0.005, 0.015)
                payload_dr = rng.uniform(-0.001, 0.001, size=3)
                quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
            if EVENT == "wind":
                wind_vec = np.array([rng.uniform(2,20),0,0]) #subject to change
                quad.aero_added_inertia(wind_vec=wind_vec)
            if EVENT == "drift":
                mass_std = rng.uniform(1e-5, 5e-5)
                intertia_std = rng.uniform(1e-8, 5e-8)
                com_std = rng.uniform(1e-5, 5e-5)
                quad.add_drift(mass_std=mass_std, intertia_std=intertia_std, com_std=com_std)

        x_true = x_true_next
        x_meas = x_meas_next

    # Arrays
    x_meas_traj = np.array(x_meas_traj)
    x_ref_traj  = np.array(x_ref_traj)
    gt_err_traj  = np.array(gt_err_traj)
    est_err_traj = np.array(est_err_traj)

    # Component errors
    pos_err_comp = x_meas_traj[:, 0:3]   - x_ref_traj[:, 0:3]
    vel_err_comp = x_meas_traj[:, 7:10]  - x_ref_traj[:, 7:10]

    qu_meas = np.roll(x_meas_traj[:, 3:7], -1, axis=1)  # (wxyz)->(xyzw)
    qu_ref  = np.roll(x_ref_traj[:,  3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler('xyz', degrees=True)
    eul_ref  = R.from_quat(qu_ref ).as_euler('xyz', degrees=True)
    orient_err_comp = eul_meas - eul_ref

    # Per-time metrics (avg over value dims; absolute for pos/vel/ori)
    abs_mean_pos_err_t = np.mean(np.abs(pos_err_comp),    axis=1)
    abs_mean_vel_err_t = np.mean(np.abs(vel_err_comp),    axis=1)
    abs_mean_ori_err_t = np.mean(np.abs(orient_err_comp), axis=1)

    return {
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "gt_force_err_t":     gt_err_traj,
        "est_force_err_t":    est_err_traj
    }

# ----------------------- Runner -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="plots_param_estimation",
                        help="Directory to save output plots")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # probe for time axis
    probe = QuadrotorDynamics()
    T = 20.0
    t = np.arange(0, T, probe.dt)

    # storage
    results = {
        algo: {
            "pos_err_trials": [],
            "vel_err_trials": [],
            "ori_err_trials": [],
            "gt_force_err_trials": [],
            "est_force_err_trials": []
        } for algo in ALGOS
    }

    for algo in ALGOS:
        print(f"\nRunning {algo} trials...")
        for trial in tqdm(range(N_TRIALS), desc=f"{algo}", leave=False):
            out = run_single_trial(algo, trial_idx=trial)
            results[algo]["pos_err_trials"].append(out["abs_mean_pos_err_t"])
            results[algo]["vel_err_trials"].append(out["abs_mean_vel_err_t"])
            results[algo]["ori_err_trials"].append(out["abs_mean_ori_err_t"])
            results[algo]["gt_force_err_trials"].append(out["gt_force_err_t"])
            results[algo]["est_force_err_trials"].append(out["est_force_err_t"])

    # to arrays (N_TRIALS, T)
    for algo in ALGOS:
        for key in results[algo]:
            results[algo][key] = np.vstack(results[algo][key])

    # trial-averaged time series
    trial_avg = {}
    for algo in ALGOS:
        trial_avg[algo] = {
            "pos":      np.nanmean(results[algo]["pos_err_trials"],      axis=0),
            "vel":      np.nanmean(results[algo]["vel_err_trials"],      axis=0),
            "ori":      np.nanmean(results[algo]["ori_err_trials"],      axis=0),
            "force_gt": np.nanmean(results[algo]["gt_force_err_trials"], axis=0),
            "force_est":np.nanmean(results[algo]["est_force_err_trials"],axis=0),
        }

    # epsilon to avoid log(0) in plots
    eps = 1e-12
    def _safe(y): return np.clip(y, eps, None)

    # --------- Plots (saved to outdir, log scale for pos/vel/ori) ----------
    # Abs mean position error
    plt.figure(figsize=(9,5))
    for algo in ALGOS:
        plt.plot(t, _safe(trial_avg[algo]["pos"]), label=f"{algo}")
    plt.grid(True, which="both"); plt.xlabel("Time [s]")
    plt.ylabel("Abs mean pos err [m]"); plt.yscale("log")
    plt.title(f"Abs Mean Position Error (avg over dims & {N_TRIALS} trials)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "abs_mean_pos_error_log.png", dpi=200)
    plt.close()

    # Abs mean velocity error
    plt.figure(figsize=(9,5))
    for algo in ALGOS:
        plt.plot(t, _safe(trial_avg[algo]["vel"]), label=f"{algo}")
    plt.grid(True, which="both"); plt.xlabel("Time [s]")
    plt.ylabel("Abs mean vel err [m/s]"); plt.yscale("log")
    plt.title(f"Abs Mean Velocity Error (avg over dims & {N_TRIALS} trials)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "abs_mean_vel_error_log.png", dpi=200)
    plt.close()

    # Abs mean orientation error (deg)
    plt.figure(figsize=(9,5))
    for algo in ALGOS:
        plt.plot(t, _safe(trial_avg[algo]["ori"]), label=f"{algo}")
    plt.grid(True, which="both"); plt.xlabel("Time [s]")
    plt.ylabel("Abs mean ori err [deg]"); plt.yscale("log")
    plt.title(f"Abs Mean Orientation Error (avg over roll/pitch/yaw & {N_TRIALS} trials)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "abs_mean_orientation_error_log.png", dpi=200)
    plt.close()

    # Mean relative force error (log y for readability)
    plt.figure(figsize=(9,5))
    for algo in ALGOS:
        y = trial_avg[algo]["force_est"]
        if np.isnan(y).all():  # none/gt before estimate
            y = trial_avg[algo]["force_gt"]
            label = f"{algo} (GT residual)"
        else:
            label = f"{algo} (Est residual)"
        plt.plot(t, moving_average(_safe(y)), label=label)
    plt.yscale("log")
    plt.grid(True, which="both"); plt.xlabel("Time [s]"); plt.ylabel("Mean relative force error")
    plt.title(f"Residual Force Error (avg over {N_TRIALS} trials)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mean_relative_force_error_log.png", dpi=200)
    plt.close()
    # --------- CDF of Position Error (all trials, all times) ----------
    plt.figure(figsize=(8,5))
    for algo in ALGOS:
        # Flatten all trials and all times into a single array
        pos_err_flat = results[algo]["pos_err_trials"].flatten()
        pos_err_flat = pos_err_flat[~np.isnan(pos_err_flat)]  # remove NaNs
        if len(pos_err_flat) == 0:
            continue
        sorted_err = np.sort(pos_err_flat)
        cdf = np.linspace(0, 1, len(sorted_err))
        plt.plot(sorted_err, cdf, label=algo)
    # --------- CDF of Velocity Error ----------
    plt.figure(figsize=(8,5))
    for algo in ALGOS:
        vel_err_flat = results[algo]["vel_err_trials"].flatten()
        vel_err_flat = vel_err_flat[~np.isnan(vel_err_flat)]
        if len(vel_err_flat) == 0:
            continue
        sorted_err = np.sort(vel_err_flat)
        cdf = np.linspace(0, 1, len(sorted_err))
        plt.plot(sorted_err, cdf, label=algo)
    plt.xlabel("Abs mean velocity error [m/s]")
    plt.ylabel("CDF")
    plt.title(f"CDF of Velocity Error (all trials, all times)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cdf_vel_error.png", dpi=200)
    plt.close()

    # --------- CDF of Orientation Error ----------
    plt.figure(figsize=(8,5))
    for algo in ALGOS:
        ori_err_flat = results[algo]["ori_err_trials"].flatten()
        ori_err_flat = ori_err_flat[~np.isnan(ori_err_flat)]
        if len(ori_err_flat) == 0:
            continue
        sorted_err = np.sort(ori_err_flat)
        cdf = np.linspace(0, 1, len(sorted_err))
        plt.plot(sorted_err, cdf, label=algo)
    plt.xlabel("Abs mean orientation error [deg]")
    plt.ylabel("CDF")
    plt.title(f"CDF of Orientation Error (all trials, all times)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cdf_ori_error.png", dpi=200)
    plt.close()

    # --------- CDF of Relative Force Error ----------
    plt.figure(figsize=(8,5))
    for algo in ALGOS:
        force_err_flat = results[algo]["est_force_err_trials"].flatten()
        force_err_flat = force_err_flat[~np.isnan(force_err_flat)]
        if len(force_err_flat) == 0:
            continue
        sorted_err = np.sort(force_err_flat)
        cdf = np.linspace(0, 1, len(sorted_err))
        plt.plot(sorted_err, cdf, label=algo)
    #limit x-axis for better visibility
    plt.xlim(0, 6)
    plt.xlabel("Relative force error")
    plt.ylabel("CDF")
    plt.title(f"CDF of Relative Force Error (all trials, all times)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cdf_force_error.png", dpi=200)
    plt.close()
    # --------- Scalar summary (also averaged over time) ----------
    print("\n=== Overall (time- & trial-averaged) metrics ===")
    print(f"(all errors are already averaged over value dimensions)")
    header = f"{'algo':<8} {'pos[m]':>10} {'vel[m/s]':>12} {'ori[deg]':>10} {'force(rel)':>12}"
    print(header)
    print("-" * len(header))
    for algo in ALGOS:
        pos_s   = float(np.nanmean(trial_avg[algo]["pos"]))
        vel_s   = float(np.nanmean(trial_avg[algo]["vel"]))
        ori_s   = float(np.nanmean(trial_avg[algo]["ori"]))
        force_s = trial_avg[algo]["force_est"]
        if np.isnan(force_s).all():
            force_s = trial_avg[algo]["force_gt"]
        force_s = float(np.nanmean(force_s))
        print(f"{algo:<8} {pos_s:10.4e} {vel_s:12.4e} {ori_s:10.4e} {force_s:12.4e}")
    print(f"\nSaved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    np.random.seed(BASE_SEED)  # for any global noise usage
    main()
