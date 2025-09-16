#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch simulator for quadrotor parameter-estimation trials.

Key features in this revision
- Adds --noise CLI (choices: none, low, high) and passes it through end-to-end.
- Refactors for readability: type hints, docstrings, smaller helpers, consistent prints.
- Saves noise level into the output .npz for traceability.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Tuple
import traceback
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from concurrent.futures import as_completed  # alongside FIRST_COMPLETED, wait

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root
# Project imports
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import apply_noise  # not used yet; kept for future wiring
from python.common.estimation_methods import *  # RLS, KF, RK, TARK, IGRK, MGRK, REK, DEKA
from python.common.make_trajectories import * 
from python.common.viz_results import visualize_3d_traj
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
np.set_printoptions(precision=7, suppress=False)


ALGOS_DEFAULT: List[str] = ["tagrk", "tagrk_5", "tagrk_10", "tagrk_15", "tagrk_20", "tagrk_25", "rk", "grk", "tark"] # "rk", "grk", "tark",
REF_CHOICES: List[str] = ["figure8", "circle", "spiral", "ellipse", "helix"] # hover, "lissajous"
NOISE_LEVELS: List[str] = ["low","medium", "high", "none"] # NOTE: controlled by cli
NoiseLevel = Literal["high", "medium", "low", "none"]



def abs_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Absolute residual ||Ax-b||."""
    r = A @ x - b
    return np.linalg.norm(r)

def rel_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Relative residual ||Ax-b|| / max(||b||, eps)."""
    r = A @ x - b
    return np.linalg.norm(r) / max(np.linalg.norm(b), eps)

def scaled_cond(A):
    """Scaled condition number kappa(A) = ||A||_F * ||A^{-1}||_2."""
    A = np.asarray(A, float)
    frob_norm = np.linalg.norm(A, 'fro')          # Frobenius norm
    inv_2norm = np.linalg.norm(np.linalg.inv(A), 2)  # spectral norm of A^{-1}
    return frob_norm * inv_2norm

def closest_spd(theta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Project inertia submatrix to nearest SPD via symmetrization + eigenvalue clamp.
    theta layout: [m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    """
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    A_sym = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A_sym)
    w = np.clip(w, epsilon, None)
    A_spd = V @ np.diag(w) @ V.T
    Ixx_spd, Iyy_spd, Izz_spd = A_spd[0, 0], A_spd[1, 1], A_spd[2, 2]
    Ixy_spd, Ixz_spd, Iyz_spd = A_spd[0, 1], A_spd[0, 2], A_spd[1, 2]
    return np.array([m, cx, cy, cz, Ixx_spd, Iyy_spd, Izz_spd, Ixy_spd, Ixz_spd, Iyz_spd])

def gate_param_update(theta, A, b, rel_resid_max=0.1, verbose=False):
    m,cx,cy,cz,Ixx,Iyy,Izz,Ixy,Ixz,Iyz = theta
    # import pdb; pdb.set_trace()

    # basic physical bounds (tune to your platform)
    if not (0.02 <= m <= 0.08): 
        if verbose:
            print(f"Reject: mass {m:.4f} out of bounds [0.02, 0.08]")
        return False
    if np.linalg.norm([cx,cy,cz]) > 0.02: 
        if verbose:
            print(f"Reject: center of mass [{cx:.4f}, {cy:.4f}, {cz:.4f}] out of bounds [-0.02, 0.02]")
        return False
    J = np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
    # SPD + magnitude clamp (tune)
    w = np.linalg.eigvalsh(0.5*(J+J.T))
    if (w < 1e-7).any() or (w > 1e-4).any(): # TODO: change lower to 1e-5
        if verbose:
            print(f"Reject: inertia matrix eigenvalues {w} out of bounds [1e-7, 1e-4]")
        return False

    # numerics
    rel = np.linalg.norm(A@theta - b) / max(np.linalg.norm(b), 1e-12)
    if rel > rel_resid_max: 
        if verbose:
            print(f"Reject: relative residual {rel:.4f} > {rel_resid_max}")
        return False
    # if np.linalg.cond(A) > condA_max: return False

    return True

TRAJ_FACTORY: Dict[str, Callable[[float, float, np.random.Generator],
                                 Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    "hover": make_traj_hover,
    "figure8": make_traj_figure8,
    "circle": make_traj_circle,
    "lissajous": make_traj_lissajous,
    "ellipse": make_traj_ellipse,
    "helix": make_traj_helix,
    "spiral": make_traj_spiral,
}


# --------------------------- small helpers ---------------------------

def get_error_state(quad: QuadrotorDynamics, x: np.ndarray, xg: np.ndarray) -> np.ndarray:
    """12D LQR error state: [pos_err(3), rpy(3), vel_err(3), omg_err(3)]."""
    pos_err = x[0:3] - xg[0:3]
    phi = quad.qtorp(x[3:7])
    vel_err = x[7:10] - xg[7:10]
    om_err = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])

def make_estimator(name: str, theta0: np.ndarray, window: int):
    """Factory for estimators; 'gt' or 'none' returns None."""
    n = int(theta0.shape[0])
    name = name.lower()

    if name == "rls_0.99":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.99, c=1000)
    if name == "rls_0.96":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.96, c=1000)
    if name == "rls_0.8":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.8, c=1000)
    if name == "rls_0.5":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.5, c=1000)
    if name == "rls_0.3":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.3, c=1000)
    if name == "kf_low":
        Q = 1e-3 * np.eye(n)
        Rm = 1e-5 * np.eye(window * 6)
        return KF(num_params=n, process_noise=Q, measurement_noise=Rm, theta_hat=theta0, c=10)
    if name == "kf_high":
        Q = 1e-1 * np.eye(n)
        Rm = 1e-3 * np.eye(window * 6)
        return KF(num_params=n, process_noise=Q, measurement_noise=Rm, theta_hat=theta0, c=10)
    if name == "rk":
        return RK(num_params=n, x0=theta0)
    if name == "tark":
        return TARK(num_params=n, x0=theta0)
    if name == "igrk":
        return IGRK(num_params=n, x0=theta0)
    if name == "mgrk":
        return MGRK(num_params=n, x0=theta0)
    if name == "rek":
        return REK(num_params=n, x0=theta0)
    if name == "deka":
        return DEKA(num_params=n, x0=theta0)
    if name == "fdbk":
        return FDBK(num_params=n, x0=theta0)
    if name == "grk":
        return GRK(num_params=n, x0=theta0)
    if name == "rk_colscaled":
        return RK_ColScaled(num_params=n, x0=theta0)
    if name == "rk_equi":
        return RK_Equi(num_params=n, x0=theta0)
    if name == "rk_tikh":
        return RK_Tikh(num_params=n, x0=theta0)
    if name == "rk_equi_tikh":
        return RK_EquiTikh(num_params=n, x0=theta0)
    if name == "grk_tikh":
        return GRK_Tikh(num_params=n, x0=theta0)
    if name == "tagrk_tikh":
        return GRK_TailAvg_Tikh(num_params=n, x0=theta0)
    if name == "tagrk":
        return GRK_TailAvg(num_params=n, x0=theta0)    
    if name == "tagrk_5":
        return GRK_TailAvg(num_params=n, x0=theta0, burnin=5)
    if name == "tagrk_10":
        return GRK_TailAvg(num_params=n, x0=theta0, burnin=10)
    if name == "tagrk_15":
        return GRK_TailAvg(num_params=n, x0=theta0, burnin=15)
    if name == "tagrk_20":
        return GRK_TailAvg(num_params=n, x0=theta0, burnin=20)
    if name == "tagrk_25":
        return GRK_TailAvg(num_params=n, x0=theta0, burnin=25)
    if name in ("gt", "none"):
        return None
    raise ValueError(f"Unknown estimator: {name}")

def make_lqr_around_hover(quad: QuadrotorDynamics) -> Tuple[LQRController, np.ndarray]:
    """Construct an LQR controller about hover and return (controller, ug)."""
    ug = quad.hover_thrust_true.copy()
    xg0 = np.hstack([np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(6)])
    A_lin, B_lin = quad.get_linearized_true(xg0, ug)

    # tuning bounds for weighting
    max_dev_x = np.array([
        0.1, 0.1, 0.1,  # pos
        0.5, 0.5, 0.05,  # rpy
        0.5, 0.5, 0.5,  # vel
        0.7, 0.7, 0.2,  # omg
    ])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])

    Q = np.diag(1.0 / max_dev_x**2)
    Rmat = np.diag(1.0 / max_dev_u**2)
    lqr = LQRController(A_lin, B_lin, Q, Rmat)
    return lqr, ug


# --------------------------- single trial ---------------------------

def run_single_trial(
    estimator_name: str,
    base_seed: int,
    seed_offset: int,
    est_freq: int,
    window: int,
    ref_type: str,
    T_sim: float = 20.0,
    noise: NoiseLevel = "none",
    *,
    verbose_ctrl: bool = False,
    verbose_est: bool = False,
) -> Dict[str, np.ndarray | int | bool | str]:
    """
    Run one simulation trial.
    Returns a dict with arrays plus:
      - 'failed' (bool) and 'fail_reason' (str)
      - 'abort_step' (int, -1 if not aborted)
    """
    rng = np.random.default_rng(base_seed + seed_offset)
    quad = QuadrotorDynamics()
    lqr, ug = make_lqr_around_hover(quad)

    dt = quad.dt
    t, pos_ref, vel_ref = TRAJ_FACTORY[ref_type](T_sim, dt, rng)

    # Initial state
    x_true = np.zeros(13)
    x_true[0:3] = pos_ref[0] + 0.05 * rng.standard_normal(3)
    x_true[3:7] = quad.rptoq(0.02 * rng.standard_normal(3))
    x_true[7:10] = vel_ref[0] + 0.01 * rng.standard_normal(3)
    x_true[10:13] = 0.01 * rng.standard_normal(3)
    x_meas = x_true.copy()

    # Randomized payload events
    add_step = rng.integers(300, 401)
    drop_step = rng.integers(700, 801)
    payload_m = rng.uniform(0.035/3, 0.035/2)
    payload_dr = rng.uniform(-0.001, 0.001, size=3)

    # Estimator
    theta0 = quad.get_true_inertial_params()
    est = make_estimator(estimator_name, theta0, window)

    # Rolling buffers
    A_buf: deque[np.ndarray] = deque(maxlen=window)
    B_buf: deque[np.ndarray] = deque(maxlen=window)

    # Storage
    x_meas_traj: List[np.ndarray] = []
    x_ref_traj: List[np.ndarray] = []
    est_err_traj: List[float] = []
    theta_gt_list: List[np.ndarray] = []
    theta_est_list: List[np.ndarray] = []
    A_snap_list: List[np.ndarray] = []
    b_snap_list: List[np.ndarray] = []

    u = ug.copy()
    abort_step = -1
    failed = False
    fail_reason = ""

    trial_seed = int(np.asarray(base_seed).item()) + int(np.asarray(seed_offset).item())
    rng = np.random.default_rng(trial_seed)

    try:
        for k in range(len(t)):
            rg, vg = pos_ref[k], vel_ref[k]
            qg, omgg = np.array([1, 0, 0, 0]), np.zeros(3)
            xg = np.hstack([rg, qg, vg, omgg])

            # ---- early deviation gate (abort counts as failure) ----
            if np.linalg.norm(x_meas[0:3] - rg) > 0.3:
                abort_step = k
                failed = True
                fail_reason = f"abort_pos_deviation@step_{k}"
                break

            x_ref_traj.append(xg.copy())
            x_meas_traj.append(x_meas.copy())

            # Control
            x_err12 = get_error_state(quad, x_meas, xg)
            u_delta = lqr.control(x_err12)
            u = np.clip(ug + u_delta, 0.0, 1.0)

            if verbose_ctrl:
                print(f"[CTRL] k={k:5d} t={t[k]:7.3f} ||e||={np.linalg.norm(x_err12):.4g}, add@{add_step}, drop@{drop_step}")
                print(f"[CTRL] ug = {ug}")
                print(f"[CTRL] u_delta = {u_delta}")
                print(f"[CTRL] u_clipped = {u}")
                print(f"[CTRL] omg_meas = {x_meas[10:13]}")

            # Propagate + measure with noise
            x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)
            x_meas_next = apply_noise(x_true_next, level=noise)

            # Build A, b from measurements
            dx_meas = (x_meas_next - x_meas) / dt
            A_mat = quad.get_data_matrix(x_meas, dx_meas)     # (6,10)
            b_vec = quad.get_force_vector(x_meas, dx_meas, u) # (6,)

            A_buf.append(A_mat)
            B_buf.append(b_vec)

            # Estimator update
            if (k % est_freq == 0) and (k > 0):
                A_stack = np.vstack(list(A_buf))               # (6*window, 10)
                b_stack = np.concatenate(list(B_buf), axis=0)  # (6*window,)

                A_snap_list.append(A_stack.copy())
                b_snap_list.append(b_stack.copy())

                theta_gt = quad.get_true_inertial_params()
                theta_est_cur = np.full_like(theta_gt, np.nan)

                if estimator_name == "gt":
                    est_err_traj.append(rel_residual(A_stack, theta_gt, b_stack))
                    ug = quad.get_hover_thrust_true()
                    A_new, B_new = quad.get_linearized_true(xg, ug)
                    lqr.update_linearized_dynamics(A_new, B_new)
                    theta_est_cur = theta_gt.copy()

                elif estimator_name == "none":
                    est_err_traj.append(rel_residual(A_stack, theta0, b_stack))

                else:
                    theta_est = est.iterate(A_stack, b_stack)
                    theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                    theta_est = closest_spd(theta_est)

                    if gate_param_update(theta_est, A_stack, b_stack):
                        quad.set_estimated_inertial_params(theta_est)
                        ug = quad.get_hover_thrust_est()
                        A_new, B_new = quad.get_linearized_est(xg, ug)
                        lqr.update_linearized_dynamics(A_new, B_new)
                    # keep K update conservative but consistent

                    if verbose_est: 
                        r_est = A_stack @ theta_est - b_stack
                        rr_est = rel_residual(A_stack, theta_est, b_stack)
                        print(f"[EST] k={k:5d} ‖r_est‖={np.linalg.norm(r_est):.4e} rel={rr_est:.4e}")
                        print(f"[EST] theta_gt = {theta_gt}")
                        print(f"[EST] theta_est= {theta_est}")
                        print(f"[EST] update controller = {gate_param_update(theta_est, A_stack, b_stack)}")
                        # print(f"[EST] scaled cond(A) = {scaled_cond(A_stack):.4e}")

                    est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))
                    theta_est_cur = theta_est.copy()

                theta_gt_list.append(theta_gt.copy())
                theta_est_list.append(theta_est_cur)

            # Payload events
            if k == add_step:  
                quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
            if k == drop_step: 
                quad.detach_payload()
            if k == drop_step - 10:
                if np.linalg.norm(x_meas[0:3] - rg) > 0.05: # greater then 5cm
                    failed = True
                    fail_reason = f"failed_to_adapt_params@step_{k}"

            x_true = x_true_next
            x_meas = x_meas_next

    except Exception as e:
        # Any exception ➜ failure, but we still return shaped arrays
        failed = True
        fail_reason = f"exception:{repr(e)}"

    # ----------------- PACK RESULTS (pad safely) -----------------
    T = len(t)
    N_done = len(x_meas_traj)
    x_meas_arr = np.full((T, 13), np.nan)
    x_ref_arr  = np.full((T, 13), np.nan)
    if N_done > 0:
        x_meas_arr[:N_done] = np.asarray(x_meas_traj)
        x_ref_arr[:N_done]  = np.asarray(x_ref_traj)

    pos_err = x_meas_arr[:, 0:3] - x_ref_arr[:, 0:3]
    vel_err = x_meas_arr[:, 7:10] - x_ref_arr[:, 7:10]
    ori_err = np.full((T, 3), np.nan)
    if N_done > 0:
        qu_meas = np.roll(x_meas_arr[:N_done, 3:7], -1, axis=1)  # [x,y,z,w]
        qu_ref  = np.roll(x_ref_arr [:N_done, 3:7], -1, axis=1)
        try:
            eul_meas = R.from_quat(qu_meas).as_euler("xyz", degrees=True)
            eul_ref  = R.from_quat(qu_ref ).as_euler("xyz", degrees=True)
            ori_err[:N_done] = eul_meas - eul_ref
        except Exception:
            pass

    with np.errstate(invalid="ignore", divide="ignore"):
        abs_mean_pos_err_t = np.nanmean(np.abs(pos_err), axis=1)
        abs_mean_vel_err_t = np.nanmean(np.abs(vel_err), axis=1)
        abs_mean_ori_err_t = np.nanmean(np.abs(ori_err), axis=1)
    # Estimator snapshots padding
    M_EST = len(t[::est_freq][1:])
    S_ROW = 6 * window
    N_PAR = 10

    def _pad2d(lst, shape_tail):
        U = len(lst)
        out = np.full((M_EST,) + shape_tail, np.nan)
        if U > 0:
            out[:min(U, M_EST)] = np.asarray(lst)[:M_EST]
        return out

    theta_gt_traj  = _pad2d(theta_gt_list,  (N_PAR,))
    theta_est_traj = _pad2d(theta_est_list, (N_PAR,))
    A_snapshots    = _pad2d(A_snap_list,    (S_ROW, N_PAR))
    b_snapshots    = _pad2d(b_snap_list,    (S_ROW,))

    est_err_arr = np.full(M_EST, np.nan)
    if len(est_err_traj) > 0:
        est_err_arr[:min(len(est_err_traj), M_EST)] = np.asarray(est_err_traj)[:M_EST]

    return {
        "failed": bool(failed),
        "fail_reason": str(fail_reason),
        "abort_step": np.array([abort_step]),
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "est_err_t": est_err_arr,
        "x_meas_traj": x_meas_arr,
        "x_ref_traj": x_ref_arr,
        "t": t,
        "event_idx": (int(add_step), int(drop_step)),
        "theta_gt_traj":  theta_gt_traj,
        "theta_est_traj": theta_est_traj,
        "A_snapshots":    A_snapshots,
        "b_snapshots":    b_snapshots,
        "seed_offset": int(seed_offset),
        "rng_seed": trial_seed,
    }

# --------------------------- worker wrapper ---------------------------

def _trial_worker(
    estimator_name: str,
    base_seed: int,
    seed_offset: int,
    est_freq: int,
    window: int,
    ref_type: str,
    T_sim: float,
    noise: NoiseLevel,
) -> Dict[str, np.ndarray | int | bool | str]:
    """
    Always returns a shaped payload with 'failed' flag.
    Exceptions are caught *inside* run_single_trial and marked failed.
    """
    out = run_single_trial(
        estimator_name=estimator_name,
        base_seed=base_seed,
        seed_offset=seed_offset,
        est_freq=est_freq,
        window=window,
        ref_type=ref_type,
        T_sim=T_sim,
        noise=noise,
    )
    return out


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results_param_estimation")
    ap.add_argument("--ntrials", type=int, default=50,
                    help="TOTAL attempts per algo per ref (includes failed).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--algos", type=str, nargs="*", default=ALGOS_DEFAULT)
    ap.add_argument("--refs", type=str, nargs="+", default=REF_CHOICES)
    ap.add_argument("--noise", type=str, choices=NOISE_LEVELS, default="none")
    ap.add_argument("--est_freq", type=int, default=20)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--save_traj", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="If >0, use ProcessPoolExecutor with this many workers.")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ALGOS = list(args.algos)
    N_TRIAL = int(args.ntrials)
    BASE_SEED = int(args.seed)
    est_freq = int(args.est_freq)
    window = int(args.window)
    ref_types = [str(r) for r in args.refs]
    workers = int(args.workers)
    T_sim = 20.0
    noise: NoiseLevel = args.noise  # type: ignore

    # Probe time axes (same dt/T for all refs by construction)
    probe = QuadrotorDynamics()
    t_all, _, _ = make_traj_figure8(T_sim, probe.dt, np.random.default_rng(0))
    t = t_all
    t_est = t[::est_freq][1:]

    # storage across ALL refs
    store: Dict[str, Dict[str, List[np.ndarray] | None]] = {
        algo: {
            "pos": [], "vel": [], "ori": [], "est": [],
            "failed": [], "fail_reason": [], "abort_step": [],
            "xm": [] if args.save_traj else None,
            "xr": [] if args.save_traj else None,
            "theta_gt": [], "theta_est": [], "A": [], "b": [],
        }
        for algo in ALGOS
    }
    event_idx_all: List[Tuple[int, int]] = []
    traj_labels: List[str] = []

    for ref_type in ref_types:
        print(f"\n=== Running ref_type='{ref_type}' (noise='{noise}') ===")
        for a_idx, algo in enumerate(ALGOS):
            print(f"[{ref_type} | {algo}] running {N_TRIAL} total trials")

            successes = 0
            failures = 0

            # Run trials (optionally parallel)
            seed_offsets = list(range(N_TRIAL))
            desc = f"[noise={noise} | ref={ref_type} | algo={algo}]"

            # Run trials (optionally parallel)
            if workers > 0:
                results = []
                with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(total=N_TRIAL, desc=desc) as pbar:
                    futures = [
                        ex.submit(
                            _trial_worker,
                            algo, BASE_SEED, seed_offset, est_freq, window, ref_type, T_sim, noise
                        )
                        for seed_offset in seed_offsets
                    ]
                    for fut in as_completed(futures):
                        res = fut.result()
                        results.append(res)
                        if res["failed"]:
                            failures += 1
                        else:
                            successes += 1
                        pbar.update(1)
                    results = sorted(results, key=lambda r: r["seed_offset"])

            else:
                results = []
                with tqdm(total=N_TRIAL, desc=desc) as pbar:
                    for seed_offset in seed_offsets:
                        res = _trial_worker(algo, BASE_SEED, seed_offset, est_freq, window, ref_type, T_sim, noise)
                        results.append(res)
                        if res["failed"]:
                            failures += 1
                        else:
                            successes += 1
                        pbar.update(1)

            # Ingest results
            for idx, out in enumerate(results):
                store[algo]["pos"].append(out["abs_mean_pos_err_t"])  # type: ignore
                store[algo]["vel"].append(out["abs_mean_vel_err_t"])  # type: ignore
                store[algo]["ori"].append(out["abs_mean_ori_err_t"])  # type: ignore
                store[algo]["est"].append(out["est_err_t"])           # type: ignore
                store[algo]["theta_gt"].append(out["theta_gt_traj"])  # type: ignore
                store[algo]["theta_est"].append(out["theta_est_traj"])# type: ignore
                store[algo]["A"].append(out["A_snapshots"])           # type: ignore
                store[algo]["b"].append(out["b_snapshots"])           # type: ignore
                store[algo]["failed"].append(np.array([1 if out["failed"] else 0], dtype=int))  # type: ignore
                store[algo]["fail_reason"].append(np.array([out["fail_reason"]], dtype="U64"))  # type: ignore
                store[algo]["abort_step"].append(out["abort_step"])  # (1,) int array
                if args.save_traj:
                    store[algo]["xm"].append(out["x_meas_traj"])     # type: ignore
                    store[algo]["xr"].append(out["x_ref_traj"])      # type: ignore

                if a_idx == 0:
                    event_idx_all.append(out["event_idx"])            # type: ignore
                    traj_labels.append(ref_type)
            
            print(f"[{ref_type} | {algo}] summary: {successes} success, {failures} failed\n")

    # Convert lists to arrays and save
    event_idx_all_arr = np.asarray(event_idx_all, dtype=int).reshape(-1, 2)
    traj_labels_arr = np.asarray(traj_labels, dtype=str)
    N_TOTAL = traj_labels_arr.shape[0]  # = N_TRIAL * len(ref_types)

    save_kwargs = {
        "algos": np.array(ALGOS),
        "ref_types": np.array(ref_types, dtype=str),
        "traj_labels": traj_labels_arr,
        "ntrials": np.array([N_TOTAL]),           # total attempts (includes failed)
        "base_seed": np.array([BASE_SEED]),
        "est_freq": np.array([est_freq]),
        "window": np.array([window]),
        "t": t,
        "t_est": t[::est_freq][1:],
        "event_idx": event_idx_all_arr,
        "event_time": t[np.clip(event_idx_all_arr, 0, len(t) - 1)],
        "event_labels": np.array(["add", "drop"]),
        "noise_level": np.array([noise], dtype="U10"),
    }

    for algo in ALGOS:
        save_kwargs[f"{algo}__pos"]   = np.stack(store[algo]["pos"], axis=0)   # (N_TOTAL, T)
        save_kwargs[f"{algo}__vel"]   = np.stack(store[algo]["vel"], axis=0)
        save_kwargs[f"{algo}__ori"]   = np.stack(store[algo]["ori"], axis=0)
        save_kwargs[f"{algo}__est"]   = np.stack(store[algo]["est"], axis=0)   # (N_TOTAL, T_est)
        save_kwargs[f"{algo}__theta_gt"] = np.stack(store[algo]["theta_gt"], axis=0)  # (N_TOTAL, T_est, 10)
        save_kwargs[f"{algo}__theta_est"] = np.stack(store[algo]["theta_est"], axis=0)
        save_kwargs[f"{algo}__A"] = np.stack(store[algo]["A"], axis=0)         # (N_TOTAL, T_est, 6*window, 10)
        save_kwargs[f"{algo}__b"] = np.stack(store[algo]["b"], axis=0)         # (N_TOTAL, T_est, 6*window)
        save_kwargs[f"{algo}__failed"] = np.concatenate(store[algo]["failed"], axis=0)  # (N_TOTAL,)
        save_kwargs[f"{algo}__fail_reason"] = np.concatenate(store[algo]["fail_reason"], axis=0)  # (N_TOTAL,)
        save_kwargs[f"{algo}__abort_step"] = np.concatenate(store[algo]["abort_step"], axis=0)  # (N_TOTAL,)
        if store[algo]["xm"] is not None:
            save_kwargs[f"{algo}__xm"] = np.stack(store[algo]["xm"], axis=0)
            save_kwargs[f"{algo}__xr"] = np.stack(store[algo]["xr"], axis=0)

    out_npz = Path(args.outdir) / "results.npz"
    np.savez_compressed(out_npz, **save_kwargs)
    print(f"\nSaved results to: {out_npz.resolve()}")


if __name__ == "__main__":
    main()
