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

# Project imports
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import apply_noise  # not used yet; kept for future wiring
from python.common.estimation_methods import *  # RLS, KF, RK, TARK, IGRK, MGRK, REK, DEKA

np.set_printoptions(precision=3, suppress=True)


def rel_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Relative residual ||Ax-b|| / max(||b||, eps)."""
    r = A @ x - b
    return np.linalg.norm(r) / max(np.linalg.norm(b), eps)

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

def gate_param_update(theta, A, b, rel_resid_max=0.25, verbose=False):
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

def make_traj_figure8(T: float, dt: float, rng: np.random.Generator):
    A = rng.uniform(0.4, 0.8)
    B = rng.uniform(0.2, 0.4)
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2 * np.pi / 20.0) * rng.uniform(0.8, 1.25)
    t = np.arange(0, T, dt)
    x = A * np.sin(omega * t)
    y = B * np.sin(2 * omega * t)
    z = z0 * np.ones_like(t)
    vx = A * omega * np.cos(omega * t)
    vy = 2 * B * omega * np.cos(2 * omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T
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

    if name == "rls_0.8":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.8, c=1000)
    if name == "rls_0.5":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.5, c=1000)
    if name == "rls_0.2":
        return RLS(num_params=n, theta_hat=theta0, forgetting_factor=0.2, c=1000)
    if name == "kf_low":
        Q = 1e-3 * np.eye(n)
        Rm = 1e-4 * np.eye(window * 6)
        return KF(num_params=n, process_noise=Q, measurement_noise=Rm, theta_hat=theta0, c=1000)
    if name == "kf_high":
        Q = 1e-2 * np.eye(n)
        Rm = 1e-3 * np.eye(window * 6)
        return KF(num_params=n, process_noise=Q, measurement_noise=Rm, theta_hat=theta0, c=1000)
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

def run_online_estimation(
    estimator_name='ours',
    seed=42,
    window=1,
    est_freq=50,
    T_sim=20.0,
    noise="low",
    verbose_ctrl: bool = False,
    verbose_est: bool = True,
):
    rng = np.random.default_rng(seed)
    quad = QuadrotorDynamics()
    lqr, ug = make_lqr_around_hover(quad)

    dt = quad.dt
    t, pos_ref, vel_ref = make_traj_figure8(T_sim, dt, rng)

    # Initial state
    x_true = np.zeros(13)
    x_true[0:3] = pos_ref[0] + 0.05 * rng.standard_normal(3)
    x_true[3:7] = quad.rptoq(0.02 * rng.standard_normal(3))
    x_true[7:10] = vel_ref[0] + 0.01 * rng.standard_normal(3)
    x_true[10:13] = 0.01 * rng.standard_normal(3)
    x_meas = x_true.copy()

    # Randomized payload events
    add_step = 200 #rng.integers(100, 201)
    drop_step = 600 #rng.integers(500, 701)
    payload_m = 0.015 #rng.uniform(0.01, 0.02)
    payload_dr = np.array([0.0005, -0.001, 0.001]) #rng.uniform(-0.002, 0.002, size=3)

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
    aborted = False
    abort_step = -1
    failed = False
    fail_reason = ""

    for k in range(len(t)):
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1, 0, 0, 0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        # ---- early deviation gate (abort counts as failure) ----
        if np.linalg.norm(x_meas[0:3] - rg) > 0.3:
            aborted = True
            abort_step = k
            failed = True
            fail_reason = f"abort_pos_deviation@step_{k}"
            break

        x_ref_traj.append(xg.copy())
        x_meas_traj.append(x_meas.copy())

        # Control
        x_err12 = get_error_state(quad, x_meas, xg)
        u_delta = lqr.control(x_err12)
        u_preclip = ug + u_delta
        u = np.clip(u_preclip, 0.0, 1.0)

        if verbose_ctrl:
            print(f"[CTRL] k={k:5d} t={t[k]:7.3f}  ||e||={np.linalg.norm(x_err12):.4g}, add@{add_step}, drop@{drop_step}")
            print(f"[CTRL] ug        = {ug}")
            print(f"[CTRL] u_delta   = {u_delta}")
            print(f"[CTRL] u_preclip = {u_preclip}")
            print(f"[CTRL] u_clipped = {u}")
            print(f"[CTRL] omg_meas  = {x_meas[10:13]}")

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
                # use ground-truth, update controller linearization around truth
                if verbose_est:
                    r_gt = A_stack @ theta_gt - b_stack
                    rr_gt = rel_residual(A_stack, theta_gt, b_stack)
                    print(f"[EST] k={k:5d}  ‖r_gt‖={np.linalg.norm(r_gt):.4e}  rel={rr_gt:.4e}")
                    print(f"[EST] theta_gt = {theta_gt}")
                
                gate_param_update(theta_gt, A_stack, b_stack)
                est_err_traj.append(rel_residual(A_stack, theta_gt, b_stack))
                ug = quad.get_hover_thrust_true()
                A_new, B_new = quad.get_linearized_true(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)
                theta_est_cur = theta_gt.copy()

            elif estimator_name == "none":
                if verbose_est:
                    r_0 = A_stack @ theta0 - b_stack
                    rr_0 = rel_residual(A_stack, theta0, b_stack)
                    print(f"[EST] k={k:5d}  (no-est) ‖r‖={np.linalg.norm(r_0):.4e}  rel={rr_0:.4e}")
                    print(f"[EST] theta_gt = {theta_gt}")
                est_err_traj.append(rel_residual(A_stack, theta0, b_stack))

            else:
                theta_est = est.iterate(A_stack, b_stack)
                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)

                theta_candidate = theta_est  # after SPD fix, symmetry, etc.
                if gate_param_update(theta_candidate, A_stack, b_stack):
                    quad.set_estimated_inertial_params(theta_candidate)
                    ug = quad.get_hover_thrust_est()              # OK to update feedforward now
                    # KEEP OLD K FOR NOW  ➜ do NOT update A,B,K yet in step 1
                else:
                    # reject: keep previous params and ug
                    pass

                if verbose_est:
                    r_est = A_stack @ theta_est - b_stack
                    rr_est = rel_residual(A_stack, theta_est, b_stack)
                    print(f"[EST] k={k:5d}  ‖r_est‖={np.linalg.norm(r_est):.4e}  rel={rr_est:.4e}")
                    print(f"[EST] theta_gt = {theta_gt}")
                    print(f"[EST] theta_est= {theta_est}")
                    print(f"[EST] udpate controller = {gate_param_update(theta_candidate, A_stack, b_stack)}")
                    print(f"[EST] scale cond(A) = {scaled_cond(A_stack):.4e}")

                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)
                est_err_traj.append(rel_residual(A_stack, theta_est, b_stack))
                theta_est_cur = theta_est.copy()

            theta_gt_list.append(theta_gt.copy())
            theta_est_list.append(theta_est_cur)

        # Payload events
        if k == add_step:
            quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
        elif k == drop_step:
            quad.detach_payload()

        x_true = x_true_next
        x_meas = x_meas_next

    # Arrays
    x_meas_traj = np.array(x_meas_traj)
    x_ref_traj = np.array(x_ref_traj)
    theta_gt_traj  = np.asarray(theta_gt_list)
    theta_est_traj = np.asarray(theta_est_list)
    A_snapshots    = np.asarray(A_snap_list)
    b_snapshots    = np.asarray(b_snap_list)

    # Errors
    pos_err = x_meas_traj[:, 0:3] - x_ref_traj[:, 0:3]
    vel_err = x_meas_traj[:, 7:10] - x_ref_traj[:, 7:10]
    qu_meas = np.roll(x_meas_traj[:, 3:7], -1, axis=1)  # [x,y,z,w]
    qu_ref  = np.roll(x_ref_traj[:, 3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler("xyz", degrees=True)
    eul_ref  = R.from_quat(qu_ref ).as_euler("xyz", degrees=True)
    ori_err  = eul_meas - eul_ref

    abs_mean_pos_err_t = np.mean(np.abs(pos_err), axis=1)
    abs_mean_vel_err_t = np.mean(np.abs(vel_err), axis=1)
    abs_mean_ori_err_t = np.mean(np.abs(ori_err), axis=1)

    return {
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "est_err_t": np.array(est_err_traj),
        "x_meas_traj": x_meas_traj,
        "x_ref_traj": x_ref_traj,
        "t": t,
        "event_idx": (int(add_step), int(drop_step)),
        "theta_gt_traj":  theta_gt_traj,
        "theta_est_traj": theta_est_traj,
        "A_snapshots":    A_snapshots,
        "b_snapshots":    b_snapshots,
    }

def scaled_cond(A):
    """Scaled condition number kappa(A) = ||A||_F * ||A^{-1}||_2."""
    A = np.asarray(A, float)
    frob_norm = np.linalg.norm(A, 'fro')          # Frobenius norm
    inv_2norm = np.linalg.norm(np.linalg.inv(A), 2)  # spectral norm of A^{-1}
    return frob_norm * inv_2norm

if __name__ == "__main__":
    METHODS = ["rk", "grk", "rls", "kf", "rk_tikh","gt", "rk_equi", "rk_colscaled"]
    results = {}

    for method in METHODS:
        name = f"dict_{method}"
        print(f"Running method: {method}")
        results[name] = run_online_estimation(estimator_name=method, noise="none")

    out_dir = Path("output/clipped_res/single_trial_0907")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.npz"
    np.savez_compressed(out_path, **results)
    print(f"Saved results to {out_path}")