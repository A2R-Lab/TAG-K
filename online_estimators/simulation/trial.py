"""
Quadrotor parameter-estimation trial runner.

Orchestrates a closed-loop simulation: trajectory tracking via LQR,
online parameter estimation using any algorithm from :mod:`online_estimators.estimators`,
and payload-change events.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from online_estimators.control.lqr import LQRController
from online_estimators.dynamics.quadrotor import QuadrotorDynamics
from online_estimators.estimators import (
    FDBK,
    GRK,
    IGRK,
    KF,
    MGRK,
    REK,
    RK,
    RLS,
    TAGK,
    TARK,
    GRK_Tikh,
    RK_ColScaled,
    RK_Equi,
    RK_EquiTikh,
    RK_Tikh,
    TAGK_Tikh,
)
from online_estimators.noise.models import apply_noise
from online_estimators.simulation.utils import abs_residual, closest_spd
from online_estimators.trajectories.generators import TRAJECTORY_REGISTRY


def make_estimator(name: str, theta0: np.ndarray, window: int, seed: int | None = None):
    """Instantiate an estimator by name.

    Parameters
    ----------
    name : str
        Algorithm identifier (case-insensitive). ``"gt"`` or ``"none"``
        returns ``None``.
    theta0 : np.ndarray, shape ``(n,)``
        Initial parameter estimate.
    window : int
        Number of stacked measurement rows (used for KF noise sizing).
    seed : int or None
        If provided, seed the internal RNG of randomised estimators for
        reproducibility.

    Returns
    -------
    estimator or None

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    n = int(theta0.shape[0])
    name = name.lower()

    registry = {
        "rls_0.8": lambda: RLS(n, theta_hat=theta0, forgetting_factor=0.8, c=1000),
        "rls_0.5": lambda: RLS(n, theta_hat=theta0, forgetting_factor=0.5, c=1000),
        "rls_0.2": lambda: RLS(n, theta_hat=theta0, forgetting_factor=0.2, c=1000),
        "kf_low": lambda: KF(
            n,
            process_noise=1e-3 * np.eye(n),
            measurement_noise=1e-4 * np.eye(window * 6),
            theta_hat=theta0,
            c=1000,
        ),
        "kf_high": lambda: KF(
            n,
            process_noise=1e-2 * np.eye(n),
            measurement_noise=1e-3 * np.eye(window * 6),
            theta_hat=theta0,
            c=1000,
        ),
        "rk": lambda: RK(n, x0=theta0),
        "tark": lambda: TARK(n, x0=theta0),
        "igrk": lambda: IGRK(n, x0=theta0),
        "mgrk": lambda: MGRK(n, x0=theta0),
        "rek": lambda: REK(n, x0=theta0),
        "fdbk": lambda: FDBK(n, x0=theta0),
        "grk": lambda: GRK(n, x0=theta0),
        "rk_colscaled": lambda: RK_ColScaled(n, x0=theta0),
        "rk_equi": lambda: RK_Equi(n, x0=theta0),
        "rk_tikh": lambda: RK_Tikh(n, x0=theta0),
        "rk_equi_tikh": lambda: RK_EquiTikh(n, x0=theta0),
        "grk_tikh": lambda: GRK_Tikh(n, x0=theta0),
        "tagk_tikh": lambda: TAGK_Tikh(n, x0=theta0),
        "tagk": lambda: TAGK(n, x0=theta0),
    }

    if name in ("gt", "none"):
        return None
    if name not in registry:
        raise ValueError(
            f"Unknown estimator '{name}'. Available: {sorted(registry)} + ['gt', 'none']"
        )
    est = registry[name]()

    # Seed internal RNG when the estimator supports it
    if seed is not None and hasattr(est, "seed_rng"):
        est.seed_rng(seed)

    return est


def gate_param_update(
    theta: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    rel_resid_max: float = 1e-4,
    verbose: bool = False,
) -> bool:
    """Safety gate: accept or reject a proposed parameter update.

    Checks physical plausibility (mass bounds, COM bounds, inertia
    eigenvalues) and numerical quality (relative residual).

    Parameters
    ----------
    theta : np.ndarray, shape ``(10,)``
    A : np.ndarray
    b : np.ndarray
    rel_resid_max : float
    verbose : bool

    Returns
    -------
    bool
        ``True`` if the update should be applied.
    """
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta

    if not (0.02 <= m <= 0.08):
        if verbose:
            print(f"Reject: mass {m:.4f} out of bounds [0.02, 0.08]")
        return False
    if np.linalg.norm([cx, cy, cz]) > 0.02:
        if verbose:
            print(f"Reject: COM [{cx:.4f}, {cy:.4f}, {cz:.4f}] out of bounds")
        return False

    J = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    w = np.linalg.eigvalsh(0.5 * (J + J.T))
    if (w < 1e-7).any() or (w > 1e-4).any():
        if verbose:
            print(f"Reject: inertia eigenvalues {w} out of bounds")
        return False

    rel = np.linalg.norm(A @ theta - b) / max(np.linalg.norm(b), 1e-12)
    if rel > rel_resid_max:
        if verbose:
            print(f"Reject: relative residual {rel:.4f} > {rel_resid_max}")
        return False

    return True


def _get_error_state(quad: QuadrotorDynamics, x: np.ndarray, xg: np.ndarray) -> np.ndarray:
    """12-D LQR error state: [pos_err, rpy, vel_err, omega_err]."""
    pos_err = x[0:3] - xg[0:3]
    phi = quad.qtorp(x[3:7])
    vel_err = x[7:10] - xg[7:10]
    om_err = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])


def _make_lqr_around_hover(quad: QuadrotorDynamics) -> Tuple[LQRController, np.ndarray]:
    """Build an LQR controller about the hover equilibrium."""
    ug = quad.hover_thrust_true.copy()
    xg0 = np.hstack([np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(6)])
    A_lin, B_lin = quad.get_linearized_true(xg0, ug)

    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])
    Q = np.diag(1.0 / max_dev_x**2)
    Rmat = np.diag(1.0 / max_dev_u**2)
    lqr = LQRController(A_lin, B_lin, Q, Rmat)
    return lqr, ug


def run_single_trial(
    estimator_name: str,
    base_seed: int,
    seed_offset: int,
    est_freq: int,
    window: int,
    ref_type: str,
    T_sim: float = 20.0,
    noise: str = "none",
    *,
    verbose_ctrl: bool = False,
    verbose_est: bool = False,
) -> Dict[str, np.ndarray | int | bool | str]:
    """Run one closed-loop quadrotor parameter-estimation trial.

    Parameters
    ----------
    estimator_name : str
        Algorithm name (see :func:`make_estimator`).
    base_seed : int
        Base random seed.
    seed_offset : int
        Offset added to *base_seed* for reproducibility across trials.
    est_freq : int
        Estimation update period (in simulation steps).
    window : int
        Number of consecutive measurement snapshots stacked per update.
    ref_type : str
        Trajectory type (key into :data:`TRAJECTORY_REGISTRY`).
    T_sim : float
        Simulation duration (s).
    noise : str
        Sensor noise level: ``"none"``, ``"low"``, ``"medium"``, ``"high"``.
    verbose_ctrl : bool
        Print per-step control debug info.
    verbose_est : bool
        Print per-update estimation debug info.

    Returns
    -------
    dict
        Result dictionary with trajectory data, errors, and metadata.
    """
    rng = np.random.default_rng(base_seed + seed_offset)
    quad = QuadrotorDynamics()
    lqr, ug = _make_lqr_around_hover(quad)

    dt = quad.dt
    t, pos_ref, vel_ref = TRAJECTORY_REGISTRY[ref_type](T_sim, dt, rng)

    # Initial state (small perturbation around reference)
    x_true = np.zeros(13)
    x_true[0:3] = pos_ref[0] + 0.05 * rng.standard_normal(3)
    x_true[3:7] = quad.rptoq(0.02 * rng.standard_normal(3))
    x_true[7:10] = vel_ref[0] + 0.01 * rng.standard_normal(3)
    x_true[10:13] = 0.01 * rng.standard_normal(3)
    x_meas = x_true.copy()

    # Randomised payload events
    add_step = rng.integers(200, 301)
    drop_step = rng.integers(600, 701)
    payload_m = rng.uniform(0.035 / 3, 0.035 / 2)
    payload_dr = rng.uniform(-0.001, 0.001, size=3)

    theta0 = quad.get_true_inertial_params()
    est_seed = int(rng.integers(0, 2**31))
    est = make_estimator(estimator_name, theta0, window, seed=est_seed)

    A_buf: deque[np.ndarray] = deque(maxlen=window)
    B_buf: deque[np.ndarray] = deque(maxlen=window)

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

    try:
        for k in range(len(t)):
            rg, vg = pos_ref[k], vel_ref[k]
            qg, omgg = np.array([1, 0, 0, 0]), np.zeros(3)
            xg = np.hstack([rg, qg, vg, omgg])

            if np.linalg.norm(x_meas[0:3] - rg) > 0.3:
                aborted = True
                abort_step = k
                failed = True
                fail_reason = f"abort_pos_deviation@step_{k}"
                break

            x_ref_traj.append(xg.copy())
            x_meas_traj.append(x_meas.copy())

            x_err12 = _get_error_state(quad, x_meas, xg)
            u_delta = lqr.control(x_err12)
            u = np.clip(ug + u_delta, 0.0, 1.0)

            x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)
            x_meas_next = apply_noise(x_true_next, level=noise, rng=rng)

            dx_meas = (x_meas_next - x_meas) / dt
            A_mat = quad.get_data_matrix(x_meas, dx_meas)
            b_vec = quad.get_force_vector(x_meas, dx_meas, u)

            A_buf.append(A_mat)
            B_buf.append(b_vec)

            if (k % est_freq == 0) and (k > 0):
                A_stack = np.vstack(list(A_buf))
                b_stack = np.concatenate(list(B_buf), axis=0)

                A_snap_list.append(A_stack.copy())
                b_snap_list.append(b_stack.copy())

                theta_gt = quad.get_true_inertial_params()
                theta_est_cur = np.full_like(theta_gt, np.nan)

                if estimator_name == "gt":
                    est_err_traj.append(abs_residual(A_stack, theta_gt, b_stack))
                    ug = quad.get_hover_thrust_true()
                    A_new, B_new = quad.get_linearized_true(xg, ug)
                    lqr.update_linearized_dynamics(A_new, B_new)
                    theta_est_cur = theta_gt.copy()
                elif estimator_name == "none":
                    est_err_traj.append(abs_residual(A_stack, theta0, b_stack))
                else:
                    theta_est = est.iterate(A_stack, b_stack)
                    theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                    theta_est = closest_spd(theta_est)

                    if gate_param_update(theta_est, A_stack, b_stack):
                        quad.set_estimated_inertial_params(theta_est)
                        ug = quad.get_hover_thrust_est()
                        A_new, B_new = quad.get_linearized_est(xg, ug)
                        lqr.update_linearized_dynamics(A_new, B_new)

                    est_err_traj.append(abs_residual(A_stack, theta_est, b_stack))
                    theta_est_cur = theta_est.copy()

                theta_gt_list.append(theta_gt.copy())
                theta_est_list.append(theta_est_cur)

            if k == add_step:
                quad.attach_payload(m_p=payload_m, delta_r_p=payload_dr)
            if k == drop_step:
                quad.detach_payload()

            x_true = x_true_next
            x_meas = x_meas_next

    except Exception as e:
        failed = True
        fail_reason = f"exception:{repr(e)}"

    # Pack results
    T_total = len(t)
    N_done = len(x_meas_traj)
    x_meas_arr = np.full((T_total, 13), np.nan)
    x_ref_arr = np.full((T_total, 13), np.nan)
    if N_done > 0:
        x_meas_arr[:N_done] = np.asarray(x_meas_traj)
        x_ref_arr[:N_done] = np.asarray(x_ref_traj)

    pos_err = x_meas_arr[:, 0:3] - x_ref_arr[:, 0:3]
    vel_err = x_meas_arr[:, 7:10] - x_ref_arr[:, 7:10]
    ori_err = np.full((T_total, 3), np.nan)
    if N_done > 0:
        qu_meas = np.roll(x_meas_arr[:N_done, 3:7], -1, axis=1)
        qu_ref = np.roll(x_ref_arr[:N_done, 3:7], -1, axis=1)
        try:
            eul_meas = R.from_quat(qu_meas).as_euler("xyz", degrees=True)
            eul_ref = R.from_quat(qu_ref).as_euler("xyz", degrees=True)
            ori_err[:N_done] = eul_meas - eul_ref
        except Exception:
            pass

    abs_mean_pos_err_t = np.nanmean(np.abs(pos_err), axis=1)
    abs_mean_vel_err_t = np.nanmean(np.abs(vel_err), axis=1)
    abs_mean_ori_err_t = np.nanmean(np.abs(ori_err), axis=1)

    M_EST = len(t[::est_freq][1:])
    S_ROW = 6 * window
    N_PAR = 10

    def _pad2d(lst, shape_tail):
        U = len(lst)
        out = np.full((M_EST,) + shape_tail, np.nan)
        if U > 0:
            out[: min(U, M_EST)] = np.asarray(lst)[:M_EST]
        return out

    theta_gt_traj = _pad2d(theta_gt_list, (N_PAR,))
    theta_est_traj = _pad2d(theta_est_list, (N_PAR,))
    A_snapshots = _pad2d(A_snap_list, (S_ROW, N_PAR))
    b_snapshots = _pad2d(b_snap_list, (S_ROW,))

    est_err_arr = np.full(M_EST, np.nan)
    if len(est_err_traj) > 0:
        est_err_arr[: min(len(est_err_traj), M_EST)] = np.asarray(est_err_traj)[:M_EST]

    return {
        "failed": bool(failed),
        "fail_reason": str(fail_reason),
        "abs_mean_pos_err_t": abs_mean_pos_err_t,
        "abs_mean_vel_err_t": abs_mean_vel_err_t,
        "abs_mean_ori_err_t": abs_mean_ori_err_t,
        "est_err_t": est_err_arr,
        "x_meas_traj": x_meas_arr,
        "x_ref_traj": x_ref_arr,
        "t": t,
        "event_idx": int(add_step),
        "theta_gt_traj": theta_gt_traj,
        "theta_est_traj": theta_est_traj,
        "A_snapshots": A_snapshots,
        "b_snapshots": b_snapshots,
        "aborted": np.array([aborted]),
        "abort_step": np.array([abort_step]),
    }
