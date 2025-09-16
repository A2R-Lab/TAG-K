#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import *   # for apply_noise
from python.common.estimation_methods import *

np.random.seed(42)
np.set_printoptions(precision=7, suppress=True)

def set_axes_equal(ax):
    """
    Set 3D plot axes to equal scale so that spheres appear as spheres.
    Works with Matplotlib 3D axes.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    centers = np.mean(limits, axis=1)
    span = max(limits[:,1] - limits[:,0]) / 2
    for center, setter in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter(center - span, center + span)

def mean_relative_error(x_hat, x_true, eps=1e-18):
    rel_err = np.abs(x_hat - x_true) / (np.abs(x_true) + eps)
    # print(f"Relative error: {rel_err}")
    return np.mean(rel_err)

def weighted_lstsq(A: np.ndarray,
                   b: np.ndarray,
                   w: np.ndarray,
                   rcond=None) -> np.ndarray:
    """
    Solve min_x || W^(1/2) (A x - b) ||₂ via ordinary least squares.

    Args:
        A      – design matrix, shape (N, p)
        b      – measurements vector, shape (N,)
        w      – non‐negative weights, shape (N,)
        rcond  – passed through to np.linalg.lstsq

    Returns:
        x_hat  – solution vector, shape (p,)
    """
    # form W^(1/2)
    sqrt_w = np.sqrt(w)
    # sqrt_w[0] *= 5
    # weight each row
    A_w = A * sqrt_w[:, None]
    b_w = b * sqrt_w
    # solve and return
    x_hat, *_ = np.linalg.lstsq(A_w, b_w, rcond=rcond)
    return x_hat

def is_valid_inertia(J):
    # 1. symmetric?
    if not np.allclose(J, J.T, atol=1e-8):
        return False, "not symmetric"
    # 2. positive eigenvalues?
    eigs = np.linalg.eigvalsh(J)
    if np.any(eigs <= 0):
        return False, "not positive-definite"
    return True, "OK"

def closest_spd(theta, epsilon=1e-6):
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]])
    # Step 1: Symmetrize
    A_sym = 0.5 * (A + A.T)

    # Step 2: Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(A_sym)

    # Step 3: Clamp eigenvalues to be > epsilon
    eigvals_clamped = np.clip(eigvals, epsilon, None)

    # Step 4: Reconstruct SPD matrix
    A_spd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

    Ixx_spd, Iyy_spd, Izz_spd = A_spd[0, 0], A_spd[1, 1], A_spd[2, 2]
    Ixy_spd, Ixz_spd, Iyz_spd = A_spd[0, 1], A_spd[0, 2], A_spd[1, 2]

    theta_spd = np.array([m, cx, cy, cz, Ixx_spd, Iyy_spd, Izz_spd, Ixy_spd, Ixz_spd, Iyz_spd])

    return theta_spd

def compute_tracking_error(x_all, x_ref_all):
    """Compute L2 position error over the trajectory"""
    errors = [np.linalg.norm(x[0:3] - x_ref[0:3])
              for x, x_ref in zip(x_all, x_ref_all)]
    return np.mean(errors), np.max(errors), np.array([x[0:3] - x_ref[0:3]
                                                       for x, x_ref in zip(x_all, x_ref_all)])

def generate_figure8(T, dt, A=0.5, B=0.25, z0=0.0, omega=2*np.pi/10):
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

def main():
    estimator_options = "tark" # gt, none, rk, rls, kf
    assert estimator_options in ["gt", "none", "lstsq", "rk", "tark", "rls", "kf"], \
        "Invalid estimator option. Choose from 'gt', 'none', 'rk', 'rls', 'kf'."

    quad = QuadrotorDynamics()
    ug   = quad.hover_thrust_true
    u = ug.copy()

    # LQR around hover
    xg0 = np.hstack([np.zeros(3), np.array([1,0,0,0]), np.zeros(6)])
    A, B = quad.get_linearized_true(xg0, ug)
    max_dev_x = np.array([0.1,0.1,0.1,  0.5,0.5,0.05,  0.5,0.5,0.5,  0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0/max_dev_x**2)
    Rmat = np.diag(1.0/max_dev_u**2)
    lqr = LQRController(A, B, Q, Rmat)

    # Trajectory
    dt = quad.dt; T = 20.0
    t, pos_ref, vel_ref = generate_figure8(T, dt,
                                           A=0.6, B=0.3, z0=0.0,
                                           omega=2*np.pi/20)

    # Initial true state
    x_true = np.zeros(13)
    x_true[0:3] = pos_ref[0] + 0.1*np.random.randn(3)
    phi0 = 0.02*np.random.randn(3)
    x_true[3:7] = quad.rptoq(phi0)
    x_true[7:10] = vel_ref[0] + 0.01*np.random.randn(3)
    x_true[10:13] = 0.01*np.random.randn(3)

    # Measurement: first sample
    x_meas = apply_noise(x_true)

    # Storage
    x_true_traj = []
    x_meas_traj = []
    x_ref_traj  = []
    meas_noise  = []
    gt_err_traj  = []   # residual using true parameters
    est_err_traj = []   # residual using estimator (if any)

    # Estimator
    theta = quad.get_true_inertial_params()
    theta_est = None
    _have_estimator = (estimator_options != "none")

    if estimator_options == "rls":
        estimator = RLS(num_params=theta.shape[0], theta_hat=theta, forgetting_factor=0.7, c=1000)
    elif estimator_options == "kf":
        estimator = KF(num_params=theta.shape[0], process_noise=1e-4*np.eye(theta.shape[0]),
                       measurement_noise=1e-4*np.eye(6), theta_hat=theta, c=1000)
    elif estimator_options == "rk":
        estimator = RK(num_params=theta.shape[0], x0=theta)
    elif estimator_options == "tark":
        estimator = TARK(num_params=theta.shape[0], x0=theta)

    event_step = 250

    # Simulation loop
    for k in range(len(t)):
        # goal state
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1,0,0,0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        # record
        x_true_traj.append(x_true.copy())
        x_meas_traj.append(x_meas.copy())
        x_ref_traj.append(xg.copy())

        # control on measured
        x_err12 = get_error_state(quad, x_meas, xg)
        last_u = u.copy()
        u = ug + lqr.control(x_err12)
        u = 0.5 * u + 0.5 * last_u  # low-pass filter

        # true propagation
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)
            
        # measurement noise
        x_meas_next = apply_noise(x_true_next)
        noise_vec   = x_meas_next - x_true_next

        # force residual on measured
        dx_meas = (x_meas_next - x_meas)/dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)
        A_norm = np.linalg.norm(A_mat)
        A_mat /= A_norm
        b_vec = quad.get_force_vector(x_meas, dx_meas, u) #+ np.random.normal(0, 1e-9, size=(A_mat.shape[0],))        
        b_vec /= A_norm

        theta_gt   = quad.get_true_inertial_params()
        err_gt     = mean_relative_error(A_mat @ theta_gt, b_vec)
        print("theta_gt:", theta_gt)
        print("A_mat @ theta_gt:", A_mat @ theta_gt)
        print("b_vec:", b_vec)
        gt_err_traj.append(err_gt)

        if _have_estimator & (k % 20 == 0):
            if estimator_options == "gt":
                # use true parameters
                theta_est = quad.get_true_inertial_params()
                ug = quad.get_hover_thrust_true()
                A_new, B_new = quad.get_linearized_true(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

            else: 
                if estimator_options == "lstsq":
                    theta_est = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                else:
                    theta_est = estimator.iterate(A_mat, b_vec)

                theta_est[4] = theta_est[5] = np.mean(theta_est[4:6])
                theta_est = closest_spd(theta_est)  # ensure SPD

                quad.set_estimated_inertial_params(theta_est)
                ug = quad.get_hover_thrust_est()
                A_new, B_new = quad.get_linearized_est(xg, ug)
                lqr.update_linearized_dynamics(A_new, B_new)

        if _have_estimator:
            err_est   = mean_relative_error(A_mat @ theta_est, b_vec)
            est_err_traj.append(err_est)
            print("theta_est:", theta_est)
            print("A_mat @ theta_est:", A_mat @ theta_est)
            print("b_vec:", b_vec)

        meas_noise.append(noise_vec.copy())

        print(f"Step {k+1}/{len(t)}: ")
                # f"True pos: {x_true_next[0:3]}, "
                # f"Measured pos: {x_meas_next[0:3]}, "
                # f"GT error: {err_gt:.4f}, "
                # f"Est error: {err_est:.4f}" if _have_estimator else "")

        # perturb theta
        if k == event_step:
            wind_vec = np.array([23.5, 10.0, 0.0])
            quad.aero_added_inertia(wind_vec)
            print(f"Wind applied at t={t[k]:.2f}s, wind_vec={wind_vec}, new theta:\n{theta}")

        # step
        x_true = x_true_next
        x_meas = x_meas_next

    # to arrays
    x_true_traj = np.array(x_true_traj)
    x_meas_traj = np.array(x_meas_traj)
    x_ref_traj  = np.array(x_ref_traj)
    meas_noise  = np.array(meas_noise)
    gt_err_traj  = np.array(gt_err_traj)
    est_err_traj = np.array(est_err_traj) if _have_estimator else None

    # compute errors
    pos_err_comp = x_meas_traj[:,0:3] - x_ref_traj[:,0:3]
    vel_err_comp = x_meas_traj[:,7:10] - x_ref_traj[:,7:10]

    # orientation errors in Euler (deg)
    qu_meas = np.roll(x_meas_traj[:,3:7], -1, axis=1)
    qu_ref  = np.roll(x_ref_traj[:,3:7], -1, axis=1)
    eul_meas = R.from_quat(qu_meas).as_euler('xyz', degrees=True)
    eul_ref  = R.from_quat(qu_ref).as_euler('xyz', degrees=True)
    orient_err_comp = eul_meas - eul_ref

    # measurement error norms
    pos_me_noise = np.linalg.norm(meas_noise[:,0:3], axis=1)
    vel_me_noise = np.linalg.norm(meas_noise[:,7:10], axis=1)
    # orientation measurement error magnitude (deg)
    qu_true = np.roll(x_true_traj[:,3:7], -1, axis=1)
    rot_err    = (R.from_quat(qu_meas) * R.from_quat(qu_true).inv()).as_rotvec()
    rot_me_noise = np.linalg.norm(rot_err, axis=1) * 180/np.pi
    angv_me_noise = np.linalg.norm(meas_noise[:,10:13], axis=1)

    def moving_average(x, window_size=20):
        return np.convolve(x, np.ones(window_size)/window_size, mode='same')

    # --- Plots ---
    # 1) Component-wise errors
    fig, axs = plt.subplots(3,1, figsize=(9,7), sharex=True)
    axs[0].plot(t, pos_err_comp); axs[0].set_ylabel('Pos error [m]'); axs[0].legend(('x','y','z')); axs[0].grid(True); axs[0].axvline(event_step * quad.dt, color='red', linestyle='--')
    axs[1].plot(t, vel_err_comp); axs[1].set_ylabel('Vel error [m/s]'); axs[1].legend(('v_x','v_y','v_z')); axs[1].grid(True); axs[1].axvline(event_step * quad.dt, color='red', linestyle='--')
    axs[2].plot(t, orient_err_comp); axs[2].set_ylabel('Ori error [deg]'); axs[2].legend(('roll','pitch','yaw')); axs[2].set_xlabel('Time [s]'); axs[2].grid(True); axs[2].axvline(event_step * quad.dt, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

    # 2) Mean errors
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    # compute mean across x,y,z for each error type
    pos_err_mean   = np.mean(pos_err_comp,   axis=1)
    vel_err_mean   = np.mean(vel_err_comp,   axis=1)
    orient_err_mean= np.mean(orient_err_comp,axis=1)

    # Panel 1: mean position error
    axs[0].plot(t, moving_average(abs(pos_err_mean)))
    axs[0].set_ylabel('Abs mean pos error [m]')
    axs[0].grid(True)
    axs[0].axvline(event_step * quad.dt, color='red', linestyle='--')

    # Panel 2: mean velocity error
    axs[1].plot(t, moving_average(abs(vel_err_mean)))
    axs[1].set_ylabel('Abs mean vel error [m/s]')
    axs[1].grid(True)
    axs[1].axvline(event_step * quad.dt, color='red', linestyle='--')

    # Panel 3: mean orientation error
    axs[2].plot(t, moving_average(abs(orient_err_mean)))
    axs[2].set_ylabel('Abs mean ori error [deg]')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True)
    axs[2].axvline(event_step * quad.dt, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

    # 3) 3d Vis
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot(x_ref_traj[:,0], x_ref_traj[:,1], x_ref_traj[:,2], 'g--', label='Reference 8')
    ax1.plot(x_meas_traj[:,0],     x_meas_traj[:,1],     x_meas_traj[:,2],     'b',   label='Tracked')
    ax1.set_title('3D Figure‑8 Tracking')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    set_axes_equal(ax1)
    # A*theta - b error norm
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title('Mean Relative Force Error')
    ax2.plot(t, moving_average(gt_err_traj), lw=1.5, color='crimson', label='GT residual')
    if _have_estimator:
        ax2.plot(t, moving_average(est_err_traj), lw=1.5, color='navy',   label='Est residual')
    ax2.legend()
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Residual Norm')
    ax2.set_yscale('log')  # Use logarithmic scale for y-axis
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
