#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController
from python.common.noise_models import *   # for apply_noise

np.random.seed(42)

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
    quad = QuadrotorDynamics()
    ug   = quad.hover_thrust_true

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
    x_true[0:3] = pos_ref[0] + 0.05*np.random.randn(3)
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
    err_force_traj = []

    theta = quad.get_true_inertial_params()

    # Simulation loop
    for k in range(len(t)):
        rg, vg = pos_ref[k], vel_ref[k]
        qg, omgg = np.array([1,0,0,0]), np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        # record
        x_true_traj.append(x_true.copy())
        x_meas_traj.append(x_meas.copy())
        x_ref_traj.append(xg.copy())

        # control on measured
        x_err12 = get_error_state(quad, x_meas, xg)
        u = ug + lqr.control(x_err12)

        # true propagation
        x_true_next = quad.dynamics_rk4_true(x_true, u, dt=dt)

        # measurement noise
        x_meas_next = x_true_next + np.random.normal(0, 3e-4, size=x_true_next.shape)
        noise_vec   = x_meas_next - x_true_next

        # force residual on measured
        dx_meas = (x_meas_next - x_meas)/dt
        A_mat = quad.get_data_matrix(x_meas, dx_meas)
        b_vec = quad.get_force_vector(x_meas, u) + np.random.normal(0, 3e-4, size=(A_mat.shape[0],))
        err_force_traj.append((A_mat @ theta - b_vec))

        meas_noise.append(noise_vec.copy())

        # step
        x_true = x_true_next
        x_meas = x_meas_next

    # to arrays
    x_true_traj = np.array(x_true_traj)
    x_meas_traj = np.array(x_meas_traj)
    x_ref_traj  = np.array(x_ref_traj)
    meas_noise  = np.array(meas_noise)
    err_force_traj = np.array(err_force_traj)

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

    # --- Plots ---
    # 1) Component-wise errors
    fig, axs = plt.subplots(3,1, figsize=(9,7), sharex=True)
    axs[0].plot(t, pos_err_comp); axs[0].set_ylabel('Pos error [m]'); axs[0].legend(('x','y','z')); axs[0].grid(True)
    axs[1].plot(t, vel_err_comp); axs[1].set_ylabel('Vel error [m/s]'); axs[1].legend(('vₓ','vᵧ','v𝓏')); axs[1].grid(True)
    axs[2].plot(t, orient_err_comp); axs[2].set_ylabel('Ori error [deg]'); axs[2].legend(('roll','pitch','yaw')); axs[2].set_xlabel('Time [s]'); axs[2].grid(True)
    plt.tight_layout()
    plt.show()

    # 2) 3d Vis
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1.plot(x_ref_traj[:,0], x_ref_traj[:,1], x_ref_traj[:,2], 'g--', label='Reference 8')
    ax1.plot(x_meas_traj[:,0],     x_meas_traj[:,1],     x_meas_traj[:,2],     'b',   label='Tracked')
    ax1.set_title('3D Figure‑8 Tracking')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()

    # error vs time
    ax2 = fig.add_subplot(1,3,2)
    ax2.plot(t, pos_me_noise, label='pos noise [m]')
    ax2.plot(t, rot_me_noise, label='rot noise [deg]')
    ax2.plot(t, vel_me_noise, label='vel noise [m/s]')
    ax2.plot(t, angv_me_noise, label='angvel noise [rad/s]')
    ax2.set_title('Measurement Noise Norms')
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Noise magnitude')
    ax2.legend(); ax2.grid(True)

    # A*theta - b error norm
    err_norms = np.linalg.norm(err_force_traj, axis=1)
    ax3 = fig.add_subplot(1,3,3)
    ax3.plot(t, err_norms, lw=1.5, color='crimson')
    ax3.set_title('||Aθ - b|| (Force Residual)')
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Residual Norm')
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
