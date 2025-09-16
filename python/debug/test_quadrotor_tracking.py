#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController

np.random.seed(42)

def compute_tracking_error(x_all, x_ref_all):
    """Compute L2 position error over the trajectory"""
    errors = [np.linalg.norm(x[0:3] - x_ref[0:3]) 
              for x, x_ref in zip(x_all, x_ref_all)]
    return np.mean(errors), np.max(errors), errors

def generate_figure8(T, dt, A=0.5, B=0.25, z0=0.0, omega=2*np.pi/10):
    """
    Parametric figure‑8 in XY plane:
        x = A * sin(ω t)
        y = B * sin(2 ω t)
        z = z0
    Returns arrays of shape (N,): t, x_ref, v_ref
    """
    t = np.arange(0, T, dt)
    x_ref = A * np.sin(omega * t)
    y_ref = B * np.sin(2*omega * t)
    z_ref = z0 * np.ones_like(t)
    # Velocities
    vx_ref = A * omega * np.cos(omega * t)
    vy_ref = 2 * B * omega * np.cos(2*omega * t)
    vz_ref = np.zeros_like(t)
    pos_ref = np.vstack([x_ref, y_ref, z_ref]).T
    vel_ref = np.vstack([vx_ref, vy_ref, vz_ref]).T
    return t, pos_ref, vel_ref

def get_error_state(quad, x, xg):
    pos_err = x[0:3] - xg[0:3]
    phi     = quad.qtorp(x[3:7])  # convert quaternion to roll-pitch-yaw
    vel_err = x[7:10] - xg[7:10]
    om_err  = x[10:13] - xg[10:13]
    return np.hstack([pos_err, phi, vel_err, om_err])

def main():
    # quadrotor and nominal hover thrust
    quad = QuadrotorDynamics()
    ug = quad.hover_thrust_true

    # LQR design around hover
    xg0 = np.hstack([np.zeros(3), np.array([1,0,0,0]), np.zeros(6)])
    A, B = quad.get_linearized_true(xg0, ug)
    max_dev_x = np.array([0.1,0.1,0.1, 0.5,0.5,0.05, 0.5,0.5,0.5, 0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0 / max_dev_x**2)
    R = np.diag(1.0 / max_dev_u**2)
    lqr = LQRController(A, B, Q, R)

    # Simulation parameters
    dt = 1/50
    T  = 20.0
    t, pos_ref, vel_ref = generate_figure8(T, dt, A=0.6, B=0.3, z0=0.0, omega=2*np.pi/20)

    # initialize state near the first ref point
    x = np.zeros(13)
    x[0:3] = pos_ref[0] + 0.05 * np.random.randn(3)
    phi0    = 0.02 * np.random.randn(3)
    x[3:7] = quad.rptoq(phi0)
    x[7:10] = vel_ref[0] + 0.01 * np.random.randn(3)
    x[10:13] = 0.01 * np.random.randn(3)

    # get initial param
    theta = quad.get_true_inertial_params()

    # storage
    x_traj     = []
    x_ref_traj = []
    err_force_traj = []

    # run closed‑loop simulation
    for k in range(len(t)):
        # build full reference state at time k
        rg = pos_ref[k]
        qg = np.array([1,0,0,0])           # keep level attitude
        vg = vel_ref[k]
        omgg = np.zeros(3)
        xg = np.hstack([rg, qg, vg, omgg])

        # error state (12-d)
        x_err12 = get_error_state(quad, x, xg)

        # control
        u_delta = lqr.control(x_err12)
        u = ug + u_delta

        # estimate A and b
        x_next = quad.dynamics_rk4_true(x, u, dt=dt)
        dx = (x_next - x) / dt
        A_mat = quad.get_data_matrix(x, dx)
        b_vec = quad.get_force_vector(x, u)
        err_force = A_mat @ theta - b_vec

        # record and step
        x_traj.append(x.copy())
        x_ref_traj.append(xg.copy())
        err_force_traj.append(err_force.copy())
        x = x_next

    x_traj     = np.array(x_traj)
    x_ref_traj = np.array(x_ref_traj)
    err_force_traj = np.array(err_force_traj)

    mean_err, max_err, errors = compute_tracking_error(x_traj, x_ref_traj)
    print(f"Mean tracking error: {mean_err:.4f} m")
    print(f"Max  tracking error: {max_err:.4f} m")

    # --- Visualization ---

    # 3D figure‑8 path
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1.plot(x_ref_traj[:,0], x_ref_traj[:,1], x_ref_traj[:,2], 'g--', label='Reference 8')
    ax1.plot(x_traj[:,0],     x_traj[:,1],     x_traj[:,2],     'b',   label='Tracked')
    ax1.set_title('3D Figure‑8 Tracking')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()

    # error vs time
    ax2 = fig.add_subplot(1,3,2)
    ax2.plot(t, errors, lw=1.5)
    ax2.set_title('Position Tracking Error')
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('L₂ Error [m]')
    ax2.grid(True)

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