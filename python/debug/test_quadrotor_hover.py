import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from python.quadrotor.quadrotor import QuadrotorDynamics
from python.common.LQR_controller import LQRController

np.random.seed(42)

def compute_hover_error(x_all, xg):
    """Compute L2 tracking error over the hover trajectory"""
    errors = [np.linalg.norm(x[0:3] - xg[0:3]) for x in x_all]
    return np.mean(errors), np.max(errors), errors

def main():
    quad = QuadrotorDynamics()

    rg = np.array([0.0, 0.0, 0.0])
    qg = np.array([1.0, 0.0, 0.0, 0.0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    ug = quad.hover_thrust_true

    A, B = quad.get_linearized_true(xg, ug)

    x0 = xg.copy()
    x0[0:3] += np.array([0.2, 0.2, -0.2])
    x0[3:7] = quad.rptoq(np.array([1.0, 0.0, 0.0]))

    max_dev_x = np.array([0.1,0.1,0.1, 0.5,0.5,0.05, 0.5,0.5,0.5, 0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0 / max_dev_x**2)
    R = np.diag(1.0 / max_dev_u**2)
    lqr = LQRController(A, B, Q, R)

    dt = 1.0 / 50.0
    T  = 100.0
    steps = int(T / dt)

    x = x0.copy()
    x_traj = []
    err_traj = []
    err_force_traj = []
    time = np.linspace(0, T, steps)
    theta = quad.get_true_inertial_params()

    for k in range(steps):
        pos_err = x[0:3] - xg[0:3]
        phi     = quad.qtorp(x[3:7])
        vel_err = x[7:10] - xg[7:10]
        om_err  = x[10:13] - xg[10:13]
        x_err12 = np.hstack([pos_err, phi, vel_err, om_err])

        u_delta = lqr.control(x_err12)
        u = ug + u_delta

        x_next = quad.dynamics_rk4_true(x, u, dt)
        dx = (x_next - x) / dt
        A_mat = quad.get_data_matrix(x, dx)
        b_vec = quad.get_force_vector(x, u)
        err_force = A_mat @ theta - b_vec

        x_traj.append(x.copy())
        err_force_traj.append(err_force.copy())
        x = x_next

    x_traj = np.array(x_traj)
    err_force_traj = np.array(err_force_traj)
    mean_err, max_err, errors = compute_hover_error(x_traj, xg)
    print(f"Mean position error: {mean_err:.4f} m")
    print(f"Max  position error: {max_err:.4f} m")

    fig = plt.figure(figsize=(15,5))

    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1.plot(x_traj[:,0], x_traj[:,1], x_traj[:,2], lw=2, label='Trajectory')
    ax1.scatter(xg[0], xg[1], xg[2], c='r', s=50, label='Hover Point')
    ax1.set_title('3D Hover Trajectory')
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
    ax1.legend()

    ax2 = fig.add_subplot(1,3,2)
    ax2.plot(time, errors, lw=1.5)
    ax2.set_title('Position Tracking Error')
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('L₂ Error [m]')
    ax2.grid(True)

    err_norms = np.linalg.norm(err_force_traj, axis=1)
    ax3 = fig.add_subplot(1,3,3)
    ax3.plot(time, err_norms, lw=1.5, color='crimson')
    ax3.set_title('||Aθ - b|| (Force Residual)')
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Residual Norm')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
