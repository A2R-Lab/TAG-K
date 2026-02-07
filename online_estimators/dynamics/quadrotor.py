"""
Full 6-DOF quadrotor dynamics with inertial parameter management.

This module implements a rigid-body quadrotor model with:

- Separate *ground-truth* and *estimated* inertial parameters
  (mass, center of mass, full 3x3 inertia tensor).
- Newton-Euler data matrix/force vector for parameter estimation
  (``A theta = b``).
- Payload attach/detach via parallel-axis theorem.
- Aerodynamic added-inertia and parameter drift utilities.
- RK4 integration and autograd-based linearisation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np
    from numpy.linalg import inv, norm
else:
    import autograd.numpy as np
    from autograd.numpy.linalg import inv, norm

from autograd import jacobian


class QuadrotorDynamics:
    """Rigid-body quadrotor with separate true/estimated inertial parameters.

    The model uses a quaternion attitude representation
    ``q = [q_w, q_x, q_y, q_z]`` and a 13-dimensional state vector:

    ``x = [r(3), q(4), v(3), omega(3)]``

    where ``r`` is the world-frame position, ``q`` the body->world
    quaternion, ``v`` the world-frame linear velocity, and ``omega`` the
    body-frame angular velocity.

    Parameters
    ----------
    r_off : array-like or None
        Initial body-frame COM offset (3-vector).  Defaults to zero.

    Attributes
    ----------
    nx : int
        State dimension (12 after quaternion reduction).
    nu : int
        Control dimension (4 motors).
    dt : float
        Default integration time-step (1/freq).
    freq : float
        Control frequency in Hz.
    """

    # -- Class constants ----------------------------------------------
    T = np.diag([1.0, -1, -1, -1])
    H = np.vstack([np.zeros((1, 3)), np.eye(3)])

    def __init__(self, r_off: np.ndarray | None = None) -> None:
        # -- Ground-truth parameters ----------------------------------
        self._mass_true: float = 0.035
        self._J_true: np.ndarray = np.array(
            [
                [1.66e-5, 0.83e-6, 0.72e-6],
                [0.83e-6, 1.66e-5, 1.8e-6],
                [0.72e-6, 1.8e-6, 2.93e-5],
            ]
        )
        self.g: float = 9.81
        self.thrustToTorque: float = 0.0008
        self.el: float = 0.046 / math.sqrt(2)
        self.scale: int = 65535
        self.kt: float = 2.245365e-6 * self.scale
        self.km: float = self.kt * self.thrustToTorque

        self._r_off_true: np.ndarray = (
            np.zeros(3) if r_off is None else np.array(r_off, dtype=float)
        )

        # Baseline copies for reset
        self._mass_base: float = float(self._mass_true)
        self._r_off_base: np.ndarray = self._r_off_true.copy()
        self._J_base: np.ndarray = self._J_true.copy()

        # Payload stack (LIFO)
        self._payloads: List[dict] = []

        self.hover_thrust_true: np.ndarray = (self._mass_true * self.g / self.kt / 4) * np.ones(4)

        # -- Estimated parameters (controller belief) -----------------
        self.mass_est: float = self._mass_true
        self.J_est: np.ndarray = self._J_true.copy()
        self.r_off_est: np.ndarray = self._r_off_true.copy()

        # -- Timing ---------------------------------------------------
        self.freq: float = 50.0
        self.dt: float = 1.0 / self.freq
        self.nx: int = 12
        self.nu: int = 4

    # ------------------------------------------------------------------
    # True / estimated parameter accessors
    # ------------------------------------------------------------------

    def set_true_inertial_params(self, theta: np.ndarray) -> None:
        """Set ground-truth inertial parameters.

        Parameters
        ----------
        theta : np.ndarray, shape ``(10,)``
            ``[m, m*cx, m*cy, m*cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``
        """
        m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
        self._mass_true = m
        self._r_off_true = np.array([cx, cy, cz]) / m
        self._J_true = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        self.hover_thrust_true = (m * self.g / self.kt / 4) * np.ones(4)

    def get_true_inertial_params(self) -> np.ndarray:
        """Return the 10-vector of ground-truth inertial parameters.

        Returns
        -------
        np.ndarray, shape ``(10,)``
            ``[m, m*cx, m*cy, m*cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``
        """
        c = self._mass_true * self._r_off_true
        J_mat = self._J_true
        return np.array(
            [
                self._mass_true,
                *c,
                J_mat[0, 0],
                J_mat[1, 1],
                J_mat[2, 2],
                J_mat[0, 1],
                J_mat[0, 2],
                J_mat[1, 2],
            ]
        )

    def set_estimated_inertial_params(self, theta: np.ndarray) -> None:
        """Update the controller's believed inertial parameters.

        Parameters
        ----------
        theta : np.ndarray, shape ``(10,)``
        """
        m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
        self.mass_est = m
        self.r_off_est = np.array([cx, cy, cz]) / m
        self.J_est = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    def get_estimated_inertial_params(self) -> np.ndarray:
        """Return the 10-vector of estimated inertial parameters.

        Returns
        -------
        np.ndarray, shape ``(10,)``
        """
        c = self.mass_est * self.r_off_est
        J_mat = self.J_est
        return np.array(
            [
                self.mass_est,
                *c,
                J_mat[0, 0],
                J_mat[1, 1],
                J_mat[2, 2],
                J_mat[0, 1],
                J_mat[0, 2],
                J_mat[1, 2],
            ]
        )

    # ------------------------------------------------------------------
    # Hover thrust helpers
    # ------------------------------------------------------------------

    def get_hover_thrust_est(self) -> np.ndarray:
        """Per-motor hover thrust from *estimated* mass."""
        return (self.mass_est * self.g / self.kt / 4) * np.ones(4)

    def get_hover_thrust_true(self) -> np.ndarray:
        """Per-motor hover thrust from *true* mass."""
        return (self._mass_true * self.g / self.kt / 4) * np.ones(4)

    # ------------------------------------------------------------------
    # Core dynamics (parameterised)
    # ------------------------------------------------------------------

    def _dynamics_param(
        self,
        x: np.ndarray,
        u: np.ndarray,
        mass: float,
        J: np.ndarray,
        r_off: np.ndarray,
        wind_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute xdot given explicit inertial parameters.

        Parameters
        ----------
        x : np.ndarray, shape ``(13,)``
        u : np.ndarray, shape ``(4,)``
        mass : float
        J : np.ndarray, shape ``(3, 3)``
        r_off : np.ndarray, shape ``(3,)``
        wind_vec : np.ndarray or None, shape ``(3,)``

        Returns
        -------
        np.ndarray, shape ``(13,)``
        """
        q = x[3:7] / norm(x[3:7])
        v = x[7:10]
        omg = x[10:13]
        Qmat = self.qtoQ(q)

        dr = v
        dq = 0.5 * self.L(q) @ self.H @ omg

        F_th = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [self.kt, self.kt, self.kt, self.kt]]) @ u
        dv_base = np.array([0, 0, -self.g]) + (1 / mass) * Qmat @ F_th
        if wind_vec is not None:
            dv_base = dv_base + wind_vec

        c = mass * r_off
        tau = (
            np.array(
                [
                    [-self.el * self.kt, -self.el * self.kt, self.el * self.kt, self.el * self.kt],
                    [-self.el * self.kt, self.el * self.kt, self.el * self.kt, -self.el * self.kt],
                    [-self.km, self.km, -self.km, self.km],
                ]
            )
            @ u
        )

        domg = inv(J) @ (-self.hat(omg) @ J @ omg + tau - self.hat(c) @ dv_base)
        dv_c = np.cross(r_off, domg) + np.cross(omg, np.cross(r_off, omg))
        dv = dv_base + dv_c

        return np.hstack([dr, dq, dv, domg])

    def _rk4(self, dyn, x, u, dt, wind_vec):
        """4th-order Runge-Kutta integrator with quaternion renormalisation."""
        f1 = dyn(x, u, wind_vec)
        f2 = dyn(x + 0.5 * dt * f1, u, wind_vec)
        f3 = dyn(x + 0.5 * dt * f2, u, wind_vec)
        f4 = dyn(x + dt * f3, u, wind_vec)
        xn = x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        qn = xn[3:7] / norm(xn[3:7])
        return np.hstack([xn[0:3], qn, xn[7:13]])

    # ------------------------------------------------------------------
    # Public: true vs. estimated dynamics
    # ------------------------------------------------------------------

    def dynamics_true(self, x: np.ndarray, u: np.ndarray, wind_vec=None) -> np.ndarray:
        """Continuous-time dynamics using *ground-truth* parameters."""
        return self._dynamics_param(x, u, self._mass_true, self._J_true, self._r_off_true, wind_vec)

    def dynamics_rk4_true(self, x: np.ndarray, u: np.ndarray, dt=None, wind_vec=None) -> np.ndarray:
        """RK4 step using *ground-truth* parameters."""
        if dt is None:
            dt = self.dt
        return self._rk4(self.dynamics_true, x, u, dt, wind_vec)

    def dynamics_est(self, x: np.ndarray, u: np.ndarray, wind_vec=None) -> np.ndarray:
        """Continuous-time dynamics using *estimated* parameters."""
        return self._dynamics_param(x, u, self.mass_est, self.J_est, self.r_off_est, wind_vec)

    def dynamics_rk4_est(self, x: np.ndarray, u: np.ndarray, dt=None, wind_vec=None) -> np.ndarray:
        """RK4 step using *estimated* parameters."""
        if dt is None:
            dt = self.dt
        return self._rk4(self.dynamics_est, x, u, dt, wind_vec)

    # ------------------------------------------------------------------
    # Linearisation
    # ------------------------------------------------------------------

    def get_linearized_true(self, x_ref: np.ndarray, u_ref: np.ndarray):
        """Linearise true dynamics about ``(x_ref, u_ref)``.

        Returns
        -------
        A : np.ndarray, shape ``(12, 12)``
        B : np.ndarray, shape ``(12, 4)``
        """
        A_j = jacobian(self.dynamics_rk4_true, 0)
        B_j = jacobian(self.dynamics_rk4_true, 1)
        A = A_j(x_ref, u_ref)
        B = B_j(x_ref, u_ref)
        return self.E(x_ref[3:7]).T @ A @ self.E(x_ref[3:7]), self.E(x_ref[3:7]).T @ B

    def get_linearized_est(self, x_ref: np.ndarray, u_ref: np.ndarray):
        """Linearise estimated dynamics about ``(x_ref, u_ref)``.

        Returns
        -------
        A : np.ndarray, shape ``(12, 12)``
        B : np.ndarray, shape ``(12, 4)``
        """
        A_j = jacobian(self.dynamics_rk4_est, 0)
        B_j = jacobian(self.dynamics_rk4_est, 1)
        A = A_j(x_ref, u_ref)
        B = B_j(x_ref, u_ref)
        return self.E(x_ref[3:7]).T @ A @ self.E(x_ref[3:7]), self.E(x_ref[3:7]).T @ B

    # ------------------------------------------------------------------
    # Newton-Euler data matrix and force vector
    # ------------------------------------------------------------------

    def get_data_matrix(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Build the 6x10 Newton-Euler data matrix ``A``.

        Constructs the matrix such that ``A theta = w`` where
        ``theta = [m, mcx, mcy, mcz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``.

        Forces and torques are expressed in the **body frame**.

        Parameters
        ----------
        x : np.ndarray, shape ``(13,)``
            State ``[r, q, v, omega]``.
        dx : np.ndarray, shape ``(13,)``
            State derivative ``[rdot, q_dot, v_dot_world, alpha_body]``.

        Returns
        -------
        np.ndarray, shape ``(6, 10)``
        """
        q = x[3:7] / np.linalg.norm(x[3:7])
        Rwb = self.qtoQ(q)
        Rbw = Rwb.T

        a_w = dx[7:10]
        a_b = Rbw @ a_w
        g_b = Rbw @ np.array([0.0, 0.0, -self.g])

        ax_b, ay_b, az_b = a_b
        gx_b, gy_b, gz_b = g_b
        wx, wy, wz = x[10:13]
        ax, ay, az = dx[10:13]

        # Expanded coefficients for omegax(omegaxc) + alphaxc
        f11, f12, f13 = (-(wy**2) - wz**2), (wx * wy - az), (wx * wz + ay)
        f21, f22, f23 = (wx * wy + az), (-(wx**2) - wz**2), (wy * wz - ax)
        f31, f32, f33 = (wx * wz - ay), (wy * wz + ax), (-(wx**2) - wy**2)

        dax, day, daz = ax_b - gx_b, ay_b - gy_b, az_b - gz_b

        # J alpha + omegax(J omega) contributions
        M_alpha = np.array(
            [
                [ax, 0.0, 0.0, ay, az, 0.0],
                [0.0, ay, 0.0, ax, 0.0, az],
                [0.0, 0.0, az, 0.0, ax, ay],
            ]
        )

        def cross_w(v):
            return np.array([wy * v[2] - wz * v[1], wz * v[0] - wx * v[2], wx * v[1] - wy * v[0]])

        M_gyro = np.column_stack(
            [
                cross_w(np.array([wx, 0.0, 0.0])),
                cross_w(np.array([0.0, wy, 0.0])),
                cross_w(np.array([0.0, 0.0, wz])),
                cross_w(np.array([wy, wx, 0.0])),
                cross_w(np.array([wz, 0.0, wx])),
                cross_w(np.array([0.0, wz, wy])),
            ]
        )

        A = np.zeros((6, 10))

        # Force rows (body frame)
        A[0, 0] = dax
        A[1, 0] = day
        A[2, 0] = daz
        A[0, 1:4] = np.array([f11, f12, f13])
        A[1, 1:4] = np.array([f21, f22, f23])
        A[2, 1:4] = np.array([f31, f32, f33])

        # Torque rows (body frame)
        A[3:6, 1:4] = np.array(
            [
                [0.0, gz_b - az_b, gy_b - ay_b],
                [az_b - gz_b, 0.0, gx_b - ax_b],
                [ay_b - gy_b, ax_b - gx_b, 0.0],
            ]
        )
        A[3:6, 4:10] = M_alpha + M_gyro

        return A

    def get_force_vector(
        self, x_curr: np.ndarray, dx: np.ndarray, u_curr: np.ndarray | None = None
    ) -> np.ndarray:
        """Build the 6x1 wrench vector ``b`` (body frame).

        The wrench is purely gravito-inertial (excludes motor forces),
        matching :meth:`get_data_matrix` so that ``A theta = b``.

        Parameters
        ----------
        x_curr : np.ndarray, shape ``(13,)``
        dx : np.ndarray, shape ``(13,)``
        u_curr : np.ndarray or None
            Unused; kept for API compatibility.

        Returns
        -------
        np.ndarray, shape ``(6,)``
        """
        q = x_curr[3:7] / np.linalg.norm(x_curr[3:7])
        Rwb = self.qtoQ(q)
        Rbw = Rwb.T

        a_w = dx[7:10]
        a_b = Rbw @ a_w
        g_b = Rbw @ np.array([0.0, 0.0, -self.g])

        omega = x_curr[10:13]
        alpha = dx[10:13]

        m = self._mass_true
        c = m * self._r_off_true
        J = self._J_true

        delta_a_b = a_b - g_b
        f_coup = np.cross(omega, np.cross(omega, c)) + np.cross(alpha, c)
        f_b = m * delta_a_b + f_coup

        n_b = J @ alpha + np.cross(omega, J @ omega) + np.cross(c, m * delta_a_b)

        return np.hstack([f_b, n_b])

    # ------------------------------------------------------------------
    # Payload events
    # ------------------------------------------------------------------

    def attach_payload(self, m_p: float, delta_r_p: np.ndarray) -> None:
        """Attach a point-mass payload via the parallel-axis theorem.

        Parameters
        ----------
        m_p : float
            Payload mass.
        delta_r_p : array-like, shape ``(3,)``
            Body-frame position of the payload relative to the vehicle origin.
        """
        m_old = self._mass_true
        r_old = self._r_off_true
        J_old = self._J_true

        m_new = m_old + m_p
        d = np.array(delta_r_p, dtype=float)
        r_new = (m_old * r_old + m_p * d) / m_new
        Jp = m_p * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        J_new = J_old + Jp

        self._mass_true = float(m_new)
        self._r_off_true = r_new
        self._J_true = J_new
        self._payloads.append({"m": float(m_p), "d": d})
        self.hover_thrust_true = (self._mass_true * self.g / self.kt / 4) * np.ones(4)

    def detach_payload(self) -> None:
        """Detach (drop) the most recently attached payload.

        Raises
        ------
        RuntimeError
            If no payloads are attached.
        ValueError
            If removing the payload would yield non-positive mass.
        """
        if not self._payloads:
            raise RuntimeError("No payload to detach.")

        entry = self._payloads.pop()
        m_p, d = entry["m"], entry["d"]
        m_old = self._mass_true
        m_new = m_old - m_p
        if m_new <= 0:
            self._payloads.append(entry)
            raise ValueError(f"Detaching payload would result in non-positive mass ({m_new}).")

        r_new = (m_old * self._r_off_true - m_p * d) / m_new
        Jp = m_p * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        J_new = self._J_true - Jp

        self._mass_true = float(m_new)
        self._r_off_true = r_new
        self._J_true = J_new
        self.hover_thrust_true = (self._mass_true * self.g / self.kt / 4) * np.ones(4)

    # ------------------------------------------------------------------
    # Inertia projection & perturbations
    # ------------------------------------------------------------------

    @staticmethod
    def closest_spd(theta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Project the inertia sub-matrix of ``theta`` to the nearest SPD matrix.

        Parameters
        ----------
        theta : np.ndarray, shape ``(10,)``
        epsilon : float
            Eigenvalue floor.

        Returns
        -------
        np.ndarray, shape ``(10,)``
        """
        m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
        A = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        A_sym = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        eigvals_clamped = np.clip(eigvals, epsilon, None)
        A_spd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
        return np.array(
            [
                m,
                cx,
                cy,
                cz,
                A_spd[0, 0],
                A_spd[1, 1],
                A_spd[2, 2],
                A_spd[0, 1],
                A_spd[0, 2],
                A_spd[1, 2],
            ]
        )

    def aero_added_inertia(self, wind_vec: np.ndarray) -> None:
        """Apply aerodynamic added-inertia perturbation from a wind vector.

        Parameters
        ----------
        wind_vec : np.ndarray, shape ``(3,)``
        """
        k = 1e-6
        wind_speed = np.linalg.norm(wind_vec)
        if wind_speed == 0:
            return
        wind_dir = wind_vec / wind_speed
        P = np.eye(3) - np.outer(wind_dir, wind_dir)
        deltaJ = k * wind_speed**2 * P
        J = self._J_true + deltaJ
        Ixx, Iyy, Izz = J[0, 0], J[1, 1], J[2, 2]
        Ixy, Ixz, Iyz = J[0, 1], J[0, 2], J[1, 2]
        c = self._mass_true * self._r_off_true
        Ixx = Iyy = (Ixx + Iyy) / 2
        theta = np.array([self._mass_true, *c, Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
        new_theta = self.closest_spd(theta)
        self._mass_true = new_theta[0]
        new_c = new_theta[1:4]
        self._r_off_true = new_c / self._mass_true
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = new_theta[4:]
        self._J_true = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    def add_drift(
        self, mass_std: float = 0.0, inertia_std: float = 0.0, com_std: float = 0.0
    ) -> None:
        """Add Gaussian drift to true inertial parameters.

        Parameters
        ----------
        mass_std : float
            Standard deviation for mass drift.
        inertia_std : float
            Standard deviation for inertia-tensor element drift.
        com_std : float
            Standard deviation for center-of-mass drift.
        """
        theta_true = self.get_true_inertial_params()
        m_true = theta_true[0]
        c_true = theta_true[1:4]
        I_true = theta_true[4:]

        m_drift = np.random.normal(0.0, mass_std)
        c_drift = np.random.normal(0.0, com_std, size=3)
        I_drift = np.random.normal(0.0, inertia_std, size=6)

        m_new = m_true + m_drift
        c_new = c_true + c_drift
        I_new = I_true + I_drift
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I_new
        Ixx = Iyy = (Ixx + Iyy) / 2
        theta = np.array([m_new, *c_new, Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
        new_theta = self.closest_spd(theta)

        self._mass_true = new_theta[0]
        new_c = new_theta[1:4]
        self._r_off_true = new_c / self._mass_true
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = new_theta[4:]
        self._J_true = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    # ------------------------------------------------------------------
    # Quaternion / rotation helpers (static & class methods)
    # ------------------------------------------------------------------

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix from 3-vector: ``hat(u) v == u x v``.

        Parameters
        ----------
        v : np.ndarray, shape ``(3,)``

        Returns
        -------
        np.ndarray, shape ``(3, 3)``
        """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    @staticmethod
    def L(q: np.ndarray) -> np.ndarray:
        """Left-quaternion multiplication matrix: ``L(q) p == q (x) p``.

        Parameters
        ----------
        q : np.ndarray, shape ``(4,)``

        Returns
        -------
        np.ndarray, shape ``(4, 4)``
        """
        s = q[0]
        v = q[1:4]
        return np.vstack(
            [
                np.hstack([s, -v]),
                np.hstack([v.reshape(3, 1), s * np.eye(3) + QuadrotorDynamics.hat(v)]),
            ]
        )

    @classmethod
    def qtoQ(cls, q: np.ndarray) -> np.ndarray:
        """Quaternion to 3x3 rotation matrix (body -> world).

        Parameters
        ----------
        q : np.ndarray, shape ``(4,)``

        Returns
        -------
        np.ndarray, shape ``(3, 3)``
        """
        return cls.H.T @ cls.T @ cls.L(q) @ cls.T @ cls.L(q) @ cls.H

    @classmethod
    def G(cls, q: np.ndarray) -> np.ndarray:
        """Quaternion kinematic Jacobian ``G(q) = L(q) H``.

        Parameters
        ----------
        q : np.ndarray, shape ``(4,)``

        Returns
        -------
        np.ndarray, shape ``(4, 3)``
        """
        return cls.L(q) @ cls.H

    @staticmethod
    def rptoq(phi: np.ndarray) -> np.ndarray:
        """Rodrigues parameters (3-vector) -> unit quaternion.

        Parameters
        ----------
        phi : np.ndarray, shape ``(3,)``

        Returns
        -------
        np.ndarray, shape ``(4,)``
        """
        return (1.0 / math.sqrt(1 + phi.T @ phi)) * np.hstack([1, phi])

    @staticmethod
    def qtorp(q: np.ndarray) -> np.ndarray:
        """Unit quaternion -> Rodrigues parameters.

        Parameters
        ----------
        q : np.ndarray, shape ``(4,)``

        Returns
        -------
        np.ndarray, shape ``(3,)``
        """
        return q[1:4] / q[0]

    @classmethod
    def E(cls, q: np.ndarray) -> np.ndarray:
        """Block-diagonal embedding for reduced-state linearisation.

        Parameters
        ----------
        q : np.ndarray, shape ``(4,)``

        Returns
        -------
        np.ndarray, shape ``(13, 12)``
        """
        return np.vstack(
            [
                np.hstack([np.eye(3), np.zeros((3, 3)), np.zeros((3, 6))]),
                np.hstack([np.zeros((4, 3)), cls.G(q), np.zeros((4, 6))]),
                np.hstack([np.zeros((6, 3)), np.zeros((6, 3)), np.eye(6)]),
            ]
        )
