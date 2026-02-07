"""
Double pendulum dynamics with RK4 integration.

Implements the equations of motion for a planar double pendulum with
autograd-based Jacobian computation for linearisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    import autograd.numpy as np

from autograd import jacobian


class DoublePendulum:
    """Planar double pendulum with autograd-based linearisation.

    State vector: ``x = [theta_1, omega_1, theta_2, omega_2]``

    Parameters
    ----------
    g : float
        Gravitational acceleration (m/s^2).
    m1, m2 : float
        Link masses (kg).
    l1, l2 : float
        Link lengths (m).
    dt : float
        Integration time-step (s).

    Examples
    --------
    >>> from online_estimators.dynamics import DoublePendulum
    >>> dp = DoublePendulum()
    >>> x0 = np.array([0.1, 0.0, 0.2, 0.0])
    >>> u = np.array([0.0])
    >>> x1 = dp.step(x0, u)
    """

    def __init__(
        self,
        g: float = 9.81,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        dt: float = 0.01,
    ) -> None:
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.dt = dt
        self.num_states = 4
        self.num_controls = 1

        # Autograd Jacobians (computed lazily on first call to linearise)
        self._A_jac = jacobian(self._rk4, 0)
        self._B_jac = jacobian(self._rk4, 1)

    def continuous_dynamics(
        self,
        state: np.ndarray,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        g: float,
        u: np.ndarray,
    ) -> np.ndarray:
        """Continuous-time equations of motion.

        Parameters
        ----------
        state : np.ndarray, shape ``(4,)``
            ``[theta_1, omega_1, theta_2, omega_2]``
        m1, m2, l1, l2, g : float
            Physical parameters.
        u : np.ndarray, shape ``(1,)``
            Torque applied to the second link.

        Returns
        -------
        np.ndarray, shape ``(4,)``
            ``[omega_1, alpha_1, omega_2, alpha_2]``
        """
        state = np.ravel(state)
        theta1, omega1, theta2, omega2 = state
        delta = theta2 - theta1

        alpha1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(delta)
        alpha2 = (l1 / l2) * np.cos(delta)

        f1 = -(l2 / l1) * (m2 / (m1 + m2)) * omega2**2 * np.sin(delta) - (g / l1) * np.sin(theta1)
        f2 = (l1 / l2) * omega1**2 * np.sin(delta) - (g / l2) * np.sin(theta2) + (u[0] / l2)

        detA = 1 - alpha1 * alpha2
        A_inv = (1 / detA) * np.array([[1, -alpha1], [-alpha2, 1]])
        rhs = np.array([f1, f2])
        angular_accels = A_inv @ rhs
        omega1_dot, omega2_dot = angular_accels

        return np.array([omega1, omega1_dot, omega2, omega2_dot])

    def _rk4(self, x: np.ndarray, u: np.ndarray, m1, m2, l1, l2, g) -> np.ndarray:
        """RK4 integration step (private, for autograd)."""
        h = self.dt
        k1 = self.continuous_dynamics(x, m1, m2, l1, l2, g, u)
        k2 = self.continuous_dynamics(x + 0.5 * h * k1, m1, m2, l1, l2, g, u)
        k3 = self.continuous_dynamics(x + 0.5 * h * k2, m1, m2, l1, l2, g, u)
        k4 = self.continuous_dynamics(x + h * k3, m1, m2, l1, l2, g, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Advance the state by one time-step using RK4.

        Parameters
        ----------
        x : np.ndarray, shape ``(4,)``
        u : np.ndarray, shape ``(1,)``

        Returns
        -------
        np.ndarray, shape ``(4,)``
        """
        return self._rk4(x, u, self.m1, self.m2, self.l1, self.l2, self.g)

    def linearize(self, x_ref: np.ndarray, u_ref: np.ndarray):
        """Linearise the discrete dynamics about ``(x_ref, u_ref)``.

        Returns
        -------
        A : np.ndarray, shape ``(4, 4)``
        B : np.ndarray, shape ``(4, 1)``
        """
        A = self._A_jac(x_ref, u_ref, self.m1, self.m2, self.l1, self.l2, self.g)
        B = self._B_jac(x_ref, u_ref, self.m1, self.m2, self.l1, self.l2, self.g)
        return A, B
