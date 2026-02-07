"""
Discrete-time infinite-horizon LQR controller.

Solves the discrete algebraic Riccati equation (DARE) to compute an
optimal state-feedback gain ``K`` for a linear system ``x_{k+1} = A x_k + B u_k``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


def dlqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete-time LQR gain.

    Solves the DARE and returns the optimal gain ``K`` such that the
    control law ``u = -K x`` minimises the infinite-horizon cost
    ``Sum (x^TQ x + u^TR u)``.

    Parameters
    ----------
    A : np.ndarray, shape ``(n, n)``
        System dynamics matrix.
    B : np.ndarray, shape ``(n, m)``
        Input matrix.
    Q : np.ndarray, shape ``(n, n)``
        State cost matrix (positive semi-definite).
    R : np.ndarray, shape ``(m, m)``
        Input cost matrix (positive definite).

    Returns
    -------
    K : np.ndarray, shape ``(m, n)``
        Optimal feedback gain.
    P : np.ndarray, shape ``(n, n)``
        Solution to the DARE.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


class LQRController:
    """Infinite-horizon discrete LQR controller.

    Implements ``u = -K x`` where ``K`` is the DARE-optimal gain.

    Parameters
    ----------
    A : np.ndarray, shape ``(n, n)``
    B : np.ndarray, shape ``(n, m)``
    Q : np.ndarray, shape ``(n, n)``
    R : np.ndarray, shape ``(m, m)``

    Examples
    --------
    >>> import numpy as np
    >>> from online_estimators.control import LQRController
    >>> A = np.eye(2) + 0.01 * np.array([[0, 1], [-1, 0]])
    >>> B = np.array([[0], [0.01]])
    >>> Q = np.eye(2)
    >>> R = np.eye(1)
    >>> ctrl = LQRController(A, B, Q, R)
    >>> x = np.array([0.5, 0.1])
    >>> u = ctrl.control(x)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K, self.P = dlqr(A, B, Q, R)

    def update_linearized_dynamics(self, A: np.ndarray, B: np.ndarray) -> None:
        """Recompute the gain for updated linearised dynamics.

        Parameters
        ----------
        A : np.ndarray, shape ``(n, n)``
        B : np.ndarray, shape ``(n, m)``
        """
        self.A = A
        self.B = B
        self.K, self.P = dlqr(A, B, self.Q, self.R)

    def control(self, x: np.ndarray) -> np.ndarray:
        """Compute the control input ``u = -K x``.

        Parameters
        ----------
        x : np.ndarray, shape ``(n,)`` or ``(n, 1)``

        Returns
        -------
        np.ndarray
            Control vector ``u``.
        """
        x = x.reshape(-1, 1)
        u = -self.K @ x
        return u.squeeze()
