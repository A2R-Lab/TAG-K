"""
Small numerical helper functions used across simulations.
"""

from __future__ import annotations

import numpy as np


def abs_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Absolute residual ``||Ax - b||_2``.

    Parameters
    ----------
    A : np.ndarray, shape ``(m, n)``
    x : np.ndarray, shape ``(n,)``
    b : np.ndarray, shape ``(m,)``

    Returns
    -------
    float
    """
    return float(np.linalg.norm(A @ x - b))


def rel_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Relative residual ``||Ax - b||_2 / max(||b||_2, eps)``.

    Parameters
    ----------
    A : np.ndarray, shape ``(m, n)``
    x : np.ndarray, shape ``(n,)``
    b : np.ndarray, shape ``(m,)``
    eps : float
        Floor to avoid division by zero.

    Returns
    -------
    float
    """
    return float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), eps))


def closest_spd(theta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project the inertia sub-matrix to the nearest SPD matrix.

    Parameters
    ----------
    theta : np.ndarray, shape ``(10,)``
        ``[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``
    epsilon : float
        Eigenvalue floor.

    Returns
    -------
    np.ndarray, shape ``(10,)``
        Parameters with inertia projected to SPD.
    """
    m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = theta
    A = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    A_sym = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A_sym)
    w = np.clip(w, epsilon, None)
    A_spd = V @ np.diag(w) @ V.T
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
