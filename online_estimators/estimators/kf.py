"""
Kalman Filter for parameter estimation (flipped model).

In the "flipped-model" formulation the *parameters* are treated as the
hidden state and the *measurements* provide an observation equation
``b = A theta + v``.  A constant-dynamics model ``theta_{k+1} = theta_k + w`` is
assumed, making this a simple parameter-tracking filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from online_estimators.estimators import _backend
from online_estimators.estimators.base import BaseEstimator


class KF(BaseEstimator):
    r"""Kalman Filter for online parameter estimation.

    State model:  ``theta_{k+1} = theta_k + w_k``, with ``w_k ~ N(0, Q)``

    Observation:  ``b_k = A_k theta_k + v_k``, with ``v_k ~ N(0, R)``

    When the C++ extension ``online_estimation`` is installed, the iterate
    method delegates to a compiled implementation for better performance.

    Parameters
    ----------
    num_params : int
        Dimension of the parameter vector.
    process_noise : np.ndarray, shape ``(n, n)``
        Process noise covariance ``Q``.
    measurement_noise : np.ndarray, shape ``(m, m)``
        Measurement noise covariance ``R``.  Must match the number of
        rows in ``A`` at each call.
    theta_hat : np.ndarray or None
        Initial estimate.  Defaults to zero.
    c : float
        Initial covariance scaling  --  ``P_0 = c * I``.

    Examples
    --------
    >>> import numpy as np
    >>> from online_estimators.estimators import KF
    >>> n, m = 4, 6
    >>> kf = KF(n, process_noise=1e-3*np.eye(n),
    ...         measurement_noise=1e-4*np.eye(m))
    >>> A = np.random.randn(m, n)
    >>> b = A @ np.ones(n) + 0.01 * np.random.randn(m)
    >>> theta = kf.iterate(A, b)
    """

    def __init__(
        self,
        num_params: int,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        theta_hat: np.ndarray | None = None,
        c: float = 10.0,
    ) -> None:
        super().__init__(num_params)
        self.theta_hat = (
            np.zeros(self.num_params)
            if theta_hat is None
            else np.asarray(theta_hat, dtype=float).reshape(-1)
        )
        self.P = np.eye(self.num_params) * float(c)
        self.Q = np.asarray(process_noise, dtype=float)
        self.R = np.asarray(measurement_noise, dtype=float)

        # C++ backend (optional)
        self._cpp: Any = None
        if _backend.HAS_CPP:
            x0_col = self.theta_hat.reshape(-1, 1) if theta_hat is not None else None
            self._cpp = _backend._CppKF(num_params, self.Q, self.R, float(c), x0_col)

    def iterate(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Perform one predict-update cycle.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
            Observation matrix.
        b : np.ndarray, shape ``(m,)``
            Measurement vector.

        Returns
        -------
        np.ndarray, shape ``(n,)``
            Updated parameter estimate.
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        if self._cpp is not None:
            result = self._cpp.iterate(A, b.reshape(-1, 1))
            self.theta_hat = np.asarray(result).ravel()
            return self.theta_hat.copy()

        m, n = A.shape
        assert n == self.num_params
        assert self.R.shape == (m, m), f"R shape {self.R.shape} doesn't match measurement dim {m}"

        # Predict
        self.P = self.P + self.Q

        # Update
        S = A @ self.P @ A.T + self.R
        K = self.P @ A.T @ np.linalg.inv(S)
        innov = b - A @ self.theta_hat
        self.theta_hat = self.theta_hat + K @ innov
        self.P = self.P - K @ A @ self.P
        return self.theta_hat.copy()
