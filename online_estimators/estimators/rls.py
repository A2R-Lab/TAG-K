"""
Recursive Least Squares (RLS) estimator with exponential forgetting.

The RLS algorithm minimises a weighted least-squares cost where older
measurements are exponentially down-weighted via a *forgetting factor*
``lam in (0, 1]``.  Smaller ``lam`` discounts old data more aggressively,
yielding faster adaptation at the expense of higher variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from online_estimators.estimators import _backend
from online_estimators.estimators.base import BaseEstimator


class RLS(BaseEstimator):
    r"""Recursive Least Squares with exponential forgetting.

    The update equations are:

    .. math::

        S_k &= A_k P_{k-1} A_k^T + \lambda I \\
        K_k &= P_{k-1} A_k^T S_k^{-1} \\
        \hat\theta_k &= \hat\theta_{k-1} + K_k (b_k - A_k \hat\theta_{k-1}) \\
        P_k &= (P_{k-1} - K_k A_k P_{k-1}) / \lambda

    When the C++ extension ``online_estimation`` is installed, the iterate
    method delegates to a compiled implementation for better performance.

    Parameters
    ----------
    num_params : int
        Dimension of the parameter vector.
    theta_hat : np.ndarray or None
        Initial parameter estimate.  Defaults to zero.
    forgetting_factor : float
        Forgetting factor ``lam``.  Typical range: ``[0.2, 1.0]``.
    c : float
        Initial covariance scaling  --  ``P_0 = c * I``.

    Examples
    --------
    >>> import numpy as np
    >>> from online_estimators.estimators import RLS
    >>> est = RLS(num_params=3, forgetting_factor=0.9)
    >>> A = np.random.randn(6, 3)
    >>> b = A @ np.array([1.0, 2.0, 3.0])
    >>> theta = est.iterate(A, b)
    """

    def __init__(
        self,
        num_params: int,
        theta_hat: np.ndarray | None = None,
        forgetting_factor: float = 0.3,
        c: float = 1000.0,
    ) -> None:
        super().__init__(num_params)
        self.P = np.eye(num_params) * float(c)
        self.theta_hat = (
            np.zeros(self.num_params)
            if theta_hat is None
            else np.asarray(theta_hat, dtype=float).reshape(-1)
        )
        self.lambda_ = float(forgetting_factor)

        # C++ backend (optional)
        self._cpp: Any = None
        if _backend.HAS_CPP:
            x0_col = self.theta_hat.reshape(-1, 1) if theta_hat is not None else None
            self._cpp = _backend._CppRLS(num_params, self.lambda_, float(c), x0_col)

    def iterate(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Perform one RLS update with a batch of measurements.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
            Regressor matrix.
        b : np.ndarray, shape ``(m,)``
            Observation vector.

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
        assert n == self.num_params, "A.shape[1] must match num_params"

        S = A @ self.P @ A.T + self.lambda_ * np.eye(m)
        K = self.P @ A.T @ np.linalg.inv(S)
        innov = b - A @ self.theta_hat
        self.theta_hat = self.theta_hat + K @ innov
        self.P = (self.P - K @ A @ self.P) / self.lambda_
        return self.theta_hat.copy()
