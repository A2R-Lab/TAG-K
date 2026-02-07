"""
Abstract base class for all online parameter estimators.

Every estimator in TAG-K follows the same ``iterate(A, b) -> theta`` protocol,
making them interchangeable in simulation loops and benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEstimator(ABC):
    """Abstract base class for online parameter estimators.

    All estimators solve the overdetermined linear system ``A theta ~= b``
    incrementally  --  each call to :meth:`iterate` ingests a new batch of
    rows and returns the current best estimate of ``theta``.

    Parameters
    ----------
    num_params : int
        Dimension of the parameter vector ``theta``.
    """

    def __init__(self, num_params: int) -> None:
        self.num_params = int(num_params)

    @abstractmethod
    def iterate(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ingest a new measurement batch and return the updated estimate.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
            Measurement (regressor) matrix for this batch.
        b : np.ndarray, shape ``(m,)``
            Observation vector for this batch.

        Returns
        -------
        np.ndarray, shape ``(n,)``
            Updated parameter estimate ``theta_hat``.
        """
        ...
