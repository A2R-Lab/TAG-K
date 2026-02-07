"""
Greedy Randomized Kaczmarz (GRK) and its variants.

GRK improves upon vanilla Randomized Kaczmarz by selecting rows with
*large* residual-to-norm ratios, concentrating updates where they reduce
the error most.  An adaptive threshold ``eps_k`` is computed each iteration
to define the "greedy" working set.

Variants in this module:

- :class:`GRK`                --  base greedy RK
- :class:`TAGK`                --  GRK + Polyak tail-averaging
- :class:`GRK_Tikh`           --  GRK on the Tikhonov-augmented system
- :class:`TAGK_Tikh`           --  GRK + Tikhonov + tail-averaging
"""

from __future__ import annotations

from typing import Any

import numpy as np

from online_estimators.estimators import _backend
from online_estimators.estimators.base import BaseEstimator


def _grk_step(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    row_norms_sq: np.ndarray,
    fro_sq_safe: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Single greedy Kaczmarz iteration.

    Computes the residual, builds a greedy working set, samples a row
    from it, and performs the projection step.

    Parameters
    ----------
    A : np.ndarray, shape ``(m, n)``
    b : np.ndarray, shape ``(m,)``
    x : np.ndarray, shape ``(n,)``
        Current iterate (modified in-place).
    row_norms_sq : np.ndarray, shape ``(m,)``
    fro_sq_safe : float
        ``max(||A||_F^2, eps)`` for stability.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Updated ``x``.
    """
    m = A.shape[0]
    r = b - A @ x
    rnorm_sq = float(r @ r)
    if rnorm_sq <= 0.0:
        return x

    with np.errstate(divide="ignore", invalid="ignore"):
        crit = np.where(row_norms_sq > 0.0, (r * r) / row_norms_sq, 0.0)

    rnorm_sq_safe = max(rnorm_sq, 1e-300)
    eps_k = 0.5 * (crit.max() / rnorm_sq_safe + 1.0 / fro_sq_safe)
    tau_mask = (r * r) >= (eps_k * rnorm_sq * row_norms_sq)

    if not np.any(tau_mask):
        i = int(np.argmax(np.abs(r)))
        tau_mask = np.zeros(m, dtype=bool)
        tau_mask[i] = True

    r_tilde = np.where(tau_mask, r, 0.0)
    rt_sq = float(r_tilde @ r_tilde)

    if rt_sq <= 0.0:
        candidates = np.flatnonzero(tau_mask)
        i_k = int(candidates[np.argmax(np.abs(r[candidates]))])
    else:
        probs = (r_tilde * r_tilde) / rt_sq
        i_k = int(rng.choice(m, p=probs))

    ai = A[i_k, :]
    den = max(float(ai @ ai), 1e-300)
    res_i = b[i_k] - float(ai @ x)
    x = x + (res_i / den) * ai
    return x


class GRK(BaseEstimator):
    """Greedy Randomized Kaczmarz (GRK).

    Selects rows whose per-row residual is large relative to the global
    residual, then samples from this "greedy set" with residual-proportional
    probabilities.

    When the C++ extension ``online_estimation`` is installed, the iterate
    method delegates to a compiled implementation for better performance.
    The C++ backend does not support the ``n_iters``, ``x0``, ``tol``, or ``rng``
    arguments -- those calls fall back to pure Python automatically.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    default_tol : float
        Default tolerance used by the C++ backend when available.

    References
    ----------
    Bai, Z. & Wu, W.  "On greedy randomized Kaczmarz method for solving
    large sparse linear systems."  *SIAM J. Sci. Comput.*, 40(1), 2018.

    Examples
    --------
    >>> from online_estimators.estimators import GRK
    >>> grk = GRK(num_params=10)
    >>> theta = grk.iterate(A, b)
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        default_tol: float = 1e-4,
    ) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )

        # C++ backend (optional)
        self._cpp: Any = None
        if _backend.HAS_CPP:
            x0_col = self.x0.reshape(-1, 1) if x0 is not None else None
            self._cpp = _backend._CppGRK(num_params, float(default_tol), x0_col)

    def seed_rng(self, seed: int) -> None:
        """Seed the internal random engine for reproducibility.

        Parameters
        ----------
        seed : int
            Seed value.
        """
        if self._cpp is not None:
            self._cpp.seed_rng(seed)

    def iterate(
        self,
        A: np.ndarray,
        b: np.ndarray,
        n_iters: int | None = None,
        x0: np.ndarray | None = None,
        tol: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run ``n_iters`` greedy Kaczmarz iterations.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        n_iters : int or None
            Number of iterations.  Defaults to ``m``.
        x0 : np.ndarray or None
            Override starting point for this call.
        tol : float
            Early-stop tolerance on ``||r||^2``.
        rng : np.random.Generator or None

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        # Use C++ backend when no overrides are requested
        if self._cpp is not None and n_iters is None and x0 is None and rng is None:
            result = self._cpp.iterate(A, b.reshape(-1, 1))
            self.x0 = np.asarray(result).ravel()
            return self.x0.copy()

        x = (self.x0 if x0 is None else np.asarray(x0, dtype=float).reshape(-1)).copy()
        row_norms_sq = (A * A).sum(axis=1)
        fro_sq_safe = max(float((A * A).sum()), 1e-300)
        if n_iters is None:
            n_iters = m
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(n_iters)):
            r = b - A @ x
            if float(r @ r) <= tol**2:
                break
            x = _grk_step(A, b, x, row_norms_sq, fro_sq_safe, rng)

        self.x0 = x
        return x.copy()


class TAGK(BaseEstimator):
    """Greedy RK with Polyak tail-averaging.

    Same greedy row selection as :class:`GRK`, but returns the average of
    iterates collected *after* a burn-in phase.

    When the C++ extension ``online_estimation`` is installed and no
    per-call overrides are requested, the iterate method delegates to the
    C++ ``TAGK`` implementation for better performance.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    burnin : int
        Iterations to skip before starting the running average.
    default_tol : float
        Default tolerance used by the C++ backend when available.
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        burnin: int = 0,
        default_tol: float = 1e-4,
    ) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )
        self.burnin = burnin

        # C++ backend (optional) -- uses the TAGK (GRK + tail averaging) impl
        self._cpp: Any = None
        if _backend.HAS_CPP:
            x0_col = self.x0.reshape(-1, 1) if x0 is not None else None
            self._cpp = _backend._CppTAGK(num_params, int(burnin), float(default_tol), x0_col)

    def seed_rng(self, seed: int) -> None:
        """Seed the internal random engine for reproducibility.

        Parameters
        ----------
        seed : int
            Seed value.
        """
        if self._cpp is not None:
            self._cpp.seed_rng(seed)

    def iterate(
        self,
        A: np.ndarray,
        b: np.ndarray,
        n_iters: int | None = None,
        x0: np.ndarray | None = None,
        tol: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run greedy Kaczmarz with tail-averaging.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        n_iters : int or None
        x0 : np.ndarray or None
        tol : float
        rng : np.random.Generator or None

        Returns
        -------
        np.ndarray, shape ``(n,)``
            Tail-averaged iterate.
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        # Use C++ backend when no overrides are requested
        if self._cpp is not None and n_iters is None and x0 is None and rng is None:
            result = self._cpp.iterate(A, b.reshape(-1, 1))
            self.x0 = np.asarray(result).ravel()
            return self.x0.copy()

        x = (self.x0 if x0 is None else np.asarray(x0, dtype=float).reshape(-1)).copy()
        x_sum = np.zeros_like(x)
        count = 0
        row_norms_sq = (A * A).sum(axis=1)
        fro_sq_safe = max(float((A * A).sum()), 1e-300)
        if n_iters is None:
            n_iters = m
        if rng is None:
            rng = np.random.default_rng()

        for ss in range(int(n_iters)):
            r = b - A @ x
            if float(r @ r) <= tol**2:
                break
            x = _grk_step(A, b, x, row_norms_sq, fro_sq_safe, rng)
            if ss >= self.burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / max(count, 1)
        self.x0 = x_avg
        return x_avg.copy()


class GRK_Tikh(BaseEstimator):
    """Greedy RK on the Tikhonov-augmented system.

    Appends ``sqrtlam * I`` rows to ``A`` for implicit regularisation,
    then applies greedy row selection.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    lam : float or None
    lam_scale : float
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        lam: float | None = None,
        lam_scale: float = 1e-5,
    ) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )
        self.lam = lam
        self.lam_scale = float(lam_scale)

    def iterate(
        self,
        A: np.ndarray,
        b: np.ndarray,
        n_iters: int | None = None,
        x0: np.ndarray | None = None,
        tol: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run Tikhonov-augmented greedy Kaczmarz.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        n_iters : int or None
        x0 : np.ndarray or None
        tol : float
        rng : np.random.Generator or None

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape

        lam = self.lam
        if lam is None:
            lam = self.lam_scale * (np.linalg.norm(A, ord="fro") ** 2 / max(m, 1))
        A = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        b = np.concatenate([b, np.zeros(n)])
        m, n = A.shape
        assert n == self.num_params

        x = (self.x0 if x0 is None else np.asarray(x0, dtype=float).reshape(-1)).copy()
        row_norms_sq = (A * A).sum(axis=1)
        fro_sq_safe = max(float((A * A).sum()), 1e-300)
        if n_iters is None:
            n_iters = m
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(n_iters)):
            r = b - A @ x
            if float(r @ r) <= tol**2:
                break
            x = _grk_step(A, b, x, row_norms_sq, fro_sq_safe, rng)

        self.x0 = x
        return x.copy()


class TAGK_Tikh(BaseEstimator):
    """Greedy RK + Tikhonov augmentation + tail-averaging.

    This is the **TAG-K** algorithm: combines greedy row selection,
    Tikhonov regularisation, and Polyak tail-averaging for robust
    online parameter estimation.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    lam : float or None
    lam_scale : float
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        lam: float | None = None,
        lam_scale: float = 1e-3,
    ) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )
        self.lam = lam
        self.lam_scale = float(lam_scale)

    def iterate(
        self,
        A: np.ndarray,
        b: np.ndarray,
        n_iters: int | None = None,
        x0: np.ndarray | None = None,
        tol: float = 0.0,
        rng: np.random.Generator | None = None,
        burnin: int = 0,
    ) -> np.ndarray:
        """Run TAG-K: greedy Kaczmarz + Tikhonov + tail-averaging.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        n_iters : int or None
        x0 : np.ndarray or None
        tol : float
        rng : np.random.Generator or None
        burnin : int

        Returns
        -------
        np.ndarray, shape ``(n,)``
            Tail-averaged iterate.
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape

        lam = self.lam
        if lam is None:
            lam = self.lam_scale * (np.linalg.norm(A, ord="fro") ** 2 / max(m, 1))
        A = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        b = np.concatenate([b, np.zeros(n)])
        m, n = A.shape
        assert n == self.num_params

        x = (self.x0 if x0 is None else np.asarray(x0, dtype=float).reshape(-1)).copy()
        x_sum = np.zeros_like(x)
        count = 0
        row_norms_sq = (A * A).sum(axis=1)
        fro_sq_safe = max(float((A * A).sum()), 1e-300)
        if n_iters is None:
            n_iters = m
        if rng is None:
            rng = np.random.default_rng()

        for ss in range(int(n_iters)):
            r = b - A @ x
            if float(r @ r) <= tol**2:
                break
            x = _grk_step(A, b, x, row_norms_sq, fro_sq_safe, rng)
            if ss >= burnin:
                x_sum += x
                count += 1

        x_avg = x_sum / max(count, 1)
        self.x0 = x
        return x_avg.copy()
