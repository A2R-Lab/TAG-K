"""
Randomized Kaczmarz (RK) and its preconditioned/augmented variants.

The basic Randomized Kaczmarz algorithm solves ``Ax = b`` by repeatedly
projecting onto randomly selected hyperplanes defined by individual rows
of ``A``.  Rows are sampled proportional to their squared norm.

This module provides:

- :class:`RK`  --  vanilla randomized Kaczmarz
- :class:`TARK`  --  tail-averaged RK (Polyak averaging over the second half)
- :class:`RK_ColScaled`  --  column-scaled (right-preconditioned) RK
- :class:`RK_Equi`  --  Ruiz-equilibrated RK (left + right preconditioning)
- :class:`RK_Tikh`  --  Tikhonov-augmented RK (ridge-like regularisation)
- :class:`RK_EquiTikh`  --  Ruiz equilibration + Tikhonov augmentation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from online_estimators.estimators import _backend
from online_estimators.estimators.base import BaseEstimator

# ------------------------------- helpers ------------------------------


def _row_norms_sq(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Squared l_2 norms of rows, with a small stabiliser."""
    return (A * A).sum(axis=1) + eps


def _row_probs(row_norms_sq: np.ndarray) -> np.ndarray:
    """Probability vector proportional to squared row norms."""
    return row_norms_sq / row_norms_sq.sum()


def ruiz_equilibrate(
    A: np.ndarray, iters: int = 5, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ruiz equilibration: iterative row/column scaling.

    Returns ``(B, Dr, Dc)`` such that ``B = diag(Dr) A diag(Dc)`` is
    approximately doubly-stochastic in row/column norms.

    Parameters
    ----------
    A : np.ndarray, shape ``(m, n)``
    iters : int
        Number of Ruiz iterations.  5 is usually sufficient.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    B : np.ndarray, shape ``(m, n)``
        Equilibrated matrix.
    Dr : np.ndarray, shape ``(m,)``
        Row scaling factors.
    Dc : np.ndarray, shape ``(n,)``
        Column scaling factors.
    """
    m, n = A.shape
    Dr = np.ones(m)
    Dc = np.ones(n)
    B = A.copy()
    for _ in range(iters):
        r = np.sqrt((B * B).sum(axis=1)) + eps
        Dr *= 1.0 / r
        B = (B.T / r).T
        c = np.sqrt((B * B).sum(axis=0)) + eps
        Dc *= 1.0 / c
        B = B / c
    return B, Dr, Dc


# ------------------------ Randomized Kaczmarz -------------------------


class RK(BaseEstimator):
    """Classic Randomized Kaczmarz.

    At each call, performs ``m`` inner Kaczmarz sweeps (one per row in
    expectation) with row-norm-proportional sampling.

    When the C++ extension ``online_estimation`` is installed, the iterate
    method delegates to a compiled implementation for better performance.

    Parameters
    ----------
    num_params : int
        Dimension of theta.
    x0 : np.ndarray or None
        Initial iterate.  Defaults to zero.

    Examples
    --------
    >>> import numpy as np
    >>> from online_estimators.estimators import RK
    >>> rk = RK(num_params=5)
    >>> A = np.random.randn(20, 5)
    >>> b = A @ np.ones(5)
    >>> theta = rk.iterate(A, b)
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None) -> None:
        super().__init__(num_params)
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

        # C++ backend (optional)
        self._cpp: Any = None
        if _backend.HAS_CPP:
            x0_col = self.x0.reshape(-1, 1) if x0 is not None else None
            self._cpp = _backend._CppRK(num_params, x0_col)

    def seed_rng(self, seed: int) -> None:
        """Seed the internal random engine for reproducibility.

        Parameters
        ----------
        seed : int
            Seed value.
        """
        if self._cpp is not None:
            self._cpp.seed_rng(seed)

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run one epoch of randomized Kaczmarz sweeps.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float
            Numerical stability constant.

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        if self._cpp is not None:
            result = self._cpp.iterate(A, b.reshape(-1, 1))
            self.x0 = np.asarray(result).ravel()
            return self.x0.copy()

        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        rn2 = _row_norms_sq(A, eps)
        probs = _row_probs(rn2)
        for _ in range(m):
            i = np.random.choice(m, p=probs)
            ai = A[i]
            x += ((b[i] - ai @ x) / rn2[i]) * ai
        self.x0 = x
        return x.copy()


# -------------------- Tail-Averaged RK (TARK) ------------------------


class TARK(BaseEstimator):
    """Tail-Averaged Randomized Kaczmarz.

    Identical to :class:`RK` but returns the Polyak average of the iterates
    *after* a configurable burn-in period.

    When the C++ extension ``online_estimation`` is installed, the iterate
    method delegates to a compiled implementation for better performance.
    Note that the C++ backend uses the ``burnin`` value from construction
    and ignores the per-call ``burnin`` argument.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    default_burnin : int
        Default burn-in value used by the C++ backend when available.

    Examples
    --------
    >>> from online_estimators.estimators import TARK
    >>> est = TARK(num_params=3)
    >>> theta = est.iterate(A, b, burnin=5)
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        default_burnin: int = 0,
    ) -> None:
        super().__init__(num_params)
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

        # C++ backend (optional)
        self._cpp: Any = None
        if _backend.HAS_CPP:
            x0_col = self.x0.reshape(-1, 1) if x0 is not None else None
            self._cpp = _backend._CppTARK(num_params, int(default_burnin), x0_col)

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
        self, A: np.ndarray, b: np.ndarray, burnin: int = 0, eps: float = 1e-12
    ) -> np.ndarray:
        """Run one epoch and return the tail-averaged iterate.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        burnin : int
            Number of initial sweeps to discard before averaging.
            Ignored when the C++ backend is active (uses constructor value).
        eps : float
            Numerical stability constant.

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        if self._cpp is not None:
            result = self._cpp.iterate(A, b.reshape(-1, 1))
            self.x0 = np.asarray(self._cpp.state).ravel()
            return np.asarray(result).ravel().copy()

        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        x_sum = np.zeros_like(x)
        count = 0
        rn2 = _row_norms_sq(A, eps)
        probs = _row_probs(rn2)
        for s in range(m):
            i = np.random.choice(m, p=probs)
            ai = A[i]
            x += ((b[i] - ai @ x) / rn2[i]) * ai
            if s >= burnin:
                x_sum += x
                count += 1
        x_avg = x_sum / max(count, 1)
        self.x0 = x
        return x_avg.copy()


# ---------------- Column-Scaled RK (right preconditioning) -----------


class RK_ColScaled(BaseEstimator):
    """Randomized Kaczmarz with column scaling (right preconditioning).

    Scales columns of ``A`` by their standard deviation before running
    Kaczmarz, then maps the solution back.  This is effective when
    columns have very different magnitudes.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run column-scaled Kaczmarz.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        col_scale = A.std(axis=0) + 1e-12
        A_col = A / col_scale
        y = (self.x0 * col_scale).copy()
        rn2 = _row_norms_sq(A_col, eps)
        probs = _row_probs(rn2)
        for _ in range(m):
            i = np.random.choice(m, p=probs)
            ai = A_col[i]
            y += ((b[i] - ai @ y) / rn2[i]) * ai
        x = y / col_scale
        self.x0 = x
        return x.copy()


# ---------------- Ruiz-Equilibrated RK -------------------------------


class RK_Equi(BaseEstimator):
    """Randomized Kaczmarz with Ruiz equilibration.

    Applies iterative row/column (Ruiz) scaling before Kaczmarz to
    improve the condition number of ``A``.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    ruiz_iters : int
        Number of Ruiz balancing iterations.
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None, ruiz_iters: int = 5) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )
        self.ruiz_iters = int(ruiz_iters)

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run Ruiz-equilibrated Kaczmarz.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        Aeq, Dr, Dc = ruiz_equilibrate(A, iters=self.ruiz_iters, eps=eps)
        beq = Dr * b
        y = (self.x0 / (Dc + 1e-18)).copy()
        rn2 = _row_norms_sq(Aeq, eps)
        probs = _row_probs(rn2)
        for _ in range(m):
            i = np.random.choice(m, p=probs)
            ai = Aeq[i]
            y += ((beq[i] - ai @ y) / rn2[i]) * ai
        x = Dc * y
        self.x0 = x
        return x.copy()


# ---------------- Tikhonov-Augmented RK ------------------------------


class RK_Tikh(BaseEstimator):
    """Randomized Kaczmarz on the Tikhonov-augmented system.

    Appends ``sqrtlam * I`` rows and zero RHS to the system, turning
    Kaczmarz into an implicit ridge regression.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    lam : float or None
        Explicit regularisation parameter.  If ``None``, it is set
        automatically as ``lam_scale * ||A||_F^2 / m``.
    lam_scale : float
        Scale factor for the automatic ``lam``.
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

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run Tikhonov-augmented Kaczmarz.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        lam = self.lam
        if lam is None:
            lam = self.lam_scale * (np.linalg.norm(A, ord="fro") ** 2 / max(m, 1))
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        b_aug = np.concatenate([b, np.zeros(n)])

        x = self.x0.copy()
        rn2 = _row_norms_sq(A_aug, eps)
        probs = _row_probs(rn2)
        m_aug = A_aug.shape[0]
        for _ in range(m_aug):
            i = np.random.choice(m_aug, p=probs)
            ai = A_aug[i]
            x += ((b_aug[i] - ai @ x) / rn2[i]) * ai
        self.x0 = x
        return x.copy()


# ---------------- Ruiz + Tikhonov RK ---------------------------------


class RK_EquiTikh(BaseEstimator):
    """RK with both Ruiz equilibration and Tikhonov augmentation.

    Combines the benefits of condition-number improvement (Ruiz) and
    regularisation (Tikhonov).

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    ruiz_iters : int
    lam : float or None
    lam_scale : float
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        ruiz_iters: int = 5,
        lam: float | None = None,
        lam_scale: float = 1e-3,
    ) -> None:
        super().__init__(num_params)
        self.x0 = (
            np.zeros(self.num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )
        self.ruiz_iters = int(ruiz_iters)
        self.lam = lam
        self.lam_scale = float(lam_scale)

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run Ruiz-equilibrated, Tikhonov-augmented Kaczmarz.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        Aeq, Dr, Dc = ruiz_equilibrate(A, iters=self.ruiz_iters, eps=eps)
        beq = Dr * b

        lam = self.lam
        if lam is None:
            lam = self.lam_scale * (np.linalg.norm(A, ord="fro") ** 2 / max(m, 1))
        A_bot = np.sqrt(lam) * np.diag(Dc)
        A_aug = np.vstack([Aeq, A_bot])
        b_aug = np.concatenate([beq, np.zeros(n)])

        y = (self.x0 / (Dc + 1e-18)).copy()
        rn2 = _row_norms_sq(A_aug, eps)
        probs = _row_probs(rn2)
        m_aug = A_aug.shape[0]
        for _ in range(m_aug):
            i = np.random.choice(m_aug, p=probs)
            ai = A_aug[i]
            y += ((b_aug[i] - ai @ y) / rn2[i]) * ai
        x = Dc * y
        self.x0 = x
        return x.copy()
