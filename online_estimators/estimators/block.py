"""
Block and specialised Kaczmarz variants.

- :class:`FDBK`  --  Fast Deterministic Block Kaczmarz
- :class:`IGRK`  --  Improved Greedy Randomized Kaczmarz
- :class:`MGRK`  --  Momentum-accelerated Greedy RK
- :class:`REK`   --  Randomized Extended Kaczmarz (handles inconsistent systems)
"""

from __future__ import annotations

import numpy as np

from online_estimators.estimators.base import BaseEstimator


class FDBK(BaseEstimator):
    """Fast Deterministic Block Kaczmarz (FDBK).

    At each iteration, constructs a *block* of greedy rows (those with
    large per-row residuals) and performs a single combined projection.
    This can converge faster than single-row methods on well-structured
    systems.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None

    References
    ----------
    Necoara, I.  "Faster randomized block Kaczmarz algorithms."
    *SIAM J. Matrix Anal. Appl.*, 40(4), 2019.
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None) -> None:
        super().__init__(num_params)
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

    def iterate(
        self, A: np.ndarray, b: np.ndarray, tol: float = 1e-4, eps_row: float = 1e-12
    ) -> np.ndarray:
        """Run one pass of Fast Deterministic Block Kaczmarz.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        tol : float
            Convergence tolerance on ``||r||^2``.
        eps_row : float
            Numerical floor.

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        row_norms_sq = (A * A).sum(axis=1)
        fro_sq = float((A * A).sum())
        x = self.x0.copy()

        for _ in range(m):
            r = b - A @ x
            rnorm_sq = float(r @ r)
            if rnorm_sq <= tol:
                break
            r_safe = max(rnorm_sq, 1e-300)
            with np.errstate(divide="ignore", invalid="ignore"):
                crit = np.where(row_norms_sq > 0, (r * r) / row_norms_sq, 0.0)
            eps_k = 0.5 * (crit.max() / r_safe + 1.0 / max(fro_sq, 1e-300))
            tau_mask = (r * r) >= (eps_k * r_safe * row_norms_sq)
            if not tau_mask.any():
                break
            eta = np.zeros(m, dtype=float)
            eta[tau_mask] = r[tau_mask]
            At_eta = A.T @ eta
            denom = max(float(At_eta @ At_eta), 1e-300)
            alpha = float(eta @ r) / denom
            x = x + alpha * At_eta

        self.x0 = x
        return x.copy()


class IGRK(BaseEstimator):
    """Improved Greedy Randomized Kaczmarz.

    Uses a stricter greedy threshold (based on crit-max + Frobenius
    contribution) and runs ``m*n`` inner iterations per call.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None) -> None:
        super().__init__(num_params)
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run ``m*n`` IGRK sweeps.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A)
        b = np.asarray(b).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        x = self.x0.copy()
        row_norms_sq = (A * A).sum(axis=1) + eps

        for _ in range(m * n):
            r = A @ x - b
            crit = (r * r) / row_norms_sq
            Nk = np.abs(r) > 0
            if not Nk.any():
                break
            Gamma_k = row_norms_sq[Nk].sum()
            thresh = 0.5 * (crit.max() + (np.dot(r, r) / max(Gamma_k, eps)))
            valid = Nk & (row_norms_sq > eps)
            Jk = np.flatnonzero(valid & (crit >= thresh))
            if Jk.size == 0:
                break
            weights = crit[Jk]
            if not np.isfinite(weights).any() or weights.sum() <= 0:
                break
            weights = weights / weights.sum()
            i = np.random.choice(Jk, p=weights)
            ai = A[i]
            x -= (r[i] / row_norms_sq[i]) * ai

        self.x0 = x
        return x.copy()


class MGRK(BaseEstimator):
    """Momentum-accelerated Greedy Randomized Kaczmarz.

    Adds a heavy-ball momentum term ``beta(x_k - x_{k-1})`` to the
    standard Kaczmarz update for faster convergence on ill-conditioned
    systems.

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None
    alpha : float
        Step size multiplier (default 1.0).
    beta : float
        Momentum coefficient (default 0.25).
    theta : float
        Threshold blending parameter (default 0.5).
    """

    def __init__(
        self,
        num_params: int,
        x0: np.ndarray | None = None,
        alpha: float = 1.0,
        beta: float = 0.25,
        theta: float = 0.5,
    ) -> None:
        super().__init__(num_params)
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        self.x_prev = self.x0.copy()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.theta = float(theta)

    def iterate(self, A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Run ``m*n`` momentum-accelerated greedy iterations.

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

        x = self.x0.copy()
        x_prev = self.x_prev.copy()
        row_norms_sq = (A * A).sum(axis=1) + eps

        for _ in range(m * n):
            r = A @ x - b
            Nk = np.abs(r) > 0
            if not Nk.any():
                break
            Gamma_k = row_norms_sq[Nk].sum()
            crit = (r * r) / row_norms_sq
            thresh = self.theta * crit.max() + (1.0 - self.theta) * (
                np.dot(r, r) / max(Gamma_k, eps)
            )
            valid = Nk & (row_norms_sq > eps)
            Sk = np.flatnonzero(valid & (crit >= thresh))
            if Sk.size == 0:
                break
            weights = crit[Sk]
            ws = weights.sum()
            p = weights / ws if np.isfinite(ws) and ws > 0 else np.ones(Sk.size) / Sk.size
            i = np.random.choice(Sk, p=p)
            ai = A[i]
            ri = ai @ x - b[i]
            grad = (ri / row_norms_sq[i]) * ai
            x_new = x - self.alpha * grad + self.beta * (x - x_prev)
            x_prev, x = x, x_new

        self.x0 = x.copy()
        self.x_prev = x.copy()
        return x.copy()


class REK(BaseEstimator):
    """Randomized Extended Kaczmarz.

    Simultaneously projects onto both row and column spaces of ``A``,
    allowing convergence even for *inconsistent* systems (where
    ``b not in range(A)``).

    Parameters
    ----------
    num_params : int
    x0 : np.ndarray or None

    References
    ----------
    Zouzias, A. & Freris, N.  "Randomized extended Kaczmarz for solving
    least squares."  *SIAM J. Matrix Anal. Appl.*, 34(2), 2013.
    """

    def __init__(self, num_params: int, x0: np.ndarray | None = None) -> None:
        super().__init__(num_params)
        self.x0 = np.zeros(num_params) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

    def iterate(
        self,
        A: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-6,
        max_passes: int = 2,
    ) -> np.ndarray:
        """Run REK until convergence or ``max_passes`` epochs.

        Parameters
        ----------
        A : np.ndarray, shape ``(m, n)``
        b : np.ndarray, shape ``(m,)``
        eps : float
            Convergence tolerance (primal + dual).
        max_passes : int
            Maximum number of passes over the rows.

        Returns
        -------
        np.ndarray, shape ``(n,)``
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        assert n == self.num_params

        row_norms_sq = (A * A).sum(axis=1)
        col_norms_sq = (A * A).sum(axis=0)
        fro2 = row_norms_sq.sum() + 1e-12

        row_probs = row_norms_sq / fro2
        col_probs = col_norms_sq / fro2
        row_probs = np.where(row_norms_sq > 0, row_probs, 0.0)
        col_probs = np.where(col_norms_sq > 0, col_probs, 0.0)
        row_probs /= row_probs.sum()
        col_probs /= col_probs.sum()

        x = self.x0.copy()
        z = b.copy()
        max_iters = max(1, int(max_passes * m))
        check_every = max(1, 8 * min(m, n))
        fro = np.sqrt(fro2) if fro2 > 0 else 1.0

        for k in range(max_iters):
            # Column space projection (dual step)
            z_prev = z  # keep z^{(k)} for the x-update below
            j = np.random.choice(n, p=col_probs)
            aj = A[:, j]
            den_c = col_norms_sq[j] if col_norms_sq[j] > 0 else 1.0
            z = z - (aj @ z) / den_c * aj

            # Row space projection (primal step) -- uses z^{(k)}, not z^{(k+1)}
            i = np.random.choice(m, p=row_probs)
            ai = A[i, :]
            den_r = row_norms_sq[i] if row_norms_sq[i] > 0 else 1.0
            x = x + (b[i] - z_prev[i] - ai @ x) / den_r * ai

            # Convergence check
            if (k + 1) % check_every == 0:
                Ax = A @ x
                xnorm = np.linalg.norm(x)
                denom_x = max(xnorm, 1e-12)
                primal = np.linalg.norm(Ax - (b - z)) / (fro * denom_x)
                dual = np.linalg.norm(A.T @ z) / ((fro * fro) * denom_x)
                if primal <= eps and dual <= eps:
                    break

        self.x0 = x
        return x.copy()
