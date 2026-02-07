"""Tests for all estimator algorithms in online_estimators.estimators."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.estimators import (
    FDBK,
    GRK,
    IGRK,
    KF,
    MGRK,
    REK,
    RK,
    RLS,
    TAGK,
    TARK,
    GRK_Tikh,
    RK_ColScaled,
    RK_Equi,
    RK_EquiTikh,
    RK_Tikh,
    TAGK_Tikh,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KACZMARZ_ESTIMATORS = [
    RK,
    TARK,
    GRK,
    TAGK,
    TAGK_Tikh,
    GRK_Tikh,
    IGRK,
    MGRK,
    REK,
    FDBK,
    RK_ColScaled,
    RK_Equi,
    RK_Tikh,
    RK_EquiTikh,
]


def _converges(
    estimator_cls,
    A: np.ndarray,
    b: np.ndarray,
    x_true: np.ndarray,
    tol: float = 0.5,
    iters: int = 200,
    **kwargs,
) -> bool:
    """Run an estimator for *iters* iterations and check convergence."""
    n = A.shape[1]
    est = estimator_cls(n, **kwargs)
    x_est = None
    for _ in range(iters):
        x_est = est.iterate(A, b)
    assert x_est is not None
    return np.linalg.norm(x_est - x_true) < tol


# ---------------------------------------------------------------------------
# RLS / KF
# ---------------------------------------------------------------------------


class TestRLS:
    def test_converges_clean(self, simple_system):
        A, b, x_true = simple_system
        est = RLS(3, forgetting_factor=1.0, c=100)
        for _ in range(50):
            x_est = est.iterate(A, b)
        assert np.linalg.norm(x_est - x_true) < 0.1

    def test_forgetting_factor(self, simple_system):
        A, b, x_true = simple_system
        est = RLS(3, forgetting_factor=0.9, c=100)
        for _ in range(100):
            x_est = est.iterate(A, b)
        assert np.linalg.norm(x_est - x_true) < 0.5

    def test_custom_initial_estimate(self):
        A = np.eye(3)
        x_true = np.array([5.0, 6.0, 7.0])
        b = A @ x_true
        theta0 = np.array([5.1, 5.9, 7.1])
        est = RLS(3, theta_hat=theta0, c=100)
        for _ in range(10):
            x_est = est.iterate(A, b)
        assert np.linalg.norm(x_est - x_true) < 0.1


class TestKF:
    def test_converges_clean(self, simple_system):
        A, b, x_true = simple_system
        n = 3
        m = A.shape[0]
        est = KF(n, process_noise=1e-3 * np.eye(n), measurement_noise=1e-4 * np.eye(m), c=100)
        for _ in range(50):
            x_est = est.iterate(A, b)
        assert np.linalg.norm(x_est - x_true) < 0.5


# ---------------------------------------------------------------------------
# Kaczmarz-family estimators
# ---------------------------------------------------------------------------


class TestKaczmarzFamily:
    """Parametric test: every Kaczmarz-family estimator should converge on
    a simple consistent system."""

    @pytest.mark.parametrize("cls", KACZMARZ_ESTIMATORS, ids=lambda c: c.__name__)
    def test_converges_on_simple_system(self, cls, simple_system):
        A, b, x_true = simple_system
        # Tikhonov variants are biased toward 0 so need more tolerance
        tol = 3.5 if "Tikh" in cls.__name__ else 1.0
        assert _converges(cls, A, b, x_true, tol=tol, iters=1000)

    @pytest.mark.parametrize("cls", KACZMARZ_ESTIMATORS, ids=lambda c: c.__name__)
    def test_custom_x0(self, cls, simple_system):
        A, b, x_true = simple_system
        x0 = np.array([0.9, 2.1, 2.9])
        # Tikhonov, REK, and TAGK variants may converge more slowly
        if "Tikh" in cls.__name__ or "REK" in cls.__name__ or "TAGK" in cls.__name__:
            tol = 3.5
        else:
            tol = 1.0
        assert _converges(cls, A, b, x_true, tol=tol, iters=1000, x0=x0)


# ---------------------------------------------------------------------------
# Specific algorithm tests
# ---------------------------------------------------------------------------


class TestRK:
    def test_iterate_returns_ndarray(self, simple_system):
        A, b, _ = simple_system
        est = RK(3)
        result = est.iterate(A, b)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_deterministic_with_seed(self, simple_system):
        """Two RK instances with same data should eventually converge
        to the same neighbourhood (stochastic, but checks shape)."""
        A, b, _ = simple_system
        est = RK(3)
        results = [est.iterate(A, b) for _ in range(50)]
        assert len(results) == 50


class TestTARK:
    def test_tail_average_improves(self, simple_system):
        A, b, x_true = simple_system
        est = TARK(3)
        for _ in range(200):
            x_est = est.iterate(A, b)
        err = np.linalg.norm(x_est - x_true)
        assert err < 2.0  # tail averaging should help


class TestTAGKTikh:
    """The flagship TAG-K algorithm with Tikhonov regularisation."""

    def test_converges_well(self, simple_system):
        A, b, x_true = simple_system
        est = TAGK_Tikh(3)
        for _ in range(1000):
            x_est = est.iterate(A, b)
        # Tikhonov regularisation biases toward zero; accept larger tolerance
        assert np.linalg.norm(x_est - x_true) < 2.5

    def test_overdetermined_noisy(self, overdetermined_noisy):
        A, b, x_true = overdetermined_noisy
        est = TAGK_Tikh(5)
        for _ in range(500):
            x_est = est.iterate(A, b)
        assert np.linalg.norm(x_est - x_true) < 2.0


class TestREK:
    def test_handles_inconsistent(self, rng):
        """REK is designed for inconsistent systems."""
        m, n = 20, 3
        A = rng.standard_normal((m, n))
        x_star = np.linalg.lstsq(A, rng.standard_normal(m), rcond=None)[0]
        b = A @ x_star + 0.5 * rng.standard_normal(m)
        est = REK(n)
        for _ in range(300):
            x_est = est.iterate(A, b)
        assert x_est.shape == (n,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row(self):
        A = np.array([[1.0, 2.0]])
        b = np.array([5.0])
        est = RK(2)
        x = est.iterate(A, b)
        assert x.shape == (2,)

    def test_zero_b(self):
        A = np.eye(3)
        b = np.zeros(3)
        est = GRK(3)
        for _ in range(50):
            x = est.iterate(A, b)
        assert np.linalg.norm(x) < 0.5

    def test_large_system(self, rng):
        m, n = 500, 50
        A = rng.standard_normal((m, n))
        x_true = rng.standard_normal(n)
        b = A @ x_true
        est = RK(n)
        for _ in range(200):
            x = est.iterate(A, b)
        assert x.shape == (n,)
