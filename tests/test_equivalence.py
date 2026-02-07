"""
Equivalence tests: verify that the new online_estimators package produce the same
output as the original python/common/estimation_methods.py implementations
when given identical inputs and random seeds.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import the *old* estimators from python/common/estimation_methods.py
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[1]
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

old_mod = importlib.import_module("common.estimation_methods")

# ---------------------------------------------------------------------------
# Import the *new* estimators from online_estimators
# ---------------------------------------------------------------------------
from online_estimators.estimators import (  # noqa: E402
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
SEED = 42
N_PARAMS = 5
N_ROWS = 20
N_ITERS = 10


def _make_problem(seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a well-conditioned (A, b, theta_true) test problem."""
    rng = np.random.RandomState(seed)
    theta_true = rng.randn(N_PARAMS)
    A = rng.randn(N_ROWS, N_PARAMS)
    b = A @ theta_true + 0.01 * rng.randn(N_ROWS)
    return A, b, theta_true


def _run_iters(est_old, est_new, n_iters: int = N_ITERS, seed: int = SEED):
    """Run n_iters iterate() calls on both estimators with identical inputs."""
    results_old = []
    results_new = []
    for i in range(n_iters):
        A, b, _ = _make_problem(seed=seed + i)
        np.random.seed(seed + 1000 + i)  # sync global RNG for both
        old_result = est_old.iterate(A, b)
        np.random.seed(seed + 1000 + i)  # reset to same state
        new_result = est_new.iterate(A, b)
        results_old.append(np.asarray(old_result).ravel().copy())
        results_new.append(np.asarray(new_result).ravel().copy())
    return results_old, results_new


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRLSEquivalence:
    """RLS: old vs new should produce identical outputs."""

    def test_default_params(self):
        old = old_mod.RLS(N_PARAMS)
        new = RLS(N_PARAMS)
        # Disable C++ backend for pure-Python comparison
        new._cpp = None
        results_old, results_new = _run_iters(old, new)
        for i, (o, n) in enumerate(zip(results_old, results_new)):
            np.testing.assert_allclose(o, n, atol=1e-12, err_msg=f"iter {i}")

    def test_custom_params(self):
        old = old_mod.RLS(N_PARAMS, forgetting_factor=0.9, c=500.0)
        new = RLS(N_PARAMS, forgetting_factor=0.9, c=500.0)
        new._cpp = None
        results_old, results_new = _run_iters(old, new)
        for i, (o, n) in enumerate(zip(results_old, results_new)):
            np.testing.assert_allclose(o, n, atol=1e-12, err_msg=f"iter {i}")


class TestKFEquivalence:
    """KF: old vs new should produce identical outputs."""

    def test_default_params(self):
        Q = 1e-3 * np.eye(N_PARAMS)
        R = 1e-4 * np.eye(N_ROWS)
        old = old_mod.KF(N_PARAMS, process_noise=Q, measurement_noise=R)
        new = KF(N_PARAMS, process_noise=Q, measurement_noise=R)
        new._cpp = None
        results_old, results_new = _run_iters(old, new)
        for i, (o, n) in enumerate(zip(results_old, results_new)):
            np.testing.assert_allclose(o, n, atol=1e-12, err_msg=f"iter {i}")


class TestRKEquivalence:
    """RK: old vs new should produce identical outputs."""

    def test_basic(self):
        old = old_mod.RK(N_PARAMS)
        new = RK(N_PARAMS)
        new._cpp = None  # pure Python
        results_old, results_new = _run_iters(old, new)
        for i, (o, n) in enumerate(zip(results_old, results_new)):
            np.testing.assert_allclose(o, n, atol=1e-12, err_msg=f"iter {i}")


class TestTARKEquivalence:
    """TARK: old vs new should produce identical outputs."""

    def test_basic(self):
        old = old_mod.TARK(N_PARAMS)
        new = TARK(N_PARAMS)
        new._cpp = None
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b, burnin=5)
        np.random.seed(SEED)
        new_result = new.iterate(A, b, burnin=5)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestRKColScaledEquivalence:
    """RK_ColScaled: old vs new."""

    def test_basic(self):
        old = old_mod.RK_ColScaled(N_PARAMS)
        new = RK_ColScaled(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestRKEquiEquivalence:
    """RK_Equi: old vs new."""

    def test_basic(self):
        old = old_mod.RK_Equi(N_PARAMS)
        new = RK_Equi(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestRKTikhEquivalence:
    """RK_Tikh: old vs new."""

    def test_basic(self):
        old = old_mod.RK_Tikh(N_PARAMS)
        new = RK_Tikh(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestRKEquiTikhEquivalence:
    """RK_EquiTikh: old vs new."""

    def test_basic(self):
        old = old_mod.RK_EquiTikh(N_PARAMS)
        new = RK_EquiTikh(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestGRKEquivalence:
    """GRK: old vs new."""

    def test_basic(self):
        old = old_mod.GRK(N_PARAMS)
        new = GRK(N_PARAMS)
        new._cpp = None
        A, b, _ = _make_problem()
        rng_old = np.random.default_rng(SEED)
        rng_new = np.random.default_rng(SEED)
        old_result = old.iterate(A, b, rng=rng_old)
        new_result = new.iterate(A, b, rng=rng_new)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestTAGKEquivalence:
    """TAGK: old vs new."""

    def test_basic(self):
        old = old_mod.GRK_TailAvg(N_PARAMS)
        new = TAGK(N_PARAMS)
        new._cpp = None
        A, b, _ = _make_problem()
        rng_old = np.random.default_rng(SEED)
        rng_new = np.random.default_rng(SEED)
        old_result = old.iterate(A, b, rng=rng_old)
        new_result = new.iterate(A, b, rng=rng_new)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestGRKTikhEquivalence:
    """GRK_Tikh: old vs new."""

    def test_basic(self):
        old = old_mod.GRK_Tikh(N_PARAMS)
        new = GRK_Tikh(N_PARAMS)
        A, b, _ = _make_problem()
        rng_old = np.random.default_rng(SEED)
        rng_new = np.random.default_rng(SEED)
        old_result = old.iterate(A, b, rng=rng_old)
        new_result = new.iterate(A, b, rng=rng_new)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestTAGKTikhEquivalence:
    """TAGK_Tikh: old vs new."""

    def test_basic(self):
        old = old_mod.GRK_TailAvg_Tikh(N_PARAMS)
        new = TAGK_Tikh(N_PARAMS)
        A, b, _ = _make_problem()
        rng_old = np.random.default_rng(SEED)
        rng_new = np.random.default_rng(SEED)
        old_result = old.iterate(A, b, rng=rng_old)
        new_result = new.iterate(A, b, rng=rng_new)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestIGRKEquivalence:
    """IGRK: old vs new."""

    def test_basic(self):
        old = old_mod.IGRK(N_PARAMS)
        new = IGRK(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestMGRKEquivalence:
    """MGRK: old vs new."""

    def test_basic(self):
        old = old_mod.MGRK(N_PARAMS)
        new = MGRK(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestREKEquivalence:
    """REK: old vs new."""

    def test_basic(self):
        old = old_mod.REK(N_PARAMS)
        new = REK(N_PARAMS)
        A, b, _ = _make_problem()
        np.random.seed(SEED)
        old_result = old.iterate(A, b)
        np.random.seed(SEED)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)


class TestFDBKEquivalence:
    """FDBK: old vs new."""

    def test_basic(self):
        old = old_mod.FDBK(N_PARAMS)
        new = FDBK(N_PARAMS)
        A, b, _ = _make_problem()
        old_result = old.iterate(A, b)
        new_result = new.iterate(A, b)
        np.testing.assert_allclose(old_result.ravel(), new_result.ravel(), atol=1e-12)
