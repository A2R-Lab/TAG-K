"""Tests for online_estimators.simulation.trial (make_estimator, gate_param_update)."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.simulation.trial import gate_param_update, make_estimator


class TestMakeEstimator:
    def test_gt_returns_none(self):
        theta0 = np.zeros(10)
        assert make_estimator("gt", theta0, window=1) is None

    def test_none_returns_none(self):
        theta0 = np.zeros(10)
        assert make_estimator("none", theta0, window=1) is None

    def test_known_names(self):
        theta0 = np.zeros(10)
        for name in ["rk", "tark", "grk", "tagk", "tagk_tikh", "rls_0.8"]:
            est = make_estimator(name, theta0, window=1)
            assert est is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            make_estimator("nonexistent_algo", np.zeros(10), window=1)


class TestGateParamUpdate:
    def test_accepts_valid(self):
        theta = np.array([0.04, 0.0, 0.0, 0.0, 2e-5, 2e-5, 1e-5, 0.0, 0.0, 0.0])
        A = np.eye(10)
        b = A @ theta
        assert gate_param_update(theta, A, b, rel_resid_max=1.0) is True

    def test_rejects_mass_too_low(self):
        theta = np.array([0.001, 0.0, 0.0, 0.0, 2e-5, 2e-5, 1e-5, 0.0, 0.0, 0.0])
        A = np.eye(10)
        b = A @ theta
        assert gate_param_update(theta, A, b) is False

    def test_rejects_mass_too_high(self):
        theta = np.array([0.5, 0.0, 0.0, 0.0, 2e-5, 2e-5, 1e-5, 0.0, 0.0, 0.0])
        A = np.eye(10)
        b = A @ theta
        assert gate_param_update(theta, A, b) is False

    def test_rejects_com_out_of_bounds(self):
        theta = np.array([0.04, 0.1, 0.0, 0.0, 2e-5, 2e-5, 1e-5, 0.0, 0.0, 0.0])
        A = np.eye(10)
        b = A @ theta
        assert gate_param_update(theta, A, b) is False
