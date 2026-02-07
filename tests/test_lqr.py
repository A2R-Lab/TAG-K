"""Tests for online_estimators.control.lqr."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.control.lqr import LQRController, dlqr


class TestDlqr:
    def test_basic(self):
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.eye(2)
        R = np.eye(1)
        K, S = dlqr(A, B, Q, R)
        assert K.shape == (1, 2)
        assert S.shape == (2, 2)
        # S should be symmetric PD
        np.testing.assert_allclose(S, S.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(S) > 0)

    def test_closed_loop_stable(self):
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = 0.1 * np.eye(1)
        K, _ = dlqr(A, B, Q, R)
        A_cl = A - B @ K
        eigs = np.abs(np.linalg.eigvals(A_cl))
        assert np.all(eigs < 1.0), f"Closed-loop eigenvalues: {eigs}"


class TestLQRController:
    @pytest.fixture()
    def lqr(self):
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.eye(2)
        R = np.eye(1)
        return LQRController(A, B, Q, R)

    def test_control_shape(self, lqr):
        x = np.array([0.5, -0.3])
        u = lqr.control(x)
        # squeeze() yields scalar for single-input systems
        assert np.isscalar(u) or u.shape in ((), (1,))

    def test_stabilizes(self, lqr):
        x = np.array([1.0, 0.5])
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        for _ in range(100):
            u = lqr.control(x)
            x = A @ x + (B * u).ravel()
        assert np.linalg.norm(x) < 0.01

    def test_update_dynamics(self, lqr):
        A2 = np.array([[1.0, 0.5], [0.0, 1.0]])
        B2 = np.array([[0.0], [0.5]])
        lqr.update_linearized_dynamics(A2, B2)
        x = np.array([1.0, 0.5])
        u = lqr.control(x)
        assert np.isscalar(u) or u.shape in ((), (1,))
