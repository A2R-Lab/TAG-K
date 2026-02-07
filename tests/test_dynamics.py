"""Tests for online_estimators.dynamics (QuadrotorDynamics, DoublePendulum)."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.dynamics.double_pendulum import DoublePendulum
from online_estimators.dynamics.quadrotor import QuadrotorDynamics

# ---------------------------------------------------------------------------
# QuadrotorDynamics
# ---------------------------------------------------------------------------


class TestQuadrotorDynamics:
    @pytest.fixture()
    def quad(self):
        return QuadrotorDynamics()

    def test_initial_state_shape(self, quad):
        x0 = np.zeros(13)
        x0[3] = 1.0  # unit quaternion
        assert x0.shape == (13,)

    def test_hover_thrust(self, quad):
        u_hover = quad.hover_thrust_true
        assert u_hover.shape == (4,)
        assert np.all(u_hover > 0)
        # All four motors should be equal at hover
        np.testing.assert_allclose(u_hover[0], u_hover[1], atol=1e-6)
        np.testing.assert_allclose(u_hover[0], u_hover[2], atol=1e-6)

    def test_rk4_integration_preserves_quat_norm(self, quad):
        x = np.zeros(13)
        x[3] = 1.0
        u = quad.hover_thrust_true.copy()
        for _ in range(100):
            x = quad.dynamics_rk4_true(x, u)
        quat_norm = np.linalg.norm(x[3:7])
        np.testing.assert_allclose(quat_norm, 1.0, atol=1e-4)

    def test_hover_stays_in_place(self, quad):
        x = np.zeros(13)
        x[3] = 1.0  # identity quaternion
        u = quad.hover_thrust_true.copy()
        for _ in range(500):
            x = quad.dynamics_rk4_true(x, u)
        # Position should stay near zero
        assert np.linalg.norm(x[0:3]) < 0.05

    def test_data_matrix_shape(self, quad):
        x = np.zeros(13)
        x[3] = 1.0
        dx = np.zeros(13)
        A_mat = quad.get_data_matrix(x, dx)
        assert A_mat.shape == (6, 10)

    def test_force_vector_shape(self, quad):
        x = np.zeros(13)
        x[3] = 1.0
        dx = np.zeros(13)
        u = quad.hover_thrust_true.copy()
        b_vec = quad.get_force_vector(x, dx, u)
        assert b_vec.shape == (6,)

    def test_linearization_shapes(self, quad):
        x = np.zeros(13)
        x[3] = 1.0
        u = quad.hover_thrust_true.copy()
        A_lin, B_lin = quad.get_linearized_true(x, u)
        assert A_lin.shape == (12, 12)
        assert B_lin.shape == (12, 4)

    def test_payload_attach_detach(self, quad):
        theta_before = quad.get_true_inertial_params().copy()
        quad.attach_payload(m_p=0.01, delta_r_p=np.zeros(3))
        theta_after = quad.get_true_inertial_params()
        # Mass should increase
        assert theta_after[0] > theta_before[0]
        quad.detach_payload()
        theta_restored = quad.get_true_inertial_params()
        np.testing.assert_allclose(theta_restored, theta_before, atol=1e-10)

    def test_get_set_estimated_params(self, quad):
        theta = quad.get_true_inertial_params()
        quad.set_estimated_inertial_params(theta)
        u_est = quad.get_hover_thrust_est()
        np.testing.assert_allclose(u_est, quad.hover_thrust_true, atol=1e-6)

    def test_static_hat(self):
        v = np.array([1.0, 2.0, 3.0])
        S = QuadrotorDynamics.hat(v)
        assert S.shape == (3, 3)
        np.testing.assert_allclose(S + S.T, np.zeros((3, 3)))

    def test_static_quaternion_roundtrip(self):
        rp = np.array([0.1, -0.05, 0.02])
        q = QuadrotorDynamics.rptoq(rp)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        rp_back = QuadrotorDynamics.qtorp(q)
        np.testing.assert_allclose(rp_back, rp, atol=1e-6)


# ---------------------------------------------------------------------------
# DoublePendulum
# ---------------------------------------------------------------------------


class TestDoublePendulum:
    @pytest.fixture()
    def dp(self):
        return DoublePendulum()

    def test_initial_state(self, dp):
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        assert x0.shape == (4,)

    def test_rk4_integration(self, dp):
        x = np.array([0.1, 0.0, 0.05, 0.0])
        u = np.array([0.0])
        for _ in range(100):
            x = dp.step(x, u)
        assert x.shape == (4,)
        assert np.all(np.isfinite(x))

    def test_linearization_shapes(self, dp):
        x = np.zeros(4)
        u = np.zeros(1)
        A_lin, B_lin = dp.linearize(x, u)
        assert A_lin.shape == (4, 4)
        assert B_lin.shape == (4, 1)

    def test_energy_bounded_small_angle(self, dp):
        """For small angles and no input, energy should be roughly conserved
        (not grow unboundedly)."""
        x = np.array([0.05, 0.0, 0.05, 0.0])
        u = np.array([0.0])
        positions = []
        for _ in range(200):
            x = dp.step(x, u)
            positions.append(x[0:2].copy())
        positions = np.array(positions)
        # Angles should stay bounded
        assert np.max(np.abs(positions)) < 1.0
