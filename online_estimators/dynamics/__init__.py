"""
Dynamical system models for parameter estimation benchmarks.

Provides:
    :class:`~online_estimators.dynamics.quadrotor.QuadrotorDynamics`
    :class:`~online_estimators.dynamics.double_pendulum.DoublePendulum`
"""

from online_estimators.dynamics.double_pendulum import DoublePendulum
from online_estimators.dynamics.quadrotor import QuadrotorDynamics

__all__ = ["QuadrotorDynamics", "DoublePendulum"]
