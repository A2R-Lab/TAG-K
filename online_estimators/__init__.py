"""
TAG-K: Tail-Averaged Greedy Kaczmarz for online parameter estimation.

A library implementing Kaczmarz-family iterative solvers and classical
estimation methods (RLS, KF) for real-time inertial parameter estimation
in robotic systems, with built-in quadrotor and double-pendulum dynamics.
"""

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

__version__ = "0.1.0"

__all__ = [
    # Core estimators
    "RLS",
    "KF",
    "RK",
    "TARK",
    "GRK",
    "TAGK",
    "TAGK_Tikh",
    "GRK_Tikh",
    "IGRK",
    "MGRK",
    "REK",
    "FDBK",
    "RK_ColScaled",
    "RK_Equi",
    "RK_EquiTikh",
    "RK_Tikh",
]
