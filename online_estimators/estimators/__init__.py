"""
Online parameter estimation algorithms.

This subpackage provides a collection of iterative solvers for the linear
system ``A theta = b``, designed for streaming/online use where new measurement
rows arrive over time.

Algorithms
----------
Classical:
    :class:`RLS`  --  Recursive Least Squares with forgetting factor
    :class:`KF`   --  Kalman Filter (flipped model, parameters as state)

Kaczmarz family:
    :class:`RK`        --  Randomized Kaczmarz
    :class:`TARK`      --  Tail-Averaged Randomized Kaczmarz
    :class:`GRK`       --  Greedy Randomized Kaczmarz
    :class:`TAGK`      --  TAG-K: GRK with Polyak tail-averaging
    :class:`TAGK_Tikh` --  TAG-K + Tikhonov regularisation
    :class:`GRK_Tikh`  --  GRK + Tikhonov augmentation
    :class:`IGRK`      --  Improved Greedy Randomized Kaczmarz
    :class:`MGRK`      --  Momentum Greedy Randomized Kaczmarz
    :class:`REK`       --  Randomized Extended Kaczmarz

Preconditioned Kaczmarz variants:
    :class:`RK_ColScaled`  --  RK with column scaling (right preconditioning)
    :class:`RK_Equi`       --  RK with Ruiz equilibration
    :class:`RK_Tikh`       --  RK with Tikhonov augmentation
    :class:`RK_EquiTikh`   --  RK with Ruiz + Tikhonov

Block Kaczmarz:
    :class:`FDBK`  --  Fast Deterministic Block Kaczmarz
"""

from online_estimators.estimators.block import FDBK, IGRK, MGRK, REK
from online_estimators.estimators.grk import (
    GRK,
    TAGK,
    GRK_Tikh,
    TAGK_Tikh,
)
from online_estimators.estimators.kf import KF
from online_estimators.estimators.rk import (
    RK,
    TARK,
    RK_ColScaled,
    RK_Equi,
    RK_EquiTikh,
    RK_Tikh,
)
from online_estimators.estimators.rls import RLS

__all__ = [
    "RLS",
    "KF",
    "RK",
    "TARK",
    "RK_ColScaled",
    "RK_Equi",
    "RK_Tikh",
    "RK_EquiTikh",
    "GRK",
    "TAGK",
    "TAGK_Tikh",
    "GRK_Tikh",
    "IGRK",
    "MGRK",
    "REK",
    "FDBK",
]
