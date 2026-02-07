"""
Optional C++ backend for estimators.

If the ``online_estimation`` C++ extension is installed, estimator classes
will delegate to compiled implementations for better performance.  The
extension is built from the ``cpp/`` directory via ``pip install ./cpp``.

When the extension is not available, all estimators fall back to their
pure-Python implementations transparently.
"""

from __future__ import annotations

from typing import Any

_CppGRK: Any
_CppKF: Any
_CppRK: Any
_CppRLS: Any
_CppTAGK: Any
_CppTARK: Any

try:
    from online_estimation._core import GRK as _CppGRK  # type: ignore[no-redef]
    from online_estimation._core import KF as _CppKF  # type: ignore[no-redef]
    from online_estimation._core import RK as _CppRK  # type: ignore[no-redef]
    from online_estimation._core import RLS as _CppRLS  # type: ignore[no-redef]
    from online_estimation._core import TAGK as _CppTAGK  # type: ignore[no-redef]
    from online_estimation._core import TARK as _CppTARK  # type: ignore[no-redef]

    HAS_CPP: bool = True
except ImportError:
    HAS_CPP = False

__all__ = [
    "HAS_CPP",
    "_CppGRK",
    "_CppKF",
    "_CppRK",
    "_CppRLS",
    "_CppTAGK",
    "_CppTARK",
]
