# TAG-K: Tail-Averaged Greedy Kaczmarz for Online Inertial Parameter Estimation

A Python library for **online inertial parameter estimation** in robotics,
featuring the Tail-Averaged Greedy Randomized Kaczmarz (TAG-K) algorithm and
a comprehensive suite of Kaczmarz-family solvers. Includes a full 13-state
quaternion quadrotor simulator (modeled after the Crazyflie 2.1), an LQR
controller, and tools for running closed-loop estimation experiments with
payload-change events and sensor noise.

TAG-K combines greedy residual-driven row selection with tail averaging to
achieve fast, stable parameter adaptation while retaining the low per-iteration
complexity inherent to the Kaczmarz framework. As compared to RLS and KF
baselines, TAG-K achieves:

- **1.5x-1.9x faster** solve times on laptop-class CPUs
- **4.8x-20.7x faster** solve times on embedded microcontrollers (Teensy 4.1)
- **25% lower** estimation error
- **~2x better** end-to-end tracking performance

For details, see our [ICRA 2026 paper](https://arxiv.org/pdf/2510.04839).

---

## Features

- **Multiple estimation algorithms** -- from classic RLS and Kalman Filter to
  Kaczmarz variants (RK, GRK, TAG-K, REK, and more)
- **Quadrotor dynamics** -- 13-state (position, quaternion, velocity, angular
  velocity) model with RK4 integration and autograd-based linearisation
- **Double-pendulum dynamics** -- 4-state underactuated system
- **LQR control** -- discrete-time infinite-horizon LQR with online re-linearisation
- **Noise models** -- AWGN, Ornstein-Uhlenbeck, and random-walk process noise
- **Trajectory generators** -- hover, figure-8, circle, Lissajous, ellipse,
  helix, and spiral reference paths
- **Simulation harness** -- run full closed-loop trials with payload
  attach/detach, gated parameter updates, and data logging
- **Visualization** -- 3-D trajectory plots, residual time-series, error CDFs
- **Optional C++ backends** -- pybind11 wrappers for core estimators (RLS, KF,
  RK, GRK, TARK, TAG-K) for higher throughput

---

## Installation

TAG-K requires **Python >= 3.10**.

### With uv (recommended)

Install it by following Astral’s official instructions: https://docs.astral.sh/uv/getting-started/installation/

```bash
git clone https://github.com/A2R-Lab/TAG-K.git
cd TAG-K
uv sync               # creates venv, installs deps
uv sync --extra dev   # + pytest, ruff, pytest-cov
```

### With pip

```bash
git clone https://github.com/A2R-Lab/TAG-K.git
cd TAG-K
pip install -e ".[dev]"
```

### C++ backends (optional)

Building the C++ extension requires a C++17 compiler, Eigen3, and pybind11:

```bash
pip install ./cpp
```

See [cpp/README.md](cpp/README.md) for details.

---

## Quick Start

### Using an estimator directly

Every estimator implements a single `iterate(A, b) -> x` interface:

```python
import numpy as np
from online_estimators.estimators import TAGK  # the TAG-K algorithm

# Overdetermined system  A x = b
rng = np.random.default_rng(0)
n = 5
A = rng.standard_normal((50, n))
x_true = rng.standard_normal(n)
b = A @ x_true

est = TAGK(n)
for _ in range(500):
    x_hat = est.iterate(A, b)

print("Error:", np.linalg.norm(x_hat - x_true))
```

### Running a quadrotor trial

```python
from online_estimators.simulation.trial import run_single_trial

result = run_single_trial(
    estimator_name="tagk",         # TAG-K
    base_seed=0,
    seed_offset=0,
    est_freq=20,                   # 50 Hz controller / 20 = 2.5 Hz estimator
    window=5,
    ref_type="figure8",
    T_sim=20.0,
    noise="low",
)

print("Failed:", result["failed"])
```

---

## Package Structure

```
online_estimators/
|-- __init__.py               # re-exports all estimators
|-- estimators/
|   |-- _backend.py           # optional C++ backend loader
|   |-- base.py               # BaseEstimator ABC
|   |-- rls.py                # Recursive Least Squares
|   |-- kf.py                 # Kalman Filter
|   |-- rk.py                 # RK, TARK, RK_ColScaled, RK_Equi, RK_Tikh, RK_EquiTikh
|   |-- grk.py                # GRK, TAGK, GRK_Tikh, TAGK_Tikh
|   |-- block.py              # FDBK, IGRK, MGRK, REK
|-- dynamics/
|   |-- quadrotor.py          # 13-state quaternion quadrotor (Crazyflie 2.1)
|   |-- double_pendulum.py    # 4-state double pendulum
|-- control/
|   |-- lqr.py                # Discrete-time LQR
|-- noise/
|   |-- models.py             # AWGN, OU, RandomWalk, apply_noise()
|-- trajectories/
|   |-- generators.py         # 7 reference trajectories
|-- simulation/
|   |-- trial.py              # run_single_trial(), make_estimator()
|   |-- utils.py              # residual helpers, closest_spd()
|-- visualization/
|   |-- plotting.py           # 3-D traj, residuals, CDFs
|-- data/
    |-- processing.py         # merge_npz() for experiment results
```

---

## Estimators

All estimators inherit from `BaseEstimator` and expose:

```python
x_hat = estimator.iterate(A, b)
```

where `A` is an `(m, n)` measurement matrix and `b` is an `(m,)` observation
vector.

| Class | Description |
|---|---|
| `RLS` | Recursive Least Squares with forgetting factor |
| `KF` | Kalman Filter (parameters-as-state formulation) |
| `RK` | Randomized Kaczmarz |
| `TARK` | Tail-Averaged Randomized Kaczmarz |
| `GRK` | Greedy Randomized Kaczmarz (max-residual row) |
| `TAGK` | GRK + tail averaging (**TAG-K**) |
| `GRK_Tikh` | GRK + Tikhonov regularisation |
| `TAGK_Tikh` | GRK + tail averaging + Tikhonov |
| `RK_ColScaled` | RK with column-norm scaling |
| `RK_Equi` | RK with Ruiz equilibration |
| `RK_Tikh` | RK + Tikhonov |
| `RK_EquiTikh` | RK + Ruiz equilibration + Tikhonov |
| `IGRK` | Block Greedy Kaczmarz (index-set greedy) |
| `MGRK` | Momentum-accelerated GRK |
| `REK` | Randomized Extended Kaczmarz (inconsistent systems) |
| `FDBK` | Fully-Determined Block Kaczmarz |

---

## Testing and Code Quality

```bash
# run the full test suite
pytest

# with coverage
pytest --cov=online_estimators --cov-report=term-missing

# single module
pytest tests/test_estimators.py -v

# lint and format
ruff check online_estimators/ tests/
ruff format online_estimators/ tests/

# type check
ty check online_estimators/
```

---

## Repository Layout

| Path | Purpose |
|---|---|
| `online_estimators/` | Installable Python package |
| `tests/` | pytest test suite |
| `cpp/` | C++ estimator implementations + pybind11 bindings |
| `python/` | Original research scripts (kept for reference) |
| `*.ipynb` | Jupyter notebooks for exploration and plotting |

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## Citation

If you use TAG-K in your research, please cite our ICRA 2026 paper:

```bibtex
@inproceedings{tagk2026,
  title     = {TAG-K: Tail-Averaged Greedy Kaczmarz for Computationally
               Efficient and Performant Online Inertial Parameter Estimation},
  author    = {Shuo Sha and Anupam Bhakta and Zhenyuan Jiang and Kevin Qiu and Ishaan Mahajan and Gabriel Bravo and Brian Plancher},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```
