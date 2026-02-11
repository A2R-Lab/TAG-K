online_estimators
=================

**TAG-K: Tail-Averaged Greedy Kaczmarz for Online Inertial Parameter Estimation**

``online_estimators`` is a Python package implementing Kaczmarz-family iterative
solvers and classical estimation methods (RLS, KF) for real-time inertial
parameter estimation in robotic systems, with built-in quadrotor and
double-pendulum dynamics.

.. note::

   Accepted at **ICRA 2026** — IEEE International Conference on Robotics and Automation.

Key Features
------------

- **16 online estimation algorithms** with a unified ``iterate(A, b) → x`` API
- **Quadrotor dynamics** — 13-state quaternion model calibrated to Crazyflie 2.1
- **Double-pendulum dynamics** — 4-state model for benchmarking
- **LQR controller** — discrete-time with automatic linearisation
- **Noise models** — AWGN, Ornstein–Uhlenbeck, random walk
- **7 reference trajectories** — hover, figure-8, circle, ellipse, helix, Lissajous, spiral
- **Optional C++ backends** via pybind11 (1.5×–1.9× faster on laptop CPUs)

Installation
------------

.. code-block:: bash

   git clone https://github.com/A2R-Lab/TAG-K.git
   cd TAG-K
   pip install -e .                         # pure-Python
   pip install -e ".[cpp]"                  # with C++ accelerated backends

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from online_estimators.estimators import TAGK

   rng = np.random.default_rng(0)
   n = 5
   A = rng.standard_normal((50, n))
   x_true = rng.standard_normal(n)
   b = A @ x_true

   est = TAGK(n)
   for _ in range(500):
       x_hat = est.iterate(A, b)

   print("Error:", np.linalg.norm(x_hat - x_true))

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/estimators
   api/dynamics
   api/control
   api/noise
   api/trajectories
   api/simulation
   api/visualization
   api/data


.. toctree::
   :maxdepth: 1
   :caption: Links

   Project Page <http://a2r-lab.org/TAG-K>
   GitHub <https://github.com/A2R-Lab/TAG-K>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
