# Python bindings for Online Estimation Methods

## Requirements
- Cmake
- numpy
- pybind11
- Eigen

## Build
To build and install, you can use:
```bash
pip install .
```

## Usage
The estimators can be imported like python classes:
```python
from online_estimation._core import RLS, RLS_Robust, KF, KF_Robust, RK, GRK, TARK, TAGK
```

Each estimator exposes an iterate method. For example, with the RLS estimator:
```python
n = 5

A = ...
b = ...
x0 = np.zeros((n, 1))

kwargs = {
    "lambda": 0.99,
    "p_coeff": 1000,
    "x0": None
}
estimator = RLS(n, **kwargs)

x_estimate = estimator.iterate(A, b, x0=x0)
```

You can also modify the internal state without passing it to the iterate function:
```python
estimator.state = x_new  # performs check on if dimensions match current state
```
