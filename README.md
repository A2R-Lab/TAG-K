# TAG-K: Tail-Averaged Greedy Kaczmarz

A Python library implementing the Tail-Averaged Greedy Kaczmarz Algorithm for online parameter estimation, optimized for quadrotor inertial parameter estimation.

## Installation

### From Source
```bash
git clone https://github.com/A2R-Lab/TAG-K
cd TAG-K
pip install -e .
```

## Quick Start

```python
from deka import OnlineParamEst, visualize_trajectory_with_est

# Create parameter estimation instance
estimator = OnlineParamEst()

# Run simulation
x_history, u_history, theta_history, theta_hat_history = (
    estimator.simulate_quadrotor_hover_with_DEKA(
        NSIM=300 # 300 simulation steps
    )
)

# Visualize results
visualize_trajectory_with_est(
    x_history,  
    u_history,
    theta_history, 
    theta_hat_history,
    "Parameter Estimation Results"
)
```

## Features

- **Online Parameter Estimation**: Real-time parameter estimation for dynamic systems with support for:
  - Sudden parameter changes (e.g., payload pickup)
  - Gradual parameter drift
  - Measurement noise handling
  
- **Multiple Estimation Methods**: 
  - Deterministic Kaczmarz Algorithm (DeKA)
  - Recursive Least Squares (RLS) 
  - Extended Kalman Filter (EKF)

- **Quadrotor Dynamics**: Built-in support for quadrotor parameter estimation with:
  - Full 6-DOF dynamics
  - Real-time control
  - Hover and trajectory tracking

- **Visualization Tools**: Comprehensive plotting utilities

## Performance

Based on empirical testing:

- **Convergence Speed**: DeKA achieves ~7x faster convergence compared to base Kaczmarz methods
- **Estimation Accuracy**: DeKA has 45% lower estimation error
than start-of-the-art variants of RLS and KF 
- **Parameter Tracking**: Superior tracking of sudden parameter changes (see examples)

## API Documentation

### DeKA Parameters

```python
DEKA(
    num_params: int,
    damping: float = 0.1,            # Update step damping (recommended: 0.1-1.0)
    regularization: float = 1e-6,    # Numerical stability term
    smoothing_factor: float = 0.9,   # Exponential smoothing (0: none, 1: max)
    tol: float = 1e-3                # Convergence tolerance
)
```


### Visualization

The package includes three main visualization functions:
- `visualize_trajectory()`: Plot position, inertia parameters, and controls
- `visualize_trajectory_with_est()`: Compare true vs estimated parameters
- `visualize_all_parameter()`: Detailed parameter tracking visualization

## Command Line Interface

```bash
python -m deka.param_est --robot quadrotor --task hover --method DEKA
```

Options:
- `--robot`: quadrotor
- `--task`: hover, tracking
- `--method`: DEKA, RLS, EKF
- `--update_type`: immediate_update, late_update, never_update

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:
```bibtex
@article{tagk2025,
    title={TAG-K: Tail-Averaged Greedy Kaczmarz for Computationally Efficient and Performant Online Inertial Parameter Estimation},
    author={Shuo Sha, Anupam Bhakta, Zhenyuan Jiang, Kevin Qiu, Ishaan Mahajan, Gabriel Bravo, Brian Plancher},
    journal={arXiv preprint},
    year={2025}
}
```

## Contact

For questions and support:
- GitHub Issues: [Create an issue](https://github.com/A2R-Lab/TAG-K/issues)
