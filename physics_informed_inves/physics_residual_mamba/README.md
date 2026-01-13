# Physics-Residual Power Mamba

A hybrid architecture combining differentiable physics layers with Mamba-based residual learning for solar power forecasting.

## Overview

This package implements a physics-informed deep learning model for solar power forecasting that combines:

1. **Differentiable Physics Layer**: Learnable PV physics model with geometric gating for Angle of Incidence (AOI) correction
2. **Mamba Residual Learner**: State-space model that learns the error (residual) of the physics model
3. **Hybrid Combination**: Final prediction = ReLU(Physics + Residual)

The key insight is that the neural network learns the **residual** (error) of the physics model, not the raw power. This constrains the learning and provides physics-informed predictions.

## Installation

```bash
cd App_v02/physics_informed_inves/PINNS
pip install -e .
```

Or install dependencies manually:

```bash
pip install torch mamba-ssm pvlib pandas numpy scikit-learn matplotlib seaborn scipy
```

## Quick Start

```python
import pandas as pd
from physics_residual_mamba import (
    PhysicsResidualMamba,
    PhysicsResidualMambaConfigs,
    run_physics_residual_cv,
    plot_model_performance
)

# Load your data
df = pd.read_csv('your_solar_data.csv')

# Configure model
configs = PhysicsResidualMambaConfigs(n_splits=4)

# Train with cross-validation
results, history = run_physics_residual_cv(df, configs)

# Visualize results
plot_model_performance(results[-1])
```

## Package Structure

```
physics_residual_mamba/
├── __init__.py              # Main package exports
├── configs/                  # Configuration classes
│   ├── __init__.py
│   └── configs.py         # BaseMambaConfigs, PhysicsResidualMambaConfigs
├── data/                     # Data preparation
│   ├── __init__.py
│   ├── datasets.py         # MultiStepDataset, prepare_rolling_folds
│   └── preprocessing.py    # Feature engineering utilities
├── models/                    # Model architectures
│   ├── __init__.py
│   ├── components.py       # RevIN, moving_avg, series_decomp
│   ├── physics_layer.py    # DifferentiablePVLayer, FeatureScaler
│   ├── base_models.py     # MMPISSM_Model_01, VanillaLSTM, VanillaMamba
│   └── hybrid_model.py    # PhysicsResidualMamba
├── losses/                    # Loss functions
│   ├── __init__.py
│   └── physics_loss.py    # PhysicsPVLoss (triple-constraint)
├── evaluation/                 # Evaluation functions
│   ├── __init__.py
│   ├── evaluators.py     # evaluate_fold_physics, evaluate_fold_base
│   └── metrics.py         # calculate_metrics, calculate_improvement, etc.
├── visualization/              # Plotting utilities
│   ├── __init__.py
│   └── plots.py          # plot_model_performance, plot_benchmark_summary, etc.
├── training/                  # Training functions
│   ├── __init__.py
│   └── trainers.py       # run_physics_residual_cv, run_base_mamba_cv
├── utils/                     # Helper functions
│   ├── __init__.py
│   ├── random.py         # set_random_seed, get_random_seed
│   ├── device.py         # get_device, get_device_info
│   └── logging.py        # setup_logging, ExperimentLogger
└── README.md                  # This file
```

## Key Features

### Physics-Informed Architecture
- **DifferentiablePVLayer**: Learnable PV physics with Faiman temperature model
- **Geometric Gating**: Learns to map (Time of Day, Season) → Tilt Factor
- **Learnable Parameters**: Efficiency (η), Heat transfer (U0, U1), Temperature coefficient (γ), STC Power (P_stc)

### Mamba-Based Residual Learning
- **MMPISSM_Model_01**: Full Mamba model with RevIN and series decomposition
- **VanillaMamba**: Simplified Mamba baseline (no RevIN, no decomposition)
- **VanillaLSTM**: Standard LSTM baseline for comparison

### Physics-Informed Loss
- **Triple-Constraint Loss**:
  1. MSE Loss (Data Fit)
  2. Night-time Penalty (Forces P → 0 when irradiance indicates night)
  3. Monotonicity Regularization (Power should rise with sunlight)

### Evaluation
- **Smart Persistence Baseline**: P_t = P_{t-24h} × (GHI_clr,t / GHI_clr,t-24h)
- **NWP Forecast Baseline**: Physics-based power from NWP weather
- **Comprehensive Metrics**: RMSE, MAE, MAPE, R², Bias, Skill Score
- **Error Analysis**: By time of day, irradiance level, weather conditions

### Visualization
- **Publication-Quality Plots**: Using seaborn-v0_8-whitegrid style
- **Multi-Model Comparison**: Time series, scatter plots, error distributions
- **Benchmark Summary**: Bar plots with improvement statistics

## Model Configuration

### PhysicsResidualMambaConfigs

```python
configs = PhysicsResidualMambaConfigs(n_splits=4)

# Sequence Configuration
configs.seq_len = 672      # ~7 days at 15-min intervals
configs.pred_len = 96       # 24 hours at 15-min intervals

# Features
configs.PAST_INPUT_COLS = [...]  # 41 features including power, weather, time encodings
configs.FUTURE_INPUT_COLS = [...]  # 4 features: irradiance, temperature, wind speed

# Model Architecture
configs.kernel_size = 25      # Moving average kernel
configs.n_embed = 128        # Mamba embedding dimension
configs.d_state = 64          # Mamba state dimension
configs.dconv = 2              # Mamba convolution kernel
configs.e_fact = 2             # Mamba expansion factor
configs.dropout = 0.2         # Dropout rate

# Training
configs.n_splits = 4         # Number of CV folds
configs.batch_size = 64      # Batch size
configs.epochs = 30          # Training epochs

# Physics Configuration
configs.T_ref = 25.0         # Reference temperature (°C)
configs.G_night_thr = 10.0   # Night irradiance threshold (W/m²)
configs.lambda_data = 1.0     # Data fit loss weight
configs.lambda_night = 0.2    # Night penalty weight
configs.lambda_mono = 0.1     # Monotonicity weight
```

## Training Strategy

### Two-Stage Training
1. **Warmup (Epochs 0-4)**: Freeze Mamba, optimize only physics layer with higher LR (1e-3)
2. **Joint Training (Epochs 5+)**: Unfreeze Mamba, fine-tune entire model with lower LR (5e-4)

### Night Filtering
Training samples with maximum irradiance < 50 W/m² are filtered out to prevent the model from learning unrealistic night-time patterns.

### Cross-Validation
TimeSeriesSplit with rolling folds ensures proper temporal ordering and prevents data leakage.

## Baseline Models

### Smart Persistence
Formula: `P_t = P_{t-24h} × (GHI_clr,t / GHI_clr,t-24h)`

Uses yesterday's power adjusted by the ratio of today's clear sky irradiance to yesterday's clear sky irradiance.

### NWP Forecast
Physics-based power simulation using pvlib with NWP weather forecasts (irradiance, temperature, wind speed).

### Base Models
- **Base Mamba**: Full MMPISSM_Model_01 without physics layer
- **Vanilla Mamba**: Simplified Mamba (no RevIN, no decomposition)
- **Vanilla LSTM**: Standard LSTM with same input construction

## Citation

If you use this code in your research, please cite:

```bibtex
@article{physics_residual_mamba_2025,
  title={Physics-Residual Power Mamba: A Hybrid Architecture for Solar Power Forecasting},
  author={Your Name},
  journal={Applied Energy},
  year={2025},
  note={Inspired by Applied Energy 2025 on Physics-Informed Solar Forecasting}
}
```

## License

This code is provided for research purposes. Please ensure you have appropriate licenses for any dependencies (mamba-ssm, pvlib, etc.).

## Contributing

Contributions are welcome! Please ensure:
- All functions have comprehensive docstrings
- Type hints are used for function signatures
- Code follows PEP 8 style guidelines
- New features include appropriate tests

## References

1. Applied Energy 2025 - Physics-Informed Solar Forecasting
2. Faiman Temperature Model for PV Cell Temperature
3. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
4. pvlib: Python library for photovoltaic system modeling

## Contact

For questions or issues, please contact: contact@example.com
