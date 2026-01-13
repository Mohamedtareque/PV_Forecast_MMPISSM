# Physics-Residual Mamba Refactoring Plan
## Executive Summary

This document outlines the refactoring strategy for the Physics-Residual Power Mamba codebase, transforming a monolithic 2037-line implementation into a modular, maintainable architecture suitable for scientific research and production deployment.

---

## 1. Current State Analysis

### 1.1 Code Organization Issues

**Monolithic File: `physics_residual_mamba.py` (2037 lines)**

The current implementation violates several software engineering principles:

| Issue | Location | Impact |
|--------|-----------|----------|
| **Single Responsibility Violation** | Lines 32-2021 | File contains utilities, models, training, evaluation, visualization, and configuration |
| **Code Duplication** | Lines 985-1038 & 1309-1400 | `evaluate_fold_physics` and `evaluate_fold_physics_v2` serve similar purposes |
| **Hardcoded Values** | Lines 374, 406 | Station-specific module/inverter hardcoded (YL250P-29b, Solaron 500kW) |
| **Mixed Abstraction Levels** | Lines 54-204 & 210-558 | High-level metadata parsing mixed with low-level physics calculations |
| **Inconsistent Naming** | Throughout | `MMPISSM_Model_01`, `PhysicsResidualMamba` - unclear naming conventions |
| **Missing Type Hints** | Throughout | Most functions lack proper type annotations |
| **No Separation of Concerns** | Throughout | Training, evaluation, and visualization intertwined with model definitions |

### 1.2 Experimental Notebooks Analysis

**Notebook v01 (3038 lines):**
- Contains duplicate implementations of functions from monolithic file
- Has alternative `PhysicsPVLoss` implementation (lines 2519-2715)
- Tests both MSE and Physics loss variants
- No systematic ablation studies
- Lacks statistical significance testing

**Notebook v02 (1332 lines):**
- Imports from monolithic file (good practice)
- Uses `BaseMambaConfigs` class
- Calls `run_full_benchmark` from monolithic file
- Limited experimental documentation
- Missing hyperparameter sweep analysis

**Scientific Rigor Assessment:**

| Criterion | Status | Details |
|------------|--------|---------|
| **Reproducibility** | ⚠️ Partial | No random seed control, no version pinning |
| **Statistical Significance** | ❌ Missing | No confidence intervals, no hypothesis testing |
| **Ablation Studies** | ❌ Missing | No systematic component removal analysis |
| **Hyperparameter Tuning** | ⚠️ Limited | Only manual tuning, no grid search or Bayesian optimization |
| **Baseline Comparisons** | ✅ Good | Smart Persistence, NWP, LSTM, Vanilla Mamba included |
| **Cross-Validation** | ✅ Good | TimeSeriesSplit with 4 folds implemented |
| **Data Leakage Prevention** | ✅ Good | Causal rolling statistics, proper train/test splits |
| **Visualization Quality** | ⚠️ Basic | Plots exist but lack publication-ready formatting |

---

## 2. Proposed Modular Architecture

### 2.1 Directory Structure

```
App_v02/physics_informed_inves/PINNS/
├── physics_residual_mamba/          # New package root
│   ├── __init__.py                 # Package initialization
│   ├── configs/                     # Configuration management
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   ├── physics_config.py
│   │   └── hyperparams.py
│   ├── data/                         # Data preparation and loading
│   │   ├── __init__.py
│   │   ├── datasets.py                # MultiStepDataset, data loading
│   │   ├── preprocessing.py           # Feature engineering, rolling stats
│   │   └── station_metadata.py       # get_station_metadata
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   ├── base_models.py            # MMPISSM_Model_01, VanillaLSTM, VanillaMamba
│   │   ├── physics_layer.py         # DifferentiablePVLayer, FeatureScaler
│   │   ├── hybrid_model.py          # PhysicsResidualMamba
│   │   └── components.py            # RevIN, series_decomp, moving_avg
│   ├── losses/                        # Loss functions
│   │   ├── __init__.py
│   │   ├── physics_loss.py           # PhysicsPVLoss
│   │   └── base_losses.py           # MSE, MAE, SmoothL1
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── trainers.py              # train_one_epoch_physics, train_one_epoch_base
│   │   └── optimizers.py           # Optimizer factories
│   ├── evaluation/                    # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── evaluators.py           # evaluate_fold_physics, evaluate_fold_base
│   │   ├── metrics.py              # RMSE, MAE, MAPE, R² calculations
│   │   └── baselines.py            # Smart Persistence, NWP baseline calculations
│   ├── visualization/                 # Plotting and reporting
│   │   ├── __init__.py
│   │   ├── plots.py                # plot_model_performance, plot_benchmark_summary
│   │   └── reports.py             # print_performance_summary, table generation
│   ├── experiments/                   # Experiment orchestration
│   │   ├── __init__.py
│   │   ├── cross_validation.py      # run_physics_residual_cv, run_base_mamba_cv
│   │   ├── benchmarks.py           # run_full_benchmark, run_full_benchmark_extended
│   │   └── ablation.py            # Ablation study utilities
│   ├── utils/                        # General utilities
│   │   ├── __init__.py
│   │   ├── random.py               # Seed control, reproducibility
│   │   ├── device.py              # Device management
│   │   └── logging.py              # Logging utilities
│   └── physics/                      # Physics calculations
│       ├── __init__.py
│       ├── clearsky.py            # calculate_clearsky_indices
│       ├── nwp_power.py           # calculate_nwp_power
│       └── pvlib_wrapper.py       # PVLib interface layer
├── experiments/                      # Experiment notebooks
│   ├── 01_baseline_comparison.ipynb
│   ├── 02_physics_ablation.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   └── 04_final_evaluation.ipynb
├── physics_residual_mamba.py          # Legacy monolithic file (deprecated)
└── REFACTORING_PLAN.md              # This document
```

### 2.2 Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| **configs/** | Hyperparameter and experiment configuration | `PhysicsResidualMambaConfigs`, `BaseMambaConfigs` |
| **data/datasets.py** | PyTorch Dataset implementations | `MultiStepDataset` |
| **data/preprocessing.py** | Feature engineering | `prepare_features`, `add_past_rolling_stats` |
| **data/station_metadata.py** | Station metadata parsing | `get_station_metadata` |
| **models/physics_layer.py** | Differentiable physics components | `DifferentiablePVLayer`, `FeatureScaler`, `compute_scaler_stats` |
| **models/base_models.py** | Baseline neural architectures | `MMPISSM_Model_01`, `VanillaLSTM`, `VanillaMamba` |
| **models/hybrid_model.py** | Physics-residual hybrid | `PhysicsResidualMamba` |
| **models/components.py** | Reusable neural components | `RevIN`, `series_decomp`, `moving_avg` |
| **losses/physics_loss.py** | Physics-informed loss | `PhysicsPVLoss` |
| **training/trainers.py** | Training loops | `train_one_epoch_physics`, `train_one_epoch_base` |
| **evaluation/evaluators.py** | Model evaluation | `evaluate_fold_physics`, `evaluate_fold_base` |
| **evaluation/baselines.py** | Baseline calculations | Smart Persistence, NWP calculations |
| **visualization/plots.py** | Plotting functions | `plot_model_performance`, `plot_benchmark_summary` |
| **experiments/cross_validation.py** | CV orchestration | `run_physics_residual_cv`, `run_base_mamba_cv` |
| **physics/clearsky.py** | Clear sky calculations | `calculate_clearsky_indices` |
| **physics/nwp_power.py** | NWP-based power simulation | `calculate_nwp_power` |

---

## 3. Detailed Refactoring Tasks

### Phase 1: Foundation (Priority: HIGH)

#### Task 1.1: Create Package Structure
- [ ] Create `physics_residual_mamba/` directory
- [ ] Create all `__init__.py` files
- [ ] Set up package imports

#### Task 1.2: Extract Configuration Classes
**Source:** Lines 1946-2021, 1200-1277 (v02)

**Actions:**
- [ ] Move `PhysicsResidualMambaConfigs` to `configs/physics_config.py`
- [ ] Move `BaseMambaConfigs` to `configs/base_config.py`
- [ ] Create `hyperparams.py` for hyperparameter definitions
- [ ] Add validation methods to config classes
- [ ] Add `to_dict()` and `from_dict()` methods for serialization

#### Task 1.3: Extract Data Components
**Source:** Lines 737-924, 54-204, 210-558, 889-1158

**Actions:**
- [ ] Move `MultiStepDataset` to `data/datasets.py`
- [ ] Move `prepare_rolling_folds` to `data/datasets.py`
- [ ] Move `get_station_metadata` to `data/station_metadata.py`
- [ ] Move `calculate_clearsky_indices` to `physics/clearsky.py`
- [ ] Move `calculate_nwp_power` to `physics/nwp_power.py`
- [ ] Move `prepare_features` to `data/preprocessing.py`
- [ ] Move `add_past_rolling_stats` to `data/preprocessing.py`
- [ ] Add type hints to all functions
- [ ] Add docstring standardization (Google style)

#### Task 1.4: Extract Model Components
**Source:** Lines 246-335, 140-240, 350-503

**Actions:**
- [ ] Move `RevIN` to `models/components.py`
- [ ] Move `moving_avg` to `models/components.py`
- [ ] Move `series_decomp` to `models/components.py`
- [ ] Move `DifferentiablePVLayer` to `models/physics_layer.py`
- [ ] Move `FeatureScaler` to `models/physics_layer.py`
- [ ] Move `compute_scaler_stats` to `models/physics_layer.py`
- [ ] Move `MMPISSM_Model_01` to `models/base_models.py`
- [ ] Move `VanillaLSTM` to `models/base_models.py`
- [ ] Move `VanillaMamba` to `models/base_models.py`
- [ ] Move `PhysicsResidualMamba` to `models/hybrid_model.py`

### Phase 2: Training & Evaluation (Priority: HIGH)

#### Task 2.1: Extract Training Functions
**Source:** Lines 930-983, 1513-1892, 2716-2789

**Actions:**
- [ ] Move `train_one_epoch_physics` to `training/trainers.py`
- [ ] Move `train_one_epoch_base` to `training/trainers.py`
- [ ] Create optimizer factory in `training/optimizers.py`
- [ ] Add early stopping support
- [ ] Add learning rate scheduling options
- [ ] Add gradient clipping configuration

#### Task 2.2: Extract Evaluation Functions
**Source:** Lines 985-1038, 1309-1400, 1161-1288, 1897-1972, 2794-2889

**Actions:**
- [ ] Move `evaluate_fold_physics` to `evaluation/evaluators.py`
- [ ] Move `evaluate_fold_physics_v2` to `evaluation/evaluators.py`
- [ ] Move `evaluate_fold_base` to `evaluation/evaluators.py`
- [ ] Move `evaluate_fold_with_baselines` to `evaluation/evaluators.py`
- [ ] Consolidate duplicate evaluation functions
- [ ] Move baseline calculations to `evaluation/baselines.py`
- [ ] Create `metrics.py` with RMSE, MAE, MAPE, R²

#### Task 2.3: Extract Visualization Functions
**Source:** Lines 1204-1507, 1401-1475, 1885-1941

**Actions:**
- [ ] Move `plot_model_performance` to `visualization/plots.py`
- [ ] Move `plot_benchmark_summary` to `visualization/plots.py`
- [ ] Move `plot_loss_curve` to `visualization/plots.py`
- [ ] Move `print_performance_summary` to `visualization/reports.py`
- [ ] Add publication-quality styling (seaborn-v0_8-whitegrid)
- [ ] Add save-to-file functionality
- [ ] Add multi-panel subplot support

### Phase 3: Experiment Orchestration (Priority: MEDIUM)

#### Task 3.1: Extract CV Functions
**Source:** Lines 1053-1199, 2005-2164, 2889-3030

**Actions:**
- [ ] Move `run_physics_residual_cv` to `experiments/cross_validation.py`
- [ ] Move `run_base_mamba_cv` to `experiments/cross_validation.py`
- [ ] Move `run_full_benchmark` to `experiments/benchmarks.py`
- [ ] Move `run_full_benchmark_extended` to `experiments/benchmarks.py`
- [ ] Move `run_generic_cv` to `experiments/cross_validation.py`
- [ ] Add experiment logging (MLflow, Weights & Biases, or TensorBoard)
- [ ] Add checkpoint management
- [ ] Add resume capability

#### Task 3.2: Create Ablation Study Framework
**New Module: `experiments/ablation.py`

**Actions:**
- [ ] Create ablation study template
- [ ] Define ablation configurations:
  - Physics layer only
  - Mamba only
  - Without RevIN
  - Without series decomposition
  - Different loss weights
- [ ] Add systematic comparison framework
- [ ] Add statistical significance testing

### Phase 4: Utilities & Improvements (Priority: MEDIUM)

#### Task 4.1: Extract Utilities
**New Module: `utils/`

**Actions:**
- [ ] Create `utils/random.py` with seed control
- [ ] Create `utils/device.py` for device management
- [ ] Create `utils/logging.py` for consistent logging
- [ ] Add `set_seed()` function for reproducibility
- [ ] Add `get_device()` function with fallback logic

#### Task 4.2: Remove Hardcoded Values
**Locations:** Lines 374, 406, 720, 726

**Actions:**
- [ ] Create `equipment_registry.py` for module/inverter specs
- [ ] Load equipment specs from metadata file
- [ ] Make equipment selection configurable
- [ ] Add validation for equipment compatibility

#### Task 4.3: Add Type Hints
**Scope:** All functions and classes

**Actions:**
- [ ] Add type hints to all function signatures
- [ ] Use `typing` module (List, Dict, Tuple, Optional)
- [ ] Add return type annotations
- [ ] Use `torch.Tensor` and `np.ndarray` types
- [ ] Enable mypy checking

#### Task 4.4: Improve Documentation
**Scope:** All modules

**Actions:**
- [ ] Standardize docstrings (Google style)
- [ ] Add module-level docstrings
- [ ] Document all parameters with types
- [ ] Document return values
- [ ] Add usage examples
- [ ] Create API documentation

### Phase 5: Scientific Rigor Enhancements (Priority: HIGH)

#### Task 5.1: Reproducibility
**Actions:**
- [ ] Add random seed control to all experiments
- [ ] Pin library versions (requirements.txt with hashes)
- [ ] Add data versioning
- [ ] Add model checkpointing with versioning
- [ ] Log all hyperparameters
- [ ] Log system information (CUDA, OS, Python version)

#### Task 5.2: Statistical Analysis
**New Module: `evaluation/statistics.py`

**Actions:**
- [ ] Add confidence interval calculation (bootstrap)
- [ ] Add hypothesis testing (paired t-test)
- [ ] Add effect size calculation (Cohen's d)
- [ ] Add statistical significance reporting
- [ ] Add cross-fold variance analysis

#### Task 5.3: Ablation Studies
**New Notebook: `experiments/02_physics_ablation.ipynb`

**Actions:**
- [ ] Design ablation matrix:
  ```
  | Configuration | RMSE | MAE | Significance |
  |-------------|------|-----|-------------|
  | Full Model   |      |     |             |
  | No Physics   |      |     |             |
  | No RevIN     |      |     |             |
  | No Decomp     |      |     |             |
  | Physics Only  |      |     |             |
  ```
- [ ] Implement systematic ablation experiments
- [ ] Generate comparison tables
- [ ] Create visualizations of ablation results

#### Task 5.4: Hyperparameter Tuning
**New Notebook: `experiments/03_hyperparameter_tuning.ipynb`

**Actions:**
- [ ] Implement grid search for key hyperparameters:
  - Learning rates: [1e-4, 5e-4, 1e-3, 5e-3]
  - Lambda weights: [(1.0, 0.2, 0.1), (1.0, 0.3, 0.2), (0.8, 0.2, 0.1)]
  - Mamba d_state: [64, 96, 128]
  - Sequence lengths: [96, 192, 384, 672, 960]
- [ ] Use Optuna or Ray Tune for Bayesian optimization
- [ ] Log all trials
- [ ] Analyze hyperparameter sensitivity
- [ ] Select best configuration with statistical validation

---

## 4. Migration Strategy

### 4.1 Backward Compatibility

During transition, maintain the monolithic file with deprecation warnings:

```python
import warnings

warnings.warn(
    "physics_residual_mamba.py is deprecated. "
    "Import from physics_residual_mamba package instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 4.2 Testing Strategy

**Unit Tests:**
- [ ] Create `tests/` directory
- [ ] Add tests for all data preprocessing functions
- [ ] Add tests for model forward passes
- [ ] Add tests for loss calculations
- [ ] Add tests for metric calculations
- [ ] Use pytest framework

**Integration Tests:**
- [ ] Test full training pipeline
- [ ] Test cross-validation
- [ ] Test baseline calculations
- [ ] Test visualization generation

### 4.3 Rollout Plan

**Week 1-2:**
- Complete Phase 1 (Foundation)
- Set up package structure
- Extract configuration classes

**Week 3-4:**
- Complete Phase 2 (Training & Evaluation)
- Extract all model components
- Extract training and evaluation functions

**Week 5-6:**
- Complete Phase 3 (Experiment Orchestration)
- Create experiment notebooks
- Set up ablation framework

**Week 7-8:**
- Complete Phase 4 (Utilities & Improvements)
- Add scientific rigor enhancements
- Implement hyperparameter tuning
- Add comprehensive documentation

---

## 5. Code Quality Standards

### 5.1 Naming Conventions

```python
# Classes: PascalCase
class PhysicsResidualMamba:
class DifferentiablePVLayer:

# Functions: snake_case
def train_one_epoch_physics():
def calculate_clearsky_indices():

# Constants: UPPER_SNAKE_CASE
DEFAULT_LEARNING_RATE = 1e-3
NIGHT_IRRADIANCE_THRESHOLD = 10.0

# Private methods: leading underscore
def _compute_rolling_statistics():
```

### 5.2 Type Hints

```python
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch and return loss and metrics."""
    pass
```

### 5.3 Documentation Standard

```python
def calculate_smart_persistence(
    power_last: torch.Tensor,
    p_clr_last: torch.Tensor,
    p_clr_future: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Smart Persistence baseline.
    
    Smart Persistence adjusts yesterday's power by the ratio of
    clear sky irradiance between yesterday and today.
    
    Formula:
        P_t = P_{t-24h} × (GHI_clr,t-24h / GHI_clr,t)
    
    Args:
        power_last: Power at t-24h [B, 1]
        p_clr_last: Clear sky irradiance at t-24h [B, 1]
        p_clr_future: Clear sky irradiance at t [B, H]
        eps: Small constant for numerical stability
        
    Returns:
        Smart persistence prediction [B, H, 1]
        
    Example:
        >>> power_last = torch.tensor([[10.0]])
        >>> p_clr_last = torch.tensor([[800.0]])
        >>> p_clr_future = torch.tensor([[1000.0, 900.0]])
        >>> result = calculate_smart_persistence(power_last, p_clr_last, p_clr_future)
        >>> print(result)
        tensor([[12.5000, 11.2500]])
    """
    return power_last * (p_clr_future / (p_clr_last + eps))
```

---

## 6. Scientific Research Recommendations

### 6.1 Critical Gaps

| Gap | Severity | Recommendation |
|------|----------|---------------|
| **No Statistical Significance Testing** | HIGH | Add bootstrap confidence intervals, paired t-tests between models |
| **Limited Ablation Studies** | HIGH | Systematically remove components to measure contribution |
| **No Hyperparameter Sensitivity Analysis** | MEDIUM | Perform grid search or Bayesian optimization |
| **Missing Cross-Station Validation** | HIGH | Test on multiple stations (00-09 available) |
| **No Temporal Generalization Test** | MEDIUM | Test on different time periods (seasonal splits) |
| **Unclear Baseline Superiority Claims** | MEDIUM | Use statistical tests to prove significance |
| **Missing Error Analysis** | MEDIUM | Analyze prediction errors by time of day, season, weather conditions |
| **No Model Interpretability** | LOW | Add attention visualization, feature importance analysis |

### 6.2 Experimental Design Improvements

**1. Multi-Station Validation:**
```python
stations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
results_by_station = {}

for station_num in stations:
    df = load_station_data(station_num)
    metadata = get_station_metadata(station_num)
    results = run_full_benchmark(df, metadata, configs)
    results_by_station[station_num] = results

# Analyze cross-station generalization
plot_station_comparison(results_by_station)
compute_statistical_significance(results_by_station)
```

**2. Temporal Splits:**
```python
# Instead of just TimeSeriesSplit, also test:
# - Winter only (Dec-Feb)
# - Summer only (Jun-Aug)
# - Transition months (Mar-May, Sep-Nov)

seasonal_splits = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11]
}
```

**3. Error Analysis Framework:**
```python
def analyze_prediction_errors(preds, actuals, features):
    """Analyze errors by conditions."""
    errors = preds - actuals
    
    # By time of day
    hour_errors = errors.groupby(features['hour']).mean()
    
    # By irradiance level
    irr_bins = [0, 100, 200, 400, 600, 800, 1000]
    irr_errors = errors.groupby(pd.cut(features['irradiance'], irr_bins)).mean()
    
    # By season
    season_errors = errors.groupby(features['season']).mean()
    
    return {
        'hourly': hour_errors,
        'irradiance': irr_errors,
        'seasonal': season_errors
    }
```

### 6.3 Publication-Ready Visualizations

**Required Plots:**
1. **Time Series Comparison** (existing, needs improvement)
   - Ground Truth (black, solid)
   - Smart Persistence (blue, dashed)
   - NWP Forecast (green, dashed)
   - Physics-Residual Model (red, solid)
   - Add confidence intervals (shaded regions)

2. **Zoomed-In Daily Profiles**
   - Select 3 representative days: clear, cloudy, mixed
   - Show sunrise-to-sunset curves
   - Highlight physics layer smoothness vs. baseline jaggedness

3. **Scatter Plot with Statistics**
   - Predicted vs Actual scatter
   - Ideal y=x line
   - Add R², RMSE, MAE annotations
   - Add density contours

4. **Error Distribution**
   - Histogram of prediction errors
   - Q-Q plot vs normal distribution
   - Box plots by time of day

5. **Ablation Results**
   - Bar chart comparing component contributions
   - Statistical significance annotations
   - Effect size indicators

6. **Hyperparameter Sensitivity**
   - Heat map of RMSE vs hyperparameters
   - Pareto frontier visualization
   - Best configuration markers

---

## 7. Implementation Priority Matrix

| Phase | Tasks | Effort | Impact | Priority |
|--------|---------|--------|----------|
| **Phase 1: Foundation** | 1.1-1.4 | High | P0 |
| **Phase 2: Training & Evaluation** | 2.1-2.3 | High | P0 |
| **Phase 5: Scientific Rigor** | 5.1-5.4 | Very High | P0 |
| **Phase 3: Experiment Orchestration** | 3.1-3.2 | Medium | P1 |
| **Phase 4: Utilities** | 4.1-4.4 | Medium | P1 |

**Legend:**
- **P0**: Critical path, must complete before production use
- **P1**: Important but can be done incrementally

---

## 8. Success Criteria

### 8.1 Code Quality
- [ ] All modules have type hints
- [ ] All functions have docstrings
- [ ] Code passes mypy strict mode
- [ ] Unit test coverage > 80%
- [ ] No linting errors (pylint, flake8)

### 8.2 Scientific Rigor
- [ ] Statistical significance testing implemented
- [ ] Ablation studies completed
- [ ] Multi-station validation performed
- [ ] Hyperparameter sensitivity analyzed
- [ ] Results reproducible with fixed seeds

### 8.3 Documentation
- [ ] API documentation generated (Sphinx/MkDocs)
- [ ] Usage examples provided
- [ ] Installation guide created
- [ ] Tutorial notebooks created

### 8.4 Performance
- [ ] Training time < baseline
- [ ] Memory usage optimized
- [ ] Inference latency measured
- [ ] Scalability tested

---

## 9. Next Steps

1. **Review and approve this plan** with stakeholders
2. **Begin Phase 1 implementation** (Foundation)
3. **Set up testing framework** before extensive refactoring
4. **Create migration guide** for existing code users
5. **Establish CI/CD pipeline** for automated testing
6. **Schedule regular code reviews** during refactoring

---

## Appendix: File-by-File Migration Map

### physics_residual_mamba.py → New Modules

| Lines | Content | Destination |
|--------|---------|-------------|
| 32-133 | `FeatureScaler`, `compute_scaler_stats` | `models/physics_layer.py` |
| 140-240 | `DifferentiablePVLayer` | `models/physics_layer.py` |
| 246-335 | `RevIN`, `moving_avg`, `series_decomp` | `models/components.py` |
| 350-503 | `MMPISSM_Model_01` | `models/base_models.py` |
| 509-625 | `PhysicsResidualMamba` | `models/hybrid_model.py` |
| 631-731 | `PhysicsPVLoss` | `losses/physics_loss.py` |
| 737-924 | `MultiStepDataset`, `prepare_rolling_folds` | `data/datasets.py` |
| 930-983 | `train_one_epoch_physics` | `training/trainers.py` |
| 985-1038 | `evaluate_fold_physics` | `evaluation/evaluators.py` |
| 1053-1199 | `run_physics_residual_cv` | `experiments/cross_validation.py` |
| 1204-1507 | `plot_model_performance`, `print_performance_summary` | `visualization/plots.py`, `visualization/reports.py` |
| 1309-1400 | `evaluate_fold_physics_v2` | `evaluation/evaluators.py` (consolidate) |
| 1675-1801 | `VanillaLSTM`, `VanillaMamba` | `models/base_models.py` |
| 1802-1884 | `run_generic_cv` | `experiments/cross_validation.py` |
| 1885-1941 | `plot_benchmark_summary` | `visualization/plots.py` |
| 1946-2021 | `PhysicsResidualMambaConfigs` | `configs/physics_config.py` |

### Notebook Functions → New Modules

| Lines | Content | Destination |
|--------|---------|-------------|
| 54-204 | `get_station_metadata` | `data/station_metadata.py` |
| 210-558 | `calculate_clearsky_indices` | `physics/clearsky.py` |
| 564-847 | `calculate_nwp_power` | `physics/nwp_power.py` |
| 889-936 | `prepare_features` | `data/preprocessing.py` |
| 955-1132 | `add_past_rolling_stats` | `data/preprocessing.py` |
| 1200-1277 | `BaseMambaConfigs` | `configs/base_config.py` |
| 1333-1428 | `RevIN`, `moving_avg`, `series_decomp` | `models/components.py` (duplicate) |
| 1513-1828 | `MMPISSM_Model_01` | `models/base_models.py` (duplicate) |
| 1839-1892 | `train_one_epoch` | `training/trainers.py` (duplicate) |
| 1897-1972 | `evaluate_fold` | `evaluation/evaluators.py` (duplicate) |
| 1977-2000 | `plot_loss_curve` | `visualization/plots.py` |
| 2005-2164 | `run_rolling_cv` | `experiments/cross_validation.py` (duplicate) |
| 2376-2503 | `MambaModelConfigs` | `configs/base_config.py` (duplicate) |
| 2519-2715 | `PhysicsPVLoss` (alternative) | `losses/physics_loss.py` (consolidate) |
| 2716-2789 | `train_one_epoch` (extended) | `training/trainers.py` (consolidate) |
| 2794-2889 | `evaluate_fold` (extended) | `evaluation/evaluators.py` (consolidate) |
| 2889-3030 | `run_rolling_cv` (extended) | `experiments/cross_validation.py` (consolidate) |

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-10  
**Status:** Ready for Review and Approval
