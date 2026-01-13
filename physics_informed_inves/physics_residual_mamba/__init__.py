"""
Physics-Residual Power Mamba: Hybrid Architecture for Solar Power Forecasting.

This package provides a modular implementation of a physics-informed deep learning
model combining differentiable physics layers with Mamba-based residual learning
for solar power forecasting.

Architecture:
    - Differentiable Physics Layer: Learnable PV physics model with geometric gating
    - Mamba Residual Learner: State-space model for error correction
    - Hybrid Combination: P_total = ReLU(P_physics + P_residual)

Key Features:
    - Physics-informed predictions with learnable parameters
    - Smart Persistence baseline calculation
    - Time series cross-validation
    - Comprehensive evaluation metrics
    - Publication-quality visualizations

Modules:
    - configs: Configuration classes for models
    - data: Dataset preparation and feature engineering
    - models: Model architectures (base, physics, hybrid)
    - losses: Physics-informed loss functions
    - evaluation: Evaluation functions and metrics
    - visualization: Plotting utilities
    - utils: Helper functions (random, device, logging)

Example Usage:
    >>> from physics_residual_mamba import (
    ...     PhysicsResidualMamba,
    ...     PhysicsResidualMambaConfigs,
    ...     PhysicsPVLoss,
    ...     run_physics_residual_cv,
    ...     plot_model_performance
    ... )
    >>> 
    >>> # Configure model
    >>> configs = PhysicsResidualMambaConfigs(n_splits=4)
    >>> 
    >>> # Train model
    >>> results, history = run_physics_residual_cv(df, configs)
    >>> 
    >>> # Visualize results
    >>> plot_model_performance(results[-1])
"""

# Configuration
from .configs import BaseMambaConfigs, PhysicsResidualMambaConfigs

# Data
from .data import MultiStepDataset, FilteredDataset, prepare_rolling_folds
from .data.preprocessing import (
    add_cyclic_features,
    add_past_rolling_stats,
    clean_dataframe,
    clip_outliers,
    normalize_features,
    denormalize_features
)

# Models
from .models import (
    RevIN,
    moving_avg,
    series_decomp,
    DifferentiablePVLayer,
    FeatureScaler,
    MMPISSM_Model_01,
    VanillaLSTM,
    VanillaMamba,
    PhysicsResidualMamba
)

# Losses
from .losses import PhysicsPVLoss

# Evaluation
from .evaluation import (
    evaluate_fold_physics,
    evaluate_fold_base,
    calculate_metrics,
    calculate_improvement,
    calculate_confidence_interval,
    calculate_skill_score,
    calculate_errors_by_time,
    calculate_errors_by_irradiance
)

# Visualization
from .visualization import (
    plot_model_performance,
    plot_loss_curve,
    plot_benchmark_summary,
    plot_error_analysis,
    plot_multi_model_comparison
)

# Utils
from .utils import (
    set_random_seed,
    get_random_seed,
    get_device,
    get_device_info,
    print_device_info,
    to_device,
    setup_logging,
    get_logger,
    ExperimentLogger
)

__version__ = '0.1.0'
__author__ = 'Physics-Residual Mamba Team'
__email__ = 'contact@example.com'

__all__ = [
    # Configuration
    'BaseMambaConfigs',
    'PhysicsResidualMambaConfigs',
    
    # Data
    'MultiStepDataset',
    'FilteredDataset',
    'prepare_rolling_folds',
    'add_cyclic_features',
    'add_past_rolling_stats',
    'clean_dataframe',
    'clip_outliers',
    'normalize_features',
    'denormalize_features',
    
    # Models
    'RevIN',
    'moving_avg',
    'series_decomp',
    'DifferentiablePVLayer',
    'FeatureScaler',
    'MMPISSM_Model_01',
    'VanillaLSTM',
    'VanillaMamba',
    'PhysicsResidualMamba',
    
    # Losses
    'PhysicsPVLoss',
    
    # Evaluation
    'evaluate_fold_physics',
    'evaluate_fold_base',
    'calculate_metrics',
    'calculate_improvement',
    'calculate_confidence_interval',
    'calculate_skill_score',
    'calculate_errors_by_time',
    'calculate_errors_by_irradiance',
    
    # Visualization
    'plot_model_performance',
    'plot_loss_curve',
    'plot_benchmark_summary',
    'plot_error_analysis',
    'plot_multi_model_comparison',
    
    # Utils
    'set_random_seed',
    'get_random_seed',
    'get_device',
    'get_device_info',
    'print_device_info',
    'to_device',
    'setup_logging',
    'get_logger',
    'ExperimentLogger',
]
