"""
Training module for solar power forecasting models.

Provides:
- train_one_epoch_physics: Train physics-informed model for one epoch
- train_one_epoch_base: Train base model for one epoch
- run_physics_residual_cv: Cross-validation training for physics model
- run_base_mamba_cv: Cross-validation training for base model
"""

from .trainers import (
    train_one_epoch_physics,
    train_one_epoch_base,
    run_physics_residual_cv,
    run_base_mamba_cv
)

__all__ = [
    'train_one_epoch_physics',
    'train_one_epoch_base',
    'run_physics_residual_cv',
    'run_base_mamba_cv',
]
