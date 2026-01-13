"""
Evaluation and metrics module for solar power forecasting.

Provides:
- evaluate_fold_physics: Evaluate physics-informed models
- evaluate_fold_base: Evaluate base models
- calculate_metrics: Compute standard metrics
- calculate_improvement: Compute percentage improvement
- calculate_confidence_interval: Bootstrap confidence intervals
"""

from .evaluators import evaluate_fold_physics, evaluate_fold_base
from .metrics import (
    calculate_metrics,
    calculate_improvement,
    calculate_confidence_interval,
    calculate_skill_score,
    calculate_errors_by_time,
    calculate_errors_by_irradiance
)

__all__ = [
    'evaluate_fold_physics',
    'evaluate_fold_base',
    'calculate_metrics',
    'calculate_improvement',
    'calculate_confidence_interval',
    'calculate_skill_score',
    'calculate_errors_by_time',
    'calculate_errors_by_irradiance',
]
