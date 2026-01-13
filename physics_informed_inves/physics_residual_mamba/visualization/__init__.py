"""
Visualization module for solar power forecasting evaluation.

Provides:
- plot_model_performance: Main performance visualization (3 plots)
- plot_loss_curve: Training/validation loss curves
- plot_benchmark_summary: Comparative bar plots
- plot_error_analysis: Error distribution and analysis
- plot_multi_model_comparison: Multi-model comparison
"""

from .plots import (
    plot_model_performance,
    plot_loss_curve,
    plot_benchmark_summary,
    plot_error_analysis,
    plot_multi_model_comparison
)

__all__ = [
    'plot_model_performance',
    'plot_loss_curve',
    'plot_benchmark_summary',
    'plot_error_analysis',
    'plot_multi_model_comparison',
]
