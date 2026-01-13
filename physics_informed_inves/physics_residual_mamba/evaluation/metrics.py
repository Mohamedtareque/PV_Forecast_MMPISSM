"""
Metric calculation utilities for solar power forecasting.

Provides:
- calculate_metrics: Compute RMSE, MAE, MAPE, etc.
- calculate_improvement: Compute percentage improvement over baseline
- calculate_confidence_interval: Bootstrap confidence interval for metrics
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def calculate_metrics(
    preds: np.ndarray,
    actuals: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate standard forecasting metrics.
    
    Computes:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - MAPE: Mean Absolute Percentage Error
        - R²: Coefficient of Determination
        - Bias: Mean prediction error (positive = overprediction)
    
    Args:
        preds: Predictions [N] or [N, H]
        actuals: Ground truth [N] or [N, H]
        mask: Optional boolean mask to exclude certain samples
        
    Returns:
        Dictionary with metric names and values
    """
    # Flatten if needed
    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        preds_flat = preds_flat[mask_flat]
        actuals_flat = actuals_flat[mask_flat]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(preds_flat) | np.isnan(actuals_flat))
    preds_flat = preds_flat[valid_mask]
    actuals_flat = actuals_flat[valid_mask]
    
    if len(preds_flat) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'bias': np.nan
        }
    
    # RMSE
    rmse = np.sqrt(np.mean((preds_flat - actuals_flat) ** 2))
    
    # MAE
    mae = np.mean(np.abs(preds_flat - actuals_flat))
    
    # MAPE (exclude near-zero values)
    nonzero_mask = np.abs(actuals_flat) > 0.1  # Exclude values < 0.1 MW
    if nonzero_mask.any():
        mape = np.mean(np.abs((actuals_flat[nonzero_mask] - preds_flat[nonzero_mask]) / actuals_flat[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # R² (Coefficient of Determination)
    ss_res = np.sum((actuals_flat - preds_flat) ** 2)
    ss_tot = np.sum((actuals_flat - np.mean(actuals_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Bias (mean error)
    bias = np.mean(preds_flat - actuals_flat)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'bias': bias
    }


def calculate_improvement(
    model_metric: float,
    baseline_metric: float,
    metric_name: str = 'RMSE'
) -> Dict[str, float]:
    """
    Calculate percentage improvement over baseline.
    
    Args:
        model_metric: Model's metric value (lower is better for RMSE/MAE)
        baseline_metric: Baseline's metric value
        metric_name: Name of metric for logging
        
    Returns:
        Dictionary with improvement statistics:
            - improvement_pct: Percentage improvement
            - absolute_diff: Absolute difference
            - ratio: Model/baseline ratio
    """
    if baseline_metric == 0:
        return {
            'improvement_pct': 0.0,
            'absolute_diff': 0.0,
            'ratio': 1.0
        }
    
    improvement_pct = (1 - model_metric / baseline_metric) * 100
    absolute_diff = baseline_metric - model_metric
    ratio = model_metric / baseline_metric
    
    return {
        'improvement_pct': improvement_pct,
        'absolute_diff': absolute_diff,
        'ratio': ratio
    }


def calculate_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for metric values.
    
    Args:
        values: Array of metric values (e.g., RMSE from multiple folds)
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples (default 1000)
        
    Returns:
        (lower_bound, upper_bound): Confidence interval bounds
    """
    if len(values) < 2:
        return values[0], values[0]
    
    # Bootstrap sampling
    n = len(values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def calculate_skill_score(
    model_rmse: float,
    persistence_rmse: float
) -> float:
    """
    Calculate Forecast Skill Score (FSS).
    
    FSS = 1 - (RMSE_model / RMSE_persistence)
    
    Positive values indicate improvement over persistence baseline.
    
    Args:
        model_rmse: Model's RMSE
        persistence_rmse: Smart persistence baseline RMSE
        
    Returns:
        Skill score (positive = better than persistence)
    """
    if persistence_rmse == 0:
        return 0.0
    
    return 1 - (model_rmse / persistence_rmse)


def calculate_errors_by_time(
    preds: np.ndarray,
    actuals: np.ndarray,
    timestamps: np.ndarray,
    bins: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by time of day.
    
    Args:
        preds: Predictions [N, H] or flattened [N*H]
        actuals: Ground truth [N, H] or flattened [N*H]
        timestamps: DatetimeIndex corresponding to predictions
        bins: Hour bins (default: [0, 6, 12, 18, 24])
        
    Returns:
        Dictionary with hour ranges as keys and metric dicts as values
    """
    if bins is None:
        bins = [0, 6, 12, 18, 24]
    
    # Flatten if needed
    if preds.ndim > 1:
        preds_flat = preds.flatten()
        actuals_flat = actuals.flatten()
    else:
        preds_flat = preds
        actuals_flat = actuals
    
    # Get hour of day for each prediction
    hours = timestamps.hour.values[:len(preds_flat)]
    
    results = {}
    for i in range(len(bins) - 1):
        start_hour = bins[i]
        end_hour = bins[i + 1]
        
        # Create mask for this time range
        mask = (hours >= start_hour) & (hours < end_hour)
        
        if mask.sum() > 0:
            metrics = calculate_metrics(preds_flat[mask], actuals_flat[mask])
            results[f"{start_hour:02d}-{end_hour:02d}h"] = metrics
    
    return results


def calculate_errors_by_irradiance(
    preds: np.ndarray,
    actuals: np.ndarray,
    irradiance: np.ndarray,
    bins: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by irradiance level.
    
    Args:
        preds: Predictions [N, H] or flattened [N*H]
        actuals: Ground truth [N, H] or flattened [N*H]
        irradiance: Irradiance values [N, H] or flattened [N*H]
        bins: Irradiance bins in W/m² (default: [0, 100, 300, 600, 1200])
        
    Returns:
        Dictionary with irradiance ranges as keys and metric dicts as values
    """
    if bins is None:
        bins = [0, 100, 300, 600, 1200]
    
    # Flatten if needed
    if preds.ndim > 1:
        preds_flat = preds.flatten()
        actuals_flat = actuals.flatten()
        irr_flat = irradiance.flatten()
    else:
        preds_flat = preds
        actuals_flat = actuals
        irr_flat = irradiance
    
    results = {}
    for i in range(len(bins) - 1):
        start_irr = bins[i]
        end_irr = bins[i + 1]
        
        # Create mask for this irradiance range
        mask = (irr_flat >= start_irr) & (irr_flat < end_irr)
        
        if mask.sum() > 0:
            metrics = calculate_metrics(preds_flat[mask], actuals_flat[mask])
            results[f"{start_irr}-{end_irr} W/m²"] = metrics
    
    return results
