"""
Visualization functions for solar power forecasting evaluation.

Provides:
- plot_model_performance: Main performance visualization (3 plots)
- plot_loss_curve: Training/validation loss curves
- plot_benchmark_summary: Comparative bar plots
- plot_error_analysis: Error distribution and analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional


def plot_model_performance(results_dict: Dict, figsize: tuple = (15, 12)):
    """
    Generate professional plots for model performance evaluation.
    
    Creates three publication-quality plots:
        Plot A: "Big Picture" - Time series comparison (first 500 steps)
        Plot B: "Zoomed-In View" - 24h window showing sunrise/sunset curves
        Plot C: Scatter Plot - Goodness of fit (predicted vs actual)
    
    Args:
        results_dict: Dictionary containing:
            - preds: Model predictions [N, H, 1]
            - actuals: Ground truth [N, H, 1]
            - baseline: Smart Persistence baseline [N, H]
            - rmse: Model RMSE
        figsize: Figure size (default (15, 12))
        
    Example:
        >>> plot_model_performance(results_dict)
        >>> plt.show()
    """
    # Use seaborn style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data from results dict
    if 'preds' not in results_dict:
        print("Error: Results dictionary missing 'preds'. Ensure evaluation was run with updated code.")
        return
    
    preds = results_dict['preds'].flatten()
    actuals = results_dict['actuals'].flatten()
    baseline = results_dict['baseline'].flatten()
    
    rmse_model = results_dict['rmse']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle('Physics-Residual Mamba Performance Evaluation', fontsize=16, fontweight='bold')
    
    # Plot A: The "Big Picture" (First 500 steps)
    limit = 500
    axes[0].plot(actuals[:limit], label='Ground Truth', color='black', linewidth=1.5)
    axes[0].plot(baseline[:limit], label='Smart Persistence', color='blue', linestyle='--', alpha=0.7, linewidth=1.2)
    axes[0].plot(preds[:limit], label='Physics-Residual Mamba', color='red', alpha=0.8, linewidth=1.5)
    axes[0].set_title(f'Model vs Baseline (First {limit} time steps)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Power (MW)', fontsize=11)
    axes[0].set_xlabel('Time Steps', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot B: Zoomed-In View (Random 96-step window)
    window_size = 96
    if len(preds) > window_size:
        # Use deterministic seed for reproducibility
        np.random.seed(42)
        start_idx = np.random.randint(0, len(preds) - window_size)
    else:
        start_idx = 0
    end_idx = start_idx + window_size
    
    axes[1].plot(actuals[start_idx:end_idx], label='Ground Truth', color='black', linewidth=2)
    axes[1].plot(baseline[start_idx:end_idx], label='Smart Persistence', color='blue', linestyle='--', linewidth=1.5)
    axes[1].plot(preds[start_idx:end_idx], label='Physics-Residual', color='red', linewidth=2)
    
    # Calculate window metrics
    win_rmse_model = np.sqrt(np.mean((preds[start_idx:end_idx] - actuals[start_idx:end_idx])**2))
    win_rmse_base = np.sqrt(np.mean((baseline[start_idx:end_idx] - actuals[start_idx:end_idx])**2))
    
    axes[1].set_title(
        f'Zoomed-In 24h Window (Model RMSE: {win_rmse_model:.2f} vs Baseline: {win_rmse_base:.2f})',
        fontsize=12, fontweight='bold'
    )
    axes[1].set_ylabel('Power (MW)', fontsize=11)
    axes[1].set_xlabel('Time Steps (24 hours)', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot C: Scatter Plot (Goodness of Fit)
    axes[2].scatter(actuals, baseline, alpha=0.1, color='blue', label='Smart Persistence', s=5)
    axes[2].scatter(actuals, preds, alpha=0.3, color='red', label='Physics-Residual', s=5)
    
    # Ideal line
    max_val = max(actuals.max(), preds.max()) if len(actuals) > 0 else 1.0
    axes[2].plot([0, max_val], [0, max_val], 'k--', label='Ideal', linewidth=1.5, alpha=0.7)
    
    axes[2].set_title(f'Goodness of Fit: Predicted vs Actual (RMSE: {rmse_model:.4f})', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Actual Power (MW)', fontsize=11)
    axes[2].set_ylabel('Predicted Power (MW)', fontsize=11)
    axes[2].legend(loc='best', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_loss_curve(train_losses: list, val_losses: list, fold_num: int):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        fold_num: Fold number for title
        
    Example:
        >>> plot_loss_curve(train_losses, val_losses, fold_num=1)
        >>> plt.show()
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    
    plt.title(f'Fold {fold_num} Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_benchmark_summary(
    base_results: list,
    phys_results: list,
    figsize: tuple = (12, 7)
):
    """
    Generate comparative bar plots with improvement statistics.
    
    Args:
        base_results: List of base model results (each with 'rmse', 'mae')
        phys_results: List of physics model results (each with 'rmse', 'mae', 'sp_rmse', 'nwp_rmse')
        figsize: Figure size
        
    Example:
        >>> plot_benchmark_summary(base_results, phys_results)
        >>> plt.show()
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Calculate Averages
    avg_base_rmse = np.nanmean([r['rmse'] for r in base_results])
    avg_phys_rmse = np.nanmean([r['rmse'] for r in phys_results])
    avg_sp_rmse = np.nanmean([r['sp_rmse'] for r in phys_results])
    avg_nwp_rmse = np.nanmean([r['nwp_rmse'] for r in phys_results])
    
    # Data for plotting
    models = ['Smart Persistence', 'NWP Forecast', 'Base Mamba', 'Physics-Residual']
    rmses = [avg_sp_rmse, avg_nwp_rmse, avg_base_rmse, avg_phys_rmse]
    colors = ['#95a5a6', '#f39c12', '#3498db', '#e74c3c']  # Gray, Orange, Blue, Red
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, rmses, color=colors, alpha=0.9, width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.05,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    
    # Calculate Improvements
    imp_base = (1 - avg_phys_rmse / avg_base_rmse) * 100
    imp_pers = (1 - avg_phys_rmse / avg_sp_rmse) * 100
    imp_nwp = (1 - avg_phys_rmse / avg_nwp_rmse) * 100
    
    # Add improvement lines/text
    def draw_improvement(idx_start, idx_end, imp_val, y_pos):
        x_start = bars[idx_start].get_x() + bars[idx_start].get_width() / 2
        x_end = bars[idx_end].get_x() + bars[idx_end].get_width() / 2
        plt.plot([x_start, x_end], [y_pos, y_pos], 'k-', lw=1.5)
        plt.text(
            (x_start + x_end) / 2, y_pos + 0.05,
            f'+{imp_val:.1f}% Better',
            ha='center', va='bottom', fontsize=11, color='green', fontweight='bold'
        )
    
    # Draw improvement vs Base (Idx 2 -> 3)
    max_y = max(rmses)
    draw_improvement(2, 3, imp_base, max_y * 1.05)
    # Draw improvement vs NWP (Idx 1 -> 3)
    draw_improvement(1, 3, imp_nwp, max_y * 1.20)
    # Draw improvement vs Persistence (Idx 0 -> 3)
    draw_improvement(0, 3, imp_pers, max_y * 1.35)
    
    plt.ylim(0, max_y * 1.5)
    plt.ylabel('RMSE (MW)', fontsize=12, fontweight='bold')
    plt.title('Benchmark Results: Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n[CONCLUSION] Physics-Residual Mamba improves RMSE by {imp_base:.1f}% vs Base Mamba "
          f"and {imp_nwp:.1f}% vs NWP.")


def plot_error_analysis(
    preds: np.ndarray,
    actuals: np.ndarray,
    baseline: np.ndarray,
    figsize: tuple = (15, 10)
):
    """
    Plot error distribution and analysis.
    
    Creates:
        1. Error histogram comparison
        2. Error vs Power scatter
        3. Cumulative error distribution
        
    Args:
        preds: Model predictions
        actuals: Ground truth
        baseline: Smart Persistence baseline
        figsize: Figure size
        
    Example:
        >>> plot_error_analysis(preds, actuals, baseline)
        >>> plt.show()
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Flatten arrays
    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()
    baseline_flat = baseline.flatten()
    
    # Calculate errors
    model_errors = preds_flat - actuals_flat
    baseline_errors = baseline_flat - actuals_flat
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Error Analysis: Model vs Baseline', fontsize=16, fontweight='bold')
    
    # Plot 1: Error Histogram
    axes[0, 0].hist(model_errors, bins=50, alpha=0.5, color='red', label='Model', density=True)
    axes[0, 0].hist(baseline_errors, bins=50, alpha=0.5, color='blue', label='Baseline', density=True)
    axes[0, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Error (MW)', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].legend(loc='best', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error vs Power
    axes[0, 1].scatter(actuals_flat, model_errors, alpha=0.3, color='red', s=5, label='Model')
    axes[0, 1].scatter(actuals_flat, baseline_errors, alpha=0.3, color='blue', s=5, label='Baseline')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Error vs Actual Power', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Power (MW)', fontsize=11)
    axes[0, 1].set_ylabel('Error (MW)', fontsize=11)
    axes[0, 1].legend(loc='best', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Error Distribution
    model_errors_sorted = np.sort(np.abs(model_errors))
    baseline_errors_sorted = np.sort(np.abs(baseline_errors))
    n = len(model_errors_sorted)
    
    axes[1, 0].plot(model_errors_sorted, np.arange(1, n+1)/n, color='red', linewidth=2, label='Model')
    axes[1, 0].plot(baseline_errors_sorted, np.arange(1, n+1)/n, color='blue', linewidth=2, label='Baseline')
    axes[1, 0].set_title('Cumulative Absolute Error', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Absolute Error (MW)', fontsize=11)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=11)
    axes[1, 0].legend(loc='best', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Box Plot
    axes[1, 1].boxplot([model_errors, baseline_errors], labels=['Model', 'Baseline'])
    axes[1, 1].set_title('Error Distribution (Box Plot)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Error (MW)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_multi_model_comparison(
    actuals: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    limit: int = 500,
    figsize: tuple = (15, 6)
):
    """
    Compare multiple models on same time series.
    
    Args:
        actuals: Ground truth
        predictions_dict: Dict mapping model names to predictions
        limit: Number of time steps to plot
        figsize: Figure size
        
    Example:
        >>> predictions_dict = {
        ...     'Base Mamba': base_preds,
        ...     'Physics-Residual': phys_preds,
        ...     'Smart Persistence': sp_preds
        ... }
        >>> plot_multi_model_comparison(actuals, predictions_dict)
        >>> plt.show()
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    
    # Plot ground truth
    plt.plot(actuals[:limit], label='Ground Truth', color='black', linewidth=1.5)
    
    # Plot each model
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(preds.flatten()[:limit], label=model_name, color=color, alpha=0.7, linewidth=1.2)
    
    plt.title(f'Multi-Model Comparison (First {limit} time steps)', fontsize=14, fontweight='bold')
    plt.ylabel('Power (MW)', fontsize=12)
    plt.xlabel('Time Steps', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
