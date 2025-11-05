"""
This module contains high-level "manager" or "helper" classes that are used throughout the pipeline to manage workflow, not to transform data.

Contents:
- class DataManager
- class ExperimentTracker
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o

def _jsonable_dict(d):
    return {k: _jsonable(v) for k, v in d.items()}


class DataManager:
    """Handles data saving, loading, and split management.
    
    When We create a DataManager object, it simply establishes the directory paths where all data (data_dir) and split definitions (splits_dir) will be stored. 
    The mkdir(exist_ok=True) command ensures these folders are created if they don't already exist, preventing errors.
    """
    
    def __init__(self, data_dir: str = "data", splits_dir: str = "splits"):
        self.data_dir = Path(data_dir)
        self.splits_dir = Path(splits_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.splits_dir.mkdir(exist_ok=True)
    
    def save_arrays(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        labels: Optional[pd.DataFrame] = None,
        filename_prefix: str = "kbins_data",
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save numpy arrays and optional labels"""
        np.save(self.data_dir / f"{filename_prefix}_X.npy", X)
        np.save(self.data_dir / f"{filename_prefix}_Y.npy", Y)
        
        if labels is not None:
            labels.to_pickle(self.data_dir / f"{filename_prefix}_labels.pkl")
        
        # Save metadata
        metadata_payload = {
            "X_shape": X.shape,
            "Y_shape": Y.shape,
            "X_dtype": str(X.dtype),
            "Y_dtype": str(Y.dtype),
            "saved_at": datetime.now().isoformat(),
            "has_labels": labels is not None,
            "feature_cols": feature_cols,
            "target_col": target_col,
        }
        if metadata is not None:
            metadata_payload.update(metadata)
        
        with open(self.data_dir / f"{filename_prefix}_metadata.json", 'w') as f:
            json.dump(metadata_payload, f, indent=2)
        
        logger.info(f"Saved arrays to {self.data_dir}/{filename_prefix}_*.npy")
        logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    def load_arrays(self, filename_prefix: str = "kbins_data") -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame], Dict[str, Any]]:
        """Load saved arrays and labels"""
        X = np.load(self.data_dir / f"{filename_prefix}_X.npy")
        Y = np.load(self.data_dir / f"{filename_prefix}_Y.npy")
        
        labels = None
        labels_path = self.data_dir / f"{filename_prefix}_labels.pkl"
        if labels_path.exists():
            labels = pd.read_pickle(labels_path)

        metadata_path = self.data_dir / f"{filename_prefix}_metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Loaded arrays from {self.data_dir}/{filename_prefix}_*.npy")
        logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        return X, Y, labels, metadata
    
    def load_rolling_splits(self, splits_file: str = "rolling_origin_splits.json") -> Dict:
        """Load rolling origin cross-validation splits
        
        It reads a predefined JSON file that contains the specific start and end dates for the training and validation sets for each "fold" or "split" of your experiment.
        """
        with open(self.splits_dir / splits_file, 'r') as f:
            splits_data = json.load(f)
        
        logger.info(f"Loaded {len(splits_data['folds'])} folds from {splits_file}")
        return splits_data
    
    def get_fold_indices(self, X: np.ndarray, labels: pd.DataFrame, 
                        fold_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get train and validation indices for a specific fold.
        
        This is a lookup function. 
        Its job is to translate the date ranges from a specific fold (loaded by load_rolling_splits) into the exact integer row numbers needed to slice your main X and Y arrays.
        """
        train_indices_str = fold_data['train_indices']
        val_indices_str = fold_data['val_indices']
        
        labels_index = labels.index
        labels_tz = labels_index.tz if isinstance(labels_index, pd.DatetimeIndex) else None

        def _align_datetimes(values):
            ts = pd.to_datetime(values)
            if not isinstance(ts, pd.DatetimeIndex):
                return ts
            if labels_tz is None and ts.tz is not None:
                return ts.tz_convert('UTC').tz_localize(None)
            if labels_tz is not None and ts.tz is None:
                return ts.tz_localize(labels_tz)
            if labels_tz is not None and ts.tz is not None:
                return ts.tz_convert(labels_tz)
            return ts
        
        train_timestamps = _align_datetimes(train_indices_str)
        val_timestamps = _align_datetimes(val_indices_str)

        if isinstance(labels_index, pd.DatetimeIndex):
            label_dates = labels_index.normalize()
            train_dates = train_timestamps.normalize() if isinstance(train_timestamps, pd.DatetimeIndex) else pd.to_datetime(train_timestamps).normalize()
            val_dates = val_timestamps.normalize() if isinstance(val_timestamps, pd.DatetimeIndex) else pd.to_datetime(val_timestamps).normalize()
            train_mask = label_dates.isin(train_dates)
            val_mask = label_dates.isin(val_dates)
        else:
            train_mask = labels_index.isin(train_timestamps)
            val_mask = labels_index.isin(val_timestamps)
        
        if not train_mask.any() or not val_mask.any():
            train_period = fold_data.get("train_period")
            val_period = fold_data.get("val_period")
            if train_period and val_period:
                train_start = pd.to_datetime(train_period["start"])
                train_end = pd.to_datetime(train_period["end"])
                val_start = pd.to_datetime(val_period["start"])
                val_end = pd.to_datetime(val_period["end"])

                if isinstance(labels_index, pd.DatetimeIndex):
                    if labels_tz is None and train_start.tzinfo is not None:
                        train_start = train_start.tz_convert('UTC').tz_localize(None)
                        train_end = train_end.tz_convert('UTC').tz_localize(None)
                        val_start = val_start.tz_convert('UTC').tz_localize(None)
                        val_end = val_end.tz_convert('UTC').tz_localize(None)
                    elif labels_tz is not None and train_start.tzinfo is None:
                        train_start = train_start.tz_localize(labels_tz)
                        train_end = train_end.tz_localize(labels_tz)
                        val_start = val_start.tz_localize(labels_tz)
                        val_end = val_end.tz_localize(labels_tz)
                    else:
                        train_start = train_start.tz_convert(labels_tz)
                        train_end = train_end.tz_convert(labels_tz)
                        val_start = val_start.tz_convert(labels_tz)
                        val_end = val_end.tz_convert(labels_tz)

                    label_dates = labels_index
                    train_mask = (label_dates >= train_start.normalize()) & (label_dates <= train_end.normalize())
                    val_mask = (label_dates >= val_start.normalize()) & (label_dates <= val_end.normalize())
                else:
                    train_mask = (labels_index >= train_start) & (labels_index <= train_end)
                    val_mask = (labels_index >= val_start) & (labels_index <= val_end)
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                "Unable to match fold indices to prepared data samples. "
                "Check that the splits were generated from the same processed dataset."
            )
        
        return train_idx, val_idx


class ExperimentTracker:
    """Track experiments, save results and models
    
    ts job is to create a self-contained, timestamped folder for every single experiment run, ensuring that all related files—configurations, models, metrics, and plots—are saved in one clean, predictable place.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        
        # Create directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "models").mkdir(exist_ok=True)
        (self.exp_dir / "plots").mkdir(exist_ok=True)
        (self.exp_dir / "metrics").mkdir(exist_ok=True)
        
        self.config = {}
        self.results = {}
    
    def save_config(self, config: Dict):
        """Save experiment configuration.
        Saves the hyperparameters (like learning rate, number of layers, etc.) to a JSON file. This is essential for knowing exactly what settings produced a given result.
        """
        self.config = config
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_model(self, model: nn.Module, fold: int = 0, is_best: bool = False):
        """Save model weights"""
        filename = f"best_model_fold_{fold}.pth" if is_best else f"model_fold_{fold}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self.config.get('model_config', {})
        }, self.exp_dir / "models" / filename)
    
    def save_metrics(self, metrics: Dict, fold: int = 0, split: str = "test"):
        """Save evaluation metrics
        Saves the trained PyTorch model's weights (state_dict). The is_best flag is a common pattern for saving only 
        the model that achieved the best performance on the validation set, which is often the one you care about most.
        """
        filename = f"metrics_fold_{fold}_{split}.json"
        with open(self.exp_dir / "metrics" / filename, 'w') as f:
            json.dump(_jsonable_dict(metrics), f, indent=2)
        
        # Update results
        if fold not in self.results:
            self.results[fold] = {}
        self.results[fold][split] = metrics
    
    def save_comparison_table(self, df: pd.DataFrame, fold: int = 0, split: str = "validation"):
        """Persist detailed comparison data (model vs NAM vs actual)."""
        if df.empty:
            logger.warning(
                "Comparison dataframe for fold %s / %s is empty; nothing to save.",
                fold,
                split,
            )
            return

        filename = self.exp_dir / "metrics" / f"comparison_fold_{fold}_{split}.csv"
        df.to_csv(filename, index=False)

    def save_nam_comparison_plot(
        self,
        df: pd.DataFrame,
        metrics: Optional[Dict[str, float]] = None,
        fold: int = 0,
        split: str = "validation",
    ):
        """Create NAM vs model vs actual plots, annotated with key metrics."""
        if df.empty:
            logger.warning(
                "Comparison dataframe for fold %s / %s is empty; skipping NAM plot.",
                fold,
                split,
            )
            return

        best_sample = None
        best_energy = -np.inf
        for sample_order, sample_df in df.groupby("sample_order"):
            daylight = sample_df[sample_df["clear_sky_ghi"] > 5.0]
            energy = daylight["clear_sky_ghi"].sum()
            if energy > best_energy and not daylight.empty:
                best_energy = energy
                best_sample = (sample_order, sample_df.sort_values("target_time"))

        if best_sample is None:
            sample_order, sample_df = next(iter(df.groupby("sample_order")))
            sample_df = sample_df.sort_values("target_time")
        else:
            sample_order, sample_df = best_sample

        times = sample_df["target_time"]
        if times.dt.tz is None:
            times = times.dt.tz_localize("UTC")
        else:
            times = times.dt.tz_convert("UTC")
        actual_ghi = sample_df["actual_ghi"]
        model_ghi = sample_df["model_ghi_pred"]
        nam_ghi = sample_df["nam_ghi"]

        daylight_mask = sample_df["clear_sky_ghi"] > 5.0
        daylight_actual = actual_ghi[daylight_mask]
        daylight_model = model_ghi[daylight_mask]
        daylight_nam = nam_ghi[daylight_mask]

        fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

        ax_ts = axes[0]
        ax_ts.plot(times, actual_ghi, label="Measured GHI", color="tab:blue", linewidth=2)
        ax_ts.plot(times, model_ghi, label="Model Forecast", color="tab:orange", linewidth=2)
        ax_ts.plot(times, nam_ghi, label="NAM Forecast", color="tab:green", linewidth=2, linestyle="--")
        ax_ts.set_ylabel("GHI [W/m²]")
        ax_ts.set_title(
            f"GHI Forecast Comparison — Fold {fold}, Sample #{sample_order}, Split: {split}"
        )
        ax_ts.grid(alpha=0.3)
        ax_ts.legend(loc="upper right")

        ax_scatter = axes[1]
        ax_scatter.scatter(
            daylight_actual,
            daylight_model,
            label="Model vs Measured",
            color="tab:orange",
            alpha=0.6,
        )
        ax_scatter.scatter(
            daylight_actual,
            daylight_nam,
            label="NAM vs Measured",
            color="tab:green",
            alpha=0.6,
            marker="s",
        )
        if not daylight_actual.empty:
            diag_min = min(
                daylight_actual.min(), daylight_model.min(), daylight_nam.min()
            )
            diag_max = max(
                daylight_actual.max(), daylight_model.max(), daylight_nam.max()
            )
            ax_scatter.plot([diag_min, diag_max], [diag_min, diag_max], "k--", alpha=0.5)
        ax_scatter.set_xlabel("Measured GHI [W/m²]")
        ax_scatter.set_ylabel("Forecast GHI [W/m²]")
        ax_scatter.set_title("Daylight Scatter — Accuracy Against Measured GHI")
        ax_scatter.grid(alpha=0.3)
        ax_scatter.legend(loc="upper left")

        if metrics:
            def fmt(key: str) -> str:
                value = metrics.get(key)
                if value is None:
                    return "n/a"
                return f"{value:.2f}"

            text_lines = [
                "Validation Metrics:",
                f"MAE (Model GHI): {fmt('model_mae_ghi')} W/m²",
                f"MAE (NAM GHI): {fmt('nam_mae_ghi')} W/m²",
                f"RMSE (Model GHI): {fmt('model_rmse_ghi')} W/m²",
                f"RMSE (NAM GHI): {fmt('nam_rmse_ghi')} W/m²",
                f"Skill (MAE, GHI): {fmt('mae_skill_ghi_pct')} %",
                f"Skill (RMSE, GHI): {fmt('rmse_skill_ghi_pct')} %",
            ]
            metrics_box = "\n".join(text_lines)
            ax_scatter.text(
                0.99,
                0.01,
                metrics_box,
                transform=ax_scatter.transAxes,
                fontsize=10,
                va="bottom",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85),
            )

        filename = self.exp_dir / "plots" / f"nam_comparison_fold_{fold}_{split}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    
    def save_training_history(self, history: Dict, fold: int = 0):
        """Save training history"""
        filename = f"history_fold_{fold}.json"
        with open(self.exp_dir / "metrics" / filename, 'w') as f:
            json.dump(_jsonable_dict(history), f)  # if history values can be numpy
    
    def plot_training_history(self, history: Dict, fold: int = 0):
        """Plot and save training curves.
        
        Generates the classic learning curve plot. This is one of the most important diagnostic tools in deep learning, as it instantly 
        shows if the model is overfitting (training loss goes down, validation loss goes up) or underfitting.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        if 'train_mae' in history:
            ax2.plot(history['train_mae'], label='Training MAE', color='blue')
            ax2.plot(history['val_mae'], label='Validation MAE', color='red')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)

        def _last(values: List[float]) -> Optional[float]:
            return float(values[-1]) if values else None

        train_rmse = None
        val_rmse = None
        if 'train_loss' in history and history['train_loss']:
            train_rmse = float(np.sqrt(history['train_loss'][-1]))
        if 'val_loss' in history and history['val_loss']:
            val_rmse = float(np.sqrt(history['val_loss'][-1]))

        train_mae = _last(history.get('train_mae', []))
        val_mae = _last(history.get('val_mae', []))

        summary_lines = [
            "Final Validation Snapshot:",
            f"MAE (Train): {train_mae:.3f}" if train_mae is not None else "MAE (Train): n/a",
            f"MAE (Validation): {val_mae:.3f}" if val_mae is not None else "MAE (Validation): n/a",
            f"RMSE (Train): {train_rmse:.3f}" if train_rmse is not None else "RMSE (Train): n/a",
            f"RMSE (Validation): {val_rmse:.3f}" if val_rmse is not None else "RMSE (Validation): n/a",
        ]

        ax2.text(
            0.99,
            0.01,
            "\n".join(summary_lines),
            transform=ax2.transAxes,
            fontsize=10,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "plots" / f"training_history_fold_{fold}.png")
        plt.close()
    
    def save_predictions_plot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            fold: int = 0, split: str = "test", sample_size: int = 500):
        """Save predictions vs actual plot.
        
        Creates a visual "sanity check" of the model's outputs. The scatter plot, especially with the red y=x line, is a powerful way to see if 
        the predictions are systematically biased (e.g., always predicting too low or too high).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Flatten if needed
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Sample if too large
        if len(y_true_flat) > sample_size:
            idx = np.random.choice(len(y_true_flat), sample_size, replace=False)
            y_true_plot = y_true_flat[idx]
            y_pred_plot = y_pred_flat[idx]
        else:
            y_true_plot = y_true_flat
            y_pred_plot = y_pred_flat
        
        # Time series plot
        ax1.plot(y_true_plot[:100], label='Actual', color='blue', alpha=0.7)
        ax1.plot(y_pred_plot[:100], label='Predicted', color='red', alpha=0.7)
        ax1.set_title(f'Time Series Comparison (first 100 points)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('CSI GHI')
        ax1.legend()
        ax1.grid(True)
        
        # Scatter plot
        ax2.scatter(y_true_plot, y_pred_plot, alpha=0.5, s=10)
        ax2.plot([y_true_plot.min(), y_true_plot.max()], 
                [y_true_plot.min(), y_true_plot.max()], 
                'r--', lw=2)
        ax2.set_title(f'Predicted vs Actual')
        ax2.set_xlabel('Actual CSI GHI')
        ax2.set_ylabel('Predicted CSI GHI')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "plots" / f"predictions_fold_{fold}_{split}.png")
        plt.close()
    
    def summarize_results(self):
        """Generate summary of all folds.
        Provide the final, high-level summary of the entire experiment, especially when using cross-validation.

        After all the individual folds of a cross-validation run are complete, this method gathers the metrics from each fold. 
        It then calculates the mean and standard deviation for each metric. 
        This is far more reliable than looking at a single train/test split, as it gives you a robust estimate of the model's 
        expected performance and its consistency across different slices of the data. 
        The final summary is saved as a single summary.json file, which serves as the experiment's "abstract."
        """
        summary = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "fold_results": self.results,
            "average_metrics": {}
        }
        
        # Calculate average metrics across folds
        if self.results:
            all_metrics = {}
            for fold, fold_results in self.results.items():
                for split, metrics in fold_results.items():
                    if split not in all_metrics:
                        all_metrics[split] = {}
                    for metric, value in metrics.items():
                        if metric not in all_metrics[split]:
                            all_metrics[split][metric] = []
                        all_metrics[split][metric].append(value)
            
            # Average
            for split, metrics in all_metrics.items():
                summary["average_metrics"][split] = {}
                for metric, values in metrics.items():
                    summary["average_metrics"][split][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
        
        # Save summary
        with open(self.exp_dir / "summary.json", 'w') as f:
            json.dump(_jsonable(summary), f, indent=2)  
        
        return summary
