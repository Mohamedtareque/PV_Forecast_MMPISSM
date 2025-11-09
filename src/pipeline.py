"""
Contains the high-level class that orchestrates the entire end-to-end process for a single experiment run.

class SolarForecastingPipeline
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict, List, Tuple, Optional, Any
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import DataManager, ExperimentTracker
from .models import ImprovedLSTM, MATNet
from .engine import train_model, evaluate_model
from .config import (
    PROCESSED_DATA_PATH,
    SITE_LATITUDE,
    SITE_LONGITUDE,
    SITE_ALTITUDE,
    SITE_TIMEZONE,
)
from .evaluation_utils import (
    SiteLocation,
    build_comparison_table,
    compute_nam_comparison_metrics,
)

from .utils import (
    fit_feature_scaler, transform_X_with_scaler,
    fit_target_scaler, transform_Y_with_scaler
)

from src.benchmarking import run_benchmark_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def _fmt_metric(v):
#     if v is None:
#         return "None"
#     # scalar-like
#     if np.isscalar(v):
#         return f"{float(v):.6f}"
#     v = np.asarray(v)
#     if v.ndim == 0:
#         return f"{float(v):.6f}"
#     # small arrays: print values; big arrays: print shape only
#     if v.size <= 10:
#         flat = v.reshape(-1)
#         return "[" + ", ".join(f"{float(x):.6f}" for x in flat) + "]"
#     return f"array{tuple(v.shape)}"


# Main Pipeline

class SolarForecastingPipeline:
    """Complete pipeline for solar forecasting experiments"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_manager = DataManager(
            data_dir=config.get('data_dir', 'data'),
            splits_dir=config.get('splits_dir', 'splits')
        )
        self.tracker = ExperimentTracker(
            experiment_name=config.get('experiment_name', 'solar_forecast')
        )
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Optional: provide the full fixed-grid dataframe used to build arrays.
        # This lets us compute NAM and Naive baselines aligned to target timestamps.
        self.reference_df = None  # set this from the runner: pipeline.reference_df = fixed_df
        # Where to save benchmark comparison CSVs/plots
        self.benchmark_dirname = "benchmarks"
        
    def create_model(self, model_type: str, input_size: int, horizon_days: int = 1, steps_per_day: int = None) -> nn.Module:
        model_config = self.config.get('model_config', {})

        if model_type == 'LSTM':
            return ImprovedLSTM(
                input_size=input_size,
                steps_per_day=steps_per_day,                                 # <-- NEW
                hidden_size=model_config.get('hidden_size', 256),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                activation_function=model_config.get('activation_function', "ReLU"),
                bidirectional=model_config.get('bidirectional', True),
                horizon_days=horizon_days,
                attn_heads=model_config.get('num_heads', 8),                 # optional: aligns with MATNet style
                use_attention=model_config.get('use_attention', False),      # <-- NEW
            )
        elif model_type == 'MATNet':
            return MATNet(
                input_size=input_size,
                d_model=model_config.get('d_model', 256),
                num_heads=model_config.get('num_heads', 8),
                num_encoder_layers=model_config.get('num_layers', 4),
                dropout=model_config.get('dropout', 0.2),
                horizon_days=horizon_days
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_fold(self, X: np.ndarray, Y: np.ndarray,
                 train_idx: np.ndarray, val_idx: np.ndarray,
                 fold_id: int,
                 labels_index: Optional[pd.Index] = None,
                 reference_df: Optional[pd.DataFrame] = None) -> Dict:
        """Run training for a single fold"""
        
        logger.info(f"\n=== Running Fold {fold_id} ===")
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Split data
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # --- Feature scaling (fit on TRAIN only) ---
        feature_scaler = fit_feature_scaler(X_train)
        X_train = transform_X_with_scaler(X_train, feature_scaler)
        X_val   = transform_X_with_scaler(X_val,   feature_scaler)

        

        # Becasue our target is "Clear Sky Index" which is already scaled.
        target_scaler = None
        if self.config.get("scale_target", False):
            target_scaler = fit_target_scaler(Y_train)
            Y_train = transform_Y_with_scaler(Y_train, target_scaler)
            Y_val   = transform_Y_with_scaler(Y_val,   target_scaler)
        
        logger.info(f"X_train scaled range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        logger.info(f"Y_train scaled range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")

        self.scaler_info = {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        }
        
        

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train).float()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(Y_val).float()
        )

        train_loader = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.config["batch_size"], shuffle=False)
        
        # Create model
        input_size   = X.shape[-1]    # features
        steps_per_day = X.shape[2]    # grid steps per day (e.g., 24, 48)
        horizon_days = Y.shape[1]

        model = self.create_model(
            self.config['model_type'],
            input_size,
            horizon_days,
            steps_per_day=steps_per_day,   # <-- pass it through
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        loss_function_name = self.config.get('loss_function', 'MSE')
        # Train model
        history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=self.config.get('num_epochs', 100),
            learning_rate=self.config.get('learning_rate', 0.001),
            device=self.device,
            loss_function=loss_function_name,
            early_stopping_patience=self.config.get('early_stopping_patience', 20)
        )
        
        # Evaluate
        scaler_info = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
        
        val_metrics, val_pred, val_true = evaluate_model(
            model,
            val_loader,
            device=self.device,
            scaler_info=scaler_info,
        )

        for k in ("MAPE", "sMAPE", "sMAPE_%", "per_horizon_MAPE"):
            val_metrics.pop(k, None)

        comparison_df = None
        # Optional legacy NAM block (off by default)
        if self.config.get("enable_legacy_nam_compare", False):
            try:
                ref_df = reference_df
                if isinstance(ref_df.index, pd.DatetimeIndex) and (ref_df.index.name == "target_time"):
                    tmp = ref_df.reset_index()
                    tmp["date"] = tmp["target_time"].dt.normalize()
                    tmp["bin_id"] = tmp.groupby("date").cumcount()
                    if "nam_csi" not in tmp.columns and {"nam_ghi", "clear_sky_ghi"}.issubset(tmp.columns):
                        tmp["nam_csi"] = tmp["nam_ghi"] / tmp["clear_sky_ghi"]
                    ref_df = tmp.set_index(["date", "bin_id"]).sort_index()

                comparison_df = build_comparison_table(
                    val_pred, val_true, val_idx, labels_index, ref_df, fold_id
                )
                nam_metrics = compute_nam_comparison_metrics(comparison_df)
                if nam_metrics:
                    val_metrics.update(nam_metrics)
            except Exception as exc:
                logger.warning("Legacy NAM compare failed (disabled by default): %s", exc)
        
        
        else:
            if labels_index is None:
                logger.warning(
                    "Labels index missing; skipping NAM comparison for fold %s.",
                    fold_id,
                )
            elif reference_df is None:
                logger.warning(
                    "Reference dataframe missing; skipping NAM comparison for fold %s.",
                    fold_id,
                )
        
        # Save results
        
        
        
        self.tracker.plot_training_history(history, fold=fold_id)
        self.tracker.save_predictions_plot(val_true, val_pred, fold=fold_id, split='validation')

        self.tracker.save_model(model, fold=fold_id, is_best=True)
        self.tracker.save_training_history(history, fold=fold_id)
        self.tracker.save_metrics(val_metrics, fold=fold_id, split='validation')

        # --- Build a benchmark report vs NAM & Naive on CSI and GHI ---
        try:
            import os

            if labels_index is None:
                logger.warning("Labels index missing; skipping benchmark for fold %s.", fold_id)
            elif self.reference_df is None:
                logger.warning("reference_df is None; set pipeline.reference_df = fixed_df before run() to enable benchmarks.")
            else:
                # 1) Flatten truth/preds from the evaluation you already did
                #    val_true / val_pred are in CSI space (thanks to scaler_info)
                y_true_csi = np.asarray(val_true, dtype=float).reshape(-1)
                y_pred_csi = np.asarray(val_pred, dtype=float).reshape(-1)

                # 2) Steps-per-day: prefer config, else infer from X shape
                spd = self.config.get("model_config", {}).get("steps_per_day", 11)
                if spd is None:
                    spd = int(X.shape[2])  # (N, history_days, steps_per_day, features)

                # 3) Build label_times of length N_val * spd
                #    Take validation *days* (one per sample) and expand by canonical intraday times
                val_days = pd.to_datetime(labels_index[val_idx], utc=True).normalize()

                ref_idx = self.reference_df.index
                if not isinstance(ref_idx, pd.DatetimeIndex):
                    ref_idx = pd.to_datetime(ref_idx, utc=True, errors="coerce")

                # Make tz-aware UTC
                if ref_idx.tz is None:
                    ref_idx = ref_idx.tz_localize("UTC")
                else:
                    ref_idx = ref_idx.tz_convert("UTC")

                # Pick the most common times-of-day; sort chronologically
                tod_counts = pd.Series(ref_idx.time).value_counts()
                top_times_sorted = sorted(list(tod_counts.index[:spd]))

                label_times_list = []
                for d in val_days:
                    d_date = d.date()  # python date
                    for t in top_times_sorted:
                        # combine as naive, then localize to UTC (no tz kwarg)
                        ts_naive = pd.Timestamp.combine(d_date, t)
                        ts = pd.Timestamp(ts_naive).tz_localize("UTC")
                        label_times_list.append(ts)

                label_times = pd.DatetimeIndex(label_times_list)

                if len(label_times) != y_true_csi.size:
                    raise RuntimeError(
                        f"Label-times length mismatch: got {len(label_times)} vs {y_true_csi.size} "
                        f"(expected N_val*steps_per_day). Check steps_per_day and reference intraday times."
                    )

                # 4) Column mapping (ensure clear-sky column is correct!)
                colmap = {
                    "truth_csi": self.config.get("truth_csi_col", "actual_csi"),
                    "truth_ghi": self.config.get("truth_ghi_col", None),
                    "ghi_cs":    self.config.get("ghi_cs_col",  "clear_sky_ghi"),
                    "nam_ghi":   self.config.get("nam_ghi_col", "nam_ghi"),
                    "nam_csi":   self.config.get("nam_csi_col", None),
                }

                # 5) Output dir and run
                bench_root = os.path.join(self.tracker.exp_dir, self.benchmark_dirname)
                fold_dir = os.path.join(bench_root, f"fold_{int(fold_id)}")
                os.makedirs(fold_dir, exist_ok=True)

                logger.info("Benchmark colmap: %s", colmap)
                logger.info("ref_df columns (first 10): %s", list(self.reference_df.columns)[:10])
                logger.info("label_times: %d  | y_true: %d | y_pred: %d", len(label_times), y_true_csi.size, y_pred_csi.size)
                out = run_benchmark_report(
                    out_dir=fold_dir,
                    label_times=label_times,
                    y_true_csi=y_true_csi,
                    y_pred_csi=y_pred_csi,
                    reference_df=self.reference_df,
                    column_map=colmap,
                    tag="validation",
                    make_plots=True,
                );
                # Add skill vs NAM / Naive (RMSE & MAE) to metrics so they appear in the summary
                # svnam = out.get("Skill_vs_NAM", {})
                # svnv  = out.get("Skill_vs_Naive", {})
                # Add “forecast skill vs Smart Persistence” for BOTH Ours and NAM
                svsp_ours = out.get("Skill_vs_SP_Ours", {})
                svsp_nam  = out.get("Skill_vs_SP_NAM",  {})
                # Keep names short and clear
                skill_updates = {
                    # Ours vs SP
                    "GHI_RMSE_skill_Ours_vs_SP": svsp_ours.get("GHI_RMSE_Skill_Ours_vs_SP"),
                    "GHI_MAE_skill_Ours_vs_SP":  svsp_ours.get("GHI_MAE_Skill_Ours_vs_SP"),
                    "CSI_RMSE_skill_Ours_vs_SP": svsp_ours.get("CSI_RMSE_Skill_Ours_vs_SP"),
                    "CSI_MAE_skill_Ours_vs_SP":  svsp_ours.get("CSI_MAE_Skill_Ours_vs_SP"),
                    # NAM vs SP
                    "GHI_RMSE_skill_NAM_vs_SP":  svsp_nam.get("GHI_RMSE_Skill_NAM_vs_SP"),
                    "GHI_MAE_skill_NAM_vs_SP":   svsp_nam.get("GHI_MAE_Skill_NAM_vs_SP"),
                    "CSI_RMSE_skill_NAM_vs_SP":  svsp_nam.get("CSI_RMSE_Skill_NAM_vs_SP"),
                    "CSI_MAE_skill_NAM_vs_SP":   svsp_nam.get("CSI_MAE_Skill_NAM_vs_SP"),
                }
                val_metrics.update({k: float(v) for k, v in skill_updates.items() if v is not None and np.isfinite(v)})
        except Exception as e:
            logger.warning("Benchmark report generation failed for fold %s: %s", fold_id, e)
        
        return {
            'history': history,
            'metrics': val_metrics,
            'model': model,
            'scalers': scaler_info
        }
    
    def run(self):
        """Run complete pipeline"""
        
        # Save configuration
        self.tracker.save_config(self.config)
        
        # Load data
        logger.info("Loading data...")
        X, Y, labels, metadata = self.data_manager.load_arrays(
            filename_prefix=self.config.get('data_prefix', 'kbins_data')
        )
        saved_features = metadata.get("feature_cols")
        expected_features = self.config.get("feature_cols")
        if expected_features is not None:
            if saved_features is not None and list(expected_features) != list(saved_features):
                raise ValueError(
                    "Feature mismatch between configuration and saved arrays. "
                    f"Config expects {expected_features}, arrays contain {saved_features}."
                )
        else:
            self.config["feature_cols"] = saved_features
        if saved_features is None:
            logger.warning("Feature metadata missing in saved arrays; proceeding without validation.")

        labels_index = labels.index if labels is not None else None
        if labels_index is None:
            logger.warning(
                "Labels dataframe not available; NAM comparison metrics will be skipped."
            )
            self.reference_df = None
        else:
            # Loading Raw data that uesed to cook the X, Y, Labels Tensors.
            from .evaluation_utils import _load_processed_dataframe
            csv_path = metadata.get("input_csv", PROCESSED_DATA_PATH) 
            base_df = _load_processed_dataframe(csv_path) 

            try:
                from .evaluation_utils import build_reference_from_existing
                logger.info("Building simple reference from existing columns (no pvlib, no regridding)...")
                # Use the SAME merged dataframe you used to build arrays (before windowing)
                self.reference_df = build_reference_from_existing(
                    base_df,                         # <-- processed/merged modeling df
                    time_col="measurement_time",
                    nam_time_col="nam_target_time",
                    meas_ghi_col="ghi",
                    nam_ghi_col="nam_ghi",
                    cs_ghi_col="GHI_cs",
                    actual_csi_col="CSI_ghi",
                )
                # enforce the correct default mapping if user didn't override
                self.config.setdefault("truth_csi_col", "actual_csi")
                self.config.setdefault("truth_ghi_col", "actual_ghi")      # optional; else reconstruct from CSI*clear_sky_ghi
                self.config.setdefault("ghi_cs_col",  "clear_sky_ghi")
                self.config.setdefault("nam_ghi_col", "nam_ghi")
                self.config.setdefault("nam_csi_col", "nam_csi")           # optional; benchmark will compute if missing
                logger.info("Reference dataframe prepared (%d rows).", len(self.reference_df))
            except Exception as exc:
                logger.warning("Failed to construct simple reference: %s", exc)
                self.reference_df = None
        
        # Load splits
        splits_data = self.data_manager.load_rolling_splits(
            self.config.get('splits_file', 'rolling_origin_splits.json')
        )
        
        # Run each fold
        fold_results = []
        max_folds = self.config.get('max_folds', len(splits_data['folds']))
        
        for i, fold_data in enumerate(splits_data['folds'][:max_folds]):
            fold_id = fold_data['fold_id']
            
            # Get indices for this fold
            if labels is not None:
                train_idx, val_idx = self.data_manager.get_fold_indices(
                    X, labels, fold_data
                )
            else:
                # If no labels, use the sizes directly
                train_size = fold_data['train_size']
                val_size = fold_data['val_size']
                train_idx = np.arange(train_size)
                val_idx = np.arange(train_size, train_size + val_size)
            
            # Run fold
            fold_result = self.run_fold(
                X,
                Y,
                train_idx,
                val_idx,
                fold_id,
                labels_index=labels_index,
                reference_df=self.reference_df,
            )
            fold_results.append(fold_result)
        
        # Summarize results
        summary = self.tracker.summarize_results()
        
        logger.info("\n=== Experiment Summary ===")
        if 'average_metrics' in summary and 'validation' in summary['average_metrics']:
            logger.info("Average Validation Metrics:")
            for metric, stats in summary['average_metrics']['validation'].items():
                logger.info(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
        
        return fold_results, summary
