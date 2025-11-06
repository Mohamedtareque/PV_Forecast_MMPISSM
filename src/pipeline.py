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

from .utils import DataManager, ExperimentTracker
from .models import ImprovedLSTM, MATNet
from .dataset import KBinsDataset
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
    build_reference_kbin_frame,
    build_comparison_table,
    build_reference_fixedgrid_frame,
    compute_nam_comparison_metrics,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.reference_df = None
        
    def create_model(self, model_type: str, input_size: int, K_bins: int, horizon_days: int = 1) -> nn.Module:
        """Create model based on type"""
        model_config = self.config.get('model_config', {})
        
        if model_type == 'LSTM':
            return ImprovedLSTM(
                input_size=input_size,
                hidden_size=model_config.get('hidden_size', 256),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                K_bins=K_bins,
                horizon_days=horizon_days,
                bidirectional=model_config.get('bidirectional', True)
            )
        elif model_type == 'MATNet':
            return MATNet(
                input_size=input_size,
                d_model=model_config.get('d_model', 256),
                num_heads=model_config.get('num_heads', 8),
                num_encoder_layers=model_config.get('num_layers', 4),
                dropout=model_config.get('dropout', 0.2),
                K_bins=K_bins,
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
        
        # Create datasets
        train_dataset = KBinsDataset(X_train, Y_train, fit_scalers=True)
        feature_scaler, target_scaler = train_dataset.get_scalers()
        
        val_dataset = KBinsDataset(X_val, Y_val, 
                                  feature_scaler=feature_scaler,
                                  target_scaler=target_scaler, fit_scalers=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=2
        )
        
        # Create model
        input_size = X.shape[-1]  # Number of features
        K_bins = X.shape[2]  # Number of bins
        horizon_days = Y.shape[1]
        model = self.create_model(
            self.config['model_type'],
            input_size,
            K_bins, 
            horizon_days
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
            model, val_loader, scaler_info, self.device
        )

        comparison_df = None
        if labels_index is not None and reference_df is not None:
            try:
                comparison_df = build_comparison_table(
                    val_pred,
                    val_true,
                    val_idx,
                    labels_index,
                    reference_df,
                    fold_id,
                )
                nam_metrics = compute_nam_comparison_metrics(comparison_df)
                if nam_metrics:
                    val_metrics.update(nam_metrics)
            except Exception as exc:
                logger.warning(
                    "Failed to compute NAM comparison metrics for fold %s: %s",
                    fold_id,
                    exc,
                )
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
        self.tracker.save_model(model, fold=fold_id, is_best=True)
        self.tracker.save_training_history(history, fold=fold_id)
        self.tracker.save_metrics(val_metrics, fold=fold_id, split='validation')
        self.tracker.plot_training_history(history, fold=fold_id)
        self.tracker.save_predictions_plot(val_true, val_pred, fold=fold_id, split='validation')
        if comparison_df is not None and not comparison_df.empty:
            self.tracker.save_comparison_table(
                comparison_df,
                fold=fold_id,
                split='validation',
            )
            self.tracker.save_nam_comparison_plot(
                comparison_df,
                metrics=val_metrics,
                fold=fold_id,
                split='validation',
            )
        
        logger.info(f"Fold {fold_id} - Validation Metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
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
            csv_path = metadata.get("input_csv", PROCESSED_DATA_PATH)
            k_bins_meta = metadata.get("k_bins", X.shape[2])
            inferred_k = None
            if k_bins_meta is not None:
                try:
                    inferred_k = int(k_bins_meta)
                except (TypeError, ValueError):
                    inferred_k = X.shape[2] # Fallback for K-Bins case

            site_config = SiteLocation(
                latitude=self.config.get('site_latitude', SITE_LATITUDE),
                longitude=self.config.get('site_longitude', SITE_LONGITUDE),
                altitude=self.config.get('site_altitude', SITE_ALTITUDE),
                timezone=self.config.get('site_timezone', SITE_TIMEZONE),
            )

            try:
                # --- FIX 2: Add logic to call the correct function ---
                if inferred_k is not None:
                    # K-BINS PATH
                    logger.info(f"Building K-BINS reference frame (K={inferred_k})...")
                    self.reference_df = build_reference_kbin_frame(
                        csv_path,
                        inferred_k,
                        site_config,
                    )
                else:
                    # FIXED-GRID PATH 
                    logger.info("Building FIXED-GRID reference frame (K_bins is None)...")
                    # Make sure you have imported build_reference_fixedgrid_frame
                    from .evaluation_utils import build_reference_fixedgrid_frame 
                    ts_col = metadata.get("timestamp_col", "measurement_time")
                    self.reference_df = build_reference_fixedgrid_frame(
                        csv_path,
                        site_config,
                        timestamp_col=ts_col
                    )
                logger.info(
                    "Reference dataframe prepared for NAM comparison "
                    "(%d rows).",
                    len(self.reference_df),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to construct reference dataframe for NAM comparison: %s",
                    exc,
                )
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
