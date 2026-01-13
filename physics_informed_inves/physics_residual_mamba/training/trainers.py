"""
Training functions for solar power forecasting models.

Provides:
- train_one_epoch_physics: Train physics-informed model for one epoch
- train_one_epoch_base: Train base model for one epoch
- run_physics_residual_cv: Cross-validation training for physics model
- run_base_mamba_cv: Cross-validation training for base model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List

from ..evaluation.evaluators import evaluate_fold_physics, evaluate_fold_base
from ..utils import get_device, set_random_seed


def train_one_epoch_physics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, Dict[str, float]]:
    """
    Train one epoch with the PhysicsResidualMamba model.
    
    Args:
        model: PhysicsResidualMamba model
        loader: Training data loader
        optimizer: Optimizer
        criterion: PhysicsPVLoss criterion
        device: torch device
        
    Returns:
        tuple: (avg_loss, phys_logs_accum)
            - avg_loss: Average loss over all batches
            - phys_logs_accum: Dictionary with loss components and physics params
    """
    model.train()
    total_loss = 0.0
    phys_logs_accum = {"L_data": 0.0, "L_night": 0.0, "L_mono": 0.0}
    n_batches = 0
    
    for batch in loader:
        x_past, y_future, x_future = batch[0], batch[1], batch[2]
        x_past = x_past.to(device)
        y_future = y_future.to(device)
        x_future = x_future.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: get both total prediction and physics prediction
        preds, p_physics = model(x_past, x_future)
        
        # Compute loss
        loss, logs = criterion(preds, y_future, x_future)
        
        # Accumulate log values
        phys_logs_accum["L_data"] += logs["L_data"]
        phys_logs_accum["L_night"] += logs["L_night"]
        phys_logs_accum["L_mono"] += logs["L_mono"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    for k in phys_logs_accum:
        phys_logs_accum[k] /= max(1, n_batches)
    
    # Add physics parameters to logs
    phys_logs_accum.update(model.get_physics_params())
    
    return avg_loss, phys_logs_accum


def train_one_epoch_base(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train one epoch with base model (Mamba, LSTM, etc.).
    
    Args:
        model: Base model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: torch device
        
    Returns:
        Average loss over all batches
    """
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        x_past, y_future, x_future = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        # NaN Guard
        if torch.isnan(x_past).any() or torch.isnan(y_future).any():
            continue
        
        optimizer.zero_grad()
        preds = model(x_past, x_future)
        
        loss = criterion(preds, y_future)
        
        if torch.isnan(loss):
            print("WARNING: NaN loss detected in Base Mamba. Skipping batch.")
            optimizer.zero_grad()
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / max(1, len(loader))


def run_physics_residual_cv(
    df,
    configs,
    use_scaler: bool = True,
    n_splits: Optional[int] = None,
    random_seed: int = 42
) -> tuple[List[Dict], Dict]:
    """
    Run rolling cross-validation with PhysicsResidualMamba.
    
    Args:
        df: DataFrame with features
        configs: PhysicsResidualMambaConfigs object
        use_scaler: If True, create FeatureScaler for physics layer
        n_splits: Optional override for number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (results, history)
            - results: List of result dicts per fold
            - history: Dict with training history per fold
    """
    from ..data import prepare_rolling_folds
    from ..models import PhysicsResidualMamba
    from ..losses import PhysicsPVLoss
    from ..models.physics_layer import FeatureScaler
    from ..visualization import plot_loss_curve
    
    # Set random seed for reproducibility
    set_random_seed(random_seed)
    
    results = []
    history = {}
    device = get_device()
    
    # Set device in configs
    configs.device = device
    
    # Resolve n_splits override
    final_n_splits = n_splits if n_splits is not None else configs.n_splits
    
    print(f"DEBUG: n_splits={final_n_splits}, batch_size={configs.batch_size}")
    
    fold_gen = prepare_rolling_folds(
        df, configs.PAST_INPUT_COLS, configs.TARGET_COL, configs.FUTURE_INPUT_COLS,
        n_splits=final_n_splits,
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        batch_size=configs.batch_size
    )
    
    for i, (train_loader, test_loader, scaler_stats) in enumerate(fold_gen):
        fold = i + 1
        print(f"\n=== Training Fold {fold}/{final_n_splits} ===")
        history[fold] = {'train': [], 'val': [], 'physics_params': [], 'scaler_stats': scaler_stats}
        
        # Create PhysicsResidualMamba model
        model = PhysicsResidualMamba(configs)
        
        # Set up FeatureScaler for physics layer
        if use_scaler:
            scaler = FeatureScaler(scaler_stats['means'], scaler_stats['stds'])
            # Do NOT set scaler on physics layer (inputs are already raw)
            print(f"  Scaler stats: G_mean={scaler_stats['means']['nwp_globalirrad']:.1f}, "
                  f"T_mean={scaler_stats['means']['nwp_temperature']:.1f}, "
                  f"W_mean={scaler_stats['means']['nwp_windspeed']:.2f}")
        
        # Move model to device
        model = model.to(device)
        
        # Optimizer placeholder (will be set in loop for warmup)
        optimizer = None
        
        # Create physics-informed loss
        criterion = PhysicsPVLoss(configs, configs.FUTURE_INPUT_COLS).to(device)
        
        for epoch in range(configs.epochs):
            # --- Warmup Strategy: Freeze Mamba for first 5 epochs ---
            if epoch < 5:
                # Freeze Mamba
                for param in model.mamba_model.parameters():
                    param.requires_grad = False
                # Optimize only physics layer (higher LR for rapid geometric adaptation)
                optimizer = optim.Adam(model.physics_layer.parameters(), lr=1e-3)
            else:
                # Unfreeze Mamba
                for param in model.mamba_model.parameters():
                    param.requires_grad = True
                # Joint optimization (fine-tuning LR)
                optimizer = optim.Adam(model.parameters(), lr=5e-4)
            
            train_loss, train_logs = train_one_epoch_physics(
                model, train_loader, optimizer, criterion, device
            )
            
            val_loss, val_rmse, val_mae, val_logs, _, _, _, _ = evaluate_fold_physics(
                model, test_loader, criterion, device, configs
            )
            
            history[fold]['train'].append(train_loss)
            history[fold]['val'].append(val_loss)
            history[fold]['physics_params'].append(model.get_physics_params())
            
            if (epoch + 1) % 5 == 0:
                phys = model.get_physics_params()
                msg = (
                    f"Epoch {epoch+1:03d} | "
                    f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
                    f"RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | "
                    f"L_data={val_logs['L_data']:.4f} L_night={val_logs['L_night']:.4f} L_mono={val_logs['L_mono']:.4f} | "
                    f"η={phys['eta']:.4f} U0={phys['U0']:.2f} γ={phys['gamma']:.5f}"
                )
                print(msg)
        
        # Final Evaluation with full arrays
        final_loss, final_rmse, final_mae, _, final_preds, final_act, final_base, final_nwp = evaluate_fold_physics(
            model, test_loader, criterion, device, configs
        )
        
        # Calculate Smart Persistence Metrics
        sp_rmse = np.sqrt(np.nanmean((final_base - final_act.squeeze())**2))
        sp_mae = np.nanmean(np.abs(final_base - final_act.squeeze()))
        
        # Calculate NWP Metrics
        nwp_rmse = np.sqrt(np.nanmean((final_nwp - final_act.squeeze())**2))
        
        print(f"--> Fold {fold} Finished: RMSE={final_rmse:.4f}, SP={sp_rmse:.4f}, NWP={nwp_rmse:.4f}")
        
        results.append({
            'fold': fold,
            'rmse': final_rmse,
            'mae': final_mae,
            'sp_rmse': sp_rmse,
            'sp_mae': sp_mae,
            'nwp_rmse': nwp_rmse,
            'physics_params': model.get_physics_params(),
            'scaler_stats': scaler_stats,
            'preds': final_preds,
            'actuals': final_act,
            'baseline': final_base,
            'nwp': final_nwp
        })
        
        plot_loss_curve(history[fold]['train'], history[fold]['val'], fold)
    
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    
    print("\n=== Final Cross-Validation Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    # Print final physics parameters from last fold
    print("\nLearned Physics Parameters (last fold):")
    final_params = results[-1]['physics_params']
    for k, v in final_params.items():
        print(f"  {k}: {v:.6f}")
    
    return results, history


def run_base_mamba_cv(
    df,
    configs,
    n_splits: Optional[int] = None,
    random_seed: int = 42
) -> List[Dict]:
    """
    Run Cross-Validation for Base Mamba (No Physics).
    
    Args:
        df: DataFrame with features
        configs: BaseMambaConfigs object
        n_splits: Optional override for number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        List of result dicts per fold
    """
    from ..data import prepare_rolling_folds
    from ..models import MMPISSM_Model_01
    
    # Set random seed
    set_random_seed(random_seed)
    
    results = []
    device = get_device()
    configs.device = device
    
    final_n_splits = n_splits if n_splits is not None else configs.n_splits
    
    print(f"\nTRAINING BASE MAMBA (MSE Loss) | Folds: {final_n_splits}")
    
    fold_gen = prepare_rolling_folds(
        df, configs.PAST_INPUT_COLS, configs.TARGET_COL, configs.FUTURE_INPUT_COLS,
        n_splits=final_n_splits,
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        batch_size=configs.batch_size
    )
    
    for i, (train_loader, test_loader, _) in enumerate(fold_gen):
        fold = i + 1
        print(f"  Fold {fold}/{final_n_splits}...", end="", flush=True)
        
        model = MMPISSM_Model_01(configs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.SmoothL1Loss()  # Robust loss
        
        for epoch in range(configs.epochs):
            train_one_epoch_base(model, train_loader, optimizer, criterion, device)
        
        _, rmse, mae, preds, acts = evaluate_fold_base(model, test_loader, criterion, device)
        print(f" Done. RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        results.append({
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'preds': preds,
            'actuals': acts,
            'model': 'Base Mamba'
        })
    
    return results
