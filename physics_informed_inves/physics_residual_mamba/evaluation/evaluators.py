"""
Evaluation functions for solar power forecasting models.

Provides:
- evaluate_fold_physics: Evaluate physics-informed model with baseline calculations
- evaluate_fold_base: Evaluate base models (Mamba, LSTM)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


def evaluate_fold_physics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    configs: Optional[object] = None
) -> tuple[float, float, float, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate physics-informed model on a fold.
    
    Returns full arrays for visualization and computes metrics including
    Smart Persistence baseline and NWP forecasts.
    
    Args:
        model: PhysicsResidualMamba model
        loader: Validation/test data loader
        criterion: PhysicsPVLoss criterion
        device: torch device
        configs: Configuration object with column indices
        
    Returns:
        tuple: (avg_loss, rmse, mae, logs, preds, actuals, baseline, nwp)
            - avg_loss: Average loss
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - logs: Dictionary with loss components and physics parameters
            - preds: Model predictions [N, H, 1]
            - actuals: Ground truth [N, H, 1]
            - baseline: Smart Persistence baseline [N, H]
            - nwp: NWP forecast [N, H]
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_actuals = []
    all_baselines = []
    all_nwp = []
    phys_logs_accum = {"L_data": 0.0, "L_night": 0.0, "L_mono": 0.0}
    n_batches = 0
    
    # Identify indices for baseline calculation
    p_idx, p_clr_past_idx = None, None
    nwp_fut_idx = -1
    
    if configs is not None:
        try:
            p_idx = configs.PAST_INPUT_COLS.index('power')
            p_clr_past_idx = configs.PAST_INPUT_COLS.index('P_CLR')
            if 'NWP_Power_MW' in configs.FUTURE_INPUT_COLS:
                nwp_fut_idx = configs.FUTURE_INPUT_COLS.index('NWP_Power_MW')
        except ValueError:
            pass
    
    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            y_future = batch[1].to(device)
            x_future = batch[2].to(device)
            b_future = batch[3].to(device)  # [B, H, 2] -> [NWP, P_CLR]
            
            # Prediction
            preds, p_physics = model(x_past, x_future)
            
            # Loss computation
            loss, logs = criterion(preds, y_future, x_future)
            phys_logs_accum["L_data"] += logs["L_data"]
            phys_logs_accum["L_night"] += logs["L_night"]
            phys_logs_accum["L_mono"] += logs["L_mono"]
            
            total_loss += loss.item()
            
            # --- Smart Persistence Baseline Calculation ---
            if p_idx is not None:
                # Past Power (lag 24h)
                p_lag = x_past[:, :, p_idx]
                # Past Clear Sky
                clr_lag = x_past[:, :, p_clr_past_idx]
                # Future Clear Sky
                clr_fut = b_future[:, :, 1]
                
                # Smart Persistence: P_t = P_{t-24h} * (GHI_clr,t / GHI_clr,t-24h)
                baseline = p_lag * (clr_fut / (clr_lag + 1.0))
                all_baselines.append(baseline.cpu().numpy())
            else:
                all_baselines.append(np.zeros_like(preds.cpu().numpy()))
            
            # --- NWP Extraction ---
            if nwp_fut_idx >= 0:
                nwp_vals = x_future[:, :, nwp_fut_idx]
                all_nwp.append(nwp_vals.cpu().numpy())
            else:
                all_nwp.append(np.zeros_like(preds.cpu().numpy().squeeze()))
            
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_future.cpu().numpy())
            n_batches += 1
    
    preds_arr = np.concatenate(all_preds, axis=0)
    actuals_arr = np.concatenate(all_actuals, axis=0)
    baseline_arr = np.concatenate(all_baselines, axis=0)
    nwp_arr = np.concatenate(all_nwp, axis=0)
    
    preds_flat = preds_arr.reshape(-1)
    actuals_flat = actuals_arr.reshape(-1)
    
    rmse = np.sqrt(np.nanmean((preds_flat - actuals_flat) ** 2))
    mae = np.nanmean(np.abs(preds_flat - actuals_flat))
    
    avg_loss = total_loss / max(1, n_batches)
    for k in phys_logs_accum:
        phys_logs_accum[k] /= max(1, n_batches)
    
    # Add physics parameters to logs
    phys_logs_accum.update(model.get_physics_params())
    
    return avg_loss, rmse, mae, phys_logs_accum, preds_arr, actuals_arr, baseline_arr, nwp_arr


def evaluate_fold_base(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate base model (Mamba, LSTM, etc.) on a fold.
    
    Args:
        model: Base model (MMPISSM_Model_01, VanillaLSTM, etc.)
        loader: Validation/test data loader
        criterion: Loss function (MSE, SmoothL1, etc.)
        device: torch device
        
    Returns:
        tuple: (avg_loss, rmse, mae, preds, actuals)
            - avg_loss: Average loss
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - preds: Model predictions [N, H, 1]
            - actuals: Ground truth [N, H, 1]
    """
    model.eval()
    all_preds = []
    all_act = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            y_future = batch[1].to(device)
            x_future = batch[2].to(device)
            
            preds = model(x_past, x_future)
            loss = criterion(preds, y_future)
            total_loss += loss.item()
            
            all_preds.append(preds.cpu().numpy())
            all_act.append(y_future.cpu().numpy())
    
    preds_arr = np.concatenate(all_preds, axis=0)
    act_arr = np.concatenate(all_act, axis=0)
    preds_flat = preds_arr.reshape(-1)
    act_flat = act_arr.reshape(-1)
    
    rmse = np.sqrt(np.nanmean((preds_flat - act_flat)**2))
    mae = np.nanmean(np.abs(preds_flat - act_flat))
    
    return total_loss / len(loader), rmse, mae, preds_arr, act_arr
