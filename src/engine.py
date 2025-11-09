"""
The "engine" is responsible for the actual mechanics of training (forward pass, backward pass, optimization, evaluation). 
The pipeline will use the engine, but it doesn't need to know the fine details of its operation.
Isolates the logic of a training loop from the rest of the pipeline.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
import logging
import torch
import torch.nn.functional as F 
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- add these helpers near the top of engine.py ---

def _move_to_device(obj, device):
    """Recursively move tensors (or collections of tensors) to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj  # leave non-tensors as-is

def _extract_xy(batch):
    """
    Accept (X, Y), [X, Y], or dict with common keys.
    Returns (X_tensor, Y_tensor).
    """
    # tuple/list
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]

    # dict patterns
    if isinstance(batch, dict):
        # feature keys to try
        x_keys = ("X", "x", "features", "inputs", "data")
        y_keys = ("Y", "y", "target", "labels")

        Xb = None
        for k in x_keys:
            if k in batch:
                Xb = batch[k]
                break
        Yb = None
        for k in y_keys:
            if k in batch:
                Yb = batch[k]
                break

        if Xb is not None and Yb is not None:
            return Xb, Yb

    raise TypeError(
        "Batch format not recognized. Expected (X, Y) or dict with keys like "
        "{'X'/'features', 'Y'/'target'}."
    )



# Training Functions

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 100, learning_rate: float = 0.001, loss_function:str = "MSE",
                device: str = 'cpu', early_stopping_patience: int = 20) -> Dict:
    """Train model with early stopping"""
    
    device = torch.device(device)
    model = model.to(device)
        
    if loss_function.lower() == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
        logger.info("Using HuberLoss (Smooth L1 Loss)")

    else:
        criterion = nn.MSELoss()
        logger.info("Using MSELoss")
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_batches = 0
        

        for batch in train_loader:
            Xb, Yb = _extract_xy(batch)
            Xb, Yb = _move_to_device(Xb, device), _move_to_device(Yb, device)

            # Prevent gradients accumulation
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, Yb)

            loss.backward()
            # prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            
            train_mae  += F.l1_loss(outputs, Yb).item()
            train_batches += 1


        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                Xb, Yb = _extract_xy(batch)
                Xb, Yb = _move_to_device(Xb, device), _move_to_device(Yb, device)

                outputs = model(Xb)
                loss = criterion(outputs, Yb)

                val_loss += loss.item()
                val_mae  += F.l1_loss(outputs, Yb).item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss   = val_loss   / max(val_batches, 1)
        avg_train_mae  = train_mae  / max(train_batches, 1)
        avg_val_mae    = val_mae    / max(val_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['lr'].append(current_lr)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Logging
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                f"Train MAE: {avg_train_mae:.6f}, Val MAE: {avg_val_mae:.6f}, "
                f"LR: {current_lr:.6f}"
            )
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


# engine.py
import numpy as np
import torch

def _maybe_inverse_scale(y, scaler):
    """
    automatically reverse the data scaling that was applied to the target variable (y) before model training.
    """
    if scaler is None:
        return y
    y2 = y.reshape(-1, 1)
    y2 = scaler.inverse_transform(y2).reshape(*y.shape)
    return y2

def _compute_metrics(y_true, y_pred, eps=1e-8):
    """
    takes the true values (y_true), the predicted values (y_pred), and an eps (epsilon) argument.
    eps=1e-8 is a very small number used to prevent division-by-zero errors, which is a critical part of the "safe MAPE" calculation.
    """
    # Works for shapes (N,H,K) or (N,H,K,1); squeeze last dim if singleton
    # squeezes the array, removing that unnecessary dimension to make it (Samples, Horizon, Features)
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    mae  = np.mean(np.abs(y_pred - y_true))
    # gives a much higher weight to large errors.
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    # safe MAPE for irradiance/CSI-like targets
    mape = np.mean(np.abs((y_pred - y_true) / (np.clip(np.abs(y_true), eps, None))))
    # per-horizon (if H>1)
    # returning an array of MAE values, one for each step in your forecast horizon
    # see exactly how model's accuracy degrades as it predicts further into the future
    per_h_mae = np.mean(np.abs(y_pred - y_true), axis=(0,2)) if y_true.ndim == 3 else None
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape),
            "per_horizon_MAE": per_h_mae}

def evaluate(model, loader, scaler_info=None, device="cuda"):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch in loader:
            # Xb: input features, Yb: true target values
            Xb, Yb = _extract_xy(batch)                      
            Xb, Yb = _move_to_device(Xb, device), _move_to_device(Yb, device)
            # Pb: Predicted Batch: model's output
            Pb = model(Xb)
            y_true_all.append(Yb.detach().cpu().numpy())
            y_pred_all.append(Pb.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    ts = (scaler_info or {}).get("target_scaler", None)
    # ts = scaler_info
    y_true = _maybe_inverse_scale(y_true, ts)
    y_pred = _maybe_inverse_scale(y_pred, ts)

    metrics = _compute_metrics(y_true, y_pred)
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}

def evaluate_model(model, loader, device="cuda", scaler_info=None):
    out = evaluate(model, loader, scaler_info=scaler_info, device=device)
    return out["metrics"], out["y_pred"], out["y_true"]

