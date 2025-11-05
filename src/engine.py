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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mean_absolute_error(
                batch_y.cpu().numpy().flatten(),
                outputs.detach().cpu().numpy().flatten()
            )
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_mae += mean_absolute_error(
                    batch_y.cpu().numpy().flatten(),
                    outputs.cpu().numpy().flatten()
                )
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_train_mae = train_mae / train_batches
        avg_val_mae = val_mae / val_batches
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


def evaluate_model(model: nn.Module, data_loader: DataLoader,
                  scaler_info: Optional[Dict] = None,
                  device: str = 'cpu') -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics"""
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Denormalize if scaler provided
    if scaler_info and 'target_scaler' in scaler_info:
        scaler = scaler_info['target_scaler']
        predictions_flat = predictions.reshape(-1, 1)
        targets_flat = targets.reshape(-1, 1)
        
        predictions = scaler.inverse_transform(predictions_flat).reshape(predictions.shape)
        targets = scaler.inverse_transform(targets_flat).reshape(targets.shape)
    
    # Flatten for metrics calculation
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Calculate metrics
    mse  = mean_squared_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(targets_flat, predictions_flat)
    r2   = r2_score(targets_flat, predictions_flat)

    metrics = {
        'mse':  float(mse),
        'rmse': float(rmse),
        'mae':  float(mae),
        'r2':   float(r2),
    }

    mask = np.abs(targets_flat) > 1e-6
    if mask.any():
        mape = np.mean(np.abs((targets_flat[mask] - predictions_flat[mask]) / targets_flat[mask])) * 100.0
        metrics['mape'] = float(mape)
    
    return metrics, predictions, targets

