"""
Physics-informed loss functions for solar power forecasting.

Provides:
- PhysicsPVLoss: Triple-constraint loss combining MSE, night-time penalty, and monotonicity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PhysicsPVLoss(nn.Module):
    """
    Triple-Constraint Physics PV Loss (Inspired by Applied Energy 2025).
    
    Combines multiple loss components to enforce physical constraints:
    1. MSE Loss (Data Fit): Standard mean squared error
    2. Night-time Penalty: Forces P → 0 when irradiance indicates night
    3. Monotonicity Regularization: Power should rise with sunlight
    
    Total Loss:
        L = λ_data * L_data + λ_night * L_night + λ_mono * L_mono
    
    This loss encourages the model to respect physical properties of solar power
    generation while maintaining good prediction accuracy.
    
    Args:
        configs: Configuration object with loss hyperparameters
        future_cols_order: Order of columns in x_future tensor
        
    Hyperparameters (from configs):
        - lambda_data: Weight for data fit loss (default 1.0)
        - lambda_night: Weight for night-time penalty (default 0.2)
        - lambda_mono: Weight for monotonicity regularization (default 0.1)
        - G_night_thr: Irradiance threshold for "night" (W/m², default 10.0)
        
    Example:
        >>> criterion = PhysicsPVLoss(configs, configs.FUTURE_INPUT_COLS)
        >>> loss, loss_dict = criterion(preds, y_true, x_future)
    """
    
    def __init__(self, configs, future_cols_order: list):
        super(PhysicsPVLoss, self).__init__()
        self.cfg = configs
        
        # Feature indices in x_future tensor [B, H, F_future]
        self.idx_G = future_cols_order.index('nwp_globalirrad')
        
        # Small constant for numerical stability
        self.eps = 1e-6
        
        # Lambda for monotonicity (add to configs if needed)
        self.lambda_mono = getattr(configs, 'lambda_mono', 0.1)
    
    def forward(
        self,
        preds: torch.Tensor,
        y_true: torch.Tensor,
        x_future: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the triple-constraint physics loss.
        
        Args:
            preds: Model predictions [B, H, 1]
            y_true: Ground truth power [B, H, 1]
            x_future: Future weather features [B, H, F_future]
            
        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: Weighted sum of all loss components
                - loss_dict: Dictionary with individual loss values for logging
        """
        # 1. MSE Loss (Data Fit)
        L_data = F.mse_loss(preds, y_true)
        
        # 2. Extract irradiance for constraints
        G = x_future[:, :, self.idx_G]  # [B, H]
        
        # 3. Night-time Penalty
        # If irradiance < threshold, should have zero power
        # Penalize positive power at night
        night_mask = (G < self.cfg.G_night_thr).unsqueeze(-1)  # bool [B,H,1]
        if night_mask.any():
            L_night = (preds[night_mask] ** 2).mean()
        else:
            L_night = preds.new_tensor(0.0)
        
        # 4. Monotonicity Regularization
        # Power should increase when irradiance increases
        # Penalize: if P goes down while G goes up
        dP = preds[:, 1:, :] - preds[:, :-1, :]  # [B, H-1, 1]
        dG = (G[:, 1:] - G[:, :-1]).unsqueeze(-1)  # [B, H-1, 1]
        
        # Penalize negative dP when dG is positive (power down while sun up)
        pos = dG > 0
        if pos.any():
            L_mono = F.relu(-dP[pos]).mean()
        else:
            L_mono = preds.new_tensor(0.0)
        
        # 5. Total Loss
        total = (
            self.cfg.lambda_data * L_data +
            self.cfg.lambda_night * L_night +
            self.lambda_mono * L_mono
        )
        
        # Return total loss and components for logging
        loss_dict = {
            "L_data": L_data.detach().item(),
            "L_night": L_night.detach().item(),
            "L_mono": L_mono.detach().item(),
        }
        
        return total, loss_dict
