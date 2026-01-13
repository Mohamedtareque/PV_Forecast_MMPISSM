"""
Physics-Residual hybrid model combining physics layer with Mamba residual learner.

Provides:
- PhysicsResidualMamba: Hybrid architecture combining differentiable physics
  and Mamba-based residual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .physics_layer import DifferentiablePVLayer
from .base_models import MMPISSM_Model_01


class PhysicsResidualMamba(nn.Module):
    """
    Physics-Residual Power Mamba: Hybrid Architecture.
    
    Combines a differentiable physics layer with a Mamba-based residual learner.
    
    Architecture:
        1. Branch A (Physics): DifferentiablePVLayer computes P_physics
        2. Branch B (Residual): MMPISSM_Model_01 (Mamba) predicts residual ε
        3. Combination: P_total = ReLU(P_physics + ε)
    
    The key insight is that the neural network learns the RESIDUAL (error)
    of the physics model, not the raw power. This constrains the learning
    and provides physics-informed predictions.
    
    Args:
        configs: Configuration object with model hyperparameters
        
    Attributes:
        physics_layer: DifferentiablePVLayer - Computes physics-based power
        mamba_model: MMPISSM_Model_01 - Learns residual error
        
    Example:
        >>> model = PhysicsResidualMamba(configs)
        >>> p_total, p_physics = model(x_past, x_future)
    """
    
    def __init__(self, configs):
        super(PhysicsResidualMamba, self).__init__()
        self.configs = configs
        
        # Get indices for physics layer from future input columns
        idx_G = configs.FUTURE_INPUT_COLS.index('nwp_globalirrad')
        idx_Ta = configs.FUTURE_INPUT_COLS.index('nwp_temperature')
        idx_WS = configs.FUTURE_INPUT_COLS.index('nwp_windspeed')
        
        # --- Geometric Features ---
        # Detect time features for geometric gating
        idx_time_feats = []
        possible_feats = ['hour_sin', 'hour_cos', 'season_sin', 'season_cos']
        for f in possible_feats:
            if f in configs.FUTURE_INPUT_COLS:
                idx_time_feats.append(configs.FUTURE_INPUT_COLS.index(f))
        
        if len(idx_time_feats) == 0:
            print("WARNING: Time features (hour_sin, etc.) not found in FUTURE_INPUT_COLS! "
                  "Geometric Gating might fail.")
        
        # Branch A: Differentiable Physics Layer
        self.physics_layer = DifferentiablePVLayer(
            idx_G=idx_G,
            idx_Ta=idx_Ta,
            idx_WS=idx_WS,
            idx_time_feats=idx_time_feats,
            T_ref=configs.T_ref,
            G_stc=1000.0,
            P_stc_init=18.0,  # 18 MW (slightly below 20 MW) to allow headroom
            eta_init=0.15
        )
        
        # Branch B: Mamba-based Residual Learner
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
    
    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining physics and residual predictions.
        
        Args:
            x_past: Past input features [B, L, F_past]
            x_future: Future input features [B, H, F_future]
            
        Returns:
            tuple: (p_total, p_physics)
                - p_total: Combined prediction [B, H, 1] (physics + residual)
                - p_physics: Physics-only prediction [B, H, 1] (for loss computation)
                
        Dimensions:
            - B: Batch size
            - L: Sequence length (seq_len)
            - H: Prediction horizon (pred_len)
            - F_past: Number of past features
            - F_future: Number of future features
        """
        # Check for NaN in inputs
        if torch.isnan(x_past).any() or torch.isinf(x_past).any():
            x_past = torch.nan_to_num(x_past, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(x_future).any() or torch.isinf(x_future).any():
            x_future = torch.nan_to_num(x_future, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Branch A: Physics-based prediction
        # Uses only future weather: [G, T_air, W]
        p_physics = self.physics_layer(x_future)  # [B, H, 1]
        
        # Check physics output
        if torch.isnan(p_physics).any() or torch.isinf(p_physics).any():
            p_physics = torch.nan_to_num(p_physics, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Branch B: Residual prediction via Mamba
        # Uses full history and future features
        # CRUCIAL: Output is interpreted as residual ε = P_true - P_physics
        p_residual = self.mamba_model(x_past, x_future)  # [B, H, 1]
        
        # Check residual output - use nan_to_num for gradient-safe replacement
        if torch.isnan(p_residual).any() or torch.isinf(p_residual).any():
            p_residual = torch.nan_to_num(p_residual, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combination: Sum physics and residual, apply ReLU
        # P_total = ReLU(P_physics + ε)
        # ReLU ensures power is non-negative (physical constraint)
        p_total = F.relu(p_physics + p_residual)  # [B, H, 1]
        
        return p_total, p_physics
    
    def get_physics_params(self) -> Dict[str, float]:
        """
        Return current physics layer parameters for monitoring.
        
        Returns:
            Dictionary with learned physics parameters:
                - eta: Module efficiency
                - U0: Heat transfer coefficient (no wind)
                - U1: Wind cooling coefficient
                - gamma: Temperature coefficient
                - P_stc: STC power (MW)
        """
        return self.physics_layer.get_params_dict()
