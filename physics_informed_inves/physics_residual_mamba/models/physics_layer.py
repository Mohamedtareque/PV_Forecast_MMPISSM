"""
Differentiable physics layer for PV power estimation.

Provides:
- DifferentiablePVLayer: Learnable physics model with geometric gating
- FeatureScaler: Normalization statistics management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FeatureScaler(nn.Module):
    """
    Stores normalization statistics and provides denormalization for physics layer.
    
    The physics equations require real-world units:
    - Irradiance G in W/m² (typically 0-1200)
    - Temperature T in °C
    - Wind speed W in m/s
    
    Neural networks work better with normalized (z-scored) inputs.
    This class bridges the gap by storing mean/std and denormalizing on demand.
    
    Args:
        feature_means: Mean values keyed by feature name
        feature_stds: Std values keyed by feature name
        
    Example:
        >>> means = {'nwp_globalirrad': 300.0, 'nwp_temperature': 15.0, 'nwp_windspeed': 4.0}
        >>> stds = {'nwp_globalirrad': 250.0, 'nwp_temperature': 10.0, 'nwp_windspeed': 2.0}
        >>> scaler = FeatureScaler(means, stds)
    """
    
    def __init__(self, feature_means: Dict[str, float], feature_stds: Dict[str, float]):
        super(FeatureScaler, self).__init__()
        
        # Store feature names for reference
        self.feature_names = list(feature_means.keys())
        
        # Register buffers for each feature (moves with model to GPU/CPU)
        for name in self.feature_names:
            mean_val = feature_means[name]
            std_val = feature_stds[name]
            # Ensure std is never zero
            if std_val == 0:
                std_val = 1.0
            self.register_buffer(f'{name}_mean', torch.tensor(mean_val, dtype=torch.float32))
            self.register_buffer(f'{name}_std', torch.tensor(std_val, dtype=torch.float32))
    
    def denormalize_feature(self, x_norm: torch.Tensor, feature_name: str) -> torch.Tensor:
        """
        Denormalize a single feature tensor.
        
        Args:
            x_norm: Normalized feature tensor [B, H] or any shape
            feature_name: Name of the feature (must match key in feature_means)
            
        Returns:
            Denormalized tensor in original units
        """
        mean = getattr(self, f'{feature_name}_mean')
        std = getattr(self, f'{feature_name}_std')
        return x_norm * std + mean
    
    def denormalize_dict(
        self,
        x_norm: torch.Tensor,
        feature_indices: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Denormalize multiple features from a tensor and return as dict.
        
        Args:
            x_norm: Normalized input tensor [B, H, F]
            feature_indices: Dict mapping feature name to index in last dim
            
        Returns:
            Dict mapping feature name to denormalized tensor [B, H]
        """
        result = {}
        for name, idx in feature_indices.items():
            if name in self.feature_names:
                result[name] = self.denormalize_feature(x_norm[..., idx], name)
            else:
                # If feature not in scaler, pass through unchanged
                result[name] = x_norm[..., idx]
        return result
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return all stored statistics as plain dict for logging."""
        return {
            name: {
                'mean': getattr(self, f'{name}_mean').item(),
                'std': getattr(self, f'{name}_std').item()
            }
            for name in self.feature_names
        }


class DifferentiablePVLayer(nn.Module):
    """
    Differentiable Physics Layer for PV Power Estimation.
    
    Implements learnable PV physics model based on:
    - Faiman temperature model for cell temperature
    - Geometric gating for Angle of Incidence (AOI) correction
    - Learnable efficiency and temperature coefficient
    
    The layer learns to correct systematic errors in the physical model
    through gradient-based optimization.
    
    Args:
        idx_G: Index of global irradiance in input tensor
        idx_Ta: Index of air temperature in input tensor
        idx_WS: Index of wind speed in input tensor
        idx_time_feats: List of indices for time features [hour_sin, hour_cos, season_sin, season_cos]
        T_ref: Reference temperature (default 25.0°C)
        G_stc: Standard test condition irradiance (default 1000.0 W/m²)
        P_stc_init: Initial STC power in MW (default 18.0)
        eta_init: Initial efficiency (default 0.15)
        
    Example:
        >>> physics_layer = DifferentiablePVLayer(
        ...     idx_G=0, idx_Ta=1, idx_WS=2,
        ...     idx_time_feats=[3, 4, 5, 6],
        ...     T_ref=25.0, G_stc=1000.0
        ... )
        >>> p_physics = physics_layer(x_future)
    """
    
    def __init__(
        self,
        idx_G: int,
        idx_Ta: int,
        idx_WS: int,
        idx_time_feats: list[int],
        T_ref: float = 25.0,
        G_stc: float = 1000.0,
        P_stc_init: float = 18.0,
        eta_init: float = 0.15
    ):
        super().__init__()
        self.idx_G, self.idx_Ta, self.idx_WS = idx_G, idx_Ta, idx_WS
        self.idx_time_feats = idx_time_feats
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6
        self.is_input_raw = True  # Flag to indicate inputs are raw (not normalized)
        
        # Learnable Physics Parameters
        # Poly-Si Temp Coeff ~ -0.45%/C. Init at -0.0045 via softplus
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4, dtype=torch.float32))
        
        # Heat transfer (U0 ~ 25, U1 ~ 6.8 for open rack)
        self.U0_raw = nn.Parameter(torch.tensor(3.2, dtype=torch.float32))  # inv_softplus(25)
        self.U1_raw = nn.Parameter(torch.tensor(1.9, dtype=torch.float32))  # inv_softplus(6.8)
        
        self.eta_raw = nn.Parameter(torch.tensor(-1.7))  # sigmoid(-1.7) ~ 0.15
        self.Pstc_raw = nn.Parameter(torch.tensor(float(P_stc_init)))  # Init at 18MW
        
        # Geometric Correction Network
        # Learns to map (Time of Day, Season) -> Tilt Factor
        # This replaces the static k_poa assumption
        self.geo_net = nn.Sequential(
            nn.Linear(len(idx_time_feats), 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus()  # Factor must be positive
        )
        
        # Parameter to enable/disable scaler (for debugging/logging, not computation if is_input_raw=True)
        self.scaler = None
    
    def set_scaler(self, scaler: FeatureScaler):
        """Register scaler for parameter logging only."""
        self.scaler = scaler
    
    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through differentiable physics layer.
        
        Args:
            x_future: Future weather features [B, H, F_future]
                      Must contain: G, T_air, W, time features
                    
        Returns:
            Physics-based power prediction [B, H, 1]
        """
        # 1. Extract Physics Inputs
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        
        # 2. Extract Time Inputs for Geometry
        time_feats = x_future[:, :, self.idx_time_feats]
        
        # 3. Compute Dynamic Tilt Factor
        # "How much stronger is the sun on the panel vs horizontal right now?"
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        
        # 4. Effective Plane-of-Array Irradiance
        G_poa = G * tilt_factor
        
        # 5. Enforce constraints on learnable parameters
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)  # Ensure negative
        P_stc = F.softplus(self.Pstc_raw)
        
        # 6. Faiman Model: T_cell = Ta + G_poa / (U0 + U1*WS)
        T_cell = Ta + G_poa / (U0 + U1 * WS + self.eps)
        
        # 7. Power Calculation
        # Poly-Si constraint: Performance drops as Temp rises
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        
        # Power = Efficiency * Capacity * (G_poa/G_stc) * TempFactor
        P_phys = eta * P_stc * (G_poa / self.G_stc) * temp_factor
        
        return F.relu(P_phys).unsqueeze(-1)
    
    def get_params_dict(self) -> Dict[str, float]:
        """
        Return learned parameters for logging.
        
        Returns:
            Dictionary with parameter names and their current values
        """
        with torch.no_grad():
            params = {
                'eta': torch.sigmoid(self.eta_raw).item(),
                'U0': F.softplus(self.U0_raw).item(),
                'U1': F.softplus(self.U1_raw).item(),
                'gamma': -F.softplus(self.gamma_raw).item(),
                'P_stc': F.softplus(self.Pstc_raw).item()
            }
        return params
