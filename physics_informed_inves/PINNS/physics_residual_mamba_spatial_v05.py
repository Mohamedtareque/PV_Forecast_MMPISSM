"""
Physics-Residual Power Mamba (Spatial V05) - Hyper-Robust Physics & Moving Average

Key Features:
- Backbone: Mamba + Moving Average Decomposition (No Wavelet)
- Physics V05: Hyper-Robust (Irrad Split + Log Cooling + Spectral)
- Physics V04: Robust (for comparison)
- Loss: Fixed Composite Loss (No Adaptive)

Models:
1. Vanilla Mamba
2. Base Power Mamba
3. Physics V04 (Robust)
4. Physics V05 (Hyper-Robust)
5. NWP Baseline
6. Smart Persistence
"""

import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import os

# Parameter Tracking
try:
    from parameter_tracker import ParameterTracker, generate_combined_comparison
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    print("Warning: parameter_tracker module not found. Parameter tracking disabled.")

# =============================================================================
# Helper Classes
# =============================================================================

class FeatureScaler(nn.Module):
    def __init__(self, feature_means: dict, feature_stds: dict):
        super(FeatureScaler, self).__init__()
        self.feature_names = list(feature_means.keys())
        for name in self.feature_names:
            mean_val = feature_means[name]
            std_val = feature_stds[name]
            if std_val == 0: std_val = 1.0
            self.register_buffer(f'{name}_mean', torch.tensor(mean_val, dtype=torch.float32))
            self.register_buffer(f'{name}_std', torch.tensor(std_val, dtype=torch.float32))
    
    def denormalize_feature(self, x_norm: torch.Tensor, feature_name: str) -> torch.Tensor:
        mean = getattr(self, f'{feature_name}_mean')
        std = getattr(self, f'{feature_name}_std')
        return x_norm * std + mean
    
    def get_stats(self) -> dict:
        return {name: {'mean': getattr(self, f'{name}_mean').item(), 'std': getattr(self, f'{name}_std').item()} for name in self.feature_names}

def compute_scaler_stats(df: pd.DataFrame, cols: list) -> tuple:
    means = {col: df[col].mean() for col in cols}
    stds = {col: df[col].std() for col in cols}
    return means, stds

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Latitude/longitude of point 1 (degrees)
        lat2, lon2: Latitude/longitude of point 2 (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    # Convert to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    
    # Haversine formula components
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate initial bearing from point 1 to point 2 in degrees [0, 360).
    Bearing is measured clockwise from north.
    
    Args:
        lat1, lon1: Latitude/longitude of starting point (degrees)
        lat2, lon2: Latitude/longitude of destination point (degrees)
    
    Returns:
        Bearing in degrees (0-360, measured clockwise from north)
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    
    dlon = lon2_rad - lon1_rad
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    return (bearing_deg + 360) % 360

# =============================================================================
# Fixed Physics Loss
# =============================================================================

class SmoothnessLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(SmoothnessLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred):
        if pred.ndim == 3:
            diff = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        else:
            diff = torch.abs(pred[:, 1:] - pred[:, :-1])
        return self.weight * diff.mean()

class PhysicsCompositeLoss_Fixed(nn.Module):
    def __init__(self, smooth_weight=0.1):
        super(PhysicsCompositeLoss_Fixed, self).__init__()
        self.smooth_loss = SmoothnessLoss(weight=smooth_weight)
        
    def forward(self, pred, target):
        loss_mse = F.mse_loss(pred, target)
        loss_smooth = self.smooth_loss(pred)
        return loss_mse + loss_smooth

# =============================================================================
# Physics Layers (V4 & V05)
# =============================================================================

class DifferentiablePVLayer_V02(nn.Module):
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, idx_elev=None, T_ref=25.0, G_stc=1000.0, P_stc_init=19.0, eta_init=0.16):
        super().__init__()
        self.idx_G, self.idx_Ta, self.idx_WS = idx_G, idx_Ta, idx_WS
        self.idx_elev = idx_elev
        self.idx_time_feats = idx_time_feats 
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4, dtype=torch.float32)) 
        self.U0_raw = nn.Parameter(torch.tensor(3.2, dtype=torch.float32)) 
        self.U1_raw = nn.Parameter(torch.tensor(1.9, dtype=torch.float32)) 
        self.eta_raw = nn.Parameter(torch.tensor(-1.66)) 
        self.P_stc_limit = float(P_stc_init)
        self.Pstc_raw = nn.Parameter(torch.tensor(6.0))

    def forward(self, x_future, x_clr=None):
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw) 
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        T_cell = Ta + G / (U0 + U1 * WS + self.eps)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = eta * P_stc * (G / self.G_stc) * temp_factor
        P_final_mw = F.relu(P_dc).unsqueeze(-1)
        if x_clr is not None:
            if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
            K_phys = P_final_mw / (x_clr + self.eps)
            return K_phys * (x_clr > 0.01).float()
        return P_final_mw

class DifferentiablePVLayer_V3(DifferentiablePVLayer_V02):
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs):
        super().__init__(idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs)
        self.geo_net = nn.Sequential(nn.Linear(len(idx_time_feats), 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

class DifferentiablePVLayer_V4(DifferentiablePVLayer_V3):
    """V4: Robust (IAM + Non-Linear Inverter)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs):
        super().__init__(idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs)
        self.b0_raw = nn.Parameter(torch.tensor(0.05))
        self.inv_k_raw = nn.Parameter(torch.tensor(1.0))
        self.inv_thresh_raw = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_future, x_clr=None):
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        time_feats = x_future[:, :, self.idx_time_feats]
        
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        G_poa = G * tilt_factor
        b0 = torch.sigmoid(self.b0_raw) * 0.2
        IAM = torch.clamp(1.0 - b0 * (1.0 - tilt_factor), 0.0, 1.0)
        G_eff = G_poa * IAM
        
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta_module = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        T_cell = Ta + G_eff / (U0 + U1 * WS + self.eps)
        P_dc = F.relu(eta_module * P_stc * (G_eff / self.G_stc) * (1.0 + gamma * (T_cell - self.T_ref)))
        
        inv_k = F.softplus(self.inv_k_raw) * 10.0
        inv_thresh = F.softplus(self.inv_thresh_raw)
        P_ac = P_dc * torch.sigmoid(inv_k * (P_dc - inv_thresh))
        P_final_mw = P_ac.unsqueeze(-1)
        
        if x_clr is not None:
             if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
             return (P_final_mw / (x_clr + self.eps)) * (x_clr > 0.01).float()
        return P_final_mw

class DifferentiablePVLayer_V05(DifferentiablePVLayer_V4):
    """V05: Hyper-Robust (Split Irrad + Log Cooling + Spectral)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, idx_elev, **kwargs):
        super().__init__(idx_G, idx_Ta, idx_WS, idx_time_feats, idx_elev=idx_elev, **kwargs)
        self.diffuse_frac_raw = nn.Parameter(torch.tensor(-2.0))
        self.U0_base_raw = nn.Parameter(torch.tensor(3.0))
        self.U_wind_raw = nn.Parameter(torch.tensor(1.5))
        self.spec_coeff_raw = nn.Parameter(torch.tensor(0.0))
        
    def calculate_air_mass(self, elevation_degrees):
        alpha = torch.clamp(elevation_degrees, 0.0, 90.0) 
        alpha_rad = torch.deg2rad(alpha)
        term1 = torch.sin(alpha_rad)
        term2 = 0.50572 * torch.pow(alpha + 6.07995, -1.6364)
        AM = 1.0 / (term1 + term2 + self.eps)
        return torch.clamp(AM, 1.0, 30.0)

    def forward(self, x_future, x_clr=None):
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        time_feats = x_future[:, :, self.idx_time_feats]
        
        diffuse_frac = torch.sigmoid(self.diffuse_frac_raw)
        G_beam = G * (1.0 - diffuse_frac)
        G_diff = G * diffuse_frac
        
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        b0 = torch.sigmoid(self.b0_raw) * 0.2
        IAM = torch.clamp(1.0 - b0 * (1.0 - tilt_factor), 0.0, 1.0)
        G_eff = (G_beam * tilt_factor * IAM) + G_diff
        
        U_dynamic = F.softplus(self.U0_base_raw) + F.softplus(self.U_wind_raw) * torch.log(1.0 + WS + self.eps)
        T_cell = Ta + G_eff / (U_dynamic + self.eps)
        
        spectral_factor = 1.0
        if self.idx_elev is not None:
            elev = x_future[:, :, self.idx_elev]
            AM = self.calculate_air_mass(elev)
            spectral_factor = 1.0 + self.spec_coeff_raw * (1.0 - AM)
            
        eta_eff = torch.sigmoid(self.eta_raw) * spectral_factor
        gamma = -F.softplus(self.gamma_raw)
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        P_dc = F.relu(eta_eff * P_stc * (G_eff / self.G_stc) * (1.0 + gamma * (T_cell - self.T_ref)))
        
        inv_k = F.softplus(self.inv_k_raw) * 10.0
        inv_thresh = F.softplus(self.inv_thresh_raw)
        P_ac = P_dc * torch.sigmoid(inv_k * (P_dc - inv_thresh))
        P_final_mw = P_ac.unsqueeze(-1)
        
        if x_clr is not None:
            if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
            K_phys = P_final_mw / (x_clr + self.eps)
            return K_phys * (x_clr > 0.01).float()
        return P_final_mw

# =============================================================================
# Advection Modules
# =============================================================================

class DynamicAdvectionLag(nn.Module):
    """
    Shifts Neighbor (S8) features based on Wind Speed & Direction.
    Lag = Clamp(Distance / (Speed_Aligned * DeltaT), Min, Max)
    """
    def __init__(self, distance_m=1200.0, delta_t_sec=900.0, bearing_deg=253.0, max_lag=12):
        super().__init__()
        self.D = distance_m
        self.dt = delta_t_sec
        self.phi = np.radians(bearing_deg)
        self.max_lag = max_lag
        self._last_lag = torch.tensor(0.0)
        self._last_alpha = torch.tensor(0.0)
        
    def forward(self, x_s8, wind_speed, wind_dir_deg):
        # x_s8: (B, L, D_feat) - Neighbor features
        # wind_speed: (B, L, 1)
        # wind_dir_deg: (B, L, 1)
        
        # Calculate Aligned Speed
        theta = torch.deg2rad(wind_dir_deg)
        v_aligned = wind_speed * torch.cos(theta - self.phi)
        
        # Avoid div by zero or negative speed (downwind)
        v_aligned = F.relu(v_aligned) + 0.01  # Min speed 0.01 m/s (Reduced from 0.1)
        
        # Use Mean Wind of sequence for stability
        mean_s = wind_speed.mean(dim=1, keepdim=True)
        mean_d = wind_dir_deg.mean(dim=1, keepdim=True)
        
        theta_m = torch.deg2rad(mean_d)
        v_m = mean_s * torch.cos(theta_m - self.phi)
        v_m = F.relu(v_m) + 0.01
        
        # Calculate Lag Steps: L = D / (v * dt)
        L_float = torch.clamp(self.D / (v_m * self.dt), 0.0, float(self.max_lag)).view(-1)  # (B,)
        self._last_lag = L_float.mean().detach()
        
        # Fractional Lag Interpolation
        L_floor = torch.floor(L_float).long()
        L_ceil = torch.ceil(L_float).long()
        alpha = (L_float - L_floor.float()).view(-1, 1, 1)  # (B, 1, 1)
        self._last_alpha = alpha.mean().detach()
        
        # Perform Shifting with Interpolation
        B, S, F_dim = x_s8.shape
        x_shifted = torch.zeros_like(x_s8)
        
        for b in range(B):
            l_f = L_floor[b].item()
            l_c = L_ceil[b].item()
            w_c = alpha[b, 0, 0]
            w_f = 1.0 - w_c
            
            # Floor Shift
            if l_f < S:
                src_f = x_s8[b, :S-l_f]
                x_shifted[b, l_f:] += w_f * src_f
                x_shifted[b, :l_f] += w_f * x_s8[b, 0:1] # Pad
            else:
                x_shifted[b, :] += w_f * x_s8[b, 0:1]
                
            # Ceil Shift
            if l_c < S:
                src_c = x_s8[b, :S-l_c]
                x_shifted[b, l_c:] += w_c * src_c
                x_shifted[b, :l_c] += w_c * x_s8[b, 0:1] # Pad
            else:
                x_shifted[b, :] += w_c * x_s8[b, 0:1]
            
        return x_shifted


class SoftAttentionGate(nn.Module):
    """
    MLP([v, cos_theta]) -> Beta (Importance Weight)
    """
    def __init__(self, bearing_deg=253.0, min_gate=0.05):
        super().__init__()
        self.phi = np.radians(bearing_deg)
        self.min_gate = min_gate
        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Beta in [0, 1]
        )
        self._last_beta = torch.tensor(0.5)
        
    def forward(self, wind_speed, wind_dir_deg):
        # wind_speed: (B, L, 1)
        # wind_dir_deg: (B, L, 1)
        theta = torch.deg2rad(wind_dir_deg)
        cos_align = torch.cos(theta - self.phi)
        
        inp = torch.cat([wind_speed, cos_align], dim=-1)  # (B, L, 2)
        beta = self.mlp(inp)  # (B, L, 1)
        
        # Force min attention (min_gate to 1.0)
        # Scale: beta * (1 - min) + min
        beta = beta * (1.0 - self.min_gate) + self.min_gate
        self._last_beta = beta.detach()
        return beta

# =============================================================================
# RevIN and Decomposition (Moving Average ONLY)
# =============================================================================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-2, affine=True):  
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine: self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(variance + self.eps).detach()
        self.stdev = torch.clamp(self.stdev, min=self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
class series_decomp(nn.Module):
    """Moving Average Decomposition"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        moving_mean = self.moving_avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using Linear fallback.")

# =============================================================================
# Mamba Backbone (Moving Average Only)
# =============================================================================

class MMPISSM_Model_01(nn.Module):
    """Standard Backbone using Moving Average Decomposition"""
    def __init__(self, configs, output_residual=False):
        super(MMPISSM_Model_01, self).__init__()
        self.configs = configs
        self.output_residual = output_residual
        self.num_past_feats = len(self.configs.PAST_INPUT_COLS)
        self.num_shadows = len(self.configs.common_features)
        self.enc_in = self.num_past_feats + self.num_shadows
        self.fut_indices = torch.tensor([self.configs.FUTURE_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.past_target_indices = torch.tensor([self.configs.PAST_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.shadow_start_idx = self.num_past_feats
        self.target_indices = [self.configs.PAST_INPUT_COLS.index(col) for col in self.configs.TARGET_COL]

        self.lin1 = nn.Linear(2 * (self.configs.seq_len + self.configs.pred_len), self.configs.seq_len)
        self.decomposition = series_decomp(self.configs.kernel_size)
        self.revin_layer = RevIN(self.enc_in)
        self.revin_layer_enc = RevIN(self.enc_in)
        self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)
        self.lin3 = nn.Linear(4 * self.configs.n_embed, self.configs.pred_len)
        self.dropout1 = nn.Dropout(self.configs.dropout)
        self.dropout2 = nn.Dropout(self.configs.dropout)

        if MAMBA_AVAILABLE:
            self.mamba1 = Mamba(d_model=self.configs.n_embed, d_state=self.configs.d_state, d_conv=self.configs.dconv, expand=self.configs.e_fact)
            self.mamba2 = Mamba(d_model=self.configs.enc_in, d_state=self.configs.d_state, d_conv=self.configs.dconv, expand=self.configs.e_fact)
        else:
            self.mamba1 = nn.Linear(self.configs.n_embed, self.configs.n_embed)
            self.mamba2 = nn.Linear(self.configs.enc_in, self.configs.enc_in)
    
    def forward(self, x, x_f):
        B = x.size(0)
        device = x.device
        L = self.configs.seq_len
        H = self.configs.pred_len
        if self.fut_indices.device != device:
            self.fut_indices = self.fut_indices.to(device)
            self.past_target_indices = self.past_target_indices.to(device)

        total_len = L + H
        x_pred = torch.zeros(B, total_len, self.enc_in, device=device)
        x_pred[:, :L, :self.num_past_feats] = x
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        for i, past_idx in enumerate(self.past_target_indices):
            x_pred[:, L:, past_idx] = shadow_forecasts[:, :, i]
        x_pred[:, L:, self.shadow_start_idx:] = shadow_forecasts
        for i, past_idx in enumerate(self.past_target_indices):
            shadow_col = self.shadow_start_idx + i
            x_pred[:, :L, shadow_col] = x_pred[:, -L:, past_idx]

        x = x_pred
        x = self.revin_layer_enc(x, 'norm')
        # MA Decomp 1
        seasonal_init, trend_init = self.decomposition(x)
        x = torch.cat([seasonal_init, trend_init], dim=1)
        x = self.lin1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.revin_layer_enc(x, 'denorm')
        x = self.revin_layer(x, 'norm')
        # MA Decomp 2
        seasonal_init, trend_init = self.decomposition(x)
        x_e = torch.cat([seasonal_init, trend_init], dim=1)
        x_e = self.lin2(x_e.permute(0, 2, 1))

        x_m = self.mamba1(self.dropout1(x_e))
        x_im = self.mamba2(self.dropout2(x_e).permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x_im, x_m, x_m + x_im, x_e], dim=2)
        x = self.lin3(x).permute(0, 2, 1)
        
        x = self.revin_layer(x, 'denorm')
        x = x[:, :, self.target_indices]
        past_stdev_target = torch.ones_like(x) 
        return x, past_stdev_target

# =============================================================================
# Configuration
# =============================================================================

class SpatialMambaConfigs:
    def __init__(self, n_splits=4):
        self.seq_len = 288
        self.pred_len = 96
        self.TARGET_COL = ['K_PV']
        self.PAST_INPUT_COLS = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'nwp_winddirection', 'power', 'K_PV', 'P_CLR', 'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation', 'K_CS', 's08_nwp_globalirrad', 's08_nwp_temperature', 's08_nwp_windspeed', 's08_K_PV', 'spatial_delta_irrad', 'spatial_delta_temp', 'spatial_delta_wind']
        self.FUTURE_INPUT_COLS = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'nwp_winddirection', 'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation', 's08_nwp_globalirrad', 's08_nwp_temperature', 's08_nwp_windspeed', 'spatial_delta_irrad', 'spatial_delta_temp', 'spatial_delta_wind']
        st07_future_cols = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation']
        self.st07_indices = [self.FUTURE_INPUT_COLS.index(c) for c in st07_future_cols]
        self.st07_future_cols = st07_future_cols
        
        self.common_features = [f for f in self.FUTURE_INPUT_COLS if f in self.PAST_INPUT_COLS]
        self.num_shadows = len(self.common_features)
        self.enc_in = len(self.PAST_INPUT_COLS) + self.num_shadows
        self.c_out = 1
        
        self.kernel_size = 25
        self.n_embed = 128
        self.d_state = 64
        self.dconv = 2
        self.e_fact = 2
        self.dropout = 0.2
        self.n_splits = n_splits
        self.batch_size = 64
        self.epochs = 30
        self.warmup_epochs = 15
        # Loss params
        self.T_ref = 25.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # NEW: Station coordinate placeholders
        self.lat_7 = None
        self.lon_7 = None
        self.lat_8 = None
        self.lon_8 = None
        
        # Advection parameters
        self.advection_distance_m = 1200.0  # Default, will be updated
        self.advection_delta_t_sec = 900.0  # Timestep in seconds (15 min)
        self.advection_bearing_deg = 253.0  # Default, will be updated
        self.advection_max_lag = 12  # Maximum 12 timesteps (3 hours)
        self.advection_min_gate = 0.05  # Reduced from 0.1
        self.advection_use_fractional_lag = True  # Enable fractional lag

    def set_station_coordinates(self, meta_s7, meta_s8):
        """Set station coordinates from metadata dictionaries"""
        self.lat_7 = float(meta_s7['Latitude'])
        self.lon_7 = float(meta_s7['Longitude'])
        self.lat_8 = float(meta_s8['Latitude'])
        self.lon_8 = float(meta_s8['Longitude'])
        
        # Calculate and store geometry
        self.advection_bearing_deg = calculate_bearing(
            self.lat_8, self.lon_8,
            self.lat_7, self.lon_7
        )
        self.advection_distance_m = haversine_distance(
            self.lat_8, self.lon_8,
            self.lat_7, self.lon_7
        )
        
        print(f"=== STATION GEOMETRY ===")
        print(f"Station 07: ({self.lat_7:.5f}°N, {self.lon_7:.5f}°E)")
        print(f"Station 08: ({self.lat_8:.5f}°N, {self.lon_8:.5f}°E)")
        print(f"Calculated Bearing (08→07): {self.advection_bearing_deg:.2f}°")
        print(f"Calculated Distance: {self.advection_distance_m:.1f}m")
        print(f"========================")

# =============================================================================
# Models
# =============================================================================

class VanillaMambaBackbone(nn.Module):
    """
    Pure Vanilla Mamba Backbone (No Decomposition, No Inverse Branch)
    Structure: Input -> RevIN -> Projection -> Mamba Block -> Projection -> RevIN -> Output
    """
    def __init__(self, configs):
        super(VanillaMambaBackbone, self).__init__()
        self.configs = configs
        self.num_past_feats = len(self.configs.PAST_INPUT_COLS)
        self.num_shadows = len(self.configs.common_features)
        self.enc_in = self.num_past_feats + self.num_shadows
        
        # Helper logic to match input construction in MMPISSM_Model_01
        self.fut_indices = torch.tensor([self.configs.FUTURE_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.past_target_indices = torch.tensor([self.configs.PAST_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.shadow_start_idx = self.num_past_feats
        self.target_indices = [self.configs.PAST_INPUT_COLS.index(col) for col in self.configs.TARGET_COL]

        self.revin_layer = RevIN(self.enc_in)
        
        # Projection and Mamba
        self.lin_in = nn.Linear(self.configs.seq_len, self.configs.n_embed)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=self.configs.n_embed, d_state=self.configs.d_state, d_conv=self.configs.dconv, expand=self.configs.e_fact)
        else:
            self.mamba = nn.Linear(self.configs.n_embed, self.configs.n_embed)
            
        self.lin_out = nn.Linear(self.configs.n_embed, self.configs.pred_len)
        self.dropout = nn.Dropout(self.configs.dropout)

    def forward(self, x, x_f):
        # === Input Construction (Same as MMPISSM_Model_01) ===
        B = x.size(0)
        device = x.device
        L = self.configs.seq_len
        H = self.configs.pred_len
        if self.fut_indices.device != device:
            self.fut_indices = self.fut_indices.to(device)
            self.past_target_indices = self.past_target_indices.to(device)

        total_len = L + H
        x_pred = torch.zeros(B, total_len, self.enc_in, device=device)
        x_pred[:, :L, :self.num_past_feats] = x
        
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        for i, past_idx in enumerate(self.past_target_indices):
            x_pred[:, L:, past_idx] = shadow_forecasts[:, :, i]
        x_pred[:, L:, self.shadow_start_idx:] = shadow_forecasts
        for i, past_idx in enumerate(self.past_target_indices):
            shadow_col = self.shadow_start_idx + i
            x_pred[:, :L, shadow_col] = x_pred[:, -L:, past_idx]

        # Use constructed input (focusing on past part or full sequence?) 
        # MMPISSM decomposes this. For Vanilla, we project the seq_len dimension to n_embed.
        # But wait, Mamba is usually (B, L, D).
        # Detailed instruction: "Input Projection: Linear layer to map inputs to n_embed."
        # The input x_pred has shape (B, L+H, enc_in).
        # To mimic standard usage, we probably want to model the sequence in channel dim or time dim.
        # However, to be "Vanilla", we usually treat time as sequence.
        # BUT, the MMPISSM Permutes (0, 2, 1) before Linear. `self.lin1(x.permute(0, 2, 1))` 
        # which means it treats TIME as features for the Linear layer?. 
        # Actually in MMPISSM: x is [B, L, C]. Permute -> [B, C, L]. Linear(2*L -> L).
        # This reduces time dimension? No, seq_len is linear output.
        
        # Let's interpret "Vanilla Mamba" in this context: 
        # Standard Mamba typically takes (B, L, D).
        # Here we likely want to map features D -> n_embed, and let Mamba handle L.
        # OR, follow the style of the "Projections" mentioned in PowerMamba which projects TIME.
        
        # Check USER REQUEST: "Input Projection: Linear layer to map inputs to n_embed."
        # "Backbone: A single (or stacked) Mamba block... Output Projection: Linear layer to map the Mamba output to pred_len."
        # This implies: Input (B, C, L) -> Linear(L->n_embed)? Or (B, L, C) -> Linear(C->n_embed)?
        
        # Most Time-Series Mambas (like iTransformer etc) invert: (B, C, L).
        # Let's stick to the PowerMamba's permutation logic if we want a fair "Backbone" comparison,
        # BUT the prompt says "Vanilla Mamba Definition: Use Mamba(d_model=..., d_state=...) wrapped in linear projections."
        # And "Input Projection: Linear layer to map inputs to n_embed."
        
        # Let's assume standard TS format: Input [B, L, C].
        # If we project C->n_embed, we get [B, L, n_embed]. Mamba processes L.
        # But we need output [B, H].
        
        # WAIT. MMPISSM Lin1 (B, C, L) -> (B, C, L).
        
        # Let's use the explicit request: "Input -> Embedding -> Mamba -> Projection -> Output".
        # This usually means Input [B, L, C] -> Linear(C->D) -> Mamba(D) -> Linear(D->C)? No, prediction length?
        
        # Let's look at the instruction again: "Output Projection: Linear layer to map the Mamba output to pred_len."
        # This Strongly implies Inverted handling (Mix time).
        # Input: [B, C, L]. Linear(L->n_embed). Mamba(n_embed). Linear(n_embed->H).
        # This processes channels independently (or mixes them if we don't separate).
        
        # Ref MMPISSM: `self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)`
        # `x_e = self.lin2(x_e.permute(0, 2, 1))` => Input to Lin2 is [B, C, 2*L].
        # So it projects TIME -> Embedding.
        
        # So VanillaBackbone should do:
        # Input [B, L, C]. Permute -> [B, C, L]. Linear(L -> n_embed). Mamba. Linear(n_embed -> pred_len).
        # Output [B, C, H]. Permute -> [B, H, C].
        
        x = x_pred[:, :L, :] # Take past context from the constructed input
        x = self.revin_layer(x, 'norm') # [B, L, C]
        
        # Project Time -> Embed
        x = x.permute(0, 2, 1) # [B, C, L]
        x = self.lin_in(x) # [B, C, n_embed]
        
        # Mamba
        x = self.mamba(self.dropout(x)) # [B, C, n_embed]
        
        # Project Embed -> Pred
        x = self.lin_out(x) # [B, C, H]
        
        x = x.permute(0, 2, 1) # [B, H, C]
        
        x = self.revin_layer(x, 'denorm')
        x = x[:, :, self.target_indices]
        return x, torch.zeros_like(x)

class MambaPureBackbone(nn.Module):
    """
    Simplified Mamba Backbone ("Pure").
    - Removes mamba2 (Inverse-Mamba).
    - Removes 4-way concatenation.
    - Uses Concat([Mamba_Out, Embed_Skip]) -> 2 * Dim.
    """
    def __init__(self, configs, output_residual=False):
        super(MambaPureBackbone, self).__init__()
        self.configs = configs
        self.output_residual = output_residual
        
        # --- Indexing & State Setup ---
        self.num_past_feats = len(self.configs.PAST_INPUT_COLS)
        self.num_shadows = len(self.configs.common_features)
        self.enc_in = self.num_past_feats + self.num_shadows

        # Indices (Reuse standard logic)
        self.fut_indices = torch.tensor([self.configs.FUTURE_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.past_target_indices = torch.tensor([self.configs.PAST_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.shadow_start_idx = self.num_past_feats
        self.target_indices = [self.configs.PAST_INPUT_COLS.index(col) for col in self.configs.TARGET_COL]

        # --- Layers ---
        # 1. Input Processing
        self.lin1 = nn.Linear(2 * (self.configs.seq_len + self.configs.pred_len), self.configs.seq_len)
        self.decompsition = series_decomp(self.configs.kernel_size)
        
        # 2. Normalization
        self.revin_layer = RevIN(self.enc_in)
        self.revin_layer_enc = RevIN(self.enc_in)
        
        # 3. Embedding
        self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)
        
        # 4. Projection Head (MODIFIED)
        # We only have [x_m, x_e] -> 2 * n_embed
        self.lin3 = nn.Linear(2 * self.configs.n_embed, self.configs.pred_len)
        
        self.dropout1 = nn.Dropout(self.configs.dropout)
        
        # 5. Core Mamba (Simplified)
        if MAMBA_AVAILABLE:
            self.mamba1 = Mamba(
                d_model=self.configs.n_embed, 
                d_state=self.configs.d_state, 
                d_conv=self.configs.dconv, 
                expand=self.configs.e_fact
            )
        else:
            self.mamba1 = nn.Linear(self.configs.n_embed, self.configs.n_embed)
            
    def forward(self, x, x_f):
        B = x.size(0)
        device = x.device
        L = self.configs.seq_len
        H = self.configs.pred_len
        
        # --- Preprocessing (Same as V02) ---
        if self.fut_indices.device != device:
            self.fut_indices = self.fut_indices.to(device)
            self.past_target_indices = self.past_target_indices.to(device)

        total_len = L + H
        x_pred = torch.zeros(B, total_len, self.enc_in, device=device)
        x_pred[:, :L, :self.num_past_feats] = x
        
        # Fill Future Shadows
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        for i, past_idx in enumerate(self.past_target_indices):
            x_pred[:, L:, past_idx] = shadow_forecasts[:, :, i]
        x_pred[:, L:, self.shadow_start_idx:] = shadow_forecasts
        
        # Fill Past Shadows (Persistence)
        for i, past_idx in enumerate(self.past_target_indices):
            shadow_col = self.shadow_start_idx + i
            x_pred[:, :L, shadow_col] = x_pred[:, -L:, past_idx]

        # --- Decomp & Embedding ---
        x = x_pred
        x = self.revin_layer_enc(x, 'norm') # (B, L+H, C)
        
        # Level 1 Decomp
        seasonal_init, trend_init = self.decompsition(x)
        x = torch.cat([seasonal_init, trend_init], dim=1) # (B, 2*(L+H), C)
        
        # Lin 1: (2*(L+H) -> L)
        x = self.lin1(x.permute(0, 2, 1)).permute(0, 2, 1) # (B, L, C)
        
        x = self.revin_layer_enc(x, 'denorm')
        x = self.revin_layer(x, 'norm')
        
        # Level 2 Decomp
        seasonal_init, trend_init = self.decompsition(x)
        x_e = torch.cat([seasonal_init, trend_init], dim=1) # (B, 2*L, C)
        
        # Embed: Time -> n_embed
        x_e = self.lin2(x_e.permute(0, 2, 1)) # (B, C, n_embed)
        
        # --- Mamba Pure (Simplified) ---
        # 1. Main Branch
        x_m = self.mamba1(self.dropout1(x_e)) # (B, C, n_embed)
        
        # 2. Simplified Fusion (Residual Connection)
        x_cat = torch.cat([x_m, x_e], dim=2) # (B, C, 2*n_embed)
        
        # 3. Projection
        x = self.lin3(x_cat).permute(0, 2, 1) # (B, C, pred_len) -> (B, pred_len, C)
        
        # --- Output ---
        x = self.revin_layer(x, 'denorm')
        x = x[:, :, self.target_indices]
        return x, torch.zeros_like(x)

class VanillaMamba(nn.Module):
    def __init__(self, configs):
        super(VanillaMamba, self).__init__()
        self.mamba_model = VanillaMambaBackbone(configs)
    def forward(self, x_past, x_future, x_clr):
        k_pred, _ = self.mamba_model(x_past, x_future)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, torch.zeros_like(k_pred)

class BasePowerMamba(nn.Module):
    """Same as Vanilla, just alias for requested naming"""
    def __init__(self, configs):
        super(BasePowerMamba, self).__init__()
        self.mamba_model = MMPISSM_Model_01(configs)
    def forward(self, x_past, x_future, x_clr):
        k_pred, _ = self.mamba_model(x_past, x_future)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, torch.zeros_like(k_pred)

class PhysicsResidualMamba_V02(nn.Module):
    """V02 Physics: Faiman Thermal + Efficiency Coefficients with MA Backbone"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V02, self).__init__()
        self.configs = configs
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = []  # V02 does not use time features
        try: 
            idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError: 
            idx_elev = None
        
        # V02 uses DifferentiablePVLayer_V02 (Faiman Thermal + Efficiency)
        self.physics_layer = DifferentiablePVLayer_V02(idx_G, idx_Ta, idx_WS, idx_time, idx_elev=idx_elev, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PhysicsResidualMamba_V04(nn.Module):
    """Robust Physics (V04) with MA Backbone"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V04, self).__init__()
        self.configs = configs
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']] 
        # V04 uses DifferentiablePVLayer_V4
        self.physics_layer = DifferentiablePVLayer_V4(idx_G, idx_Ta, idx_WS, idx_time, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PowerMambaPhysicsV05(nn.Module):
    """Hyper-Robust Physics (V05) with Power Mamba Backbone (MMPISSM)"""
    def __init__(self, configs):
        super(PowerMambaPhysicsV05, self).__init__()
        self.configs = configs
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']] 
        try: idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError: idx_elev = None

        self.physics_layer = DifferentiablePVLayer_V05(
            idx_G, idx_Ta, idx_WS, idx_time, idx_elev, 
            T_ref=configs.T_ref, P_stc_init=19.0
        )
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)

    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PowerMambaPhysicsV05_Advection(nn.Module):
    """
    Hyper-Robust Physics (V05) with Advection Lag + Soft Gating
    Combines:
    - V05 Physics: Diffuse split + Log cooling + Spectral
    - Advection: Dynamic lag based on wind
    - Soft Gating: Learned importance weighting
    - MMPISSM Backbone: Moving average decomposition + dual Mamba
    """
    
    def __init__(self, configs):
        super(PowerMambaPhysicsV05_Advection, self).__init__()
        self.configs = configs
        self.training_step = 0
        
        # Diagnostic tracking
        self.lag_history = []
        self.gate_history = []
        self.wind_align_history = []
        self.alpha_history = []
        
        # 1. V05 Physics Layer (Local S7)
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']]
        try:
            idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError:
            idx_elev = None

        self.physics_layer = DifferentiablePVLayer_V05(
            idx_G, idx_Ta, idx_WS, idx_time, idx_elev,
            T_ref=configs.T_ref, P_stc_init=19.0
        )
        
        # 2. Advection Modules
        self.lag_module = DynamicAdvectionLag(
            distance_m=getattr(configs, 'advection_distance_m', 1200.0),
            delta_t_sec=getattr(configs, 'advection_delta_t_sec', 900.0),
            bearing_deg=getattr(configs, 'advection_bearing_deg', 253.0),
            max_lag=getattr(configs, 'advection_max_lag', 12)
        )
        self.soft_gate = SoftAttentionGate(
            bearing_deg=getattr(configs, 'advection_bearing_deg', 253.0),
            min_gate=getattr(configs, 'advection_min_gate', 0.05)
        )
        
        # NEW: Learnable advection strength
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.alpha = torch.sigmoid(self.alpha_raw)  # Will be in [0, 1]
        
        # 3. Store S8 column indices for advection
        self.s8_cols = [c for c in configs.PAST_INPUT_COLS if c.startswith('s08_')]
        self.s7_cols = [c for c in configs.PAST_INPUT_COLS if not c.startswith('s08_') and not c.startswith('spatial_')]
        self.s8_indices = [configs.PAST_INPUT_COLS.index(c) for c in self.s8_cols]
        self.s7_indices = [configs.PAST_INPUT_COLS.index(c) for c in self.s7_cols]
        
        # 4. Wind feature indices
        try:
            self.wind_speed_idx = configs.PAST_INPUT_COLS.index('nwp_windspeed')
            self.wind_dir_idx = configs.PAST_INPUT_COLS.index('nwp_winddirection')
            self.use_past_wind = True
        except ValueError:
            self.use_past_wind = False
            try:
                self.wind_speed_idx = configs.FUTURE_INPUT_COLS.index('nwp_windspeed')
                self.wind_dir_idx = configs.FUTURE_INPUT_COLS.index('nwp_winddirection')
            except ValueError:
                self.wind_speed_idx = None
                self.wind_dir_idx = None
        
        # 5. MMPISSM Backbone (Preserve moving average decomposition)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
        print(f"[V05_Advection] Initialized with bearing={getattr(configs, 'advection_bearing_deg', 253.0):.2f}°, distance={getattr(configs, 'advection_distance_m', 1200.0):.1f}m")

    def forward(self, x_past, x_future, x_clr):
        # x_past: (B, L, D_past) - Contains S7 + S8 + Deltas
        # x_future: (B, H, D_future) - Future NWP features
        # x_clr: (B, H, 1) - Clear sky power
        
        B, L, D = x_past.shape
        
        # Extract S8 features for advection
        if len(self.s8_indices) > 0:
            x_s8 = x_past[:, :, self.s8_indices]  # (B, L, D_s8)
        else:
            x_s8 = None
        
        # Extract wind features
        if self.wind_speed_idx is not None and self.wind_dir_idx is not None:
            if self.use_past_wind:
                wind_speed = x_past[:, :, self.wind_speed_idx:self.wind_speed_idx+1]  # (B, L, 1)
                wind_dir = x_past[:, :, self.wind_dir_idx:self.wind_dir_idx+1]  # (B, L, 1)
            else:
                # Use future wind, but need to match sequence length to L
                wind_speed = x_future[:, :L, self.wind_speed_idx:self.wind_speed_idx+1]  # (B, L, 1)
                wind_dir = x_future[:, :L, self.wind_dir_idx:self.wind_dir_idx+1]  # (B, L, 1)
        else:
            wind_speed = torch.ones(B, L, 1, device=x_past.device) * 5.0  # Default 5 m/s
            wind_dir = torch.ones(B, L, 1, device=x_past.device) * 253.0  # Default bearing
        
        # Apply advection if S8 features exist
        if x_s8 is not None and x_s8.shape[2] > 0:
            # 1. Apply Advection Lag to S8 features (with interpolation)
            x_s8_shifted = self.lag_module(x_s8, wind_speed, wind_dir)
            
            # 2. Apply Learnable Alpha Blending
            alpha = torch.sigmoid(self.alpha_raw)  # (1,)
            x_s8_mixed = alpha * x_s8_shifted + (1.0 - alpha) * x_s8
            
            # 3. Apply Soft Attention Gating
            beta = self.soft_gate(wind_speed, wind_dir)  # (B, L, 1)
            beta = beta.expand(-1, -1, x_s8.shape[-1])  # (B, L, D_s8)
            x_s8_gated = x_s8_mixed * beta
            
            # 4. Reconstruct x_past with advected S8 features
            x_past_advected = x_past.clone()
            x_past_advected[:, :, self.s8_indices] = x_s8_gated
            
            # Update Alpha Property for logging
            self.alpha = alpha
        else:
            x_past_advected = x_past
            self.alpha = torch.sigmoid(self.alpha_raw)
        
        # 5. Physics-based prediction (using V05 physics)
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        
        # 6. Mamba residual (using advected past features)
        mamba_k, _ = self.mamba_model(x_past_advected, x_future)
        
        # 7. Fusion
        k_total = p_physics_k + mamba_k
        
        # 8. Power conversion
        if x_clr.ndim == 2:
            x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        
        # 9. Log diagnostics every 100 steps
        if self.training and self.training_step % 100 == 0:
            with torch.no_grad():
                # Log alpha
                self.alpha_history.append(self.alpha.item())
                
                # Log lag from module
                if hasattr(self.lag_module, '_last_lag'):
                    self.lag_history.append(self.lag_module._last_lag.cpu().item())
                
                # Log gate
                if hasattr(self.soft_gate, '_last_beta'):
                    self.gate_history.append(self.soft_gate._last_beta.mean().cpu().item())
                
                # Log wind alignment
                if self.wind_speed_idx is not None:
                    theta = torch.deg2rad(wind_dir)
                    phi = self.soft_gate.phi
                    cos_align = torch.cos(theta - phi)
                    current_align = cos_align.mean().item()
                    self.wind_align_history.append(current_align)
                    
                    print(f"[Step {self.training_step}] === ADVECTION DIAGNOSTICS ===")
                    print(f"  Alpha: {self.alpha.item():.3f} (advection strength)")
                    print(f"  Lag: {self.lag_module._last_lag.item():.2f} timesteps")
                    print(f"  Gate: {self.soft_gate._last_beta.mean().item():.3f} (avg)")
                    print(f"  Wind Align: {current_align:.3f}")
                    print(f"  ==========================================")
        
        if self.training:
            self.training_step += 1
        
        return p_total_mw, p_physics_k

class MambaPhysicsV05(nn.Module):
    """Hyper-Robust Physics (V05) with Vanilla Mamba Backbone"""
    def __init__(self, configs):
        super(MambaPhysicsV05, self).__init__()
        self.configs = configs
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']] 
        try: idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError: idx_elev = None

        self.physics_layer = DifferentiablePVLayer_V05(
            idx_G, idx_Ta, idx_WS, idx_time, idx_elev, 
            T_ref=configs.T_ref, P_stc_init=19.0
        )
        # Use Vanilla Backbone
        self.mamba_model = VanillaMambaBackbone(configs)

    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        
        # Vanilla Backbone Forward
        mamba_k, _ = self.mamba_model(x_past, x_future)
        
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PureMambaPhysicsV05(nn.Module):
    """Hyper-Robust Physics (V05) with Pure Mamba Backbone"""
    def __init__(self, configs):
        super(PureMambaPhysicsV05, self).__init__()
        self.configs = configs
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']] 
        try: idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError: idx_elev = None

        self.physics_layer = DifferentiablePVLayer_V05(
            idx_G, idx_Ta, idx_WS, idx_time, idx_elev, 
            T_ref=configs.T_ref, P_stc_init=19.0
        )
        # Use Pure Backbone
        self.mamba_model = MambaPureBackbone(configs, output_residual=True)

    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        
        mamba_k, _ = self.mamba_model(x_past, x_future)
        
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class NWPModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        try: self.nwp_idx = configs.FUTURE_INPUT_COLS.index('NWP_Power_MW')
        except ValueError: self.nwp_idx = 0
    def forward(self, x_past, x_future, x_clr):
        nwp_mw = x_future[:, :, self.nwp_idx].unsqueeze(-1)
        return nwp_mw, torch.zeros_like(nwp_mw)

class SmartPersistenceModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        try: self.k_idx = configs.PAST_INPUT_COLS.index('K_PV')
        except ValueError: self.k_idx = 0 
    def forward(self, x_past, x_future, x_clr):
        lag = 96
        H = self.configs.pred_len
        k_past = x_past[:, :, self.k_idx]
        if k_past.shape[1] >= lag: k_pred = k_past[:, -H:].unsqueeze(-1)
        else: k_pred = torch.zeros(x_past.shape[0], H, 1, device=x_past.device)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, k_pred

# =============================================================================
# Data & Training Steps
# =============================================================================

class V02Dataset(Dataset):
    def __init__(self, X, y, P_clr, Bf, seq, pred):
        self.X=X; self.y=y; self.P_clr=P_clr; self.Bf=Bf; self.seq=seq; self.pred=pred
    def __len__(self): return len(self.X) - self.seq - self.pred + 1
    def __getitem__(self, i):
        start_fut = i + self.seq
        end_fut = start_fut + self.pred
        x_clr = self.P_clr[start_fut:end_fut]
        if x_clr.ndim==1: x_clr=x_clr[:,None]
        y_fut = self.y[start_fut:end_fut]
        if y_fut.ndim==1: y_fut=y_fut[:,None]
        return (self.X[i:i+self.seq].astype(np.float32), 
                y_fut.astype(np.float32), 
                self.Bf[start_fut:end_fut].astype(np.float32),
                np.zeros_like(y_fut), 
                x_clr.astype(np.float32))

def prepare_rolling_folds(df, input_cols, target_col, prediction_cols, n_splits=2, seq_len=96, pred_len=96):
    common_features = [f for f in prediction_cols if f in input_cols]
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date_time' in df.columns:
            df = df.copy()
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.set_index('date_time')
    df = df.fillna(0)
    X_raw = df[input_cols].values.astype(np.float32)
    P_raw = df['P_CLR'].values.astype(np.float32) if 'P_CLR' in df.columns else np.zeros(len(df), dtype=np.float32)
    y_raw = df[target_col].values.astype(np.float32)
    Bf_raw = df[common_features].values.astype(np.float32)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (train_index, test_index) in enumerate(tscv.split(X_raw)):
        train_start, train_end = df.index[train_index[0]], df.index[train_index[-1]]
        test_start, test_end = df.index[test_index[0]], df.index[test_index[-1]]
        if isinstance(df.index, pd.DatetimeIndex):
            test_ts_raw = df.index[test_index]
            test_timestamps = test_ts_raw.tz_localize('UTC') if test_ts_raw.tz is None else test_ts_raw.tz_convert('UTC')
        else: test_timestamps = None
        dates = {'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end, 'test_timestamps': test_timestamps}
        X_train, P_train, y_train = X_raw[train_index], P_raw[train_index], y_raw[train_index]
        X_test, P_test, y_test = X_raw[test_index], P_raw[test_index], y_raw[test_index]
        Bf_train, Bf_test = Bf_raw[train_index], Bf_raw[test_index]
        X_test_final = np.concatenate([X_train[-seq_len:], X_test], axis=0)
        P_test_final = np.concatenate([P_train[-seq_len:], P_test], axis=0)
        y_test_final = np.concatenate([y_train[-seq_len:], y_test], axis=0)
        Bf_test_final = np.concatenate([Bf_train[-seq_len:], Bf_test], axis=0)
        train_ds = V02Dataset(X_train, y_train, P_train, Bf_train, seq_len, pred_len)
        test_ds = V02Dataset(X_test_final, y_test_final, P_test_final, Bf_test_final, seq_len, pred_len)
        yield (DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0),
               DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0), {}, dates)

def train_one_epoch(model, loader, optimizer, device, loss_fn, warmup=False):
    model.train()
    if hasattr(model, 'physics_layer') and hasattr(model, 'mamba_model'):
        if warmup:
            for param in model.mamba_model.parameters(): param.requires_grad = False
            for param in model.physics_layer.parameters(): param.requires_grad = True
        else:
            for param in model.mamba_model.parameters(): param.requires_grad = True
            for param in model.physics_layer.parameters(): param.requires_grad = True
            
    total_loss = 0.0
    for batch in loader:
        x_past, y_k_target, x_fut, _, x_clr = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3], batch[4].to(device)
        optimizer.zero_grad()
        p_mw_pred, _ = model(x_past, x_fut, x_clr)
        p_mw_target = y_k_target * (x_clr + 1e-6)
        loss = loss_fn(p_mw_pred, p_mw_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, capacity=20.0):
    model.eval()
    preds_mw, acts_mw = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x_past, y_k_target, x_fut, _, x_clr = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3], batch[4].to(device)
            p_mw_pred, _ = model(x_past, x_fut, x_clr)
            p_mw_target = y_k_target * (x_clr + 1e-6)
            total_loss += F.mse_loss(p_mw_pred, p_mw_target).item()
            preds_mw.append(p_mw_pred.cpu().numpy())
            acts_mw.append(p_mw_target.cpu().numpy())
    preds = np.concatenate(preds_mw, axis=0).flatten()
    acts = np.concatenate(acts_mw, axis=0).flatten()
    rmse = np.sqrt(np.mean((preds - acts)**2))
    mae = np.mean(np.abs(preds - acts))
    return {'rmse': rmse, 'mae': mae, 'nrmse': (rmse/capacity)*100, 'nmae': (mae/capacity)*100, 'mse_loss': total_loss/len(loader)}, preds, acts, np.concatenate(preds_mw, axis=0), np.concatenate(acts_mw, axis=0)

def plot_fold_validation(fold_i, preds_dict, acts, dates_str, timestamps=None):
    fig = go.Figure()
    if timestamps is not None and len(timestamps) == len(acts):
        fig.add_trace(go.Scatter(x=timestamps, y=acts, mode='lines', name='Actual MW', line=dict(color='black', width=2), opacity=0.6))
    else:
        fig.add_trace(go.Scatter(y=acts, mode='lines', name='Actual MW', line=dict(color='black', width=2), opacity=0.6))
    colors = {
        'Vanilla Mamba': 'gray', 'Base Power Mamba': 'blue', 
        'Physics V02 (Faiman Thermal)': 'green',
        'Physics V04 (Robust)': 'orange', 
        'Power Mamba Physics V05 (Hyper-Robust)': 'red',
        'Power Mamba Physics V05 + Advection': 'darkred',
        'Pure Mamba Physics V05 (Hyper-Robust)': 'lime',
        'Mamba Physics V05 (Hyper-Robust)': 'purple',
        'Smart Persistence': 'cyan', 'NWP Baseline': 'magenta'
    }
    for name, pred in preds_dict.items():
        c = colors.get(name, 'black')
        if timestamps is not None and len(timestamps) == len(pred):
            fig.add_trace(go.Scatter(x=timestamps, y=pred, mode='lines', name=name, line=dict(color=c, width=1.5), opacity=0.8))
        else:
            fig.add_trace(go.Scatter(y=pred, mode='lines', name=name, line=dict(color=c, width=1.5), opacity=0.8))
    fig.update_layout(title=f'Fold {fold_i} Validation: {dates_str}', xaxis_title='Time', yaxis_title='Power (MW)', template='plotly_white')
    save_path = f"results/fold_{fold_i}_spatial_v05.html"
    fig.write_html(save_path)
    print(f"Validation plot saved to {save_path}")

def plot_fold_convergence_comparison(fold_i, fold_histories_dict, save_path=None):
    plt.figure(figsize=(12, 8))
    colors = {
        'Vanilla Mamba': 'gray', 'Base Power Mamba': 'blue',
        'Physics V02 (Faiman Thermal)': 'green',
        'Physics V04 (Robust)': 'orange', 
        'Power Mamba Physics V05 (Hyper-Robust)': 'red',
        'Power Mamba Physics V05 + Advection': 'darkred',
        'Pure Mamba Physics V05 (Hyper-Robust)': 'lime',
        'Mamba Physics V05 (Hyper-Robust)': 'purple',
        'Smart Persistence': 'cyan'
    }
    
    for name, history in fold_histories_dict.items():
        if name in ['Smart Persistence', 'NWP Baseline']: continue
        train_loss = history['train']
        val_loss = history['val']
        epochs = range(1, len(train_loss) + 1)
        c = colors.get(name, 'black')
        plt.plot(epochs, train_loss, label=f'{name} Train', color=c, linestyle='-', linewidth=1.5)
        plt.plot(epochs, val_loss, label=f'{name} Val', color=c, linestyle='--', linewidth=1.5, alpha=0.7)

    plt.title(f'Fold {fold_i}: Convergence V04 vs V05')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = save_path or f"results/fold_{fold_i}_convergence_v05.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Benchmark
# =============================================================================

def collect_advection_diagnostics(model, loader, device):
    """
    Collect lag, gate, alpha and wind alignment diagnostics over dataset.
    
    Args:
        model: Trained PowerMambaPhysicsV05_Advection model
        loader: DataLoader for evaluation
        device: torch device
    
    Returns:
        Dictionary with diagnostic arrays
    """
    model.eval()
    lags, gates, alphas, wind_aligns = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            x7, x8, w, xf, xk = batch[0], batch[1], batch[2], batch[4], batch[3] # mapping from prepare_folds: X, y, Bf, zeros, P_clr
            # Wait, verify loader format from prepare_rolling_folds (dataset __getitem__)
            # ds returns: X, y, Bf, zeros, P_clr
            # train_one_epoch: x_past, y_k, x_fut, _, x_clr
            
            x7 = batch[0].to(device)
            # x8 is not separate, it's inside x7 (past features)
            # w (wind) is inside x7 or xf
            
            xf = batch[2].to(device) # Bf (Future)
            xc = batch[4].to(device) # P_clr
            
            # Forward pass to populate diagnostics
            _ = model(x7, xf, xc)
            
            # Collect from modules
            if hasattr(model.lag_module, '_last_lag'):
                lags.append(model.lag_module._last_lag.cpu().numpy())
            
            if hasattr(model.lag_module, '_last_alpha'):
                alphas.append(model.lag_module._last_alpha.cpu().numpy())
            
            if hasattr(model.soft_gate, '_last_beta'):
                gates.append(model.soft_gate._last_beta.mean(dim=1).cpu().numpy())
            
            # Wind alignment
            # Need to extract wind from x7 or xf based on model config
            if hasattr(model, 'wind_speed_idx') and model.wind_speed_idx is not None:
                if model.use_past_wind:
                    w_dir = x7[:, :, model.wind_dir_idx:model.wind_dir_idx+1]
                else:
                    w_dir = xf[:, :model.configs.seq_len, model.wind_dir_idx:model.wind_dir_idx+1]
                
                theta = torch.deg2rad(w_dir)
                phi = model.soft_gate.phi
                cos_align = torch.cos(theta - phi)
                wind_aligns.append(cos_align.mean(dim=1).cpu().numpy())
    
    return {
        'lags': np.array(lags) if lags else None,
        'gates': np.concatenate(gates) if gates else None,
        'alphas': np.array(alphas) if alphas else None,
        'wind_aligns': np.concatenate(wind_aligns) if wind_aligns else None
    }

def plot_advection_diagnostics(diagnostics, save_path='advection_diagnostics.html'):
    """
    Create interactive Plotly diagnostic plots.
    
    Args:
        diagnostics: Dictionary with lag, gate, alpha, wind_align arrays
        save_path: Path to save HTML file
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Lag Distribution', 'Gate Distribution',
        'Wind Alignment', 'Alpha Evolution'
    ])
    
    # 1. Lag Distribution
    if diagnostics['lags'] is not None:
        fig.add_trace(go.Histogram(
            x=diagnostics['lags'].flatten(),
            name='Lag (timesteps)',
            marker_color='rgba(100, 150, 50, 0.7)',
            marker_line_color='rgba(100, 150, 50, 1.0)'
        ), row=1, col=1)
        fig.update_xaxes(title_text="Lag Distribution", row=1, col=1)
    
    # 2. Gate Distribution
    if diagnostics['gates'] is not None:
        fig.add_trace(go.Histogram(
            x=diagnostics['gates'].flatten(),
            name='Gate β',
            marker_color='rgba(150, 100, 50, 0.7)',
            marker_line_color='rgba(150, 100, 50, 1.0)'
        ), row=1, col=2)
        fig.update_xaxes(title_text="Gate Distribution", row=1, col=2)
    
    # 3. Wind Alignment Distribution
    if diagnostics['wind_aligns'] is not None:
        fig.add_trace(go.Histogram(
            x=diagnostics['wind_aligns'].flatten(),
            name='cos(θ-φ)',
            marker_color='rgba(50, 150, 100, 0.7)',
            marker_line_color='rgba(50, 150, 100, 1.0)'
        ), row=2, col=1)
        fig.update_xaxes(title_text="Wind Alignment", row=2, col=1)
    
    # 4. Alpha Evolution
    if diagnostics['alphas'] is not None:
        fig.add_trace(go.Scatter(
            y=diagnostics['alphas'].flatten(),
            mode='lines',
            name='Alpha',
            line=dict(color='green', width=2)
        ), row=2, col=2)
        fig.update_yaxes(title_text="Alpha Evolution", row=2, col=2)
        
        # Add reference lines
        fig.add_hline(y=0.5, line_dash="dot", line_color="blue",
                     annotation_text="Midpoint: 0.5", row=2, col=2)
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                     annotation_text="Pure Advection: 1.0", row=2, col=2)
        fig.add_hline(y=0.0, line_dash="solid", line_color="green",
                     annotation_text="No Advection: 0.0", row=2, col=2)
    
    fig.update_layout(
        title="Advection Model Diagnostics",
        showlegend=True,
        hovermode='x unified',
        height=800,
        width=1200
    )
    fig.write_html(save_path)
    print(f"Diagnostics saved to {save_path}")

def run_comparative_benchmark(df, n_splits=2, custom_config=None):
    print(">>> Running V05 Benchmark (No Wavelet, Fixed Loss, Adaptive Removed)")
    
    # Create output directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # NEW: Load metadata for geometry
    try:
        # Try to find metadata.csv relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to PINNS parent, then to PVODdatasets_v1
        meta_path = os.path.join(current_dir, "../PVODdatasets_v1/metadata.csv")
        if not os.path.exists(meta_path):
             # Try alternate location
             meta_path = os.path.join(current_dir, "metadata.csv")
             
        if os.path.exists(meta_path):
            print(f"Loading metadata from {meta_path}")
            metaData = pd.read_csv(meta_path)
            s07_meta = metaData[metaData['Station_ID'] == 'station07'].iloc[0].to_dict()
            s08_meta = metaData[metaData['Station_ID'] == 'station08'].iloc[0].to_dict()
        else:
            print("Warning: metadata.csv not found. Using default geometry.")
            s07_meta, s08_meta = None, None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        s07_meta, s08_meta = None, None
    
    if custom_config is not None:
        configs = custom_config
        if not hasattr(configs, 'st07_indices'):
            st07_future_cols = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation']
            configs.st07_indices = [configs.FUTURE_INPUT_COLS.index(c) for c in st07_future_cols if c in configs.FUTURE_INPUT_COLS]
            configs.st07_future_cols = st07_future_cols
        
        # Set coordinates if available
        if s07_meta is not None and s08_meta is not None and hasattr(configs, 'set_station_coordinates'):
            configs.set_station_coordinates(s07_meta, s08_meta)
            
    else:
        configs = SpatialMambaConfigs(n_splits=n_splits)
        # Set coordinates if available
        if s07_meta is not None and s08_meta is not None:
            configs.set_station_coordinates(s07_meta, s08_meta)
        
    device = configs.device
    total_epochs = getattr(configs, 'epochs', 30)
    warmup_epochs = getattr(configs, 'warmup_epochs', 5)
    
    model_classes = {
        'Vanilla Mamba': VanillaMamba,
        'Base Power Mamba': BasePowerMamba,
        'Physics V02 (Faiman Thermal)': PhysicsResidualMamba_V02,
        'Physics V04 (Robust)': PhysicsResidualMamba_V04,
        'Power Mamba Physics V05 (Hyper-Robust)': PowerMambaPhysicsV05,
        'Power Mamba Physics V05 + Advection': PowerMambaPhysicsV05_Advection,
        'Pure Mamba Physics V05 (Hyper-Robust)': PureMambaPhysicsV05,
        'Mamba Physics V05 (Hyper-Robust)': MambaPhysicsV05,
        'Smart Persistence': SmartPersistenceModel,
        'NWP Baseline': NWPModel
    }
    
    models, optimizers, loss_fns = {}, {}, {}
    print(f"Initializing {len(model_classes)} models on {device}...")
    for name, ModelClass in model_classes.items():
        print(f"  -> Initializing {name}...", end="", flush=True)
        t0 = time.time()
        m = ModelClass(configs).to(device)
        print(f" Done ({time.time()-t0:.2f}s)")
        
        models[name] = m
        if name in ['Smart Persistence', 'NWP Baseline']:
            optimizers[name] = None
            loss_fns[name] = F.mse_loss
        elif 'Physics' in name:
            optimizers[name] = torch.optim.Adam(m.parameters(), lr=1e-3)
            loss_fns[name] = PhysicsCompositeLoss_Fixed(smooth_weight=0.1).to(device)
        else:
            optimizers[name] = torch.optim.Adam(m.parameters(), lr=1e-3)
            loss_fns[name] = F.mse_loss

    final_metrics = {k: {'rmse': [], 'mae': [], 'nrmse': [], 'nmae': [], 'time_s': []} for k in model_classes.keys()}
    all_fold_histories = {}
    
    # Initialize parameter trackers for physics models
    param_trackers = {}
    if TRACKING_AVAILABLE:
        for name in model_classes.keys():
            if 'Physics' in name:
                param_trackers[name] = ParameterTracker(name)
        print(f"[ParameterTracker] Tracking enabled for {len(param_trackers)} physics models") 
    
    fold_gen = prepare_rolling_folds(df, configs.PAST_INPUT_COLS, 'K_PV', configs.FUTURE_INPUT_COLS, n_splits=n_splits, seq_len=configs.seq_len, pred_len=configs.pred_len)
    
    for fold_i, (train_loader, test_loader, _, dates) in enumerate(fold_gen):
        print(f"\n=== Fold {fold_i+1}/{n_splits} ===")
        fold_preds, acts = {}, None
        fold_histories = {}
        
        for name, model in models.items():
            print(f"Training {name}...", end="")
            start_time = time.time()
            epochs = 0 if name in ['Smart Persistence', 'NWP Baseline'] else total_epochs
            
            fold_histories[name] = {'train': [], 'val': []}
            
            for ep in range(epochs):
                warmup = ('Physics' in name and ep < warmup_epochs)
                train_loss = train_one_epoch(model, train_loader, optimizers[name], device, loss_fn=loss_fns[name], warmup=warmup)
                fold_histories[name]['train'].append(train_loss)
                
                model.eval()
                with torch.no_grad():
                    val_mets, _, _, _, _ = evaluate(model, test_loader, device)
                    fold_histories[name]['val'].append(val_mets['mse_loss'])
                model.train()
                
                # Track parameters for physics models
                if TRACKING_AVAILABLE and name in param_trackers:
                    param_trackers[name].record_epoch(model, fold_i, ep)
            
            # Post-training: Collect Advection Diagnostics (If applicable)
            if name == 'Power Mamba Physics V05 + Advection':
                print(" [Collecting Diagnostics] ", end="")
                try:
                    diagnostics = collect_advection_diagnostics(model, test_loader, device)
                    plot_advection_diagnostics(
                        diagnostics,
                        save_path=f"results/advection_diagnostics_fold{fold_i+1}.html"
                    )
                    
                    # Print summary
                    if diagnostics['lags'] is not None:
                        print(f"\n    Lag: μ={diagnostics['lags'].mean():.2f}, σ={diagnostics['lags'].std():.2f}")
                        print(f"    Gate: μ={diagnostics['gates'].mean():.3f}")
                        print(f"    Alpha: μ={diagnostics['alphas'].mean():.3f}, Final={diagnostics['alphas'][-1]:.3f}")
                        print(f"    Align: μ={diagnostics['wind_aligns'].mean():.3f}")
                except Exception as e:
                    print(f"[Error collecting diagnostics] {e}")

            mets, pred, act, pred_raw, act_raw = evaluate(model, test_loader, device)
            fold_preds[name] = pred_raw[:, 0, 0]
            if acts is None: acts = act_raw[:, 0, 0]
            
            final_metrics[name]['rmse'].append(mets['rmse'])
            final_metrics[name]['mae'].append(mets['mae'])
            final_metrics[name]['nrmse'].append(mets['nrmse'])
            final_metrics[name]['nmae'].append(mets['nmae'])
            final_metrics[name]['time_s'].append(time.time() - start_time)
            print(f" RMSE: {mets['rmse']:.3f} MW | MAE: {mets['mae']:.3f} MW | nRMSE: {mets['nrmse']:.2f}%")
            
            # Save Model
            if name not in ['Smart Persistence', 'NWP Baseline']:
                torch.save(model.state_dict(), f"saved_models/{name.replace(' ', '_')}_fold{fold_i+1}.pth")
        
        # Save Histories
        all_fold_histories[fold_i+1] = fold_histories
        
        # Save Histories
        import json
        with open(f"results/history_fold{fold_i+1}.json", 'w') as f:
            # Helper to make tensors serializable
            def default(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, torch.Tensor): return obj.item()
                return str(obj)
            json.dump(all_fold_histories[fold_i+1], f, default=default)
        plot_fold_convergence_comparison(fold_i+1, fold_histories, save_path=f"results/fold_{fold_i+1}_convergence_v05.png")
        
        try:
            test_ts = dates['test_timestamps']
            flat_ts = test_ts[:len(acts)].tz_convert('Asia/Shanghai') if test_ts is not None and len(test_ts) >= len(acts) else None
        except:
            flat_ts = None
        plot_fold_validation(fold_i+1, fold_preds, acts, f"{dates['test_start']} - {dates['test_end']}", timestamps=flat_ts)
    
    print("\n>>> FINAL RESULTS <<<")
    aggregated = {}
    for name in model_classes.keys():
        aggregated[name] = {
            'rmse': np.mean(final_metrics[name]['rmse']), 
            'mae': np.mean(final_metrics[name]['mae']),
            'nrmse': np.mean(final_metrics[name]['nrmse']),
            'nmae': np.mean(final_metrics[name]['nmae'])
        }
        print(f"{name}: RMSE={aggregated[name]['rmse']:.3f} | MAE={aggregated[name]['mae']:.3f} | nRMSE={aggregated[name]['nrmse']:.2f}% | nMAE={aggregated[name]['nmae']:.2f}%")
    
    # Generate parameter tracking plots
    if TRACKING_AVAILABLE and param_trackers:
        tracking_dir = 'results/parameter_tracking'
        os.makedirs(tracking_dir, exist_ok=True)
        for name, tracker in param_trackers.items():
            tracker.generate_all_plots(tracking_dir)
        # Generate combined comparison
        generate_combined_comparison(param_trackers, tracking_dir)
        print(f"\n[ParameterTracker] All plots saved to {tracking_dir}/")
    return aggregated

if __name__ == "__main__":
    print("Spatial Physics-Residual Mamba V05 (Pure V05/V04 comparison, No Wavelet)")
