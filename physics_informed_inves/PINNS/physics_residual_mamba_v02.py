"""
Physics-Residual Power Mamba Architecture (V02 - K_PV Variant) & 7-Way Benchmark Suite

Models:
1. VanillaMamba: Simple Mamba/LSTM baseline (K_PV)
2. BasePowerMamba: Data-Only (K_PV) using MMPISSM backbone
3. PhysicsResidualMamba_V02: Hybrid (Basic Physics + Residual K_PV)
4. PhysicsResidualMamba_V3: Hybrid (Geometric Gating Tilt Fix + Residual K_PV)
5. PhysicsResidualMamba_V4: Hybrid (Robust IAM/Inverter + Residual K_PV)
6. SmartPersistenceModel: K_PV(t) = K_PV(t-24h)
7. NWPModel: Direct NWP Power Forecast

Benchmark:
- Compares all 7 on K_PV target.
- Metrics: RMSE (MW), MAE (MW), nRMSE (%), Time (s).
- Plots: Loss Curves, Metric Bars, Interactive Per-Fold Plots (Plotly), HTML Output.
- Training: Rolling/Persistent (Weights are NOT reset between folds).
"""

import numpy as np


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# Reuse Helper Classes from original if possible, but defining here for standalone usage
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


# =============================================================================
# Part 1: Layers & Base Blocks
# =============================================================================

class SmoothnessLoss(nn.Module):
    """Total Variation Penalty on Time Dimension."""
    def __init__(self, weight=0.1):
        super(SmoothnessLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred):
        # pred: (B, T, 1) or (B, T)
        if pred.ndim == 3:
            diff = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        else:
            diff = torch.abs(pred[:, 1:] - pred[:, :-1])
        return self.weight * diff.mean()

class PhysicsCompositeLoss(nn.Module):
    """MSE + Smoothness Penalty"""
    def __init__(self, smooth_weight=0.1):
        super(PhysicsCompositeLoss, self).__init__()
        self.smooth_loss = SmoothnessLoss(weight=smooth_weight)
        
    def forward(self, pred, target):
        loss_mse = F.mse_loss(pred, target)
        loss_smooth = self.smooth_loss(pred)
        return loss_mse + loss_smooth


class DifferentiablePVLayer_V02(nn.Module):
    """Basic Physics Layer (V02)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, idx_elev=None, T_ref=25.0, G_stc=1000.0, P_stc_init=19.0, eta_init=0.16):
        super().__init__()
        self.idx_G, self.idx_Ta, self.idx_WS = idx_G, idx_Ta, idx_WS
        self.idx_elev = idx_elev
        self.idx_time_feats = idx_time_feats 
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6
        
        # Base Physics Parameters (V0 Logic)
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4, dtype=torch.float32)) 
        self.U0_raw = nn.Parameter(torch.tensor(3.2, dtype=torch.float32)) 
        self.U1_raw = nn.Parameter(torch.tensor(1.9, dtype=torch.float32)) 
        self.eta_raw = nn.Parameter(torch.tensor(-1.66)) 
        
        # Boundary Constraint: P_stc <= P_stc_init
        self.P_stc_limit = float(P_stc_init)
        self.Pstc_raw = nn.Parameter(torch.tensor(6.0)) # Start near max (sigmoid(6) ~ 0.99) 
        self.scaler = None

    def forward(self, x_future, x_clr=None):
        # 1. Extract Physics Inputs
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        
        # 2. Parameter Activation
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw) 
        
        # Enforce P_stc <= P_stc_limit
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        # 3. Faiman Temperature Model
        T_cell = Ta + G / (U0 + U1 * WS + self.eps)

        # 4. DC Power Generation
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = eta * P_stc * (G / self.G_stc) * temp_factor
        
        P_final_mw = F.relu(P_dc).unsqueeze(-1)
        
        # --- K_PV Mode: Return K_phys (Index) ---
        if x_clr is not None:
             if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
             # Avoid div by zero
             K_phys = P_final_mw / (x_clr + self.eps)
             
             # Night Mask
             night_mask = (x_clr > 0.01).float()
             
             # Astronomical Boundary Constraint (Strict Night Zero)
             if self.idx_elev is not None:
                 elev = x_future[:, :, self.idx_elev]
                 day_mask = (elev > 0.0).float().unsqueeze(-1)
                 K_phys = K_phys * day_mask
                 
             K_phys = K_phys * night_mask
             
             return K_phys # Dimensionless

        # Astronomical Boundary Constraint (Strict Night Zero for Power Mode)
        if self.idx_elev is not None:
             elev = x_future[:, :, self.idx_elev]
             day_mask = (elev > 0.0).float().unsqueeze(-1)
             P_final_mw = P_final_mw * day_mask

        return P_final_mw # Fallback

class DifferentiablePVLayer_V3(DifferentiablePVLayer_V02):
    """V3: Geometric Gating (Tilt Fix)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs):
        super().__init__(idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs)
        # GeoNet MLP for Tilt Factor
        self.geo_net = nn.Sequential(
            nn.Linear(len(idx_time_feats), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Strictly positive tilt factor
        )
        
    def forward(self, x_future, x_clr=None):
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        time_feats = x_future[:, :, self.idx_time_feats]
        
        # --- V3 Logic: Calculate G_POA ---
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        G_poa = G * tilt_factor
        
        # Parameters
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)
        
        # Enforce P_stc <= P_stc_limit
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        # Faiman with G_POA
        T_cell = Ta + G_poa / (U0 + U1 * WS + self.eps)
        
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = eta * P_stc * (G_poa / self.G_stc) * temp_factor
        
        P_final_mw = F.relu(P_dc).unsqueeze(-1)
        
        if x_clr is not None:
             if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
             K_phys = P_final_mw / (x_clr + self.eps)
             night_mask = (x_clr > 0.01).float()
             K_phys = K_phys * night_mask
             return K_phys
        return P_final_mw

class DifferentiablePVLayer_V4(DifferentiablePVLayer_V3):
    """V4: Robust (IAM + Non-Linear Inverter)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs):
        super().__init__(idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs)
        # V4 Extra Parameters
        self.b0_raw = nn.Parameter(torch.tensor(0.05)) # IAM Coefficient
        self.inv_k_raw = nn.Parameter(torch.tensor(1.0)) # Inverter Steepness
        self.inv_thresh_raw = nn.Parameter(torch.tensor(0.1)) # Inverter Threshold (MW)

    def forward(self, x_future, x_clr=None):
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        time_feats = x_future[:, :, self.idx_time_feats]
        
        # --- V3 Logic: Tilt ---
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        G_poa = G * tilt_factor
        
        # --- V4 Logic A: IAM ---
        # IAM = 1 - b0 * (1 - tilt_factor) (approx)
        b0 = torch.sigmoid(self.b0_raw) * 0.2 # Bound small positive
        IAM = 1.0 - b0 * (1.0 - tilt_factor)
        IAM = torch.clamp(IAM, 0.0, 1.0)
        
        G_eff = G_poa * IAM # Effective Irradiance hitting cells
        
        # Parameters
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta_module = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)
        
        # Enforce P_stc <= P_stc_limit
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        # DC Power
        T_cell = Ta + G_eff / (U0 + U1 * WS + self.eps)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = eta_module * P_stc * (G_eff / self.G_stc) * temp_factor
        P_dc = F.relu(P_dc)
        
        # --- V4 Logic B: Inverter Efficiency ---
        # eta_inv = Sigmoid(k * (P_dc - P_thresh))
        inv_k = F.softplus(self.inv_k_raw) * 10.0 # Make it sharp
        inv_thresh = F.softplus(self.inv_thresh_raw)
        
        eta_inv = torch.sigmoid(inv_k * (P_dc - inv_thresh))
        P_ac = P_dc * eta_inv
        
        P_final_mw = P_ac.unsqueeze(-1)
        
        if x_clr is not None:
             if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
             K_phys = P_final_mw / (x_clr + self.eps)
             night_mask = (x_clr > 0.01).float()
             K_phys = K_phys * night_mask
             return K_phys
        return P_final_mw

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
    print("Warning: mamba_ssm not available. Taking Identity.")

class MMPISSM_Model_01(nn.Module):
    """Base Mamba Backbone"""
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
        self.decompsition = series_decomp(self.configs.kernel_size)
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
        seasonal_init, trend_init = self.decompsition(x)
        x = torch.cat([seasonal_init, trend_init], dim=1)
        x = self.lin1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.revin_layer_enc(x, 'denorm')
        x = self.revin_layer(x, 'norm')
        seasonal_init, trend_init = self.decompsition(x)
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
# Part 2: Models
# =============================================================================

class PhysicsResidualMamba(nn.Module):
    """V02 Basic"""
    def __init__(self, configs):
        super(PhysicsResidualMamba, self).__init__()
        self.configs = configs
        idx_G = configs.common_features.index('nwp_globalirrad')
        idx_Ta = configs.common_features.index('nwp_temperature')
        idx_WS = configs.common_features.index('nwp_windspeed')
        
        # Try to find elevation index if available
        try:
            idx_elev = configs.common_features.index('sun_elevation')
        except ValueError:
            idx_elev = None
            
        self.physics_layer = DifferentiablePVLayer_V02(idx_G, idx_Ta, idx_WS, [], idx_elev=idx_elev, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        p_physics_k = self.physics_layer(x_future, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PhysicsResidualMamba_V3(nn.Module):
    """V3 Geometric"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V3, self).__init__()
        self.configs = configs
        idx_G = configs.common_features.index('nwp_globalirrad')
        idx_Ta = configs.common_features.index('nwp_temperature')
        idx_WS = configs.common_features.index('nwp_windspeed')
        # Time feats: hour_sin, hour_cos...
        idx_time = [configs.FUTURE_INPUT_COLS.index(c) for c in ['hour_sin', 'hour_cos']] 
        
        self.physics_layer = DifferentiablePVLayer_V3(idx_G, idx_Ta, idx_WS, idx_time, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        p_physics_k = self.physics_layer(x_future, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PhysicsResidualMamba_V4(nn.Module):
    """V4 Robust"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V4, self).__init__()
        self.configs = configs
        idx_G = configs.common_features.index('nwp_globalirrad')
        idx_Ta = configs.common_features.index('nwp_temperature')
        idx_WS = configs.common_features.index('nwp_windspeed')
        idx_time = [configs.FUTURE_INPUT_COLS.index(c) for c in ['hour_sin', 'hour_cos']] 
        
        self.physics_layer = DifferentiablePVLayer_V4(idx_G, idx_Ta, idx_WS, idx_time, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        p_physics_k = self.physics_layer(x_future, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class BasePowerMamba(nn.Module):
    """Base Model: Training directly on K_PV"""
    def __init__(self, configs):
        super(BasePowerMamba, self).__init__()
        self.mamba_model = MMPISSM_Model_01(configs)
        
    def forward(self, x_past, x_future, x_clr):
        k_pred, _ = self.mamba_model(x_past, x_future)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, torch.zeros_like(k_pred)

class VanillaMamba(nn.Module):
    """Vanilla Model"""
    def __init__(self, configs):
        super(VanillaMamba, self).__init__()
        self.mamba_model = MMPISSM_Model_01(configs)
        
    def forward(self, x_past, x_future, x_clr):
        k_pred, _ = self.mamba_model(x_past, x_future)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, torch.zeros_like(k_pred)

class SmartPersistenceModel(nn.Module):
    """Smart Persistence: K_PV(t) = K_PV(t-24h)"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        # Attempt to find K_PV in PAST inputs; default to 0 if not found
        try:
            self.k_idx = configs.PAST_INPUT_COLS.index('K_PV')
        except ValueError:
            self.k_idx = 0 
            
    def forward(self, x_past, x_future, x_clr):
        # We need the LAST 'pred_len' steps of info from x_past if they represent -24h?
        # Using 96 steps as 24h lag assumption for 15min data
        lag = 96
        # If x_past is long enough (e.g. 960), we take the segment that corresponds to t-24h for each step?
        # Persistence Logic: Prediction[t] = Observation[t-lag]
        # x_past[:, -lag:] is the observation[t-lag ... t] approx
        
        # Taking the last 'pred_len' from x_past as basic persistence
        # (Assuming tomorrow looks like yesterday)
        H = self.configs.pred_len
        k_past = x_past[:, :, self.k_idx]
        
        # Safe slice
        if k_past.shape[1] >= lag:
            # Approx: Just take the last H steps of known K_PV 
            # This effectively says "Forecast = Recent History"
            # For 24h persistence specifically, we'd need aligned sequence.
            # Using simple persistence: Forecast = Last known values
            k_pred = k_past[:, -H:].unsqueeze(-1)
        else:
            k_pred = torch.zeros(x_past.shape[0], H, 1, device=x_past.device)
            
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, k_pred

class NWPModel(nn.Module):
    """Direct NWP: Prediction = NWP_Power_MW"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        try:
            self.nwp_idx = configs.FUTURE_INPUT_COLS.index('NWP_Power_MW')
        except ValueError:
            self.nwp_idx = 0
            
    def forward(self, x_past, x_future, x_clr):
        # x_future has NWP_Power_MW
        nwp_mw = x_future[:, :, self.nwp_idx].unsqueeze(-1)
        return nwp_mw, torch.zeros_like(nwp_mw)

# =============================================================================
# Part 3: Dataset & Configs
# =============================================================================

class PhysicsResidualMambaConfigs:
    def __init__(self, n_splits=4):
        self.seq_len = 96
        self.pred_len = 96
        self.TARGET_COL = ['K_PV'] # All models target K_PV
        
        self.PAST_INPUT_COLS = [
            'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 
            'power', 'K_PV', 'P_CLR', 'NWP_Power_MW', 
            'hour_sin', 'hour_cos'
        ]
        self.FUTURE_INPUT_COLS = [
            'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed',
            'NWP_Power_MW', 'hour_sin', 'hour_cos'
        ]
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
        self.lambda_data = 1.0
        self.lambda_night = 0.2
        self.lambda_mono = 0.1
        self.T_ref = 25.0
        self.G_night_thr = 10.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # ===== FIX 1: Ensure df has a DatetimeIndex =====
    # Check if index is already a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"DEBUG: df.index is {type(df.index).__name__}, not DatetimeIndex.")
        # Try to set 'date_time' column as index if it exists
        if 'date_time' in df.columns:
            print("DEBUG: Found 'date_time' column. Setting as index.")
            df = df.copy()
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.set_index('date_time')
        else:
            print("WARNING: No 'date_time' column found. Timestamps will be incorrect!")
    else:
        print(f"DEBUG: df.index is DatetimeIndex. Range: {df.index.min()} to {df.index.max()}")
    
    df = df.fillna(0)
    X_raw = df[input_cols].values.astype(np.float32)
    
    if 'P_CLR' in df.columns:
        P_raw = df['P_CLR'].values.astype(np.float32)
    else:
        P_raw = np.zeros((len(df), 1), dtype=np.float32)

    y_raw = df[target_col].values.astype(np.float32)
    Bf_raw = df[common_features].values.astype(np.float32)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for i, (train_index, test_index) in enumerate(tscv.split(X_raw)):
        # ===== FIX 2: Debug timestamp extraction =====
        print(f"DEBUG: Fold {i+1} - test_index head: {df.index[test_index][:5]}")
        print(f"DEBUG: Fold {i+1} - test_index tail: {df.index[test_index][-5:]}")
        
        # Dates Logging
        train_start = df.index[train_index[0]]
        train_end = df.index[train_index[-1]]
        test_start = df.index[test_index[0]]
        test_end = df.index[test_index[-1]]
        
        # ===== FIX 3: Robust timestamp extraction =====
        # Only apply timezone logic if we have a valid DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            test_ts_raw = df.index[test_index]
            if test_ts_raw.tz is None:
                test_timestamps = test_ts_raw.tz_localize('UTC')
            else:
                test_timestamps = test_ts_raw.tz_convert('UTC')
        else:
            # Fallback: use integer indices (will result in generic x-axis)
            test_timestamps = None
            print("WARNING: Cannot extract proper timestamps. Using index positions.")
        
        dates = {
            'train_start': train_start, 'train_end': train_end,
            'test_start': test_start, 'test_end': test_end,
            'test_timestamps': test_timestamps
        }
        
        X_train, P_train, y_train = X_raw[train_index], P_raw[train_index], y_raw[train_index]
        X_test, P_test, y_test = X_raw[test_index], P_raw[test_index], y_raw[test_index]
        Bf_train = Bf_raw[train_index]; Bf_test = Bf_raw[test_index]
        
        # Concat Lookback for Test
        X_test_final = np.concatenate([X_train[-seq_len:], X_test], axis=0)
        P_test_final = np.concatenate([P_train[-seq_len:], P_test], axis=0)
        y_test_final = np.concatenate([y_train[-seq_len:], y_test], axis=0)
        Bf_test_final = np.concatenate([Bf_train[-seq_len:], Bf_test], axis=0)
        
        train_ds = V02Dataset(X_train, y_train, P_train, Bf_train, seq_len, pred_len)
        test_ds = V02Dataset(X_test_final, y_test_final, P_test_final, Bf_test_final, seq_len, pred_len)
        
        yield (DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0),
               DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0),
               {}, dates)

# =============================================================================
# Part 4: Training & Evaluation
# =============================================================================

def train_one_epoch_v02(model, loader, optimizer, device, loss_fn=F.mse_loss, warmup=False):
    model.train()
    
    # Warmup Logic: Freeze Mamba, Train Physics Only
    if hasattr(model, 'physics_layer') and hasattr(model, 'mamba_model'):
        if warmup:
            for param in model.mamba_model.parameters():
                param.requires_grad = False
            for param in model.physics_layer.parameters():
                param.requires_grad = True
        else:
            for param in model.mamba_model.parameters():
                param.requires_grad = True
            for param in model.physics_layer.parameters():
                param.requires_grad = True

    total_loss = 0.0
    for batch in loader:
        x_past = batch[0].to(device)
        y_k_target = batch[1].to(device)
        x_fut = batch[2].to(device)
        x_clr = batch[4].to(device)
        
        optimizer.zero_grad()
        p_mw_pred, _ = model(x_past, x_fut, x_clr)
        p_mw_target = y_k_target * (x_clr + 1e-6) # Reconstruct Target to MW
        
        loss = loss_fn(p_mw_pred, p_mw_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_v02(model, loader, device, capacity=20.0):
    model.eval()
    preds_mw = []
    acts_mw = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            y_k_target = batch[1].to(device)
            x_fut = batch[2].to(device)
            x_clr = batch[4].to(device)
            
            p_mw_pred, _ = model(x_past, x_fut, x_clr)
            p_mw_target = y_k_target * (x_clr + 1e-6)
            
            # Loss Calculation
            loss = F.mse_loss(p_mw_pred, p_mw_target)
            total_loss += loss.item()
            
            preds_mw.append(p_mw_pred.cpu().numpy())
            acts_mw.append(p_mw_target.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    preds = np.concatenate(preds_mw, axis=0).flatten()
    acts = np.concatenate(acts_mw, axis=0).flatten()
    
    # Simple Metrics
    mse = np.mean((preds - acts)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - acts))
    nrmse = (rmse / capacity) * 100
    nmae = (mae / capacity) * 100
    
    # Return Raw (N, P, 1) as well for plotting
    preds_raw = np.concatenate(preds_mw, axis=0)
    acts_raw = np.concatenate(acts_mw, axis=0)
    
    return {'rmse': rmse, 'mae': mae, 'nrmse': nrmse, 'nmae': nmae, 'mse_loss': avg_loss}, preds, acts, preds_raw, acts_raw

import plotly.graph_objects as go

def plot_fold_validation(fold_i, preds_dict, acts, dates_str, timestamps=None):
    """Plot Validation Predictions for this fold using Plotly"""
    fig = go.Figure()
    
    # Actuals
    if timestamps is not None and len(timestamps) == len(acts):
        fig.add_trace(go.Scatter(x=timestamps, y=acts, mode='lines', name='Actual MW', line=dict(color='black', width=2), opacity=0.6))
        xaxis_title = 'Time'
    else:
        fig.add_trace(go.Scatter(y=acts, mode='lines', name='Actual MW', line=dict(color='black', width=2), opacity=0.6))
        xaxis_title = 'Time Steps (Flattened)'
    
    colors = {'Vanilla Mamba': 'gray', 'Base Power Mamba': 'blue', 
              'Physics V02 (Basic)': 'green', 'Physics V03 (Geo)': 'orange', 
              'Physics V04 (Robust)': 'red', 'Smart Persistence': 'cyan', 
              'NWP Baseline': 'magenta', 'Physics Pure': 'gold'}
    
    for name, pred in preds_dict.items():
        c = colors.get(name, 'purple')
        if timestamps is not None and len(timestamps) == len(pred):
             fig.add_trace(go.Scatter(x=timestamps, y=pred, mode='lines', name=name, line=dict(color=c, width=1.5), opacity=0.8))
        else:
             fig.add_trace(go.Scatter(y=pred, mode='lines', name=name, line=dict(color=c, width=1.5), opacity=0.8))
        
    fig.update_layout(
        title=f'Fold {fold_i} Validation: {dates_str}',
        xaxis_title=xaxis_title,
        yaxis_title='Power (MW)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Save to HTML (Robust fallback)
    filename = f"fold_{fold_i}_validation.html"
    fig.write_html(filename)
    print(f"Plot saved to {filename}")
    
    try:
        fig.show()
    except:
        print("Inline plot failed. Please open the generated HTML file.")

def plot_benchmark_results(losses, val_losses, metrics):
    # 1. Loss Curves (Train & Val)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Train Loss
    for name, loss_hist in losses.items():
        if len(loss_hist) > 0: 
            axes[0].plot(loss_hist, label=f'{name}')
    axes[0].set_xlabel('Global Epoch')
    axes[0].set_ylabel('MSE Loss (MW)')
    axes[0].set_title('Training Loss Convergence')
    axes[0].legend()
    axes[0].grid(True)
    
    # Val Loss
    for name, loss_hist in val_losses.items():
        if len(loss_hist) > 0: 
            axes[1].plot(loss_hist, label=f'{name}')
    axes[1].set_xlabel('Global Epoch')
    axes[1].set_ylabel('MSE Loss (MW)')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Metrics Bar Chart
    models = list(metrics.keys())
    rmses = [m['rmse'] for m in metrics.values()]
    nrmses = [m['nrmse'] for m in metrics.values()]
    times = [m['time_s'] for m in metrics.values()]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # RMSE
    axes[0].bar(models, rmses, color=['gray', 'blue', 'green', 'orange', 'red', 'cyan', 'magenta'])
    axes[0].set_title('Avg RMSE (MW)')
    axes[0].set_ylabel('MW')
    axes[0].tick_params(axis='x', rotation=45)
    
    # nRMSE
    axes[1].bar(models, nrmses, color=['gray', 'blue', 'green', 'orange', 'red', 'cyan', 'magenta'])
    axes[1].set_title('Avg nRMSE (%)')
    axes[1].set_ylabel('%')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Time
    axes[2].bar(models, times, color='purple')
    axes[2].set_title('Avg Training Time/Fold (s)')
    axes[2].set_ylabel('Seconds')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def run_comparative_benchmark(df, n_splits=2, custom_config=None):
    # Try Import Here to avoid Circular Dependency
    try:
        from physics_residual_mamba_pure import PhysicsResidualMamba_Pure
    except ImportError:
        print("Warning: PhysicsResidualMamba_Pure not found.")
        PhysicsResidualMamba_Pure = None
        
    print(">>> Running 7-Way Comparative Benchmark (Target: K_PV, Persistent Rolling)")
    
    if custom_config is not None:
        print("Using Custom Configuration Provided by User")
        configs = custom_config
        # Ensure critical physics params exist if missing from user config
        if not hasattr(configs, 'T_ref'): configs.T_ref = 25.0
        if not hasattr(configs, 'common_features'): 
            configs.common_features = [f for f in configs.FUTURE_INPUT_COLS if f in configs.PAST_INPUT_COLS]
        
        # FORCE TARGET to K_PV to align with benchmark strategy
        if hasattr(configs, 'TARGET_COL') and configs.TARGET_COL != ['K_PV']:
            print(f"Comparison Override: Changing TARGET_COL from {configs.TARGET_COL} to ['K_PV']")
            configs.TARGET_COL = ['K_PV']
    else:
        print("Using Default Configuration")
        configs = PhysicsResidualMambaConfigs(n_splits=n_splits)
        
    device = configs.device
    
    # Extract training params
    total_epochs = getattr(configs, 'epochs', 30) # Default if used standalone without custom_config
    warmup_epochs = getattr(configs, 'warmup_epochs', 5)
    print(f"DEBUG: Using Total Epochs={total_epochs}, Warmup={warmup_epochs}")

    # Definitions
    model_classes = {
        'Vanilla Mamba': VanillaMamba,
        'Base Power Mamba': BasePowerMamba,
        'Physics V02 (Basic)': PhysicsResidualMamba,
        'Physics V03 (Geo)': PhysicsResidualMamba_V3,
        'Physics V04 (Robust)': PhysicsResidualMamba_V4,
        'Smart Persistence': SmartPersistenceModel,
        'NWP Baseline': NWPModel
    }
    
    if PhysicsResidualMamba_Pure is not None:
        model_classes['Physics Pure'] = PhysicsResidualMamba_Pure
    
    # --- INSTANTIATE MODELS ONCE (PERSISTENT WEIGHTS) ---
    models = {}
    optimizers = {}
    loss_fns = {}
    
    for name, ModelClass in model_classes.items():
        m = ModelClass(configs).to(device)
        models[name] = m
        if len(list(m.parameters())) > 0:
            optimizers[name] = torch.optim.Adam(m.parameters(), lr=1e-3)
        else:
            optimizers[name] = None
            
        # Select Loss Function
        if 'Physics' in name or 'Robust' in name:
             # Use Composite Loss for Physics models
             loss_fns[name] = PhysicsCompositeLoss(smooth_weight=0.1).to(device)
        else:
             # Standard MSE for Data-Driven / Baselines
             loss_fns[name] = F.mse_loss
    
    final_metrics = {k: {'rmse': [], 'mae': [], 'nrmse': [], 'nmae': [], 'time_s': []} for k in model_classes.keys()}
    loss_histories = {k: [] for k in model_classes.keys()}
    val_loss_histories = {k: [] for k in model_classes.keys()}
    
    # Run CV
    fold_gen = prepare_rolling_folds(
        df, 
        configs.PAST_INPUT_COLS, 
        'K_PV', 
        configs.FUTURE_INPUT_COLS, 
        n_splits=n_splits, 
        seq_len=configs.seq_len, 
        pred_len=configs.pred_len
    )
    
    for fold_i, (train_loader, test_loader, stats, dates) in enumerate(fold_gen):
        print(f"\n=== Fold {fold_i+1}/{n_splits} ===")
        print(f"Train: {dates['train_start']} -> {dates['train_end']}")
        print(f"Test : {dates['test_start']} -> {dates['test_end']}")
        
        fold_preds = {}
        acts = None
        
        for name, model in models.items():
            print(f"Training {name}...", end="")
            optimizer = optimizers[name]
            loss_fn = loss_fns[name] # specialized loss
            
            start_time = time.time()
            epoch_losses = []
            
            # Epochs per fold (Skip training for baseline models)
            is_baseline = name in ['Smart Persistence', 'NWP Baseline']
            is_physics = 'Physics' in name
            
            if is_baseline:
                epochs = 0
            else:
                 # Use configured epochs
                epochs = total_epochs
            
            for ep in range(epochs):
                # Check Warmup State
                warmup = False
                if is_physics and ep < warmup_epochs: 
                    warmup = True
                
                # Pass specialized loss_fn
                loss = train_one_epoch_v02(model, train_loader, optimizer, device, loss_fn=loss_fn, warmup=warmup)
                loss_histories[name].append(loss) 
                
                # Val Loss Tracking (Per Epoch)
                # Fast Eval (MSE Only - using test loader as proxy for learning curve)
                model.eval()
                with torch.no_grad():
                    val_mets, _, _, _, _ = evaluate_v02(model, test_loader, device)
                    val_loss_histories[name].append(val_mets['mse_loss'])
                model.train()

            train_time = time.time() - start_time
            
            # Evaluate
            mets, pred, act, pred_raw, act_raw = evaluate_v02(model, test_loader, device)
            
            # Use 1-Step Ahead for Plotting (Index 0)
            # pred_raw shape: (N, PredLen, 1)
            # We want the series of "Time t+1" predictions.
            # This corresponds to diag(0) effectively if we had overlap, 
            # but given the loader structure, it is simply [:, 0, 0]
            fold_preds[name] = pred_raw[:, 0, 0]
            
            if acts is None: acts = act_raw[:, 0, 0]
            
            # Store
            final_metrics[name]['rmse'].append(mets['rmse'])
            final_metrics[name]['mae'].append(mets['mae'])
            final_metrics[name]['nrmse'].append(mets['nrmse'])
            final_metrics[name]['nmae'].append(mets['nmae'])
            final_metrics[name]['time_s'].append(train_time)
            
            print(f" Done. RMSE: {mets['rmse']:.3f} MW | nRMSE: {mets['nrmse']:.2f}% | Time: {train_time:.1f}s")
        
        # Prepare Timestamps for Plotting (1-Step Ahead)
        try:
            test_ts = dates['test_timestamps']
            
            if test_ts is None:
                print("DEBUG: test_timestamps is None, skipping timestamp alignment.")
                flat_timestamps = None
            else:
                # ===== FIX 4: Correct Timestamp Alignment =====
                # The test_ds is created with X_test_final which has seq_len prepended from train.
                # V02Dataset creates samples from index 0 to len(data) - seq - pred + 1.
                # Sample 0: Input [0:seq_len], Target [seq_len:seq_len+pred_len]
                # 
                # Since X_test_final = [train[-seq_len:], X_test], the first valid prediction
                # corresponds to the FIRST element of the original test set.
                # 
                # For step 0 prediction ([:, 0, 0]), we need:
                # - Sample i predicts step (seq_len + i) in X_test_final
                # - This corresponds to step i in the original test set (after offset)
                # 
                # Number of samples = len(X_test_final) - seq_len - pred_len + 1 = len(X_test) - pred_len + 1
                
                Num_Samples = len(acts)  # Should match pred_raw[:,0,0]
                
                print(f"DEBUG: Num_Samples={Num_Samples}, len(test_ts)={len(test_ts)}")
                
                # The prediction for sample i targets time step i in the original test set
                # So we need test_ts[0 : Num_Samples]
                if Num_Samples <= len(test_ts):
                    flat_timestamps = test_ts[:Num_Samples]
                    # Convert to local time (UTC+8) for proper visualization
                    # (Station is in China, so solar noon should appear around 12:00)
                    flat_timestamps = flat_timestamps.tz_convert('Asia/Shanghai')
                    print(f"DEBUG: Timestamps aligned. First: {flat_timestamps[0]}, Last: {flat_timestamps[-1]}")
                    print(f"DEBUG: Timestamps shape: {flat_timestamps.shape}, Acts shape: {acts.shape}")
                else:
                    # This can happen if pred_len is small or test set is short
                    flat_timestamps = None
                    print(f"WARNING: More samples ({Num_Samples}) than timestamps ({len(test_ts)}). Skipping timestamps.")

        except Exception as e:
            print(f"Comparison Plot Warning: Could not generate timestamps ({e})")
            import traceback
            traceback.print_exc()
            flat_timestamps = None

        # Plot Fold Validation
        plot_fold_validation(fold_i+1, fold_preds, acts, f"{dates['test_start']} - {dates['test_end']}", timestamps=flat_timestamps)

    # Aggregate
    aggregated_metrics = {}
    print("\n>>> FINAL RESULTS (Average across folds) <<<")
    for name in model_classes.keys():
        aggregated_metrics[name] = {
            'rmse': np.mean(final_metrics[name]['rmse']),
            'mae': np.mean(final_metrics[name]['mae']),
            'nrmse': np.mean(final_metrics[name]['nrmse']),
            'nmae': np.mean(final_metrics[name]['nmae']),
            'time_s': np.mean(final_metrics[name]['time_s'])
        }
        print(f"{name}: RMSE={aggregated_metrics[name]['rmse']:.3f} | nRMSE={aggregated_metrics[name]['nrmse']:.2f}%")
        
    plot_benchmark_results(loss_histories, val_loss_histories, aggregated_metrics)
    return aggregated_metrics
