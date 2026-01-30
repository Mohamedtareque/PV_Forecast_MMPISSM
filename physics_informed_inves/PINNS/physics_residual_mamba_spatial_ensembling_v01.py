"""
Physics-Residual Power Mamba (Spatial Ensembling V01)

Extends Spatial V02 with Inverse-RMSE Weighted Ensemble.

Key Features:
1. Physics Isolation Principle (preserved from Spatial V02)
   - Mamba Model Inputs: Full [Station07, Station08, Spatial_Deltas]
   - Physics Layer Inputs: Sliced [Station07_Features] only

2. Inverse-RMSE Weighted Ensemble
   - Combines: Physics V02, V03, V04, Pure, and NWP Baseline
   - Weight_i = (1/RMSE_i) / Σ(1/RMSE_j)
   - Pred_ensemble = Σ(Weight_i × Pred_i)
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

# =============================================================================
# Helper Classes (From V02)
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

class PhysicsCompositeLoss(nn.Module):
    def __init__(self, smooth_weight=0.1):
        super(PhysicsCompositeLoss, self).__init__()
        self.smooth_loss = SmoothnessLoss(weight=smooth_weight)
        
    def forward(self, pred, target):
        loss_mse = F.mse_loss(pred, target)
        loss_smooth = self.smooth_loss(pred)
        return loss_mse + loss_smooth

# =============================================================================
# Physics Layers
# =============================================================================

class DifferentiablePVLayer_V02(nn.Module):
    """Basic Physics Layer (V02) - Uses SLICED St07 indices"""
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
        self.scaler = None

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
            night_mask = (x_clr > 0.01).float()
            if self.idx_elev is not None:
                elev = x_future[:, :, self.idx_elev]
                day_mask = (elev > 0.0).float().unsqueeze(-1)
                K_phys = K_phys * day_mask
            K_phys = K_phys * night_mask
            return K_phys

        if self.idx_elev is not None:
            elev = x_future[:, :, self.idx_elev]
            day_mask = (elev > 0.0).float().unsqueeze(-1)
            P_final_mw = P_final_mw * day_mask
        return P_final_mw

class DifferentiablePVLayer_V3(DifferentiablePVLayer_V02):
    """V3: Geometric Gating (Tilt Fix)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs):
        super().__init__(idx_G, idx_Ta, idx_WS, idx_time_feats, **kwargs)
        self.geo_net = nn.Sequential(
            nn.Linear(len(idx_time_feats), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, x_future, x_clr=None):
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        time_feats = x_future[:, :, self.idx_time_feats]
        
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        G_poa = G * tilt_factor
        
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
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
        IAM = 1.0 - b0 * (1.0 - tilt_factor)
        IAM = torch.clamp(IAM, 0.0, 1.0)
        G_eff = G_poa * IAM
        
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta_module = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        T_cell = Ta + G_eff / (U0 + U1 * WS + self.eps)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = eta_module * P_stc * (G_eff / self.G_stc) * temp_factor
        P_dc = F.relu(P_dc)
        
        inv_k = F.softplus(self.inv_k_raw) * 10.0
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

# =============================================================================
# RevIN and Decomposition
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
# Mamba Backbone (Spatial-Aware)
# =============================================================================

class MMPISSM_Model_01(nn.Module):
    """Base Mamba Backbone - Accepts full spatial tensor"""
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
# Spatial Configuration
# =============================================================================

class SpatialMambaConfigs:
    """Configuration with Station 08 spatial features and Physics Isolation indices"""
    def __init__(self, n_splits=4):
        self.seq_len = 96
        self.pred_len = 96
        self.TARGET_COL = ['K_PV']
        
        # Station 07 Features (original)
        self.PAST_INPUT_COLS = [
            'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 
            'power', 'K_PV', 'P_CLR', 'NWP_Power_MW', 
            'hour_sin', 'hour_cos', 'sun_elevation',
            # Station 08 Features
            's08_nwp_globalirrad', 's08_nwp_temperature', 's08_nwp_windspeed', 's08_K_PV',
            # Spatial Deltas
            'spatial_delta_irrad', 'spatial_delta_temp', 'spatial_delta_wind'
        ]
        
        self.FUTURE_INPUT_COLS = [
            'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed',
            'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation',
            # Station 08 Features
            's08_nwp_globalirrad', 's08_nwp_temperature', 's08_nwp_windspeed',
            # Spatial Deltas
            'spatial_delta_irrad', 'spatial_delta_temp', 'spatial_delta_wind'
        ]
        
        # CRITICAL: Station 07-only indices for Physics layers (including sun_elevation)
        st07_future_cols = [
            'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed',
            'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation'
        ]
        self.st07_indices = [self.FUTURE_INPUT_COLS.index(c) for c in st07_future_cols]
        self.st07_future_cols = st07_future_cols  # Store for reference
        
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
        self.warmup_epochs = 5
        self.lambda_data = 1.0
        self.lambda_night = 0.2
        self.lambda_mono = 0.1
        self.T_ref = 25.0
        self.G_night_thr = 10.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Models with Physics Isolation
# =============================================================================

class PhysicsResidualMamba(nn.Module):
    """V02 Basic - With Physics Isolation Slicing"""
    def __init__(self, configs):
        super(PhysicsResidualMamba, self).__init__()
        self.configs = configs
        
        # Physics indices are RELATIVE to the SLICED st07 tensor
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        
        try:
            idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError:
            idx_elev = None
            
        self.physics_layer = DifferentiablePVLayer_V02(idx_G, idx_Ta, idx_WS, [], idx_elev=idx_elev, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        # PHYSICS ISOLATION: Slice to Station 07 only
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        
        # Physics uses SLICED input
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        
        # Mamba uses FULL spatial input
        mamba_k, _ = self.mamba_model(x_past, x_future)
        
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PhysicsResidualMamba_V3(nn.Module):
    """V3 Geometric - With Physics Isolation Slicing"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V3, self).__init__()
        self.configs = configs
        
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        # Time feats indices in SLICED tensor
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']] 
        
        self.physics_layer = DifferentiablePVLayer_V3(idx_G, idx_Ta, idx_WS, idx_time, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

class PhysicsResidualMamba_V4(nn.Module):
    """V4 Robust - With Physics Isolation Slicing"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V4, self).__init__()
        self.configs = configs
        
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        idx_time = [configs.st07_future_cols.index(c) for c in ['hour_sin', 'hour_cos']] 
        
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

class PhysicsResidualMamba_Pure(nn.Module):
    """Pure Physics - With Physics Isolation Slicing"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_Pure, self).__init__()
        self.configs = configs
        
        idx_G = configs.st07_future_cols.index('nwp_globalirrad')
        idx_Ta = configs.st07_future_cols.index('nwp_temperature')
        idx_WS = configs.st07_future_cols.index('nwp_windspeed')
        
        try:
            idx_elev = configs.st07_future_cols.index('sun_elevation')
        except ValueError:
            idx_elev = None
            
        self.physics_layer = DifferentiablePVLayer_V02(idx_G, idx_Ta, idx_WS, [], idx_elev=idx_elev, T_ref=configs.T_ref, P_stc_init=19.0)
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        x_future_s07 = x_future[:, :, self.configs.st07_indices]
        p_physics_k = self.physics_layer(x_future_s07, x_clr=x_clr)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        k_total = p_physics_k + mamba_k
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

# Non-Physics Models (No isolation needed)
class BasePowerMamba(nn.Module):
    def __init__(self, configs):
        super(BasePowerMamba, self).__init__()
        self.mamba_model = MMPISSM_Model_01(configs)
        
    def forward(self, x_past, x_future, x_clr):
        k_pred, _ = self.mamba_model(x_past, x_future)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, torch.zeros_like(k_pred)

class VanillaMamba(nn.Module):
    def __init__(self, configs):
        super(VanillaMamba, self).__init__()
        self.mamba_model = MMPISSM_Model_01(configs)
        
    def forward(self, x_past, x_future, x_clr):
        k_pred, _ = self.mamba_model(x_past, x_future)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, torch.zeros_like(k_pred)

class SmartPersistenceModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        try:
            self.k_idx = configs.PAST_INPUT_COLS.index('K_PV')
        except ValueError:
            self.k_idx = 0 
            
    def forward(self, x_past, x_future, x_clr):
        lag = 96
        H = self.configs.pred_len
        k_past = x_past[:, :, self.k_idx]
        if k_past.shape[1] >= lag:
            k_pred = k_past[:, -H:].unsqueeze(-1)
        else:
            k_pred = torch.zeros(x_past.shape[0], H, 1, device=x_past.device)
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_pred) * x_clr
        return p_total_mw, k_pred

class NWPModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        try:
            self.nwp_idx = configs.FUTURE_INPUT_COLS.index('NWP_Power_MW')
        except ValueError:
            self.nwp_idx = 0
            
    def forward(self, x_past, x_future, x_clr):
        nwp_mw = x_future[:, :, self.nwp_idx].unsqueeze(-1)
        return nwp_mw, torch.zeros_like(nwp_mw)

# =============================================================================
# Dataset & Data Prep
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
        else:
            test_timestamps = None
        
        dates = {'train_start': train_start, 'train_end': train_end, 
                 'test_start': test_start, 'test_end': test_end, 'test_timestamps': test_timestamps}
        
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

# =============================================================================
# Training & Evaluation
# =============================================================================

def train_one_epoch_v02(model, loader, optimizer, device, loss_fn=F.mse_loss, warmup=False):
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

def evaluate_v02(model, loader, device, capacity=20.0):
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
    
    colors = {'Vanilla Mamba': 'gray', 'Base Power Mamba': 'blue', 'Physics V02 (Basic)': 'green', 
              'Physics V03 (Geo)': 'orange', 'Physics V04 (Robust)': 'red', 'Smart Persistence': 'cyan', 
              'NWP Baseline': 'magenta', 'Physics Pure': 'gold'}
    
    for name, pred in preds_dict.items():
        c = colors.get(name, 'purple')
        if timestamps is not None and len(timestamps) == len(pred):
            fig.add_trace(go.Scatter(x=timestamps, y=pred, mode='lines', name=name, line=dict(color=c, width=1.5), opacity=0.8))
        else:
            fig.add_trace(go.Scatter(y=pred, mode='lines', name=name, line=dict(color=c, width=1.5), opacity=0.8))
    
    fig.update_layout(title=f'Fold {fold_i} Spatial Validation: {dates_str}', xaxis_title='Time', yaxis_title='Power (MW)', template='plotly_white')
    fig.write_html(f"fold_{fold_i}_spatial.html")
    try: fig.show()
    except: pass

# =============================================================================
# Inverse-RMSE Weighted Ensemble
# =============================================================================

def calculate_inverse_rmse_ensemble(
    rmse_dict: dict, 
    pred_dict: dict, 
    target_acts: np.ndarray, 
    model_names: list, 
    capacity: float = 20.0
) -> tuple:
    """
    Calculate Inverse-RMSE Weighted Ensemble prediction.
    
    Args:
        rmse_dict: {model_name: rmse_value} for all models
        pred_dict: {model_name: prediction_array} for all models
        target_acts: Ground truth array (flattened)
        model_names: List of model names to include in ensemble
        capacity: Plant capacity for nRMSE calculation
        
    Returns:
        (ensemble_pred, ensemble_rmse, ensemble_nrmse, weights_dict)
    """
    print("\n" + "="*60)
    print("INVERSE-RMSE WEIGHTED ENSEMBLE")
    print("="*60)
    
    # Calculate inverse weights: w_i = (1/RMSE_i) / Σ(1/RMSE_j)
    inv_rmse = {n: 1.0 / (rmse_dict[n] + 1e-8) for n in model_names}
    total_inv = sum(inv_rmse.values())
    weights = {n: inv_rmse[n] / total_inv for n in model_names}
    
    # Print weight table
    print(f"{'Model':<25} | {'RMSE (MW)':<12} | {'Weight':<10}")
    print("-" * 50)
    for n in model_names:
        print(f"{n:<25} | {rmse_dict[n]:<12.4f} | {weights[n]:<10.4f}")
    print("-" * 50)
    
    # Weighted sum: Pred_ensemble = Σ(w_i × Pred_i)
    ensemble_pred = np.zeros_like(pred_dict[model_names[0]])
    for n in model_names:
        ensemble_pred += weights[n] * pred_dict[n]
    
    # Calculate ensemble metrics
    ensemble_rmse = np.sqrt(np.mean((ensemble_pred - target_acts)**2))
    ensemble_nrmse = (ensemble_rmse / capacity) * 100
    
    print(f"\n>>> ENSEMBLE RMSE: {ensemble_rmse:.4f} MW | nRMSE: {ensemble_nrmse:.2f}%")
    print("="*60 + "\n")
    
    return ensemble_pred, ensemble_rmse, ensemble_nrmse, weights

# =============================================================================
# Benchmark Runner (with Ensembling)
# =============================================================================

def run_comparative_benchmark(df, n_splits=2, custom_config=None):
    """
    Run spatial-aware comparative benchmark with Physics Isolation and Inverse-RMSE Ensemble.
    
    Returns:
        aggregated: Per-model metrics
        ensemble_results: {'ensemble_rmse', 'ensemble_nrmse', 'weights', 'ensemble_pred'}
    """
    print(">>> Running Spatial 8-Way Benchmark + Ensemble (Target: K_PV, Physics Isolation)")
    
    if custom_config is not None:
        configs = custom_config
        if not hasattr(configs, 'T_ref'): configs.T_ref = 25.0
        if not hasattr(configs, 'common_features'): 
            configs.common_features = [f for f in configs.FUTURE_INPUT_COLS if f in configs.PAST_INPUT_COLS]
        if not hasattr(configs, 'st07_indices'):
            st07_future_cols = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'NWP_Power_MW', 'hour_sin', 'hour_cos', 'sun_elevation']
            configs.st07_indices = [configs.FUTURE_INPUT_COLS.index(c) for c in st07_future_cols if c in configs.FUTURE_INPUT_COLS]
            configs.st07_future_cols = [c for c in st07_future_cols if c in configs.FUTURE_INPUT_COLS]
    else:
        configs = SpatialMambaConfigs(n_splits=n_splits)
        
    device = configs.device
    total_epochs = getattr(configs, 'epochs', 30)
    warmup_epochs = getattr(configs, 'warmup_epochs', 5)
    
    model_classes = {
        'Vanilla Mamba': VanillaMamba, 'Base Power Mamba': BasePowerMamba,
        'Physics V02 (Basic)': PhysicsResidualMamba, 'Physics V03 (Geo)': PhysicsResidualMamba_V3,
        'Physics V04 (Robust)': PhysicsResidualMamba_V4, 'Physics Pure': PhysicsResidualMamba_Pure,
        'Smart Persistence': SmartPersistenceModel, 'NWP Baseline': NWPModel
    }
    
    models, optimizers, loss_fns = {}, {}, {}
    for name, ModelClass in model_classes.items():
        m = ModelClass(configs).to(device)
        models[name] = m
        optimizers[name] = torch.optim.Adam(m.parameters(), lr=1e-3) if len(list(m.parameters())) > 0 else None
        loss_fns[name] = PhysicsCompositeLoss(smooth_weight=0.1).to(device) if 'Physics' in name else F.mse_loss
    
    final_metrics = {k: {'rmse': [], 'mae': [], 'nrmse': [], 'time_s': []} for k in model_classes.keys()}
    
    # =========================================================================
    # STEP A: Initialize prediction accumulators for ensembling
    # =========================================================================
    all_fold_preds = {name: [] for name in model_classes.keys()}
    all_fold_acts = []
    
    fold_gen = prepare_rolling_folds(df, configs.PAST_INPUT_COLS, 'K_PV', configs.FUTURE_INPUT_COLS, n_splits=n_splits, seq_len=configs.seq_len, pred_len=configs.pred_len)
    
    for fold_i, (train_loader, test_loader, _, dates) in enumerate(fold_gen):
        print(f"\n=== Fold {fold_i+1}/{n_splits} ===")
        fold_preds, acts = {}, None
        fold_ground_truth = None  # Safer ground truth capture
        
        for name, model in models.items():
            print(f"Training {name}...", end="")
            start_time = time.time()
            is_baseline = name in ['Smart Persistence', 'NWP Baseline']
            epochs = 0 if is_baseline else total_epochs
            
            for ep in range(epochs):
                warmup = ('Physics' in name and ep < warmup_epochs)
                train_one_epoch_v02(model, train_loader, optimizers[name], device, loss_fn=loss_fns[name], warmup=warmup)
            
            mets, pred, act, pred_raw, act_raw = evaluate_v02(model, test_loader, device)
            fold_preds[name] = pred_raw[:, 0, 0]
            
            # =========================================================================
            # STEP B: Accumulate flattened predictions for ensembling
            # =========================================================================
            all_fold_preds[name].append(pred_raw.flatten())
            
            # Capture ground truth once per fold (safer method per user suggestion)
            if fold_ground_truth is None:
                fold_ground_truth = act_raw.flatten()
            
            if acts is None: acts = act_raw[:, 0, 0]
            
            final_metrics[name]['rmse'].append(mets['rmse'])
            final_metrics[name]['mae'].append(mets['mae'])
            final_metrics[name]['nrmse'].append(mets['nrmse'])
            final_metrics[name]['time_s'].append(time.time() - start_time)
            print(f" RMSE: {mets['rmse']:.3f} MW | nRMSE: {mets['nrmse']:.2f}%")
        
        # Append fold ground truth after model loop
        all_fold_acts.append(fold_ground_truth)
        
        try:
            test_ts = dates['test_timestamps']
            flat_ts = test_ts[:len(acts)].tz_convert('Asia/Shanghai') if test_ts is not None and len(test_ts) >= len(acts) else None
        except:
            flat_ts = None
        plot_fold_validation(fold_i+1, fold_preds, acts, f"{dates['test_start']} - {dates['test_end']}", timestamps=flat_ts)
    
    # =========================================================================
    # STEP C: Concatenate predictions from all folds
    # =========================================================================
    combined_preds = {n: np.concatenate(all_fold_preds[n]) for n in model_classes.keys()}
    combined_acts = np.concatenate(all_fold_acts)
    
    # Calculate Global RMSE (on full concatenated test set)
    global_rmse = {n: np.sqrt(np.mean((combined_preds[n] - combined_acts)**2)) for n in model_classes.keys()}
    
    print("\n>>> FINAL RESULTS (Per-Fold Average) <<<")
    aggregated = {}
    for name in model_classes.keys():
        aggregated[name] = {'rmse': np.mean(final_metrics[name]['rmse']), 'nrmse': np.mean(final_metrics[name]['nrmse'])}
        print(f"{name}: RMSE={aggregated[name]['rmse']:.3f} | nRMSE={aggregated[name]['nrmse']:.2f}%")
    
    # =========================================================================
    # STEP D: Calculate Inverse-RMSE Weighted Ensemble
    # =========================================================================
    ensemble_members = [
        'Physics V02 (Basic)', 
        'Physics V03 (Geo)', 
        'Physics V04 (Robust)', 
        'Physics Pure', 
        'NWP Baseline'
    ]
    
    ens_pred, ens_rmse, ens_nrmse, weights = calculate_inverse_rmse_ensemble(
        global_rmse, combined_preds, combined_acts, ensemble_members
    )
    
    # Return both individual model results and ensemble results
    ensemble_results = {
        'ensemble_rmse': ens_rmse,
        'ensemble_nrmse': ens_nrmse,
        'weights': weights,
        'ensemble_pred': ens_pred,
        'combined_acts': combined_acts
    }
    
    return aggregated, ensemble_results

if __name__ == "__main__":
    print("Spatial Physics-Residual Mamba Ensembling V01")
    print("Features: Physics Isolation + Inverse-RMSE Weighted Ensemble")
    print(f"Ensemble Members: Physics V02, V03, V04, Pure, NWP Baseline")
    c = SpatialMambaConfigs()
    print(f"St07 Indices: {c.st07_indices}")
    print(f"St07 Cols: {c.st07_future_cols}")
