"""
Physics-Residual Power Mamba Architecture

This module implements a hybrid architecture combining:
1. A Differentiable Physics Layer (inference of implicit temperature and power)
2. A Deep Learning Residual Learner (Mamba-based MMPISSM_Model_01)

The core hypothesis: Instead of predicting raw Power, the Neural Network
predicts the residual error of a physical model.

Final prediction: P_total = ReLU(P_physics + P_residual)

References:
- Applied Energy 2025 (Physics-Informed Solar Forecasting)
- Faiman Temperature Model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


# =============================================================================
# Feature Scaler: Handles Normalization Statistics for Physics Layer
# =============================================================================

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
        feature_means (dict): Mean values keyed by feature name
        feature_stds (dict): Std values keyed by feature name
        
    Example:
        >>> means = {'nwp_globalirrad': 300.0, 'nwp_temperature': 15.0, 'nwp_windspeed': 4.0}
        >>> stds = {'nwp_globalirrad': 250.0, 'nwp_temperature': 10.0, 'nwp_windspeed': 2.0}
        >>> scaler = FeatureScaler(means, stds)
    """
    def __init__(self, feature_means: dict, feature_stds: dict):
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
    
    def denormalize_dict(self, x_norm: torch.Tensor, feature_indices: dict) -> dict:
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
    
    def get_stats(self) -> dict:
        """Return all stored statistics as plain dict for logging."""
        return {
            name: {
                'mean': getattr(self, f'{name}_mean').item(),
                'std': getattr(self, f'{name}_std').item()
            }
            for name in self.feature_names
        }


def compute_scaler_stats(df: pd.DataFrame, cols: list) -> tuple:
    """
    Compute mean and std for columns.
    """
    means = {col: df[col].mean() for col in cols}
    stds = {col: df[col].std() for col in cols}
    return means, stds


# =============================================================================
# Part 1: Differentiable Physics Layer
# =============================================================================


class DifferentiablePVLayer(nn.Module):
    """
    Differentiable Physics Layer for PV Power Estimation.
    
    Includes:
    1. Geometric Gating (Dynamic Tilt/AOI)
    2. Optical Loss (Incidence Angle Modifier)
    3. Electrical Loss (Inverter Efficiency Curve)
    4. System Loss (Soiling/Degradation)
    """
    def __init__(
        self,
        idx_G, idx_Ta, idx_WS, 
        idx_time_feats, 
        T_ref=25.0,
        G_stc=1000.0,
        P_stc_init=19.0, # Station 07
        eta_init=0.16    # Station 07
    ):
        super().__init__()
        self.idx_G, self.idx_Ta, self.idx_WS = idx_G, idx_Ta, idx_WS
        self.idx_time_feats = idx_time_feats 
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6

        # --- Base Physics Parameters ---
        # Poly-Si Temp Coeff ~ -0.45%/C -> softplus(-5.4) ~ 0.0045
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4, dtype=torch.float32)) 
        
        # Heat transfer (U0 ~ 25, U1 ~ 6.8)
        self.U0_raw = nn.Parameter(torch.tensor(3.2, dtype=torch.float32)) 
        self.U1_raw = nn.Parameter(torch.tensor(1.9, dtype=torch.float32)) 
        
        # Efficiency (Sigmoid(-1.66) ~ 0.16)
        self.eta_raw = nn.Parameter(torch.tensor(-1.66)) 
        
        # P_stc Constraint: P_stc <= P_stc_init
        # We start with a high sigmoid value (e.g. 6.0 -> 0.9975) to correspond to near-max capacity
        self.P_stc_limit = float(P_stc_init)
        self.Pstc_raw = nn.Parameter(torch.tensor(6.0, dtype=torch.float32)) 
        
        self.scaler = None
        
        self.scaler = None

    def set_scaler(self, scaler):
        """Register scaler for parameter logging."""
        self.scaler = scaler

    def forward(self, x_future):
        # 1. Extract Physics Inputs
        G = x_future[:, :, self.idx_G]
        Ta = x_future[:, :, self.idx_Ta]
        WS = x_future[:, :, self.idx_WS]
        
        # 2. Parameter Activation
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw) 
        
        # Enforce Boundary: P_stc <= P_stc_limit
        P_stc = self.P_stc_limit * torch.sigmoid(self.Pstc_raw)
        
        # 3. Faiman Temperature Model
        # T_cell = T_a + G / (U0 + U1*WS)
        T_cell = Ta + G / (U0 + U1 * WS + self.eps)

        # 4. DC Power Generation (Thermal corrected)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        
        # P_dc = eta * P_stc * (G/G_stc) * temp_factor
        P_dc = eta * P_stc * (G / self.G_stc) * temp_factor
        
        # 5. Output (MW)
        # No system loss, no inverter - simplified V0 logic
        P_final_mw = F.relu(P_dc).unsqueeze(-1)
        
        return P_final_mw

    def get_params_dict(self):
        with torch.no_grad():
            params = {
                'eta': torch.sigmoid(self.eta_raw).item(),
                'U0': F.softplus(self.U0_raw).item(),
                'gamma': -F.softplus(self.gamma_raw).item(),
                'P_stc': (self.P_stc_limit * torch.sigmoid(self.Pstc_raw)).item()
            }
        return params


# =============================================================================
# Part 2: Helper Classes (from original codebase)
# =============================================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    """
    def __init__(self, num_features: int, eps=1e-2, affine=True):  
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
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


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# =============================================================================
# Part 2 (continued): MMPISSM_Model_01 (Mamba Model)
# =============================================================================

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using placeholder.")


class MMPISSM_Model_01(nn.Module):
    """
    Power Mamba Model (Base model for residual learning).
    """
    def __init__(self, configs, output_residual=False):
        super(MMPISSM_Model_01, self).__init__()
        self.configs = configs
        self.output_residual = output_residual
        
        self.num_past_feats = len(self.configs.PAST_INPUT_COLS)
        self.num_shadows = len(self.configs.common_features)
        self.enc_in = self.num_past_feats + self.num_shadows

        self.fut_indices = torch.tensor(
            [self.configs.FUTURE_INPUT_COLS.index(f) for f in self.configs.common_features],
            dtype=torch.long
        )
        self.past_target_indices = torch.tensor(
            [self.configs.PAST_INPUT_COLS.index(f) for f in self.configs.common_features],
            dtype=torch.long
        )

        self.shadow_start_idx = self.num_past_feats

        self.lin1 = nn.Linear(2 * (self.configs.seq_len + self.configs.pred_len), self.configs.seq_len)
        self.target_indices = [self.configs.PAST_INPUT_COLS.index(col) for col in self.configs.TARGET_COL]
        self.decompsition = series_decomp(self.configs.kernel_size)

        self.revin_layer = RevIN(self.enc_in)
        self.revin_layer_enc = RevIN(self.enc_in)

        self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)
        self.lin3 = nn.Linear(4 * self.configs.n_embed, self.configs.pred_len)

        self.dropout1 = nn.Dropout(self.configs.dropout)
        self.dropout2 = nn.Dropout(self.configs.dropout)

        if MAMBA_AVAILABLE:
            self.mamba1 = Mamba(
                d_model=self.configs.n_embed, d_state=self.configs.d_state,
                d_conv=self.configs.dconv, expand=self.configs.e_fact
            )
            self.mamba2 = Mamba(
                d_model=self.configs.enc_in, d_state=self.configs.d_state,
                d_conv=self.configs.dconv, expand=self.configs.e_fact
            )
        else:
            self.mamba1 = nn.Identity()
            self.mamba2 = nn.Identity()
    
    def forward(self, x, x_f, mode='v4_robust'):
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

        x = torch.permute(x, (0, 2, 1))
        x = self.lin1(x)

        x = torch.permute(x, (0, 2, 1))
        x = self.revin_layer_enc(x, 'denorm')
        
        x = self.revin_layer(x, 'norm')
        seasonal_init, trend_init = self.decompsition(x)

        x_e = torch.cat([seasonal_init, trend_init], dim=1)
        x_e = torch.permute(x_e, (0, 2, 1))
        x_e = self.lin2(x_e)

        x_m = self.dropout1(x_e)
        x_m = self.mamba1(x_m)

        x_im = self.dropout2(x_e)
        x_im = torch.permute(x_im, (0, 2, 1))
        x_im = self.mamba2(x_im)
        x_im = torch.permute(x_im, (0, 2, 1))

        x = torch.cat([x_im, x_m, x_m + x_im, x_e], dim=2)

        x = self.lin3(x)
        x = torch.permute(x, (0, 2, 1))
        
        # --- BRANCHING LOGIC for Output Normalization ---
        # V0/V1: "Double Trend" -> Denormalize fully (Prediction = Mean + Trend + Residual)
        # V2+: "Trend-Aware" -> Return Z-Score (Prediction = Physics_Trend + Scaled_Residual)
        
        if mode in ['v1_baseline', 'v0_base']:
            # Original Behavior: Full Denormalization
            x = self.revin_layer(x, 'denorm')
            
            # Select target columns
            x = x[:, :, self.target_indices]
            
            # Dummy stdev (not used in V1 logic, but returned for API consistency)
            past_stdev_target = torch.ones_like(x) 
            
            return x, past_stdev_target
            
        else:
            # New Behavior (V2, V3, V4): Return Z-Score + Stdev
            # x is currently in "Normalized Space" (approx Z-scores)
            
            # We need to extract the target STDEV
            past_stdev = self.revin_layer.stdev # [B, 1, F]

            # Select target columns only
            x = x[:, :, self.target_indices]
            past_stdev_target = past_stdev[:, :, self.target_indices]

            return x, past_stdev_target


# =============================================================================
# Part 2 (final): Physics-Residual Mamba Hybrid Model
# =============================================================================

class PhysicsResidualMamba(nn.Module):
    """
    Physics-Residual Power Mamba: Hybrid Architecture.
    """
    def __init__(self, configs):
        super(PhysicsResidualMamba, self).__init__()
        self.configs = configs
        
        # Correctly map indices based on the subset used in x_future (common_features)
        # x_future only contains features that are in both PAST and FUTURE lists
        idx_G = configs.common_features.index('nwp_globalirrad')
        idx_Ta = configs.common_features.index('nwp_temperature')
        idx_WS = configs.common_features.index('nwp_windspeed')
        
        # --- Geometric Features ---
        idx_time_feats = []
        possible_feats = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'season_sin', 'season_cos']
        for f in possible_feats:
            if f in configs.FUTURE_INPUT_COLS:
                idx_time_feats.append(configs.FUTURE_INPUT_COLS.index(f))
                
        if len(idx_time_feats) == 0:
             print("WARNING: Time features not found!")

        self.physics_layer = DifferentiablePVLayer(
            idx_G=idx_G,
            idx_Ta=idx_Ta,
            idx_WS=idx_WS,
            idx_time_feats=idx_time_feats,
            T_ref=configs.T_ref,
            G_stc=1000.0,
            P_stc_init=19.0, # Station 07: Start below 20MW
            eta_init=0.16    # Station 07: Poly-Si
        )
        
        self.mamba_model = MMPISSM_Model_01(configs, output_residual=True)
        
    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, x_clr: torch.Tensor = None):
        if torch.isnan(x_past).any() or torch.isinf(x_past).any():
            x_past = torch.nan_to_num(x_past, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(x_future).any() or torch.isinf(x_future).any():
            x_future = torch.nan_to_num(x_future, nan=0.0, posinf=0.0, neginf=0.0)
        
        mode = self.configs.experiment_mode
        
        # 1. Branch A: Physics (The Trend)
        # Physics predicts P_phys (MW)
        p_physics = self.physics_layer(x_future)
        
        if torch.isnan(p_physics).any() or torch.isinf(p_physics).any():
            p_physics = torch.nan_to_num(p_physics, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Branch B: Mamba (The Fluctuation)
        # In K_PV mode, this is Delta K (Index Residual)
        mamba_raw, past_stdev = self.mamba_model(x_past, x_future, mode=mode)
        
        if torch.isnan(mamba_raw).any() or torch.isinf(mamba_raw).any():
            mamba_raw = torch.nan_to_num(mamba_raw, nan=0.0, posinf=0.0, neginf=0.0)
            
        # 3. Combine Logic (Branching)
        # V0 Revert: Simple Additive Residual
        # Prediction = ReLU(Physics_Power + Mamba_Residual)
        p_total = F.relu(p_physics + mamba_raw)
        
        return p_total, p_physics


    
    def get_physics_params(self) -> dict:
        return self.physics_layer.get_params_dict()


# =============================================================================
# Part 3: Physics-Informed Loss Function
# =============================================================================

class PhysicsPVLoss(nn.Module):
    """
    Triple-Constraint Physics PV Loss.
    """
    def __init__(self, configs, future_cols_order: list):
        super(PhysicsPVLoss, self).__init__()
        self.cfg = configs
        self.idx_G = future_cols_order.index('nwp_globalirrad')
        self.eps = 1e-6
        self.lambda_mono = getattr(configs, 'lambda_mono', 0.1)
        
    def forward(self, preds, y_true, x_future):
        G = x_future[:, :, self.idx_G]
        
        # --- Night Masking (GHI < 1.0 -> 0 Loss) ---
        # Don't learn from noise in the dark
        night_mask = (G < 1.0)
        
        # Only compute MSE on valid daytime samples
        diff = preds - y_true
        diff[night_mask] = 0.0 # Zero out night errors
        L_data = (diff ** 2).sum() / torch.clamp((~night_mask).sum(), min=1.0)
        
        # L_night: Penalize predictions during night (Strict)
        # Using a slightly higher threshold for penalty to encourage 0.0 earlier
        night_penalty_mask = (G < self.cfg.G_night_thr).unsqueeze(-1)
        if night_penalty_mask.any():
            L_night = (preds[night_penalty_mask] ** 2).mean()
        else:
            L_night = preds.new_tensor(0.0)
        
        # Penalize negative dP when dG is positive (power down while sun up)
        dP = preds[:, 1:, :] - preds[:, :-1, :]
        dG = (G[:, 1:] - G[:, :-1]).unsqueeze(-1)

        pos = dG > 0
        if pos.any():
            L_mono = F.relu(-dP[pos]).mean()
        else:
            L_mono = preds.new_tensor(0.0)
        
        total = (
            self.cfg.lambda_data * L_data +
            self.cfg.lambda_night * L_night +
            self.lambda_mono * L_mono
        )
        
        loss_dict = {
            "L_data": L_data.detach().item(),
            "L_night": L_night.detach().item(),
            "L_mono": L_mono.detach().item(),
        }
        
        return total, loss_dict


# =============================================================================
# Part 4: Dataset Class
# =============================================================================

class MultiStepDataset(Dataset):
    def __init__(self, X, Y, P, B_future, B_last, seq_len=96, pred_len=96, idx_clr=-1):
        self.X = X
        self.y = Y
        self.P = P
        self.Bf = B_future
        self.Bl = B_last
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.idx_clr = idx_clr

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        x_past = self.X[i : i + self.seq_len]
        start_fut = i + self.seq_len
        end_fut = start_fut + self.pred_len

        # Targets
        y_future = self.y[start_fut : end_fut]
        if y_future.ndim == 1: y_future = y_future[:, None]
            
        # Physics Baseline (P_phys)
        p_phys = self.P[start_fut : end_fut]
        if p_phys.ndim == 1: p_phys = p_phys[:, None]

        # Future Features (Bf)
        b_future = self.Bf[start_fut : end_fut]
        
        # Clear Sky (Extract from Bf if idx_clr is set, else zeroes)
        if self.idx_clr != -1 and self.idx_clr < b_future.shape[-1]:
             x_clr = b_future[:, self.idx_clr]
             if x_clr.ndim == 1: x_clr = x_clr[:, None]
        else:
             # Fallback: Try to find 'P_CLR' if passed, otherwise zeros
             # Given constraints, we return zeros if not found
             x_clr = np.zeros_like(y_future)

        # Lagged/Meta (Bl)
        b_last = self.Bl[i + self.seq_len - 1]

        # Return Order:
        # 0: Past Input (X)
        # 1: Future Target (Y_true)
        # 2: Future Features (X_future) - Critical for Generic Models
        # 3: Physics Baseline (P_phys)
        # 4: Clear Sky (P_cs)
        
        return (
            x_past.astype(np.float32),
            y_future.astype(np.float32),
            b_future.astype(np.float32), # Batch[2] = Features
            p_phys.astype(np.float32),   # Batch[3] = Physics
            x_clr.astype(np.float32),    # Batch[4] = Clear Sky
        )


def prepare_rolling_folds(df, input_cols, target_col, prediction_cols,
                          n_splits=2, seq_len=96, pred_len=96, batch_size=32,
                          physics_cols=None):
    if physics_cols is None:
        physics_cols = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed']
    
    # Calculate common features (Shadows) dynamically
    # input_cols = PAST, prediction_cols = FUTURE
    common_features = [f for f in prediction_cols if f in input_cols]
    
    df = df.fillna(0)
    
    X_raw = df[input_cols].values.astype(np.float32)
    P_raw = df[prediction_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)

    # Bf = Future Features (Shadows) -> Must match configs.common_features
    Bf_raw = df[common_features].values.astype(np.float32)
    
    # Bl = Lagged Meta (Power, P_CLR) - Keep as is for meta
    # Ensure P_CLR exists, else just Power
    meta_cols = ['power']
    if 'P_CLR' in df.columns:
        meta_cols.append('P_CLR')
    Bl_raw = df[meta_cols].values.astype(np.float32)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"DEBUG: prepare_rolling_folds - X_raw shape: {X_raw.shape}, n_splits={n_splits}")
    print(f"DEBUG: Common Features (Bf): {len(common_features)} {common_features}")

    fold_idx = 1
    for train_index, test_index in tscv.split(X_raw):
        print(f"DEBUG: Generator entering Fold {fold_idx} loop...")
        X_train, P_train, y_train = X_raw[train_index], P_raw[train_index], y_raw[train_index]
        X_test, P_test, y_test = X_raw[test_index], P_raw[test_index], y_raw[test_index]
        Bf_train, Bl_train = Bf_raw[train_index], Bl_raw[train_index]
        Bf_test, Bl_test = Bf_raw[test_index], Bl_raw[test_index]
        
        train_df = df.iloc[train_index]
        means, stds = compute_scaler_stats(train_df, physics_cols)
        scaler_stats = {'means': means, 'stds': stds}

        X_lookback = X_train[-seq_len:]
        P_lookback = P_train[-seq_len:]
        y_lookback = y_train[-seq_len:]
        Bf_lookback = Bf_train[-seq_len:]
        Bl_lookback = Bl_train[-seq_len:]

        X_test_final = np.concatenate([X_lookback, X_test], axis=0)
        P_test_final = np.concatenate([P_lookback, P_test], axis=0)
        y_test_final = np.concatenate([y_lookback, y_test], axis=0)
        Bf_test_final = np.concatenate([Bf_lookback, Bf_test], axis=0)
        Bl_test_final = np.concatenate([Bl_lookback, Bl_test], axis=0)
        
        try:
            irr_idx = prediction_cols.index('nwp_globalirrad')
            valid_indices = []
            total_samples = len(X_train) - seq_len - pred_len + 1
            
            for i in range(total_samples):
                start_fut = i + seq_len
                end_fut = start_fut + pred_len
                irr_window = P_train[start_fut:end_fut, irr_idx]
                if np.max(irr_window) > 50.0:
                    valid_indices.append(i)
            
            x_list, y_list, p_list, bf_list, bl_list = [], [], [], [], []
            for i in valid_indices:
                start_past = i
                end_past = i + seq_len
                start_fut = end_past
                end_fut = start_fut + pred_len
                
                x_list.append(X_train[start_past:end_past])
                y_list.append(y_train[start_fut:end_fut])
                p_list.append(P_train[start_fut:end_fut])
                bf_list.append(Bf_train[start_fut:end_fut])
                bl_list.append(Bl_train[end_past-1])
            
            if len(x_list) > 0:
                class FilteredDataset(Dataset):
                    def __init__(self, x_l, y_l, p_l, bf_l, bl_l, idx_clr=1):
                        self.x = x_l
                        self.y = y_l
                        self.p = p_l
                        self.bf = bf_l
                        self.idx_clr = idx_clr
                    
                    def __len__(self): return len(self.x)
                    
                    def __getitem__(self, idx):
                        x_p = torch.tensor(self.x[idx], dtype=torch.float)
                        
                        y_f = torch.tensor(self.y[idx], dtype=torch.float)
                        if y_f.ndim == 1: y_f = y_f.unsqueeze(-1)
                        
                        p_ph = torch.tensor(self.p[idx], dtype=torch.float)
                        if p_ph.ndim == 1: p_ph = p_ph.unsqueeze(-1)
                        
                        b_f = torch.tensor(self.bf[idx], dtype=torch.float)
                        
                        # Extract Clear Sky (P_CLR at idx_clr)
                        if self.idx_clr != -1 and self.idx_clr < b_f.shape[-1]:
                             x_c = b_f[:, self.idx_clr]
                             if x_c.ndim == 1: x_c = x_c.unsqueeze(-1)
                        else:
                             x_c = torch.zeros_like(y_f)

                        # Return Order: X, Y, Features, Physics, ClearSky
                        return x_p, y_f, b_f, p_ph, x_c
                
                train_dataset = FilteredDataset(x_list, y_list, p_list, bf_list, bl_list, idx_clr=-1)
                print(f"Fold {fold_idx}: Filtered {total_samples - len(valid_indices)} night/invalid samples. Training on {len(valid_indices)} samples.")
            else:
                # P_CLR is index 1 in Bf (['NWP_Power_MW', 'P_CLR'])
                train_dataset = MultiStepDataset(X_train, y_train, P_train, Bf_train, Bl_train, seq_len, pred_len, idx_clr=-1)
        
        except ValueError:
            train_dataset = MultiStepDataset(X_train, y_train, P_train, Bf_train, Bl_train, seq_len, pred_len, idx_clr=-1)
            
        fold_idx += 1
        test_dataset = MultiStepDataset(X_test_final, y_test_final, P_test_final, Bf_test_final, Bl_test_final, seq_len, pred_len, idx_clr=-1)

        yield (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
            scaler_stats,
        )

# =============================================================================

def train_one_epoch_physics(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    phys_logs_accum = {"L_data": 0.0, "L_night": 0.0, "L_mono": 0.0}
    n_batches = 0

    for batch in loader:
        # Unpack: 0:x_past, 1:y_future, 2:x_features, 3:p_phys, 4:x_clr
        x_past = batch[0].to(device)
        y_future = batch[1].to(device)
        x_features = batch[2].to(device)
        p_phys_base = batch[3].to(device)
        x_clr = batch[4].to(device)

        optimizer.zero_grad()
        
        # Forward Pass with K_PV Reconstruction
        preds, p_physics_out = model(x_past, x_features, x_clr=x_clr)
        
        # Criterion expects predictions (MW) and targets (MW)
        # We pass x_features for monotonicity checks etc if needed
        loss, logs = criterion(preds, y_future, x_features)
        
        phys_logs_accum["L_data"] += logs["L_data"]
        phys_logs_accum["L_night"] += logs["L_night"]
        phys_logs_accum["L_mono"] += logs["L_mono"]

        loss.backward()
        
        if model.configs.experiment_mode == 'v4_robust':
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        else:
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
             
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    for k in phys_logs_accum:
        phys_logs_accum[k] /= max(1, n_batches)
    phys_logs_accum.update(model.get_physics_params())
    return avg_loss, phys_logs_accum


def evaluate_fold_physics(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_actuals = []
    phys_logs_accum = {"L_data": 0.0, "L_night": 0.0, "L_mono": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            y_future = batch[1].to(device)
            x_features = batch[2].to(device)
            x_clr = batch[4].to(device)

            preds, p_physics_index = model(x_past, x_features, x_clr=x_clr)
            loss, logs = criterion(preds, y_future, x_features)
            
            phys_logs_accum["L_data"] += logs["L_data"]
            phys_logs_accum["L_night"] += logs["L_night"]
            phys_logs_accum["L_mono"] += logs["L_mono"]

            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_future.cpu().numpy())
            n_batches += 1

    preds_arr = np.concatenate(all_preds, axis=0)
    actuals_arr = np.concatenate(all_actuals, axis=0)
    preds_flat = preds_arr.reshape(-1)
    actuals_flat = actuals_arr.reshape(-1)

    rmse = np.sqrt(np.mean((preds_flat - actuals_flat) ** 2))
    mae = np.mean(np.abs(preds_flat - actuals_flat))

    avg_loss = total_loss / max(1, n_batches)
    for k in phys_logs_accum:
        phys_logs_accum[k] /= max(1, n_batches)

    return avg_loss, rmse, mae, phys_logs_accum


def plot_loss_curve(train_losses, val_losses, fold_num):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title(f'Fold {fold_num} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_physics_residual_cv(df, configs, use_scaler=True, n_splits=None):
    results = []
    history = {}
    device = configs.device

    final_n_splits = n_splits if n_splits is not None else configs.n_splits
    if n_splits is not None:
        print(f"DEBUG: Overriding configs.n_splits ({configs.n_splits}) with argument {n_splits}")

    print(f"DEBUGGING CONFIGS: n_splits={final_n_splits}, batch_size={configs.batch_size}")
    
    fold_gen = prepare_rolling_folds(
        df, configs.PAST_INPUT_COLS, configs.TARGET_COL, configs.FUTURE_INPUT_COLS,
        n_splits=final_n_splits,
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        batch_size=configs.batch_size
    )

    for i, (train_loader, test_loader, scaler_stats) in enumerate(fold_gen):
        fold = i + 1
        print(f"\n=== Training Fold {fold}/{configs.n_splits} ===")
        history[fold] = {'train': [], 'val': [], 'physics_params': [], 'scaler_stats': scaler_stats}

        model = PhysicsResidualMamba(configs)
        
        if use_scaler:
            scaler = FeatureScaler(scaler_stats['means'], scaler_stats['stds'])
            print(f"  Scaler stats: G_mean={scaler_stats['means']['nwp_globalirrad']:.1f}, "
                  f"T_mean={scaler_stats['means']['nwp_temperature']:.1f}, "
                  f"W_mean={scaler_stats['means']['nwp_windspeed']:.2f}")
        
        model = model.to(device)
        optimizer = None
        criterion = PhysicsPVLoss(configs, configs.FUTURE_INPUT_COLS).to(device)

        for epoch in range(configs.epochs):
            if epoch < 5:
                for param in model.mamba_model.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.Adam(model.physics_layer.parameters(), lr=1e-3)
            else:
                for param in model.mamba_model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            
            train_loss, train_logs = train_one_epoch_physics(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_rmse, val_mae, val_logs = evaluate_fold_physics(
                model, test_loader, criterion, device
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
                    f"L_data={val_logs['L_data']:.4f} L_night={val_logs['L_night']:.4f} | "
                    f"η={phys['eta']:.4f} U0={phys['U0']:.2f} γ={phys['gamma']:.5f}"
                )
                print(msg)

        final_loss, final_rmse, final_mae, _ = evaluate_fold_physics(
            model, test_loader, criterion, device
        )
        print(f"--> Fold {fold} Finished: RMSE={final_rmse:.4f}, MAE={final_mae:.4f}")

        results.append({
            'fold': fold, 
            'rmse': final_rmse, 
            'mae': final_mae,
            'physics_params': model.get_physics_params(),
            'scaler_stats': scaler_stats
        })

        plot_loss_curve(history[fold]['train'], history[fold]['val'], fold)

    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    
    print("\n=== Final Cross-Validation Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    print("\nLearned Physics Parameters (last fold):")
    final_params = results[-1]['physics_params']
    for k, v in final_params.items():
        print(f"  {k}: {v:.6f}")

    return results, history


class PhysicsResidualMambaConfigs:
    """
    Configuration for Physics-Residual Mamba model.
    """
    def __init__(self, n_splits=4):
        self.seq_len = 96
        self.pred_len = 96
        self.experiment_mode = 'v4_robust' # Options: 'v1_baseline', 'v2_trend', 'v3_advanced', 'v4_robust'
        self.TARGET_COL = ['power']
        self.PAST_INPUT_COLS = [
            'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity',
            'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure', 'lmd_totalirrad',
            'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure',
            'lmd_winddirection', 'lmd_windspeed', 'power', 'GHI_clr', 'K_CS_day',
            'K_CS_dayNight', 'K_CS', 'P_CLR', 'K_PV', 'NWP_Power_MW', 'hour_sin',
            'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'season_sin', 'season_cos'
        ]
        self.FUTURE_INPUT_COLS = [
            'nwp_globalirrad',
            'nwp_temperature',
            'nwp_windspeed',
            'NWP_Power_MW', # Added for NWP benchmark
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'season_sin', 'season_cos'
        ]
        self.common_features = [f for f in self.FUTURE_INPUT_COLS if f in self.PAST_INPUT_COLS]
        self.num_shadows = len(self.common_features)
        self.include_pred = 1 if len(self.FUTURE_INPUT_COLS) > 0 else 0
        self.enc_in = len(self.PAST_INPUT_COLS) + self.num_shadows
        self.c_out = len(self.TARGET_COL)
        
        self.kernel_size = 25
        self.n_embed = 128
        self.d_state = 128
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

if __name__ == "__main__":
    print("Physics-Residual Power Mamba Architecture")
    print("=" * 50)

# =============================================================================
# Part 5: Base Mamba & Benchmarking (Re-added)
# =============================================================================

def train_one_epoch_base(model, loader, optimizer, criterion, device):
    """
    Train one epoch for Base Mamba (Pure Data-Driven).
    Uses SmoothL1Loss and NaN guards.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        # Unpack 4-tuple: (x_past, y_future, x_features, p_phys)
        x_past = batch[0].to(device)
        y_future = batch[1].to(device)
        x_future = batch[2].to(device) 
        
        # NaN Guard
        if torch.isnan(x_past).any() or torch.isnan(y_future).any():
            continue

        optimizer.zero_grad()
        
        # Base Mamba Forward
        # Returns (preds, stdev_from_revin) - ignore stdev here
        preds, _ = model(x_past, x_future)
        
        loss = criterion(preds, y_future)
        
        # Loss NaNs guard
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def evaluate_fold_base(model, loader, criterion, device):
    """
    Evaluate Base Mamba.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_actuals = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            # Unpack 5-tuple
            x_past = batch[0].to(device)
            y_future = batch[1].to(device)
            x_future = batch[2].to(device)

            preds, _ = model(x_past, x_future)
            loss = criterion(preds, y_future)

            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_future.cpu().numpy())
            n_batches += 1

    preds_arr = np.concatenate(all_preds, axis=0)
    actuals_arr = np.concatenate(all_actuals, axis=0)
    preds_flat = preds_arr.reshape(-1)
    actuals_flat = actuals_arr.reshape(-1)

    # Robust Metrics
    rmse = np.sqrt(np.nanmean((preds_flat - actuals_flat) ** 2))
    mae = np.nanmean(np.abs(preds_flat - actuals_flat))

    return total_loss / max(1, n_batches), rmse, mae, preds_arr, actuals_arr

def run_base_mamba_cv(df, configs, n_splits=4):
    """
    Run CV for Base Mamba.
    """
    results = []
    device = configs.device
    print(f"\nTRAINING BASE MAMBA | Folds: {n_splits}")
    
    fold_gen = prepare_rolling_folds(
        df, configs.PAST_INPUT_COLS, configs.TARGET_COL, configs.FUTURE_INPUT_COLS,
        n_splits=n_splits, seq_len=configs.seq_len, pred_len=configs.pred_len,
        batch_size=configs.batch_size
    )
    
    for i, (train_loader, test_loader, _) in enumerate(fold_gen):
        fold = i + 1
        print(f"  Fold {fold}/{n_splits}...", end="", flush=True)
        
        # Base mamba is just MMPISSM_Model_01 without residual flag
        model = MMPISSM_Model_01(configs, output_residual=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Slightly lower LR
        criterion = torch.nn.SmoothL1Loss() 
        
        for epoch in range(configs.epochs):
            train_one_epoch_base(model, train_loader, optimizer, criterion, device)
            
        _, rmse, mae, preds, acts = evaluate_fold_base(model, test_loader, criterion, device)
        print(f" Done. RMSE={rmse:.4f}")
        
        results.append({
            'fold': fold, 'rmse': rmse, 'mae': mae, 
            'preds': preds, 'actuals': acts
        })
        
    return results

def evaluate_fold_physics_benchmark(model, loader, criterion, device, configs):
    """
    Evaluate Physics model (Benchmark Version).
    Returns: avg_loss, rmse, mae, phys_logs, preds, actuals, baselines, nwp
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_actuals = []
    all_baselines = []
    all_nwp = []
    phys_logs_accum = {"L_data": 0.0, "L_night": 0.0, "L_mono": 0.0}
    n_batches = 0
    
    # Indices for Smart Persistence
    try:
        p_idx = configs.PAST_INPUT_COLS.index('power')
        pclr_past_idx = configs.PAST_INPUT_COLS.index('P_CLR')
    except ValueError:
        p_idx = -1
        pclr_past_idx = -1

    # Index for NWP
    nwp_fut_idx = -1
    if 'NWP_Power_MW' in configs.FUTURE_INPUT_COLS:
        nwp_fut_idx = configs.FUTURE_INPUT_COLS.index('NWP_Power_MW')

    with torch.no_grad():
        for batch in loader:
            # Unpack all 4 elements
            x_past = batch[0].to(device)
            y_future = batch[1].to(device)
            x_future = batch[2].to(device)
            # batch[3] is p_phys, not needed here directly unless for logging
            b_future = x_future # Alias for compatibility
            
            preds, p_physics = model(x_past, x_future)
            loss, logs = criterion(preds, y_future, x_future)
            
            phys_logs_accum["L_data"] += logs["L_data"]
            phys_logs_accum["L_night"] += logs["L_night"]
            phys_logs_accum["L_mono"] += logs["L_mono"]
            total_loss += loss.item()
            
            # --- Smart Persistence Calculation ---
            # P_pred[t] = P_actual[t-24h] * (CLR[t] / CLR[t-24h])
            # Assuming seq_len=96 (24h) and pred_len=96 (24h) matches perfectly.
            if p_idx >= 0 and pclr_past_idx >= 0:
                p_hist = x_past[:, :, p_idx]       # P[t-24h]
                clr_hist = x_past[:, :, pclr_past_idx] # CLR[t-24h]
                clr_fut = b_future[:, :, 1]        # CLR[t]
                
                # Avoid div by zero
                ratio = clr_fut / (clr_hist + 1e-4)
                baseline = p_hist * ratio
                all_baselines.append(baseline.cpu().numpy())
            else:
                all_baselines.append(np.zeros_like(preds.cpu().numpy()))

            # --- NWP Extraction ---
            if nwp_fut_idx >= 0:
                all_nwp.append(x_future[:, :, nwp_fut_idx].cpu().numpy())
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
    
    # Metrics
    rmse = np.sqrt(np.nanmean((preds_flat - actuals_flat) ** 2))
    mae = np.nanmean(np.abs(preds_flat - actuals_flat))

    avg_loss = total_loss / max(1, n_batches)
    for k in phys_logs_accum:
        phys_logs_accum[k] /= max(1, n_batches)
    phys_logs_accum.update(model.get_physics_params())
    
    return avg_loss, rmse, mae, phys_logs_accum, preds_arr, actuals_arr, baseline_arr, nwp_arr

def plot_benchmark_summary(base_results, phys_results):
    import matplotlib.pyplot as plt
    
    avg_base_rmse = np.nanmean([r['rmse'] for r in base_results])
    avg_phys_rmse = np.nanmean([r['rmse'] for r in phys_results])
    avg_sp_rmse = np.nanmean([r['sp_rmse'] for r in phys_results])
    avg_nwp_rmse = np.nanmean([r['nwp_rmse'] for r in phys_results])
    
    models = ['SP', 'NWP', 'Base', 'Physics']
    rmses = [avg_sp_rmse, avg_nwp_rmse, avg_base_rmse, avg_phys_rmse]
    colors = ['gray', 'orange', 'blue', 'red']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, rmses, color=colors)
    plt.ylabel('RMSE (MW)')
    plt.title('Performance Comparison')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')
    plt.show()

def run_full_benchmark(df, base_configs, phys_configs, n_splits=4):
    print("=== BENCHMARK START ===")
    
    # Base Mamba
    base_results = run_base_mamba_cv(df, base_configs, n_splits=n_splits)
    
    # Physics Mamba (Must use upgraded runner/eval logic, tricky)
    # We need to monkey-patch or update run_physics_residual_cv to use v2 eval
    # Or just write a new runner here?
    # I'll write 'run_physics_benchmark_cv' here.
    
    phys_results, _ = run_physics_benchmark_cv(df, phys_configs, n_splits=n_splits)
    
    plot_benchmark_summary(base_results, phys_results)
    return base_results, phys_results

def run_physics_benchmark_cv(df, configs, n_splits=4):
    """
    Dedicated runner for benchmark that uses evaluate_fold_physics_v2
    """
    results = []
    device = configs.device
    print(f"\nTRAINING PHYSICS (BENCHMARK) | Folds: {n_splits}")
    
    fold_gen = prepare_rolling_folds(
         df, configs.PAST_INPUT_COLS, configs.TARGET_COL, configs.FUTURE_INPUT_COLS,
         n_splits=n_splits, seq_len=configs.seq_len, pred_len=configs.pred_len,
         batch_size=configs.batch_size
    )
    
    history = {} # Global history accumulator

    for i, (train_loader, test_loader, scaler_stats) in enumerate(fold_gen):
        fold = i + 1
        print(f"  Fold {fold}...", end="")
        
        history[fold] = {
            'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': [],
            'eta_hist': [], 'U0_hist': [], 'gamma_hist': [] 
        }

        # 1. Init Model
        model = PhysicsResidualMamba(configs).to(device)
        criterion = PhysicsPVLoss(configs, configs.FUTURE_INPUT_COLS).to(device)
        
        # 2. Two-Phase Training Loop
        optimizer = None
        best_rmse = float('inf')
        patience = 7
        counter = 0
        best_model_path = f"best_model_fold_{fold}.pth"
        
        # V0 Strategy: Warm-Up Physics First
        warmup_epochs = 5
        
        for epoch in range(configs.epochs):
            # --- Warm-Up Switch Logic ---
            if epoch < warmup_epochs:
                # Phase 1: Warm-Up (Freeze Mamba, Train Physics)
                for param in model.mamba_model.parameters():
                    param.requires_grad = False
                for param in model.physics_layer.parameters():
                    param.requires_grad = True
                
                # Standard LR for Physics Warmup
                optimizer = torch.optim.Adam(model.physics_layer.parameters(), lr=1e-2)
                phase_name = "WARMUP"
            else:
                # Phase 2: Joint Training (Unfreeze Mamba)
                for param in model.mamba_model.parameters():
                    param.requires_grad = True
                for param in model.physics_layer.parameters():
                    param.requires_grad = True # Joint Training in V0
                
                # Joint Optimizer (Mamba dominates, Physics tweaks slowly)
                optimizer = torch.optim.Adam([
                    {'params': model.mamba_model.parameters(), 'lr': 5e-4},
                    {'params': model.physics_layer.parameters(), 'lr': 1e-3}
                ])
                phase_name = "JOINT"
            
            # Train
            train_loss, logs = train_one_epoch_physics(model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss, val_rmse, val_mae, _, _, _, _, _ = evaluate_fold_physics_benchmark(
                model, test_loader, criterion, device, configs
            )
            
            # Record History
            history[fold]['train_loss'].append(train_loss)
            history[fold]['val_loss'].append(val_loss)
            history[fold]['val_rmse'].append(val_rmse)
            history[fold]['val_mae'].append(val_mae)
            
            # Record Physics Params
            params = model.get_physics_params()
            history[fold]['eta_hist'].append(params['eta'])
            history[fold]['U0_hist'].append(params['U0'])
            history[fold]['gamma_hist'].append(params['gamma'])
            
            # Phase Report at end of Warmup
            if epoch == warmup_epochs:
                p = model.get_physics_params()
                print(f"\n[WARMUP COMPLETE] Learned Params: Eta={p['eta']:.3f}, U0={p['U0']:.2f}")

            # --- Checkpointing & Early Stopping ---
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f" [Ep{epoch+1} {phase_name}: L={train_loss:.4f} Val={val_loss:.4f} *New Best*]", end="")
            else:
                counter += 1
                if (epoch + 1) % 5 == 0:
                     print(f" [Ep{epoch+1} {phase_name}: L={train_loss:.4f} Val={val_loss:.4f}]", end="")
            
            if counter >= patience:
                print(f"\nExample early stopping triggered at epoch {epoch+1}")
                break
        
        # 3. Final Benchmark Evaluation
        # RELOAD BEST MODEL
        print(f"\nReloading Best Model (RMSE={best_rmse:.4f})...")
        model.load_state_dict(torch.load(best_model_path))
        
        avg_loss, rmse, mae, logs, preds, acts, bases, nwp = evaluate_fold_physics_benchmark(
            model, test_loader, criterion, device, configs
        )
        
        # Metrics
        sp_flat = bases.reshape(-1)
        nwp_flat = nwp.reshape(-1)
        act_flat = acts.reshape(-1)
        
        sp_rmse = np.sqrt(np.nanmean((sp_flat - act_flat)**2))
        nwp_rmse = np.sqrt(np.nanmean((nwp_flat - act_flat)**2))
        
        print(f" Done. RMSE={rmse:.4f} SP={sp_rmse:.4f}")
        
        results.append({
            'fold': fold, 'rmse': rmse, 'mae': mae, 
            'sp_rmse': sp_rmse, 'nwp_rmse': nwp_rmse,
            'preds': preds, 'actuals': acts, 
            'smart_persistence': bases, 'nwp_physical': nwp
        })
        
        # Plotting (Only for last fold to save time/space)
        if fold == n_splits:
             plot_station_benchmark(results[-1])
             plot_training_dynamics(history)
        
    return results, history


def plot_training_dynamics(history):
    """
    Plots Train/Val Loss coverage, Val RMSE, and Physics Parameter Evolution.
    Shows the Phase Switch at Epoch 10.
    """
    import matplotlib.pyplot as plt
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('bmh')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss
    for fold, data in history.items():
        axes[0].plot(data['train_loss'], label=f'Fold {fold} Train', linestyle='--')
        axes[0].plot(data['val_loss'], label=f'Fold {fold} Val')
    axes[0].axvline(x=15, color='k', linestyle=':', label='Phase Switch') # Epoch 15
    axes[0].set_title('Loss Convergence')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot 2: RMSE
    for fold, data in history.items():
        axes[1].plot(data['val_rmse'], label=f'Fold {fold}')
    axes[1].axvline(x=15, color='k', linestyle=':')
    axes[1].set_title('Validation RMSE (MW)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    # Plot 3: Physics Params (Eta & U0)
    # We plot the last fold's evolution for clarity
    last_fold = max(history.keys())
    data = history[last_fold]
    
    ax3 = axes[2]
    ln1 = ax3.plot(data['eta_hist'], label='Efficiency (η)', color='green')
    ax3.set_ylabel('Efficiency')
    ax3.set_xlabel('Epoch')
    
    ax3_twin = ax3.twinx()
    ln2 = ax3_twin.plot(data['U0_hist'], label='Heat Transfer (U0)', color='red')
    ax3_twin.set_ylabel('U0')
    
    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=0)
    
    ax3.axvline(x=5, color='k', linestyle=':', label='Warm-Up End')
    ax3.set_title(f'Physics Parameter Evolution (Fold {last_fold})')
    

# =============================================================================
# End of Core Physics-Residual Mamba Architecture
# =============================================================================

def plot_station_benchmark(fold_result):
    """
    Report specific to Station 07 requirements:
    1. 3-Day Zoom (Truth, Model, SP, NWP)
    2. Scatter with R2
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('bmh')

    preds = fold_result['preds'].reshape(-1)
    acts = fold_result['actuals'].reshape(-1)
    sp = fold_result['smart_persistence'].reshape(-1)
    nwp = fold_result['nwp_physical'].reshape(-1)
    
    # 1. 3-Day Zoom (approx 3 * 96 = 288 steps)
    start_idx = 96  # Skip first day
    window = 288
    end_idx = start_idx + window
    
    if len(preds) > end_idx:
        plt.figure(figsize=(15, 6))
        plt.plot(acts[start_idx:end_idx], 'k-', label='Ground Truth', linewidth=2, alpha=0.8)
        plt.plot(sp[start_idx:end_idx], 'b--', label='Smart Persistence', alpha=0.5)
        plt.plot(nwp[start_idx:end_idx], 'g:', label='NWP Baseline', alpha=0.5)
        plt.plot(preds[start_idx:end_idx], 'r-', label='Phys-Res Model', linewidth=2)
        
        plt.title("Station 07 Benchmark: 3-Day Zoom")
        plt.ylabel("Power (MW)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 2. Scatter Plot with R2
    valid_mask = ~np.isnan(acts) & ~np.isnan(preds)
    r2 = r2_score(acts[valid_mask], preds[valid_mask])
    
    plt.figure(figsize=(8, 8))
    plt.scatter(acts, preds, alpha=0.1, color='darkred', s=10)
    plt.plot([0, 25], [0, 25], 'k--', linewidth=2) # 20MW capacity, slightly larger axis
    
    plt.text(1.0, 22.0, f'$R^2 = {r2:.4f}$', fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f"Goodness of Fit ($R^2={r2:.3f}$)")
    plt.xlabel("Actual Power (MW)")
    plt.ylabel("Predicted Power (MW)")
    plt.grid(True)
    plt.xlim(0, 24)
    plt.ylim(0, 24)
    plt.show()

if __name__ == "__main__":
    print("Module Recreated with Benchmarking Support.")
