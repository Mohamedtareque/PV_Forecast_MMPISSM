"""
Physics-Residual Power Mamba (Pure Variant)
"The Pure Physics Implementation"

Description:
This variant simplifies the benchmark architecture by removing the Inverse-Mamba block.
It uses a strictly straightforward Mamba backbone with a consistent residual stream,
providing a "Pure" baseline for the Physics-Residual concept.

Features:
- Single-Branch Mamba (No iMamba).
- Simplified Fusion: [Mamba, Skip] instead of 4-way concatenation.
- Uses Standard DifferentiablePVLayer_V02.

Author: Antigravity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from physics_residual_mamba_v02 import (
    DifferentiablePVLayer_V02,
    PhysicsResidualMambaConfigs,
    RevIN, 
    series_decomp,
    V02Dataset, # for consistency if needed
    train_one_epoch_v02 # for consistency
)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using Linear fallback.")

# =============================================================================
# 1. MambaPureBackbone
# =============================================================================

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
        
        # --- Indexing & State Setup (Same as V02) ---
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
        # Takes [Seasonal, Trend] -> 2 * enc_in? No, decomposition returns (B, L, enc_in).
        # V02 logic: x = cat([seasonal, trend], dim=1) -> B, 2*L, C -> lin1 -> B, L, C?
        # Check v02: 
        #   seasonal, trend = (B, N, L) [permuted]
        #   cat -> (B, 2N, L) ?? No.
        #   Let's check V02 forward pass carefully.
        #   V02 forward: 
        #      seasonal, trend = decomp(x) -> (B, L, C) each.
        #      cat(dim=1) -> (B, 2*L, C).
        #      lin1 (2*L -> L). OK.
        
        #      Then later...
        #      x_e = cat([seasonal, trend], dim=1) -> (B, 2*seq, C) ??
        #      lin2 (2*seq -> n_embed).
        #      Wait, lin2 weights in V02 is (2 * seq_len, n_embed).
        #      This implies it projects the TIME dimension into EMBED dimension?
        #      x_e = lin2(x_e.permute(0, 2, 1)) -> Input (B, C, 2*L). Output (B, C, n_embed).
        
        self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)
        
        # 4. Projection Head (MODIFIED)
        # V02: 4 * n_embed -> pred_len.
        # "Pure": We only have [x_m, x_e]. 
        # x_e is (B, C, n_embed).
        # x_m is mamba(x_e). Same shape.
        # Cat dim=2 -> (B, C, 2*n_embed).
        # So we need 2 * n_embed input size.
        self.lin3 = nn.Linear(2 * self.configs.n_embed, self.configs.pred_len)
        
        self.dropout1 = nn.Dropout(self.configs.dropout)
        # self.dropout2 = nn.Dropout(...) # Removed mamba2 branch
        
        # 5. Core Mamba (Simplified)
        if MAMBA_AVAILABLE:
            self.mamba1 = Mamba(
                d_model=self.configs.n_embed, # Acting on Embedding Dim?
                # V02: mamba1(d_model=n_embed).
                # Note: Logic seems to process (B, Channels, Embed_Dim). Mamba scans across Embed_Dim? 
                # Or is it scanning across Channels?
                # V02 x_e shape before mamba: lin2 output is (B, C, n_embed).
                # Mamba expects (B, L, D). Here L=Channels, D=n_embed? Or L=n_embed?
                # Standard Mamba is Sequence model.
                # If V02 projects Time -> Embed, then "Channels" become the Sequence length?
                # That allows Mixing between Variables. Correct.
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

        # --- Decomp & Embedding (Same as V02) ---
        x = x_pred
        x = self.revin_layer_enc(x, 'norm') # (B, L+H, C)
        
        # Level 1 Decomp
        seasonal_init, trend_init = self.decompsition(x)
        x = torch.cat([seasonal_init, trend_init], dim=1) # (B, 2*(L+H), C)
        
        # Lin 1: (2*(L+H) -> L)
        # Input to lin1 needs to be (B, C, 2*(L+H)).
        x = self.lin1(x.permute(0, 2, 1)).permute(0, 2, 1) # (B, L, C)
        
        x = self.revin_layer_enc(x, 'denorm')
        x = self.revin_layer(x, 'norm')
        
        # Level 2 Decomp
        seasonal_init, trend_init = self.decompsition(x)
        x_e = torch.cat([seasonal_init, trend_init], dim=1) # (B, 2*L, C)
        
        # Embed: Time -> n_embed
        # Input to lin2: (B, C, 2*L)
        x_e = self.lin2(x_e.permute(0, 2, 1)) # (B, C, n_embed)
        
        # --- Mamba Pure (Simplified) ---
        # 1. Main Branch
        # Input: (B, C, n_embed) -> Scans across variables (C)
        x_m = self.mamba1(self.dropout1(x_e)) # (B, C, n_embed)
        
        # REMOVED: iMamba Branch (mamba2)
        
        # 2. Simplified Fusion
        # V02: [x_im, x_m, sum, x_e] (4 parts)
        # Pure: [x_m, x_e] (Residual Connection)
        x_cat = torch.cat([x_m, x_e], dim=2) # (B, C, 2*n_embed)
        
        # 3. Projection
        x = self.lin3(x_cat).permute(0, 2, 1) # (B, C, pred_len) -> (B, pred_len, C)
        
        # --- Output ---
        x = self.revin_layer(x, 'denorm')
        
        # Extract Target
        x = x[:, :, self.target_indices]
        past_stdev_target = torch.ones_like(x) 
        
        return x, past_stdev_target

# =============================================================================
# 2. PhysicsResidualMamba_Pure
# =============================================================================

class PhysicsResidualMamba_Pure(nn.Module):
    """
    Physics-Residual Mamba (Pure).
    Combines DifferentiablePVLayer (V02) with MambaPureBackbone.
    """
    def __init__(self, configs):
        super(PhysicsResidualMamba_Pure, self).__init__()
        self.configs = configs
        
        # 1. Physics Layer (V02 - Standard)
        idx_G = configs.common_features.index('nwp_globalirrad')
        idx_Ta = configs.common_features.index('nwp_temperature')
        idx_WS = configs.common_features.index('nwp_windspeed')
        
        # Try to find elevation index if available
        try:
            idx_elev = configs.common_features.index('sun_elevation')
        except ValueError:
            idx_elev = None

        # Note: V02 uses P_stc_init=19.0 default.
        self.physics_layer = DifferentiablePVLayer_V02(
            idx_G, idx_Ta, idx_WS, idx_time_feats=[], 
            idx_elev=idx_elev,
            T_ref=configs.T_ref, P_stc_init=19.0
        )
        
        # 2. Residual Learner (Pure)
        self.mamba_model = MambaPureBackbone(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        # 1. Physics Prediction (K_PV)
        # x_future contains NWP data for the forecast horizon
        p_physics_k = self.physics_layer(x_future, x_clr=x_clr)
        
        # 2. Residual Prediction (Delta K_PV)
        # mamba_k shape: (B, H, 1)
        mamba_k, _ = self.mamba_model(x_past, x_future)
        
        # 3. Additive Residual
        k_total = p_physics_k + mamba_k
        
        # 4. Clearsky Scaling -> Power
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        
        # Physical Rectification (Power >= 0)
        p_total_mw = F.relu(k_total) * x_clr
        
        return p_total_mw, p_physics_k

# =============================================================================
# 3. Usage Example / Runner
# =============================================================================

def train_pure_variant(df, n_splits=2):
    """
    Example runner for the Pure variant.
    """
    print(">>> Initializing PhysicsResidualMamba_Pure...")
    configs = PhysicsResidualMambaConfigs(n_splits=n_splits)
    model = PhysicsResidualMamba_Pure(configs).to(configs.device)
    
    print(f"Model Created. Params: {sum(p.numel() for p in model.parameters())}")
    
    # Example Dummy Input Check
    B, L, H = 2, 96, 96
    D_in = len(configs.PAST_INPUT_COLS)
    D_fut = len(configs.FUTURE_INPUT_COLS)
    
    x_past = torch.randn(B, L, D_in).to(configs.device)
    x_fut = torch.randn(B, H, D_fut).to(configs.device)
    x_clr = torch.rand(B, H).to(configs.device) # Clearsky
    
    print("Running Forward Pass...")
    try:
        p_pred, p_phys = model(x_past, x_fut, x_clr)
        print(f"Success. Output Shape: {p_pred.shape}") # Should be (B, H, 1)
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # If run as script
    pass
