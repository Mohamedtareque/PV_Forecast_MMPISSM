"""
Spatiotemporal Physics-Residual Mamba V04 (Relational & Wind-Gated)
"The Relational Model"

Design:
1. Feature Engineering: Discards S8 Raw NWP. Computes Spatial Delta dS(t) = K_PV_S7(t) - K_PV_S8(t).
2. Advection Gate: WindGatingModule(WindSpeed, WindDir) -> Alpha (0-1).
3. Mamba Input: [S7_Features, Alpha * dS_Features].
4. Physics Branch: V4 Robust (IAM + Inverter).

Author: Antigravity (Google DeepMind)
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
import physics_residual_mamba_v02 as v02_module # Reuse base classes

# =============================================================================
# Part 1: Relational Layers
# =============================================================================

class WindGatingModule(nn.Module):
    """
    Learns 'Advection Relevance' (Alpha) from Wind Vector.
    Input: [WindSpeed, WindDir_Sin, WindDir_Cos] (or similar) of Target S7.
    Output: Alpha (0 to 1) scalar for each timestep.
    """
    def __init__(self, input_dim=3, hidden_dim=16):
        super(WindGatingModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, wind_feats):
        # wind_feats: (B, L, D)
        alpha = self.net(wind_feats) # (B, L, 1)
        return alpha

class DifferentiablePVLayer_V4(nn.Module):
    """V4: Robust (IAM + Non-Linear Inverter) - Copied from V02 code base"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, T_ref=25.0, G_stc=1000.0, P_stc_init=19.0):
        super().__init__()
        self.idx_G = idx_G
        self.idx_Ta = idx_Ta 
        self.idx_WS = idx_WS
        self.idx_time_feats = idx_time_feats
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6
        
        # Physics Parameters
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4)) 
        self.U0_raw = nn.Parameter(torch.tensor(3.2)) 
        self.U1_raw = nn.Parameter(torch.tensor(1.9)) 
        self.eta_raw = nn.Parameter(torch.tensor(-1.66)) 
        self.Pstc_raw = nn.Parameter(torch.tensor(float(P_stc_init))) 
        
        # V4 Extra
        self.b0_raw = nn.Parameter(torch.tensor(0.05)) # IAM
        self.inv_k_raw = nn.Parameter(torch.tensor(1.0)) # Inv Steepness
        self.inv_thresh_raw = nn.Parameter(torch.tensor(0.1)) 

        # GeoNet
        self.geo_net = nn.Sequential(
            nn.Linear(len(idx_time_feats), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x_future, x_clr=None):
        # Extract features robustly
        def get_feat(idx, default=0.0):
            if idx is not None and idx < x_future.shape[2]:
                return x_future[:, :, idx]
            return torch.zeros_like(x_future[:, :, 0]) + default
            
        G = get_feat(self.idx_G)
        Ta = get_feat(self.idx_Ta, 25.0)
        WS = get_feat(self.idx_WS, 1.0)
        
        # Time feats
        time_feats_list = []
        for idx in self.idx_time_feats:
            if idx is not None and idx < x_future.shape[2]:
                time_feats_list.append(x_future[:, :, idx].unsqueeze(-1))
            else:
                time_feats_list.append(torch.zeros_like(x_future[:, :, 0:1]))
        if time_feats_list:
            time_feats = torch.cat(time_feats_list, dim=-1)
        else:
            time_feats = torch.zeros(G.shape[0], G.shape[1], 1, device=G.device)
        
        # --- Physics Logic ---
        # 1. Tilt Correction
        tilt_factor = self.geo_net(time_feats).squeeze(-1)
        G_poa = G * tilt_factor
        
        # 2. IAM
        b0 = torch.sigmoid(self.b0_raw) * 0.2
        IAM = 1.0 - b0 * (1.0 - tilt_factor)
        IAM = torch.clamp(IAM, 0.0, 1.0)
        G_eff = G_poa * IAM
        
        # 3. DC Power
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta_module = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw)
        P_stc = F.softplus(self.Pstc_raw)
        
        T_cell = Ta + G_eff / (U0 + U1 * WS + self.eps)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = F.relu(eta_module * P_stc * (G_eff / self.G_stc) * temp_factor)
        
        # 4. Inverter
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

class MMPISSM_Model_Relational(nn.Module):
    """
    Relational Mamba Backbone.
    Input: S7_Features (Concatenated with) (Alpha * dS_Features)
    """
    def __init__(self, configs, enc_in):
        super(MMPISSM_Model_Relational, self).__init__()
        self.configs = configs
        self.enc_in = enc_in
        
        self.lin1 = nn.Linear(self.enc_in, self.configs.seq_len) # Projection from Input Feats -> SeqLen dim? 
        # Wait, standard MMPISSM projects Time Dimension. 
        # RevIN input is (B, L, D).
        # MMPISSM_Model_01 linear layers are:
        # lin1: 2 * (seq+pred) -> seq
        # It permutes (B, L, D) -> (B, D, L).
        
        # We will wrap standard Mamba methodology
        self.lin1 = nn.Linear(2 * (self.configs.seq_len + self.configs.pred_len), self.configs.seq_len)
        self.decompsition = v02_module.series_decomp(self.configs.kernel_size)
        self.revin_layer = v02_module.RevIN(self.enc_in)
        self.revin_layer_enc = v02_module.RevIN(self.enc_in)
        self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)
        self.lin3 = nn.Linear(4 * self.configs.n_embed, self.configs.pred_len)
        self.dropout1 = nn.Dropout(self.configs.dropout)
        self.dropout2 = nn.Dropout(self.configs.dropout)

        try: 
            from mamba_ssm import Mamba
            self.mamba1 = Mamba(d_model=self.configs.n_embed, d_state=self.configs.d_state, d_conv=self.configs.dconv, expand=self.configs.e_fact)
            self.mamba2 = Mamba(d_model=self.configs.enc_in, d_state=self.configs.d_state, d_conv=self.configs.dconv, expand=self.configs.e_fact)
        except:
            self.mamba1 = nn.Linear(self.configs.n_embed, self.configs.n_embed)
            self.mamba2 = nn.Linear(self.configs.enc_in, self.configs.enc_in)
            
    def forward(self, x):
        # x shape: (B, L+H, D_combined) 
        # We assume x is already composed of [S7, Gated_Spatial] and padded for future
        
        # 1. RevIN
        x = self.revin_layer_enc(x, 'norm')
        
        # 2. Decomp
        seasonal_init, trend_init = self.decompsition(x)
        x = torch.cat([seasonal_init, trend_init], dim=1) 
        
        # 3. Mixing
        x = self.lin1(x.permute(0, 2, 1)).permute(0, 2, 1) # (B, L, D) -> (B, D, L) ???
        # Wait, the original code:
        # x = torch.cat([seasonal_init, trend_init], dim=1) # (B, 2L, D) (Actually L+H padded?)
        # lin1 input is size 2*(L+H)? 
        # In original MMPISSM_Model_01:
        # x_pred is (B, L+H, D)
        # x = torch.cat([seasonal_init, trend_init], dim=1) -> (B, 2*(L+H), D) ? No dim=1 is time? 
        # The decomp returns (B, Length, D).
        # cat dim=1 -> (B, 2*Length, D).
        # permute(0, 2, 1) -> (B, D, 2*Length).
        # lin1( ... ) -> (B, D, seq_len). 
        
        x = self.revin_layer_enc(x, 'denorm') # Why denorm here? (From Original Code)
        x = self.revin_layer(x, 'norm') # Re-norm?
        
        seasonal_init, trend_init = self.decompsition(x)
        x_e = torch.cat([seasonal_init, trend_init], dim=1) # (B, 2*L, D)
        x_e = self.lin2(x_e.permute(0, 2, 1)) # (B, D, 2L) -> (B, D, n_embed)
        
        # 4. Mamba
        x_m = self.mamba1(self.dropout1(x_e))
        x_im = self.mamba2(self.dropout2(x_e).permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x_im, x_m, x_m + x_im, x_e], dim=2)
        
        # 5. Projection
        x = self.lin3(x).permute(0, 2, 1) # (B, H, D)
        
        x = self.revin_layer(x, 'denorm')
        
        # We only care about the target column (index 0 usually, or specified)
        # Assuming the first channel is the target K_PV
        x = x[:, :, 0:1] # (B, H, 1)
        return x

class PhysicsResidualMamba_V04(nn.Module):
    def __init__(self, configs):
        super(PhysicsResidualMamba_V04, self).__init__()
        self.configs = configs
        
        # 1. Physics Branch (Local S7 V4)
        cols = configs.S7_COLS
        idx_G = cols.index('nwp_globalirrad') if 'nwp_globalirrad' in cols else 0
        idx_Ta = cols.index('nwp_temperature') if 'nwp_temperature' in cols else 0
        idx_WS = cols.index('nwp_windspeed') if 'nwp_windspeed' in cols else 0
        idx_time = [cols.index(c) for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'] if c in cols]
        
        self.physics_layer = DifferentiablePVLayer_V4(idx_G, idx_Ta, idx_WS, idx_time, T_ref=configs.T_ref)
        
        # 2. Wind Gate
        # Heuristic: Find wind cols for gating
        self.gate_indices = [cols.index(c) for c in ['nwp_windspeed', 'nwp_winddirection_sin', 'nwp_winddirection_cos'] if c in cols]
        if not self.gate_indices:
             # Fallback if direction not present, just speed
             self.gate_indices = [idx_WS]
        
        self.gate = WindGatingModule(input_dim=len(self.gate_indices))
        
        # 3. Mamba
        # Input Dim = S7_Features + Delta_Features
        # S7 features count:
        self.n_s7 = len(configs.S7_COLS)
        self.n_delta = len(configs.DELTA_COLS)
        self.enc_in = self.n_s7 + self.n_delta
        
        self.mamba_model = MMPISSM_Model_Relational(configs, enc_in=self.enc_in)
        
    def forward(self, x_past_s7, x_past_delta, x_fut_s7, x_fut_delta, x_clr):
        # x_past_s7: (B, L, D_s7)
        # x_past_delta: (B, L, D_delta)
        # x_fut_s7: (B, H, D_s7_nwp)
        
        # 1. Physics (Local)
        p_physics_k = self.physics_layer(x_fut_s7, x_clr=x_clr)
        
        # 2. Wind Gating
        # Gate operates on History + Future?
        # Mamba needs combined history (L) and future (H)
        # We need to construct the full sequence of gating signals?
        # Or just apply to history? 
        # Ideally we apply to everything.
        
        # Combine Past/Fut for simple processing
        # Note: x_fut_s7 usually has fewer features (no lagged targets), we need to handle alignment.
        # Simple Approach: Apply Gate only where we use Delta features.
        # We have x_past_delta (calculated from observed K_PV). We do NOT have x_fut_delta (unknown future difference).
        # However, Mamba inputs often pad future with 0 or NWP.
        # Logic: Mamba inputs = [Past(S7+Delta), Future(S7+ZeroDelta?)]
        
        # Gate calculation
        # Extract wind features from x_past_s7
        wind_past = x_past_s7[:, :, self.gate_indices]
        alpha_past = self.gate(wind_past) # (B, L, 1)
        
        # Apply Gate to Delta
        gated_delta_past = x_past_delta * alpha_past
        
        # Combine for Mamba
        # Prepare Future Placeholder for Delta (Zero assumption for residual input?)
        # Or use Alpha_Future * 0 = 0.
        B, L, _ = x_past_s7.shape
        H = x_fut_s7.shape[1]
        
        # Construct Full S7 Sequence (Past + Future Padded)
        # Note: MMPISSM expects fixed size inputs usually.
        # We'll align dimensions. S7 Future usually missing some cols (pwr).
        # We zero pad missing cols in future for S7 to match enc_in part?
        # Actually, let's just feed the Past into Mamba, and let Mamba recurse or use the custom forward.
        # My MMPISSM_Model_Relational forward expects (B, L+H, D).
        
        # Pad S7 Future
        x_fut_s7_padded = torch.zeros(B, H, self.n_s7, device=x_past_s7.device)
        # Map available NWP cols
        # Assumes S7_COLS order is maintained in Future and Past
        # This mapping is complex. Simplified:
        # Padded with zeros is standard for missing future vars.
        # We fill what we have.
        common_indices_s7 = [i for i, c in enumerate(self.configs.S7_COLS) if c in self.configs.S7_FUT_COLS]
        fut_indices_in_s7_fut = [self.configs.S7_FUT_COLS.index(self.configs.S7_COLS[i]) for i in common_indices_s7]
        
        x_fut_s7_padded[:, :, common_indices_s7] = x_fut_s7[:, :, fut_indices_in_s7_fut]
        # Copy last known values for others? Or zero. Zero is safer for "missing info".
        
        x_s7_full = torch.cat([x_past_s7, x_fut_s7_padded], dim=1) # (B, L+H, D_s7)
        
        # Pad Delta Future (Zero - we don't know future delta)
        x_fut_delta = torch.zeros(B, H, self.n_delta, device=x_past_s7.device)
        x_delta_full = torch.cat([gated_delta_past, x_fut_delta], dim=1) # (B, L+H, D_delta)
        
        # Concatenate
        x_mamba_in = torch.cat([x_s7_full, x_delta_full], dim=2) # (B, L+H, Enc_In)
        
        # Mamba Infer
        mamba_k = self.mamba_model(x_mamba_in) # (B, H, 1)
        
        # 3. Fusion
        k_total = p_physics_k + mamba_k
        
        # 4. Recon
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k, alpha_past.mean()

# =============================================================================
# Part 2: Data & Benchmark
# =============================================================================

class RelationalConfig:
    def __init__(self, s7_cols, delta_cols, s7_fut_cols):
        self.S7_COLS = s7_cols
        self.DELTA_COLS = delta_cols
        self.S7_FUT_COLS = s7_fut_cols
        
        self.seq_len = 96
        self.pred_len = 96
        self.kernel_size = 25
        self.n_embed = 128
        self.d_state = 64
        self.dconv = 2
        self.e_fact = 2
        self.dropout = 0.2
        self.T_ref = 25.0
        self.epochs = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TARGET_COL = ['K_PV']

class V04Dataset(Dataset):
    def __init__(self, X_s7, X_delta, Y, P_clr, seq, pred, Fut_s7):
        self.X_s7 = X_s7; self.X_delta = X_delta; self.Y = Y; self.P_clr = P_clr
        self.Fut_s7 = Fut_s7; self.seq = seq; self.pred = pred
    def __len__(self): return len(self.X_s7) - self.seq - self.pred + 1
    def __getitem__(self, i):
        start_fut = i + self.seq
        end_fut = start_fut + self.pred
        
        # Inputs
        x_past_s7 = self.X_s7[i : i+self.seq]
        x_past_delta = self.X_delta[i : i+self.seq]
        
        # Future Inputs (NWP) - Needed for Physics Loop
        # We need the Future S7 features (NWP) for the prediction horizon
        idx_fut_start = start_fut
        idx_fut_end = end_fut
        # Check boundary? (Handled by len)
        x_fut_s7 = self.Fut_s7[idx_fut_start : idx_fut_end]
        
        y_fut = self.Y[start_fut : end_fut]
        x_clr = self.P_clr[start_fut : end_fut]
        if x_clr.ndim==1: x_clr=x_clr[:,None]
        if y_fut.ndim==1: y_fut=y_fut[:,None]

        return (x_past_s7.astype(np.float32), 
                x_past_delta.astype(np.float32),
                x_fut_s7.astype(np.float32),
                y_fut.astype(np.float32), 
                x_clr.astype(np.float32))

def prepare_relational_fold(df_merged, target_cols, neighbor_cols, fut_cols, n_splits=4, seq_len=96, pred_len=96):
    """
    Computes S7 Features and Spatial Delta Features (dS).
    S7 Cols: target_cols
    Delta Cols: dS, lagged dS.
    """
    print("Preparing Relational Features (V04)...")
    df = df_merged.copy().fillna(0)
    
    # 1. Compute Spatial Delta
    if 'K_PV_s8' in df.columns and 'K_PV' in df.columns:
        df['dS'] = df['K_PV'] - df['K_PV_s8']
    else:
        print("Warning: K_PV or K_PV_s8 missing. dS set to 0.")
        df['dS'] = 0.0
        
    # 2. Delta Features (Lags?) - Mamba handles temporal sequence, so just dS at time t is enough.
    # The sequence of dS constitutes the "Lagged Delta".
    delta_cols = ['dS'] 
    
    # 3. S7 Features
    # Use target_cols provided.
    
    # 4. S7 Future Features (NWP)
    # Filter fut_cols to only those in target_cols (S7 only)
    # Assumption: neighbor columns in df have '_s8' suffix, target don't.
    s7_fut_cols = [c for c in fut_cols if '_s8' not in c]
    
    # Arrays
    X_s7 = df[target_cols].values.astype(np.float32)
    X_delta = df[delta_cols].values.astype(np.float32)
    Y = df['K_PV'].values.astype(np.float32)
    P_clr = df['P_CLR'].values.astype(np.float32) if 'P_CLR' in df.columns else np.zeros((len(df),1))
    
    # Future Array (NWP for S7)
    # We need a parallel array for Future S7 features matching the full timeline
    X_fut_s7_full = df[s7_fut_cols].values.astype(np.float32)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X_s7):
         # Dates
        dates = {
            'train_start': df.index[train_idx[0]], 'train_end': df.index[train_idx[-1]],
            'test_start': df.index[test_idx[0]], 'test_end': df.index[test_idx[-1]]
        }
        
        # Split
        def split_arr(arr): return arr[train_idx], arr[test_idx]
        
        tr_s7, te_s7 = split_arr(X_s7)
        tr_d, te_d = split_arr(X_delta)
        tr_y, te_y = split_arr(Y)
        tr_p, te_p = split_arr(P_clr)
        tr_f, te_f = split_arr(X_fut_s7_full)
        
        # Concat Test Lookback
        def concat_lookback(train, test): return np.concatenate([train[-seq_len:], test], axis=0)
        
        te_s7_c = concat_lookback(tr_s7, te_s7)
        te_d_c = concat_lookback(tr_d, te_d)
        te_y_c = concat_lookback(tr_y, te_y)
        te_p_c = concat_lookback(tr_p, te_p)
        te_f_c = concat_lookback(tr_f, te_f)
        
        train_ds = V04Dataset(tr_s7, tr_d, tr_y, tr_p, seq_len, pred_len, tr_f)
        test_ds = V04Dataset(te_s7_c, te_d_c, te_y_c, te_p_c, seq_len, pred_len, te_f_c)
        
        yield (DataLoader(train_ds, batch_size=64, shuffle=True),
               DataLoader(test_ds, batch_size=64, shuffle=False),
               {}, dates, delta_cols, s7_fut_cols)

def train_one_epoch_v04(model, loader, optimizer, device, warmup=False):
    model.train()
    # Lock-down Strategy
    if warmup:
        # Train Physics, Freeze Mamba & Gate
        for p in model.mamba_model.parameters(): p.requires_grad = False
        for p in model.gate.parameters(): p.requires_grad = False
        for p in model.physics_layer.parameters(): p.requires_grad = True
    else:
        # Train All
        for p in model.mamba_model.parameters(): p.requires_grad = True
        for p in model.gate.parameters(): p.requires_grad = True
        for p in model.physics_layer.parameters(): p.requires_grad = True
        
    total_loss = 0.0
    total_alpha = 0.0
    
    for batch in loader:
        x_past_s7 = batch[0].to(device)
        x_past_delta = batch[1].to(device)
        x_fut_s7 = batch[2].to(device)
        y = batch[3].to(device)
        x_clr = batch[4].to(device)
        
        optimizer.zero_grad()
        
        p_pred, _, alpha_mean = model(x_past_s7, x_past_delta, x_fut_s7, x_past_delta, x_clr)
        
        # Loss (Target = K_PV * P_CLR)
        p_target = y * (x_clr + 1e-6)
        loss = F.mse_loss(p_pred, p_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_alpha += alpha_mean.item()
        
    return total_loss / len(loader), total_alpha / len(loader)

def evaluate_v04(model, loader, device):
    model.eval()
    preds, acts = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x_past_s7 = batch[0].to(device)
            x_past_delta = batch[1].to(device)
            x_fut_s7 = batch[2].to(device)
            y = batch[3].to(device)
            x_clr = batch[4].to(device)
            
            p_pred, _, _ = model(x_past_s7, x_past_delta, x_fut_s7, x_past_delta, x_clr)
            p_target = y * (x_clr + 1e-6)
            
            loss = F.mse_loss(p_pred, p_target)
            total_loss += loss.item()
            preds.append(p_pred.cpu().numpy())
            acts.append(p_target.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0).flatten()
    acts = np.concatenate(acts, axis=0).flatten()
    mse = np.mean((preds-acts)**2)
    rmse = np.sqrt(mse)
    nrmse = (rmse/20.0)*100
    return {'rmse': rmse, 'nrmse': nrmse, 'mse_loss': total_loss/len(loader)}, preds, acts

def run_relational_benchmark(df_merged, target_cols, neighbor_cols, n_splits=4, custom_config=None):
    """
    Run V04 Relational Benchmark.
    """
    print(">>> Running V04 Relational Benchmark (Wind-Gated Spatiotemporal)")
    
    # 1. Determine Columns
    # Assuming 'nwp_windspeed' etc are in target_cols
    # Future cols: intersection of nwp in target
    future_native = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'NWP_Power_MW', 'hour_sin', 'hour_cos']
    fut_cols_all = [c for c in df_merged.columns if any(x in c for x in future_native)] # Crude
    
    # 2. Config Placeholder
    config = RelationalConfig(target_cols, ['dS'], []) # Updated inside loop
    
    # 2b. User Overrides (Columns & Hyperparams)
    if custom_config:
        print("Applying User Configuration Overrides...")
        if hasattr(custom_config, 'epochs'): config.epochs = custom_config.epochs
        if hasattr(custom_config, 'warmup_epochs'): config.warmup_epochs = custom_config.warmup_epochs
        if hasattr(custom_config, 'device'): config.device = custom_config.device
        
        # Column Overrides
        if hasattr(custom_config, 'PAST_INPUT_COLS'):
            print(f"Overriding Target Columns with {len(custom_config.PAST_INPUT_COLS)} features")
            target_cols = custom_config.PAST_INPUT_COLS
            config.S7_COLS = target_cols 
            
        if hasattr(custom_config, 'FUTURE_INPUT_COLS'):
            print(f"Overriding Future Columns with {len(custom_config.FUTURE_INPUT_COLS)} features")
            fut_cols_all = custom_config.FUTURE_INPUT_COLS
    
    # 3. Data Prep
    fold_gen = prepare_relational_fold(
        df_merged, target_cols, neighbor_cols, fut_cols_all, 
        n_splits=n_splits, seq_len=config.seq_len, pred_len=config.pred_len
    )
    
    metrics = {'rmse': [], 'nrmse': [], 'alpha': []}
    
    for fold_i, (train_loader, test_loader, _, dates, delta_cols, s7_fut_cols) in enumerate(fold_gen):
        print(f"\n=== Fold {fold_i+1} V04 Relational ===")
        # Update config with dynamic cols
        config.DELTA_COLS = delta_cols
        config.S7_FUT_COLS = s7_fut_cols
        
        # Init Model
        model = PhysicsResidualMamba_V04(config).to(config.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        epochs = config.epochs
        warmup = getattr(config, 'warmup_epochs', 5)
        
        print(f"Schedule: {warmup} Warmup + {epochs-warmup} Joint")
        t0 = time.time()
        
        for ep in range(epochs):
            is_warmup = ep < warmup
            loss, alpha = train_one_epoch_v04(model, train_loader, opt, config.device, warmup=is_warmup)
            if ep % 5 == 0:
                print(f" Ep {ep}: Loss {loss:.4f} | Gate Alpha: {alpha:.3f}")
                
        # Eval
        met, preds, acts = evaluate_v04(model, test_loader, config.device)
        
        metrics['rmse'].append(met['rmse'])
        metrics['nrmse'].append(met['nrmse'])
        metrics['alpha'].append(alpha) # Last alpha
        
        print(f" Fold {fold_i+1}: RMSE={met['rmse']:.3f} | nRMSE={met['nrmse']:.2f}% | Final Alpha={alpha:.3f}")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=acts, name='Actual', line=dict(color='black'), opacity=0.5))
        fig.add_trace(go.Scatter(y=preds, name='V04 Relational', line=dict(color='purple')))
        fig.update_layout(title=f"V04 Relational Fold {fold_i+1}")
        fig.write_html(f"v04_relational_fold_{fold_i+1}.html")
        
    print(f"\nAvg RMSE: {np.mean(metrics['rmse']):.3f} MW")
    return metrics
