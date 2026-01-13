"""
Spatiotemporal Physics-Residual Mamba V04 (Robust)
"The Robust Model"

Addresses:
1. Negative Spatial Gain -> via Upwind Residuals (Wind Alignment).
2. Overshooting -> via Inverter Saturation Cap.
3. Instability -> via NWP Anchoring & Smoothness Penalty.

Enhanced: Dual-Model Comparison, Loss Curves, Metric Plots.

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
from plotly.subplots import make_subplots
import time
import physics_residual_mamba_v02 as v02_module # Reuse base classes

# =============================================================================
# Part 1: Geometric & Feature Utils
# =============================================================================

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing from Point 1 to Point 2.
    Formula: Forward Azimuth.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - \
        np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    
    bearing_rad = np.arctan2(y, x)
    bearing_deg = np.degrees(bearing_rad)
    return (bearing_deg + 360) % 360

# =============================================================================
# Part 2: Custom Layers
# =============================================================================

class InverterCapLayer(nn.Module):
    """
    Learnable Tanh-based Cap to prevent Overshooting.
    P_final = P_cap * Tanh(Input_Scaled)
    """
    def __init__(self, capacity_mw=20.0):
        super(InverterCapLayer, self).__init__()
        self.cap_raw = nn.Parameter(torch.tensor(float(capacity_mw)))
        self.scale_raw = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        cap = F.softplus(self.cap_raw)
        scale = F.softplus(self.scale_raw)
        return cap * torch.tanh(x * scale / cap)

class SmoothnessLoss(nn.Module):
    """Total Variation Penalty on Time Dimension."""
    def __init__(self, weight=0.1):
        super(SmoothnessLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred):
        if pred.shape[1] > 1:
            diff = torch.abs(pred[:, 1:] - pred[:, :-1])
            return self.weight * diff.mean()
        return torch.tensor(0.0, device=pred.device)

# =============================================================================
# Part 3: Architecture
# =============================================================================

class CleanMambaBlock(nn.Module):
    """Minimal wrapper for Mamba SSM"""
    def __init__(self, d_model, seq_len, pred_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        except ImportError:
            print("Warning: Mamba not found, using Linear fallback.")
            self.mamba = nn.Linear(d_model, d_model)
        
        self.head = nn.Linear(d_model * seq_len, pred_len)
        self.seq_len = seq_len
        self.activation = nn.SiLU()
        
    def forward(self, x):
        B, L, D = x.shape
        x = self.norm(x)
        x = self.mamba(x)
        x = self.activation(x)
        x_flat = x.reshape(B, -1)
        out = self.head(x_flat)
        return out.unsqueeze(-1)

class DifferentiablePVLayer_V4(nn.Module):
    """V4: Robust (IAM + Non-Linear Inverter)"""
    def __init__(self, idx_G, idx_Ta, idx_WS, idx_time_feats, T_ref=25.0, G_stc=1000.0, P_stc_init=19.0):
        super().__init__()
        self.idx_G = idx_G
        self.idx_Ta = idx_Ta 
        self.idx_WS = idx_WS
        self.idx_time_feats = idx_time_feats
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6
        
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4)) 
        self.U0_raw = nn.Parameter(torch.tensor(3.2)) 
        self.U1_raw = nn.Parameter(torch.tensor(1.9)) 
        self.eta_raw = nn.Parameter(torch.tensor(-1.66)) 
        self.Pstc_raw = nn.Parameter(torch.tensor(float(P_stc_init))) 
        
        self.b0_raw = nn.Parameter(torch.tensor(0.05))
        self.inv_k_raw = nn.Parameter(torch.tensor(1.0))
        self.inv_thresh_raw = nn.Parameter(torch.tensor(0.1)) 

        self.geo_net = nn.Sequential(
            nn.Linear(len(idx_time_feats), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x_future, x_clr=None):
        def get(idx, default_val=0.0):
            if idx is not None and idx < x_future.shape[2]:
                return x_future[:, :, idx]
            return torch.zeros_like(x_future[:, :, 0]) + default_val
            
        G = get(self.idx_G)
        Ta = get(self.idx_Ta, 25.0)
        WS = get(self.idx_WS, 1.0)
        
        t_list = []
        for i in self.idx_time_feats:
            if i < x_future.shape[2]: t_list.append(x_future[:, :, i:i+1])
            else: t_list.append(torch.zeros_like(G).unsqueeze(-1))
        if t_list: time_feats = torch.cat(t_list, dim=-1)
        else: time_feats = torch.zeros(G.shape[0], G.shape[1], 1).to(G.device)
        
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
        P_stc = F.softplus(self.Pstc_raw)
        
        T_cell = Ta + G_eff / (U0 + U1 * WS + self.eps)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = F.relu(eta_module * P_stc * (G_eff / self.G_stc) * temp_factor)
        
        inv_k = F.softplus(self.inv_k_raw) * 10.0
        inv_thresh = F.softplus(self.inv_thresh_raw)
        eta_inv = torch.sigmoid(inv_k * (P_dc - inv_thresh))
        P_ac = P_dc * eta_inv
        
        P_final_mw = P_ac.unsqueeze(-1)
        
        if x_clr is not None:
             if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
             K_phys = P_final_mw / (x_clr + self.eps)
             return K_phys * (x_clr > 0.01).float()
        return P_final_mw

class PhysicsResidualMamba_V04_Robust(nn.Module):
    def __init__(self, configs, meta_s7):
        super(PhysicsResidualMamba_V04_Robust, self).__init__()
        self.configs = configs
        
        cols = configs.S7_COLS
        idx_G = cols.index('nwp_globalirrad') if 'nwp_globalirrad' in cols else 0
        idx_Ta = cols.index('nwp_temperature') if 'nwp_temperature' in cols else 0
        idx_WS = cols.index('nwp_windspeed') if 'nwp_windspeed' in cols else 0
        idx_time = [cols.index(c) for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'] if c in cols]
        
        self.physics_layer = DifferentiablePVLayer_V4(idx_G, idx_Ta, idx_WS, idx_time, T_ref=configs.T_ref)
        
        cap_mw = meta_s7.get('Capacity', 20000) / 1000.0
        self.inv_cap = InverterCapLayer(capacity_mw=cap_mw)
        
        self.enc_in = len(configs.S7_COLS) + len(configs.INTERACTION_COLS)
        self.mamba = CleanMambaBlock(d_model=self.enc_in, seq_len=configs.seq_len, pred_len=configs.pred_len)
        
    def forward(self, x_past, x_fut_nwp_k, x_clr):
        delta_k = self.mamba(x_past)
        k_total = x_fut_nwp_k + delta_k
        
        if x_clr.ndim == 1: x_clr = x_clr.unsqueeze(-1)
        p_unbounded = k_total * x_clr
        p_final = self.inv_cap(p_unbounded)
        
        return p_final, delta_k 

# =============================================================================
# Part 4: Data Prep
# =============================================================================

class RobustConfig:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.epochs = 30
        self.lr = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.S7_COLS = []
        self.INTERACTION_COLS = []
        self.T_ref = 25.0
        
        # V02 Compatibility Attributes
        self.PAST_INPUT_COLS = []
        self.FUTURE_INPUT_COLS = []
        self.common_features = []
        self.num_shadows = 0
        self.enc_in = 0
        self.c_out = 1
        
        # Model hyperparameters for V02 models
        self.kernel_size = 25
        self.n_embed = 64
        self.d_state = 64
        self.dconv = 2
        self.e_fact = 2
        self.dropout = 0.2
    
    def update_for_v02(self, past_cols, future_cols):
        """Update config with V02-compatible settings."""
        self.PAST_INPUT_COLS = past_cols
        self.FUTURE_INPUT_COLS = future_cols
        self.common_features = [f for f in future_cols if f in past_cols]
        self.num_shadows = len(self.common_features)
        self.enc_in = len(past_cols) + self.num_shadows
        self.TARGET_COL = ['K_PV']

class RobustDataset(Dataset):
    def __init__(self, X, X_nwp_k, Y, P_clr, seq, pred):
        self.X=X; self.X_nwp_k=X_nwp_k; self.Y=Y; self.P_clr=P_clr
        self.seq=seq; self.pred=pred
    def __len__(self): return len(self.X) - self.seq - self.pred + 1
    def __getitem__(self, i):
        start_fut = i + self.seq
        end_fut = start_fut + self.pred
        
        x_past = self.X[i:i+self.seq]
        x_fut_k = self.X_nwp_k[start_fut:end_fut]
        y_fut = self.Y[start_fut:end_fut]
        x_clr = self.P_clr[start_fut:end_fut]
        
        if x_clr.ndim==1: x_clr=x_clr[:,None]
        if y_fut.ndim==1: y_fut=y_fut[:,None]
        if x_fut_k.ndim==1: x_fut_k=x_fut_k[:,None]
        
        return (x_past.astype(np.float32), 
                x_fut_k.astype(np.float32), 
                y_fut.astype(np.float32),
                x_clr.astype(np.float32))

def prepare_robust_fold(df_merged, meta_s7, meta_s8, target_cols, neighbor_cols, n_splits=4, seq_len=96, pred_len=96, include_interaction=True):
    """Prepare data folds. If include_interaction=False, returns single-location data."""
    print(f"Preparing Features (Spatial={include_interaction})...")
    
    lat7, lon7 = meta_s7['Latitude'], meta_s7['Longitude']
    lat8, lon8 = meta_s8['Latitude'], meta_s8['Longitude']
    bearing = calculate_bearing(lat8, lon8, lat7, lon7)
    if include_interaction: print(f"Bearing S8->S7: {bearing:.1f} deg")
    
    df = df_merged.copy().fillna(0)
    
    # K_NWP for S8
    if 'NWP_Power_MW_s8' in df.columns and 'P_CLR_s8' in df.columns:
        df['K_NWP_s8'] = df['NWP_Power_MW_s8'] / (df['P_CLR_s8'] + 1e-6)
    else:
        df['K_NWP_s8'] = 0 
        
    # Delta E at S8
    if 'K_PV_s8' in df.columns:
        df['DeltaE_s8'] = df['K_PV_s8'] - df['K_NWP_s8']
    else:
        df['DeltaE_s8'] = 0
        
    # Wind Alignment
    if 'nwp_winddirection' in df.columns:
        wd = np.radians(df['nwp_winddirection'])
        br = np.radians(bearing)
        df['WindAlign'] = np.cos(wd - br).clip(lower=0.0)
    else:
        df['WindAlign'] = 0
        
    # Interaction Feature
    df['Interaction'] = df['DeltaE_s8'] * df['WindAlign']
    
    # K_NWP for S7
    if 'NWP_Power_MW' in df.columns and 'P_CLR' in df.columns:
        df['K_NWP'] = df['NWP_Power_MW'] / (df['P_CLR'] + 1e-6)
    else:
        df['K_NWP'] = 0
    
    # Select Columns
    if include_interaction:
        interaction_cols = ['Interaction']
        final_cols = target_cols + interaction_cols
    else:
        interaction_cols = []
        final_cols = target_cols
    
    X = df[final_cols].values.astype(np.float32)
    X_nwp_k = df['K_NWP'].values.astype(np.float32)
    Y = df['K_PV'].values.astype(np.float32)
    P_clr = df['P_CLR'].values.astype(np.float32)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X):
        dates = {
            'train_start': df.index[train_idx[0]], 'train_end': df.index[train_idx[-1]],
            'test_start': df.index[test_idx[0]], 'test_end': df.index[test_idx[-1]]
        }
        
        def split(arr): return arr[train_idx], arr[test_idx]
        def concat_lookback(tr, te): return np.concatenate([tr[-seq_len:], te], axis=0)
        
        tr_X, te_X = split(X)
        tr_K, te_K = split(X_nwp_k)
        tr_Y, te_Y = split(Y)
        tr_P, te_P = split(P_clr)
        
        te_X_c = concat_lookback(tr_X, te_X)
        te_K_c = concat_lookback(tr_K, te_K)
        te_Y_c = concat_lookback(tr_Y, te_Y)
        te_P_c = concat_lookback(tr_P, te_P)
        
        train_ds = RobustDataset(tr_X, tr_K, tr_Y, tr_P, seq_len, pred_len)
        test_ds = RobustDataset(te_X_c, te_K_c, te_Y_c, te_P_c, seq_len, pred_len)
        
        yield (DataLoader(train_ds, batch_size=64, shuffle=True),
               DataLoader(test_ds, batch_size=64, shuffle=False),
               dates, interaction_cols)

# =============================================================================
# Part 5: Training & Evaluation
# =============================================================================

def train_one_epoch_robust(model, loader, optimizer, device, loss_fn_smooth):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        x_past = batch[0].to(device)
        x_fut_k = batch[1].to(device)
        y = batch[2].to(device)
        x_clr = batch[3].to(device)
        
        optimizer.zero_grad()
        
        p_pred, delta_k = model(x_past, x_fut_k, x_clr)
        p_target = y * (x_clr + 1e-6)
        
        loss_mse = F.mse_loss(p_pred, p_target)
        loss_smooth = loss_fn_smooth(p_pred)
        
        loss = loss_mse + loss_smooth
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate_robust(model, loader, device):
    """Returns RMSE, nRMSE, MAE + predictions."""
    model.eval()
    preds, acts = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            x_fut_k = batch[1].to(device)
            y = batch[2].to(device)
            x_clr = batch[3].to(device)
            
            p_pred, _ = model(x_past, x_fut_k, x_clr)
            p_target = y * (x_clr + 1e-6)
            
            total_loss += F.mse_loss(p_pred, p_target).item()
            preds.append(p_pred.cpu().numpy())
            acts.append(p_target.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0).flatten()
    acts = np.concatenate(acts, axis=0).flatten()
    rmse = np.sqrt(np.mean((preds-acts)**2))
    mae = np.mean(np.abs(preds-acts))
    nrmse = (rmse/20.0)*100
    val_loss = total_loss / len(loader)
    return {'rmse': rmse, 'nrmse': nrmse, 'mae': mae, 'val_loss': val_loss}, preds, acts

def train_model_with_curves(model, train_loader, test_loader, optimizer, device, epochs, loss_fn_smooth):
    """Train and return epoch-level loss curves."""
    train_losses = []
    val_losses = []
    
    for ep in range(epochs):
        train_loss = train_one_epoch_robust(model, train_loader, optimizer, device, loss_fn_smooth)
        met, _, _ = evaluate_robust(model, test_loader, device)
        val_loss = met['val_loss']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if ep % 10 == 0:
            print(f"  Ep {ep}: Train={train_loss:.4f} Val={val_loss:.4f}")
            
    return train_losses, val_losses

# =============================================================================
# Part 6: Comparison Benchmark Runner
# =============================================================================

def run_robust_benchmark(df_merged, meta_s7, meta_s8, target_cols, neighbor_cols, n_splits=4, custom_config=None):
    """
    Enhanced Benchmark: Compares Single-Location vs Spatial models.
    Generates loss curves and metric bar charts.
    """
    print(">>> Running V04 Robust Benchmark (Single vs Spatial Comparison)")
    
    # Config
    config = RobustConfig()
    if custom_config:
        if hasattr(custom_config, 'epochs'): config.epochs = custom_config.epochs
        if hasattr(custom_config, 'PAST_INPUT_COLS'): target_cols = custom_config.PAST_INPUT_COLS
    
    config.S7_COLS = target_cols
    
    # Results Storage
    results = {
        'single': {'rmse': [], 'nrmse': [], 'mae': [], 'train_loss': [], 'val_loss': []},
        'spatial': {'rmse': [], 'nrmse': [], 'mae': [], 'train_loss': [], 'val_loss': []}
    }
    
    # Prepare folds for BOTH models
    fold_gen_single = list(prepare_robust_fold(df_merged, meta_s7, meta_s8, target_cols, neighbor_cols, n_splits=n_splits, include_interaction=False))
    fold_gen_spatial = list(prepare_robust_fold(df_merged, meta_s7, meta_s8, target_cols, neighbor_cols, n_splits=n_splits, include_interaction=True))
    
    all_preds = {'single': [], 'spatial': []}
    all_acts = []
    
    for fold_i in range(n_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_i+1}/{n_splits}")
        print(f"{'='*60}")
        
        # === SINGLE-LOCATION MODEL ===
        train_loader_s, test_loader_s, dates_s, int_cols_s = fold_gen_single[fold_i]
        config.INTERACTION_COLS = int_cols_s  # Empty list
        
        print(f"\n[Single-Location] Test: {dates_s['test_start']} -> {dates_s['test_end']}")
        model_single = PhysicsResidualMamba_V04_Robust(config, meta_s7).to(config.device)
        opt_single = torch.optim.Adam(model_single.parameters(), lr=1e-3)
        loss_smooth = SmoothnessLoss(weight=0.1)
        
        train_losses_s, val_losses_s = train_model_with_curves(
            model_single, train_loader_s, test_loader_s, opt_single, config.device, config.epochs, loss_smooth
        )
        met_s, preds_s, acts_s = evaluate_robust(model_single, test_loader_s, config.device)
        print(f"  Result: RMSE={met_s['rmse']:.3f} | nRMSE={met_s['nrmse']:.2f}% | MAE={met_s['mae']:.3f}")
        
        results['single']['rmse'].append(met_s['rmse'])
        results['single']['nrmse'].append(met_s['nrmse'])
        results['single']['mae'].append(met_s['mae'])
        results['single']['train_loss'].append(train_losses_s)
        results['single']['val_loss'].append(val_losses_s)
        all_preds['single'].append(preds_s)
        
        # === SPATIAL MODEL ===
        train_loader_sp, test_loader_sp, dates_sp, int_cols_sp = fold_gen_spatial[fold_i]
        config.INTERACTION_COLS = int_cols_sp  # ['Interaction']
        
        print(f"\n[Spatial] Test: {dates_sp['test_start']} -> {dates_sp['test_end']}")
        model_spatial = PhysicsResidualMamba_V04_Robust(config, meta_s7).to(config.device)
        opt_spatial = torch.optim.Adam(model_spatial.parameters(), lr=1e-3)
        
        train_losses_sp, val_losses_sp = train_model_with_curves(
            model_spatial, train_loader_sp, test_loader_sp, opt_spatial, config.device, config.epochs, loss_smooth
        )
        met_sp, preds_sp, acts_sp = evaluate_robust(model_spatial, test_loader_sp, config.device)
        print(f"  Result: RMSE={met_sp['rmse']:.3f} | nRMSE={met_sp['nrmse']:.2f}% | MAE={met_sp['mae']:.3f}")
        
        results['spatial']['rmse'].append(met_sp['rmse'])
        results['spatial']['nrmse'].append(met_sp['nrmse'])
        results['spatial']['mae'].append(met_sp['mae'])
        results['spatial']['train_loss'].append(train_losses_sp)
        results['spatial']['val_loss'].append(val_losses_sp)
        all_preds['spatial'].append(preds_sp)
        all_acts.append(acts_sp)
    
    # ==========================================================================
    # GENERATE PLOTS
    # ==========================================================================
    
    # 1. Loss Curves (Matplotlib)
    fig, axes = plt.subplots(2, n_splits, figsize=(5*n_splits, 8), sharex=True)
    for fold_i in range(n_splits):
        # Training Loss
        axes[0, fold_i].plot(results['single']['train_loss'][fold_i], label='Single', color='blue')
        axes[0, fold_i].plot(results['spatial']['train_loss'][fold_i], label='Spatial', color='green')
        axes[0, fold_i].set_title(f'Fold {fold_i+1} - Train Loss')
        axes[0, fold_i].legend()
        axes[0, fold_i].grid(True, alpha=0.3)
        
        # Validation Loss
        axes[1, fold_i].plot(results['single']['val_loss'][fold_i], label='Single', color='blue')
        axes[1, fold_i].plot(results['spatial']['val_loss'][fold_i], label='Spatial', color='green')
        axes[1, fold_i].set_title(f'Fold {fold_i+1} - Val Loss')
        axes[1, fold_i].set_xlabel('Epoch')
        axes[1, fold_i].legend()
        axes[1, fold_i].grid(True, alpha=0.3)
        
    axes[0, 0].set_ylabel('Training Loss')
    axes[1, 0].set_ylabel('Validation Loss')
    plt.tight_layout()
    plt.savefig('v04_robust_loss_curves.png', dpi=150)
    print("\nSaved: v04_robust_loss_curves.png")
    
    # 2. Metric Bar Chart (Plotly)
    avg_single = {k: np.mean(v) for k, v in results['single'].items() if k in ['rmse', 'nrmse', 'mae']}
    avg_spatial = {k: np.mean(v) for k, v in results['spatial'].items() if k in ['rmse', 'nrmse', 'mae']}
    
    fig_bar = make_subplots(rows=1, cols=3, subplot_titles=['RMSE (MW)', 'nRMSE (%)', 'MAE (MW)'])
    
    metrics_to_plot = ['rmse', 'nrmse', 'mae']
    for i, m in enumerate(metrics_to_plot):
        fig_bar.add_trace(go.Bar(name='Single', x=['Single'], y=[avg_single[m]], marker_color='blue', showlegend=(i==0)), row=1, col=i+1)
        fig_bar.add_trace(go.Bar(name='Spatial', x=['Spatial'], y=[avg_spatial[m]], marker_color='green', showlegend=(i==0)), row=1, col=i+1)
        
    fig_bar.update_layout(title='V04 Robust: Single vs Spatial Comparison', barmode='group')
    fig_bar.write_html('v04_robust_metric_comparison.html')
    print("Saved: v04_robust_metric_comparison.html")
    
    # 3. Prediction vs Actual (Plotly)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=np.concatenate(all_acts), mode='lines', name='Actual', line=dict(color='black'), opacity=0.5))
    fig_pred.add_trace(go.Scatter(y=np.concatenate(all_preds['single']), mode='lines', name='Single', line=dict(color='blue')))
    fig_pred.add_trace(go.Scatter(y=np.concatenate(all_preds['spatial']), mode='lines', name='Spatial', line=dict(color='green')))
    fig_pred.update_layout(title='V04 Robust: Predictions vs Actual (All Folds)', xaxis_title='Time Step', yaxis_title='Power (MW)')
    fig_pred.write_html('v04_robust_predictions.html')
    print("Saved: v04_robust_predictions.html")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Single-Location: Avg RMSE={avg_single['rmse']:.3f} MW | nRMSE={avg_single['nrmse']:.2f}% | MAE={avg_single['mae']:.3f} MW")
    print(f"Spatial:         Avg RMSE={avg_spatial['rmse']:.3f} MW | nRMSE={avg_spatial['nrmse']:.2f}% | MAE={avg_spatial['mae']:.3f} MW")
    
    spatial_gain = (avg_single['nrmse'] - avg_spatial['nrmse']) / avg_single['nrmse'] * 100
    print(f"\nSpatial Gain: {spatial_gain:.1f}% reduction in nRMSE")
    
    return results

# =============================================================================
# Part 7: Full Multi-Model Comparison Benchmark
# =============================================================================

class FullComparisonDataset(Dataset):
    """Dataset that provides all required tensors for both V02 and V04 models."""
    def __init__(self, X_past, X_future, X_nwp_k, Y, P_clr, seq, pred):
        self.X_past = X_past
        self.X_future = X_future
        self.X_nwp_k = X_nwp_k
        self.Y = Y
        self.P_clr = P_clr
        self.seq = seq
        self.pred = pred
        
    def __len__(self):
        return len(self.X_past) - self.seq - self.pred + 1
    
    def __getitem__(self, i):
        start_fut = i + self.seq
        end_fut = start_fut + self.pred
        
        x_past = self.X_past[i:i+self.seq]
        x_future = self.X_future[start_fut:end_fut]
        x_fut_k = self.X_nwp_k[start_fut:end_fut]
        y_fut = self.Y[start_fut:end_fut]
        x_clr = self.P_clr[start_fut:end_fut]
        
        # Ensure correct shapes
        if x_clr.ndim == 1: x_clr = x_clr[:, None]
        if y_fut.ndim == 1: y_fut = y_fut[:, None]
        if x_fut_k.ndim == 1: x_fut_k = x_fut_k[:, None]
        
        return (x_past.astype(np.float32),
                x_future.astype(np.float32),
                x_fut_k.astype(np.float32),
                y_fut.astype(np.float32),
                x_clr.astype(np.float32))

def prepare_full_comparison_fold(df_merged, meta_s7, meta_s8, target_cols, future_cols, neighbor_cols, n_splits=4, seq_len=96, pred_len=96):
    """Prepare folds with all data needed for both V02 and V04 models."""
    print("Preparing Full Comparison Features...")
    
    lat7, lon7 = meta_s7['Latitude'], meta_s7['Longitude']
    lat8, lon8 = meta_s8['Latitude'], meta_s8['Longitude']
    bearing = calculate_bearing(lat8, lon8, lat7, lon7)
    print(f"Bearing S8->S7: {bearing:.1f} deg")
    
    df = df_merged.copy().fillna(0)
    
    # Compute interaction features
    if 'NWP_Power_MW_s8' in df.columns and 'P_CLR_s8' in df.columns:
        df['K_NWP_s8'] = df['NWP_Power_MW_s8'] / (df['P_CLR_s8'] + 1e-6)
    else:
        df['K_NWP_s8'] = 0
        
    if 'K_PV_s8' in df.columns:
        df['DeltaE_s8'] = df['K_PV_s8'] - df['K_NWP_s8']
    else:
        df['DeltaE_s8'] = 0
        
    if 'nwp_winddirection' in df.columns:
        wd = np.radians(df['nwp_winddirection'])
        br = np.radians(bearing)
        df['WindAlign'] = np.cos(wd - br).clip(lower=0.0)
    else:
        df['WindAlign'] = 0
        
    df['Interaction'] = df['DeltaE_s8'] * df['WindAlign']
    
    if 'NWP_Power_MW' in df.columns and 'P_CLR' in df.columns:
        df['K_NWP'] = df['NWP_Power_MW'] / (df['P_CLR'] + 1e-6)
    else:
        df['K_NWP'] = 0
    
    # Prepare arrays
    spatial_cols = target_cols + ['Interaction']
    
    X_past = df[target_cols].values.astype(np.float32)
    X_past_spatial = df[spatial_cols].values.astype(np.float32)
    X_future = df[future_cols].values.astype(np.float32)
    X_nwp_k = df['K_NWP'].values.astype(np.float32)
    Y = df['K_PV'].values.astype(np.float32)
    P_clr = df['P_CLR'].values.astype(np.float32)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X_past):
        dates = {
            'train_start': df.index[train_idx[0]], 'train_end': df.index[train_idx[-1]],
            'test_start': df.index[test_idx[0]], 'test_end': df.index[test_idx[-1]]
        }
        
        def split(arr): return arr[train_idx], arr[test_idx]
        def concat_lookback(tr, te): return np.concatenate([tr[-seq_len:], te], axis=0)
        
        # Split all arrays
        tr_past, te_past = split(X_past)
        tr_past_sp, te_past_sp = split(X_past_spatial)
        tr_fut, te_fut = split(X_future)
        tr_K, te_K = split(X_nwp_k)
        tr_Y, te_Y = split(Y)
        tr_P, te_P = split(P_clr)
        
        # Concat lookback for test
        te_past_c = concat_lookback(tr_past, te_past)
        te_past_sp_c = concat_lookback(tr_past_sp, te_past_sp)
        te_fut_c = concat_lookback(tr_fut, te_fut)
        te_K_c = concat_lookback(tr_K, te_K)
        te_Y_c = concat_lookback(tr_Y, te_Y)
        te_P_c = concat_lookback(tr_P, te_P)
        
        # Create datasets for single and spatial
        train_ds_single = FullComparisonDataset(tr_past, tr_fut, tr_K, tr_Y, tr_P, seq_len, pred_len)
        test_ds_single = FullComparisonDataset(te_past_c, te_fut_c, te_K_c, te_Y_c, te_P_c, seq_len, pred_len)
        
        train_ds_spatial = FullComparisonDataset(tr_past_sp, tr_fut, tr_K, tr_Y, tr_P, seq_len, pred_len)
        test_ds_spatial = FullComparisonDataset(te_past_sp_c, te_fut_c, te_K_c, te_Y_c, te_P_c, seq_len, pred_len)
        
        yield {
            'single': (DataLoader(train_ds_single, batch_size=64, shuffle=True),
                      DataLoader(test_ds_single, batch_size=64, shuffle=False)),
            'spatial': (DataLoader(train_ds_spatial, batch_size=64, shuffle=True),
                       DataLoader(test_ds_spatial, batch_size=64, shuffle=False)),
            'dates': dates
        }

def evaluate_v02_model(model, loader, device):
    """Evaluate V02-style models that use (x_past, x_future, x_clr) signature."""
    model.eval()
    preds, acts = [], []
    
    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            x_future = batch[1].to(device)
            y = batch[3].to(device)
            x_clr = batch[4].to(device)
            
            p_pred, _ = model(x_past, x_future, x_clr)
            p_target = y * (x_clr + 1e-6)
            
            preds.append(p_pred.cpu().numpy())
            acts.append(p_target.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0).flatten()
    acts = np.concatenate(acts, axis=0).flatten()
    rmse = np.sqrt(np.mean((preds - acts)**2))
    mae = np.mean(np.abs(preds - acts))
    nrmse = (rmse / 20.0) * 100
    return {'rmse': rmse, 'nrmse': nrmse, 'mae': mae}, preds, acts

def train_v02_model(model, train_loader, test_loader, optimizer, device, epochs):
    """Train V02-style models."""
    train_losses = []
    val_losses = []
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            x_past = batch[0].to(device)
            x_future = batch[1].to(device)
            y = batch[3].to(device)
            x_clr = batch[4].to(device)
            
            optimizer.zero_grad()
            p_pred, _ = model(x_past, x_future, x_clr)
            p_target = y * (x_clr + 1e-6)
            loss = F.mse_loss(p_pred, p_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_losses.append(total_loss / len(train_loader))
        
        # Validation
        met, _, _ = evaluate_v02_model(model, test_loader, device)
        val_losses.append(met['rmse'])
        
        if ep % 10 == 0:
            print(f"    Ep {ep}: Train={train_losses[-1]:.4f} Val RMSE={met['rmse']:.3f}")
            
    return train_losses, val_losses

def run_full_comparison_benchmark(df_merged, meta_s7, meta_s8, target_cols, future_cols, neighbor_cols, n_splits=4, custom_config=None):
    """
    Full Comparison Benchmark: Compares 6 models:
    1. Smart Persistence (SP)
    2. NWP Model
    3. Vanilla Mamba
    4. V04 Robust Single
    5. V04 Robust Spatial
    """
    print("="*70)
    print("FULL COMPARISON BENCHMARK: V04 Robust vs Baselines")
    print("="*70)
    
    # Config
    config = RobustConfig()
    if custom_config:
        if hasattr(custom_config, 'epochs'): config.epochs = custom_config.epochs
        if hasattr(custom_config, 'PAST_INPUT_COLS'): target_cols = custom_config.PAST_INPUT_COLS
        if hasattr(custom_config, 'FUTURE_INPUT_COLS'): future_cols = custom_config.FUTURE_INPUT_COLS
    
    config.S7_COLS = target_cols
    config.PAST_INPUT_COLS = target_cols
    config.FUTURE_INPUT_COLS = future_cols
    config.update_for_v02(target_cols, future_cols)  # Set V02 compatibility attributes
    
    # Model names
    model_names = ['SP', 'NWP', 'Vanilla Mamba', 'V04 Single', 'V04 Spatial']
    colors = ['gray', 'orange', 'purple', 'blue', 'green']
    
    # Results storage
    results = {name: {'rmse': [], 'nrmse': [], 'mae': []} for name in model_names}
    all_preds = {name: [] for name in model_names}
    all_acts = []
    
    # Prepare folds
    fold_gen = list(prepare_full_comparison_fold(
        df_merged, meta_s7, meta_s8, target_cols, future_cols, neighbor_cols, n_splits=n_splits
    ))
    
    for fold_i, fold_data in enumerate(fold_gen):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_i+1}/{n_splits}: {fold_data['dates']['test_start']} -> {fold_data['dates']['test_end']}")
        print(f"{'='*60}")
        
        train_loader_single, test_loader_single = fold_data['single']
        train_loader_spatial, test_loader_spatial = fold_data['spatial']
        
        # =====================================================================
        # 1. Smart Persistence (No Training)
        # =====================================================================
        print("\n[1] Smart Persistence...")
        sp_model = v02_module.SmartPersistenceModel(config).to(config.device)
        met_sp, preds_sp, acts = evaluate_v02_model(sp_model, test_loader_single, config.device)
        print(f"    RMSE={met_sp['rmse']:.3f} | nRMSE={met_sp['nrmse']:.2f}% | MAE={met_sp['mae']:.3f}")
        results['SP']['rmse'].append(met_sp['rmse'])
        results['SP']['nrmse'].append(met_sp['nrmse'])
        results['SP']['mae'].append(met_sp['mae'])
        all_preds['SP'].append(preds_sp)
        all_acts.append(acts)
        
        # =====================================================================
        # 2. NWP Model (No Training)
        # =====================================================================
        print("\n[2] NWP Model...")
        nwp_model = v02_module.NWPModel(config).to(config.device)
        met_nwp, preds_nwp, _ = evaluate_v02_model(nwp_model, test_loader_single, config.device)
        print(f"    RMSE={met_nwp['rmse']:.3f} | nRMSE={met_nwp['nrmse']:.2f}% | MAE={met_nwp['mae']:.3f}")
        results['NWP']['rmse'].append(met_nwp['rmse'])
        results['NWP']['nrmse'].append(met_nwp['nrmse'])
        results['NWP']['mae'].append(met_nwp['mae'])
        all_preds['NWP'].append(preds_nwp)
        
        # =====================================================================
        # 3. Vanilla Mamba (Train)
        # =====================================================================
        print("\n[3] Vanilla Mamba (Training)...")
        vanilla_model = v02_module.VanillaMamba(config).to(config.device)
        opt_vanilla = torch.optim.Adam(vanilla_model.parameters(), lr=1e-3)
        _, _ = train_v02_model(vanilla_model, train_loader_single, test_loader_single, opt_vanilla, config.device, config.epochs)
        met_vanilla, preds_vanilla, _ = evaluate_v02_model(vanilla_model, test_loader_single, config.device)
        print(f"    RMSE={met_vanilla['rmse']:.3f} | nRMSE={met_vanilla['nrmse']:.2f}% | MAE={met_vanilla['mae']:.3f}")
        results['Vanilla Mamba']['rmse'].append(met_vanilla['rmse'])
        results['Vanilla Mamba']['nrmse'].append(met_vanilla['nrmse'])
        results['Vanilla Mamba']['mae'].append(met_vanilla['mae'])
        all_preds['Vanilla Mamba'].append(preds_vanilla)
        
        # =====================================================================
        # 4. V04 Robust Single (Train)
        # =====================================================================
        print("\n[4] V04 Robust Single (Training)...")
        config.INTERACTION_COLS = []
        model_single = PhysicsResidualMamba_V04_Robust(config, meta_s7).to(config.device)
        opt_single = torch.optim.Adam(model_single.parameters(), lr=1e-3)
        loss_smooth = SmoothnessLoss(weight=0.1)
        
        for ep in range(config.epochs):
            train_one_epoch_robust(model_single, train_loader_single, opt_single, config.device, loss_smooth)
            if ep % 10 == 0:
                met_tmp, _, _ = evaluate_robust(model_single, test_loader_single, config.device)
                print(f"    Ep {ep}: Val RMSE={met_tmp['rmse']:.3f}")
                
        met_single, preds_single, _ = evaluate_robust(model_single, test_loader_single, config.device)
        print(f"    RMSE={met_single['rmse']:.3f} | nRMSE={met_single['nrmse']:.2f}% | MAE={met_single['mae']:.3f}")
        results['V04 Single']['rmse'].append(met_single['rmse'])
        results['V04 Single']['nrmse'].append(met_single['nrmse'])
        results['V04 Single']['mae'].append(met_single['mae'])
        all_preds['V04 Single'].append(preds_single)
        
        # =====================================================================
        # 5. V04 Robust Spatial (Train)
        # =====================================================================
        print("\n[5] V04 Robust Spatial (Training)...")
        config.INTERACTION_COLS = ['Interaction']
        model_spatial = PhysicsResidualMamba_V04_Robust(config, meta_s7).to(config.device)
        opt_spatial = torch.optim.Adam(model_spatial.parameters(), lr=1e-3)
        
        for ep in range(config.epochs):
            train_one_epoch_robust(model_spatial, train_loader_spatial, opt_spatial, config.device, loss_smooth)
            if ep % 10 == 0:
                met_tmp, _, _ = evaluate_robust(model_spatial, test_loader_spatial, config.device)
                print(f"    Ep {ep}: Val RMSE={met_tmp['rmse']:.3f}")
                
        met_spatial, preds_spatial, _ = evaluate_robust(model_spatial, test_loader_spatial, config.device)
        print(f"    RMSE={met_spatial['rmse']:.3f} | nRMSE={met_spatial['nrmse']:.2f}% | MAE={met_spatial['mae']:.3f}")
        results['V04 Spatial']['rmse'].append(met_spatial['rmse'])
        results['V04 Spatial']['nrmse'].append(met_spatial['nrmse'])
        results['V04 Spatial']['mae'].append(met_spatial['mae'])
        all_preds['V04 Spatial'].append(preds_spatial)
    
    # ==========================================================================
    # GENERATE COMPARISON PLOTS
    # ==========================================================================
    print("\n" + "="*60)
    print("GENERATING PLOTS...")
    print("="*60)
    
    # 1. Metric Bar Chart
    avg_metrics = {name: {m: np.mean(results[name][m]) for m in ['rmse', 'nrmse', 'mae']} for name in model_names}
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=['RMSE (MW)', 'nRMSE (%)', 'MAE (MW)'])
    
    for i, metric in enumerate(['rmse', 'nrmse', 'mae']):
        for j, name in enumerate(model_names):
            fig.add_trace(
                go.Bar(name=name, x=[name], y=[avg_metrics[name][metric]], 
                       marker_color=colors[j], showlegend=(i==0)),
                row=1, col=i+1
            )
    
    fig.update_layout(title='Full Comparison: All Models', barmode='group', height=400)
    fig.write_html('v04_full_comparison.html')
    print("Saved: v04_full_comparison.html")
    
    # 2. Predictions Plot
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=np.concatenate(all_acts), mode='lines', name='Actual', 
                                   line=dict(color='black', width=1), opacity=0.5))
    for j, name in enumerate(model_names):
        fig_pred.add_trace(go.Scatter(y=np.concatenate(all_preds[name]), mode='lines', name=name,
                                       line=dict(color=colors[j])))
    fig_pred.update_layout(title='Full Comparison: Predictions vs Actual', 
                           xaxis_title='Time Step', yaxis_title='Power (MW)')
    fig_pred.write_html('v04_full_comparison_predictions.html')
    print("Saved: v04_full_comparison_predictions.html")
    
    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'RMSE (MW)':<12} {'nRMSE (%)':<12} {'MAE (MW)':<12}")
    print("-"*56)
    for name in model_names:
        r = avg_metrics[name]['rmse']
        n = avg_metrics[name]['nrmse']
        m = avg_metrics[name]['mae']
        print(f"{name:<20} {r:<12.3f} {n:<12.2f} {m:<12.3f}")
    
    # Best model
    best_model = min(model_names, key=lambda x: avg_metrics[x]['nrmse'])
    print(f"\nBest Model: {best_model} (nRMSE = {avg_metrics[best_model]['nrmse']:.2f}%)")
    
    # Spatial Gain
    if avg_metrics['V04 Single']['nrmse'] > 0:
        spatial_gain = (avg_metrics['V04 Single']['nrmse'] - avg_metrics['V04 Spatial']['nrmse']) / avg_metrics['V04 Single']['nrmse'] * 100
        print(f"Spatial Gain (V04 Single -> Spatial): {spatial_gain:.1f}% reduction in nRMSE")
    
    return results, avg_metrics

