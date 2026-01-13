"""
Spatiotemporal Physics-Residual Mamba Architecture (V03)
Integrates Adjacent Station Data (e.g., S7 + S8) for Spatial Reasoning.

Design:
1. Physics Branch: Focuses strictly on Target Station (S7) NWP features to produce the "Local Trend".
2. Mamba Branch: Sees BOTH S7 and S8 History/Future features to learn spatiotemporal residuals (e.g. cloud motion).
3. Fusion: P_Total = P_Phys(S7) + Mamba(S7, S8)

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
import re
from pvlib import location, pvsystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# =============================================================================
# Part 1: Data Alignment & Metadata Helpers (Copied & Adapted)
# =============================================================================

def get_station_metadata(station_num, df):
    """Parses station metadata based on Station_ID."""
    station_id_str = f"station{str(station_num).zfill(2)}"
    station_row = df[df['Station_ID'] == station_id_str]
    
    if station_row.empty:
        raise ValueError(f"Station ID {station_id_str} not found.")
    
    row = station_row.iloc[0]
    
    def parse_kv_string(text_block):
        data = {}
        if isinstance(text_block, str):
            for item in text_block.split('\n'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    data[k.strip()] = v.strip()
        return data

    def extract_num(val, default=0.0):
        if pd.isna(val): return default
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else default

    module_items = parse_kv_string(row.get('Module', ''))
    inverter_items = parse_kv_string(row.get('Inverters', ''))
    layout_items = parse_kv_string(row.get('Layout', ''))

    metadata = {
        'Station_ID': station_id_str,
        'Longitude': float(row['Longitude']),
        'Latitude': float(row['Latitude']),
        'Array_Tilt': row['Array_Tilt'],
        'Capacity': float(row['Capacity']),
        'Panel_Size': float(row.get('Panel_Size', 1.62)),
        'Module_Pmax': extract_num(module_items.get('Pmax'), default=250),
        'Modules_per_String': int(extract_num(layout_items.get('modules per string'), default=20)),
        'Strings_per_Inverter': int(extract_num(layout_items.get('strings per inverter'), default=100)),
    }
    return metadata

def calculate_clearsky_indices(metadata, df):
    """Calculates GHI_clr, P_CLR, K_CS, K_PV for a station."""
    print(f"--- Processing {metadata['Station_ID']} Physics ---")
    
    # 1. Location
    lat, lon = metadata['Latitude'], metadata['Longitude']
    site = location.Location(lat, lon, tz='UTC')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df['date_time'] = pd.to_datetime(df['date_time'])
            df.set_index('date_time', inplace=True)
        except:
            df.index = pd.to_datetime(df.index)

    # 2. Clear Sky GHI
    cs = site.get_clearsky(df.index, model='ineichen')
    df['GHI_clr'] = cs['ghi']
    
    # 3. K_CS (Irradiance Index)
    if 'lmd_totalirrad' in df.columns:
        df['K_CS'] = np.where(df['GHI_clr'] > 10, df['lmd_totalirrad'] / df['GHI_clr'], 0.0)
        df['K_CS'] = df['K_CS'].clip(0.0, 1.5)

    # 4. PV Simulation for P_CLR (Clear Sky Power)
    try:
        tilt_str = str(metadata.get('Array_Tilt'))
        tilt_match = re.search(r"[\d.]+", tilt_str)
        tilt = float(tilt_match.group()) if tilt_match else 33.0
    except: tilt = 33.0
    
    # Simple setup using 'sandia' or 'cec' as generic fallback
    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    ivt_para = pvsystem.retrieve_sam('cecinverter')['Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_']
    # Adjust generic inverter to match capacity
    ivt_para["Pdco"] = 567000 # Default fallback
    
    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = pvsystem.PVSystem(
        surface_tilt=tilt, surface_azimuth=180,
        module_parameters=module_params, inverter_parameters=ivt_para,
        temperature_model_parameters=temp_model,
        modules_per_string=metadata['Modules_per_String'],
        strings_per_inverter=metadata['Strings_per_Inverter']
    )
    
    mc = ModelChain(system, site, transposition_model='perez', solar_position_method='nrel_numpy', aoi_model='physical', spectral_model='no_loss')
    mc.run_model(cs)
    
    # Scale to Station Capacity
    block_dc_watts = (metadata['Modules_per_String'] * metadata['Strings_per_Inverter'] * metadata['Module_Pmax'])
    total_cap_watts = metadata['Capacity'] * 1000
    if block_dc_watts > 0: factor = total_cap_watts / block_dc_watts
    else: factor = 1.0
        
    p_clr_watts = mc.results.ac.fillna(0) * factor
    df['P_CLR'] = p_clr_watts / 1_000_000 # MW
    
    # 5. K_PV (Power Index)
    if 'power' in df.columns:
        meas = df['power']
        if meas.max() > 500: meas = meas / 1000.0 # Convert kW to MW
        df['K_PV'] = np.where(df['P_CLR'] > 1.0, meas / df['P_CLR'], 0.0)
        df['K_PV'] = df['K_PV'].clip(0.0, 1.5)
        
    return df

def merge_stations(df_target, df_neighbor, suffix='_s8'):
    """Aligns Target and Neighbor dataframes on index."""
    print("Merging spatial data...")
    # Suffix neighbor columns
    neighbor_renamed = df_neighbor.add_suffix(suffix)
    
    # Merge on Index (Time)
    # Merge on Index (Time)
    # Fix: Ensure Timezone Awareness Matches
    target_tz = df_target.index.tz
    neighbor_tz = neighbor_renamed.index.tz
    
    if target_tz is not None and neighbor_tz is None:
        # Target is Aware, Neighbor is Naive -> Assume Neighbor is UTC, then align
        print(f"Aligning Neighbor (Naive) to Target TZ ({target_tz})...")
        neighbor_renamed.index = neighbor_renamed.index.tz_localize('UTC').tz_convert(target_tz)
        
    elif target_tz is None and neighbor_tz is not None:
         # Target is Naive, Neighbor is Aware -> Convert Neighbor to UTC then make Naive
         print(f"Aligning Neighbor (Aware) to Target (Naive) via UTC...")
         neighbor_renamed.index = neighbor_renamed.index.tz_convert('UTC').tz_localize(None)
         
    elif target_tz != neighbor_tz:
        # Both aware but different
        print(f"Aligning Neighbor TZ ({neighbor_tz}) to Target TZ ({target_tz})...")
        neighbor_renamed.index = neighbor_renamed.index.tz_convert(target_tz)

    df_merged = df_target.join(neighbor_renamed, how='inner') # Inner join strict on Time Match
    
    print(f"Original Target Shape: {df_target.shape}")
    print(f"Original Neighbor Shape: {df_neighbor.shape}")
    print(f"Merged Shape: {df_merged.shape}")
    return df_merged

# =============================================================================
# Part 2: Architecture (V03 Spatial)
# =============================================================================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):  
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

class MMPISSM_Model_V03(nn.Module):
    """Spatiotemporal Mamba Backbone (Handles S7 + S8 features)"""
    def __init__(self, configs, output_residual=False):
        super(MMPISSM_Model_V03, self).__init__()
        self.configs = configs
        self.output_residual = output_residual
        
        # Spatial Feature Handling
        # input_dim = S7_Past_Cols + S8_Past_Cols (implied in PAST_INPUT_COLS)
        self.num_past_feats = len(self.configs.PAST_INPUT_COLS)
        self.num_shadows = len(self.configs.common_features)
        
        # Total Encoding Dim
        self.enc_in = self.num_past_feats + self.num_shadows

        # Indices for Shadowing Future Feats (S7 and S8 NWPs)
        self.fut_indices = torch.tensor([self.configs.FUTURE_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.past_target_indices = torch.tensor([self.configs.PAST_INPUT_COLS.index(f) for f in self.configs.common_features], dtype=torch.long)
        self.shadow_start_idx = self.num_past_feats
        
        # Output Head (Target S7 K_PV Only)
        # Note: configs.TARGET_COL should be ['K_PV'] (Dataset targets S7)
        self.target_indices = [self.configs.PAST_INPUT_COLS.index(col) for col in self.configs.TARGET_COL]

        # Core Layers
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

        # 1. Shadow Feature Construction (S7+S8 inputs)
        # We append Future NWP (S7+S8) to the end of Past (S7+S8)
        total_len = L + H
        x_pred = torch.zeros(B, total_len, self.enc_in, device=device)
        x_pred[:, :L, :self.num_past_feats] = x
        
        # Future Context (NWP)
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        
        # Place Future NWP into "Placeholder" slots for horizon
        for i, past_idx in enumerate(self.past_target_indices):
            x_pred[:, L:, past_idx] = shadow_forecasts[:, :, i]
            
        # Place Future NWP into Dedicated Shadow Channels (Full Sequence)
        x_pred[:, L:, self.shadow_start_idx:] = shadow_forecasts
        for i, past_idx in enumerate(self.past_target_indices):
            shadow_col = self.shadow_start_idx + i
            x_pred[:, :L, shadow_col] = x_pred[:, -L:, past_idx]

        # 2. RevIN & Decomposition
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

        # 3. Mamba Blocks (Spatial & Temporal Mixing)
        # Mamba sees S8 features here and learns cross-correlations
        x_m = self.mamba1(self.dropout1(x_e))
        x_im = self.mamba2(self.dropout2(x_e).permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x_im, x_m, x_m + x_im, x_e], dim=2)
        x = self.lin3(x).permute(0, 2, 1)
        
        # 4. Final Projection
        x = self.revin_layer(x, 'denorm')
        x = x[:, :, self.target_indices] # ONLY extract S7 K_PV prediction
        
        return x, None

class DifferentiablePVLayer_V02(nn.Module):
    """Basic Physics Layer (Reused from V02) - Robust to missing cols"""
    def __init__(self, idx_G, idx_Ta=None, idx_WS=None, T_ref=25.0, G_stc=1000.0, P_stc_init=19.0):
        super().__init__()
        self.idx_G = idx_G
        self.idx_Ta = idx_Ta 
        self.idx_WS = idx_WS
        self.T_ref, self.G_stc = float(T_ref), float(G_stc)
        self.eps = 1e-6
        
        self.gamma_raw = nn.Parameter(torch.tensor(-5.4, dtype=torch.float32)) 
        self.U0_raw = nn.Parameter(torch.tensor(3.2, dtype=torch.float32)) 
        self.U1_raw = nn.Parameter(torch.tensor(1.9, dtype=torch.float32)) 
        self.eta_raw = nn.Parameter(torch.tensor(-1.66)) 
        self.Pstc_raw = nn.Parameter(torch.tensor(float(P_stc_init))) 

    def forward(self, x_future, x_clr=None):
        # Robust G Extraction
        if self.idx_G is not None and self.idx_G < x_future.shape[2]:
            G = x_future[:, :, self.idx_G]
        else:
            # Fallback (Should typically not happen if config is sane, but safe)
            G = x_future[:, :, 0] 

        # Robust Ta Extraction
        if self.idx_Ta is not None and self.idx_Ta < x_future.shape[2]:
            Ta = x_future[:, :, self.idx_Ta]
        else:
            Ta = torch.zeros_like(G) + 25.0 # Default 25C

        # Robust WS Extraction
        if self.idx_WS is not None and self.idx_WS < x_future.shape[2]:
            WS = x_future[:, :, self.idx_WS]
        else:
            WS = torch.zeros_like(G) + 1.0 # Default 1m/s
        
        U0 = F.softplus(self.U0_raw)
        U1 = F.softplus(self.U1_raw)
        eta = torch.sigmoid(self.eta_raw)
        gamma = -F.softplus(self.gamma_raw) 
        P_stc = F.softplus(self.Pstc_raw)
        
        T_cell = Ta + G / (U0 + U1 * WS + self.eps)
        delta_T = T_cell - self.T_ref
        temp_factor = 1.0 + gamma * delta_T
        P_dc = eta * P_stc * (G / self.G_stc) * temp_factor
        
        P_final_mw = F.relu(P_dc).unsqueeze(-1)
        
        if x_clr is not None:
             if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
             K_phys = P_final_mw / (x_clr + self.eps)
             night_mask = (x_clr > 0.01).float()
             K_phys = K_phys * night_mask
             return K_phys
        return P_final_mw

class PhysicsResidualMamba_V03(nn.Module):
    """Spatiotemporal V03: Single-Station Physics + Multi-Station Mamba"""
    def __init__(self, configs):
        super(PhysicsResidualMamba_V03, self).__init__()
        self.configs = configs
        
        # Physics Layer sees ONLY S7 NWP
        # Intelligent Index Finding
        try:
            str_cols = configs.FUTURE_INPUT_COLS
            idx_G = str_cols.index('nwp_globalirrad') if 'nwp_globalirrad' in str_cols else 0
            
            idx_Ta = str_cols.index('nwp_temperature') if 'nwp_temperature' in str_cols else None
            idx_WS = str_cols.index('nwp_windspeed') if 'nwp_windspeed' in str_cols else None
            
        except ValueError:
            print("Warning: Could not resolve Physics Columns. Using Defaults.")
            idx_G, idx_Ta, idx_WS = 0, None, None
            
        self.physics_layer = DifferentiablePVLayer_V02(idx_G, idx_Ta, idx_WS, T_ref=configs.T_ref)
        
        # Mamba Model sees EVERYTHING (S7 + S8)
        self.mamba_model = MMPISSM_Model_V03(configs, output_residual=True)
        
    def forward(self, x_past, x_future, x_clr):
        # 1. Physics Branch (Local Trend)
        p_physics_k = self.physics_layer(x_future, x_clr=x_clr)
        
        # 2. Mamba Branch (Spatiotemporal Residuals)
        # x_past includes S8 history. x_future includes S8 NWP.
        mamba_k, _ = self.mamba_model(x_past, x_future)
        
        # 3. Fusion
        k_total = p_physics_k + mamba_k
        
        # 4. Reconstruction
        if x_clr.ndim == 2: x_clr = x_clr.unsqueeze(-1)
        p_total_mw = F.relu(k_total) * x_clr
        return p_total_mw, p_physics_k

# =============================================================================
# Part 3: Dataset & Configs
# =============================================================================

class SpatiotemporalConfigs:
    def __init__(self, target_cols, neighbor_cols, n_splits=4):
        self.seq_len = 96
        self.pred_len = 96
        self.TARGET_COL = ['K_PV'] 
        
        # Combine S7 and S8 columns for inputs
        # Example S7: ['nwp_G', 'K_PV', ...]
        # Example S8: ['nwp_G_s8', 'K_PV_s8', ...]
        self.PAST_INPUT_COLS = target_cols + neighbor_cols
        
        # Future inputs usually just NWP from both
        self.FUTURE_INPUT_COLS = [c for c in self.PAST_INPUT_COLS if 'nwp' in c or 'sin' in c or 'cos' in c]
        
        self.common_features = [f for f in self.FUTURE_INPUT_COLS if f in self.PAST_INPUT_COLS]
        self.num_shadows = len(self.common_features)
        
        # Dimensions
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
        self.T_ref = 25.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class V03Dataset(Dataset):
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

def prepare_spatiotemporal_fold(df_merged, target_input_cols, neighbor_input_cols, 
                                future_target_cols, future_neighbor_cols,
                                n_splits=2, seq_len=96, pred_len=96):
                                
    # 1. Define Complete Feature Sets
    past_cols = target_input_cols + neighbor_input_cols
    future_cols = future_target_cols + future_neighbor_cols
    
    # Common features (Shadowing)
    common_features = [f for f in future_cols if f in past_cols]
    
    print(f"Input Features: {len(past_cols)} | Future Features: {len(future_cols)}")
    
    # 2. Extract Data
    df = df_merged.fillna(0)
    X_raw = df[past_cols].values.astype(np.float32)
    y_raw = df['K_PV'].values.astype(np.float32) # Target S7 K_PV
    
    if 'P_CLR' in df.columns: P_raw = df['P_CLR'].values.astype(np.float32)
    else: P_raw = np.zeros((len(df), 1), dtype=np.float32)
        
    Bf_raw = df[common_features].values.astype(np.float32)
    
    # 3. TimeSeries Split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for i, (train_index, test_index) in enumerate(tscv.split(X_raw)):
        # Dates
        dates = {
            'train_start': df.index[train_index[0]], 'train_end': df.index[train_index[-1]],
            'test_start': df.index[test_index[0]], 'test_end': df.index[test_index[-1]]
        }
        
        X_train, P_train, y_train = X_raw[train_index], P_raw[train_index], y_raw[train_index]
        X_test, P_test, y_test = X_raw[test_index], P_raw[test_index], y_raw[test_index]
        Bf_train = Bf_raw[train_index]; Bf_test = Bf_raw[test_index]
        
        # Concat Lookback
        X_test_final = np.concatenate([X_train[-seq_len:], X_test], axis=0)
        P_test_final = np.concatenate([P_train[-seq_len:], P_test], axis=0)
        y_test_final = np.concatenate([y_train[-seq_len:], y_test], axis=0)
        Bf_test_final = np.concatenate([Bf_train[-seq_len:], Bf_test], axis=0)
        
        train_ds = V03Dataset(X_train, y_train, P_train, Bf_train, seq_len, pred_len)
        test_ds = V03Dataset(X_test_final, y_test_final, P_test_final, Bf_test_final, seq_len, pred_len)
        
        yield (DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0),
               DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0),
               {}, dates)

# =============================================================================
# Part 4: Training Runner
# =============================================================================

def train_one_epoch_v03(model, loader, optimizer, device, warmup=False):
    model.train()
    if warmup:
        for p in model.mamba_model.parameters(): p.requires_grad = False
        for p in model.physics_layer.parameters(): p.requires_grad = True
    else:
        for p in model.mamba_model.parameters(): p.requires_grad = True
        for p in model.physics_layer.parameters(): p.requires_grad = True
            
    total_loss = 0.0
    for batch in loader:
        x_past = batch[0].to(device)
        y = batch[1].to(device)
        x_fut = batch[2].to(device)
        x_clr = batch[4].to(device)
        
        optimizer.zero_grad()
        p_pred, _ = model(x_past, x_fut, x_clr)
        
        p_target = y * (x_clr + 1e-6)
        loss = F.mse_loss(p_pred, p_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_v03(model, loader, device):
    model.eval()
    preds, acts = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x_past = batch[0].to(device)
            y = batch[1].to(device) # K_PV
            x_fut = batch[2].to(device)
            x_clr = batch[4].to(device) # P_CLR
            
            p_pred, _ = model(x_past, x_fut, x_clr)
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

# =============================================================================
# Part 5: One-vs-All Comparison (V02 Single vs V03 Spatial)
# =============================================================================
import physics_residual_mamba_v02 as v02_module

def run_comparative_spatial_benchmark(df_merged, target_cols, neighbor_cols, n_splits=4, custom_config=None):
    """
    Compares Single-Station Physics Mamba (V02) vs Spatiotemporal Physics Mamba (V03).
    Ensures identical train/test splits for fair comparison.
    Accepts custom_config to override hyperparameters (seq_len, d_model, epochs, etc.).
    """
    # --- APPLY USER OVERRIDES ---
    # 1. Column Overrides (Check first so they propagate to Configs)
    if custom_config is not None:
        if hasattr(custom_config, 'PAST_INPUT_COLS'):
            print(f"Overriding Target Columns with {len(custom_config.PAST_INPUT_COLS)} features")
            target_cols = custom_config.PAST_INPUT_COLS
            
        if hasattr(custom_config, 'FUTURE_INPUT_COLS'):
            print(f"Overriding Future Target Columns with {len(custom_config.FUTURE_INPUT_COLS)} features")
            # We assume these are S7 future cols. We need to derive S8 future cols too?
            # Or we assume the user just wants to control S7. 
            # We will use this list to replace the auto-derived 'fut_target'
            # Note: We must leave 'future_native' definition for neighbor derivation unless we change that too.
            pass

    # 1b. Configs
    # Spatial Config (S7 + S8)
    # Re-derive future lists based on potentially updated target_cols
    future_native = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'NWP_Power_MW', 'hour_sin', 'hour_cos']
    
    if custom_config and hasattr(custom_config, 'FUTURE_INPUT_COLS'):
        fut_target = custom_config.FUTURE_INPUT_COLS
    else:
        fut_target = [c for c in target_cols if any(x in c for x in future_native)]
        
    fut_neighbor = [c for c in neighbor_cols if any(x in c.replace('_s8','') for x in future_native)]
    
    config_spatial = SpatiotemporalConfigs(target_cols, neighbor_cols, n_splits=n_splits)
    
    # Single Config (S7 Only)
    config_single = v02_module.PhysicsResidualMambaConfigs(n_splits=n_splits)

    # --- APPLY USER OVERRIDES ---
    if custom_config is not None:
        print("Applying User Configuration Overrides...")
        # List of attrs to copy
        attrs = ['seq_len', 'pred_len', 'n_embed', 'd_state', 'dconv', 'e_fact', 'dropout', 'batch_size', 'epochs', 'device', 'kernel_size']
        for attr in attrs:
            if hasattr(custom_config, attr):
                val = getattr(custom_config, attr)
                setattr(config_spatial, attr, val)
                setattr(config_single, attr, val)
        
    config_spatial.FUTURE_INPUT_COLS = fut_target + fut_neighbor
    config_spatial.common_features = [f for f in config_spatial.FUTURE_INPUT_COLS if f in config_spatial.PAST_INPUT_COLS]
    config_spatial.num_shadows = len(config_spatial.common_features)
    config_spatial.enc_in = len(config_spatial.PAST_INPUT_COLS) + config_spatial.num_shadows
    
    # Override cols to match S7 subset of spatial df
    config_single.PAST_INPUT_COLS = target_cols
    config_single.FUTURE_INPUT_COLS = fut_target
    config_single.common_features = [f for f in config_single.FUTURE_INPUT_COLS if f in config_single.PAST_INPUT_COLS]
    config_single.num_shadows = len(config_single.common_features)
    config_single.enc_in = len(config_single.PAST_INPUT_COLS) + config_single.num_shadows
    config_single.TARGET_COL = ['K_PV']
    # Use spatial dimensions for consistency
    config_single.seq_len = config_spatial.seq_len
    config_single.pred_len = config_spatial.pred_len

    device = config_spatial.device
    print(f"Configuration: Seq_Len={config_spatial.seq_len}, Pred_Len={config_spatial.pred_len}, Epochs={config_spatial.epochs}")
    
    # 2. Models
    # V03 (Spatial)
    model_spatial = PhysicsResidualMamba_V03(config_spatial).to(device)
    opt_spatial = torch.optim.Adam(model_spatial.parameters(), lr=1e-3)
    
    # V02 (Single) - Reusing class from v02 module
    model_single = v02_module.PhysicsResidualMamba(config_single).to(device)
    opt_single = torch.optim.Adam(model_single.parameters(), lr=1e-3)

    # 3. Data Prep
    fold_gen = prepare_spatiotemporal_fold(
        df_merged, target_cols, neighbor_cols, 
        fut_target, fut_neighbor, 
        n_splits=n_splits, seq_len=config_spatial.seq_len, pred_len=config_spatial.pred_len
    )
    
    metrics = {
        'single': {'rmse': [], 'nrmse': []},
        'spatial': {'rmse': [], 'nrmse': []}
    }
    
    for fold_i, (loader_spatial_train, loader_spatial_test, _, dates) in enumerate(fold_gen):
        print(f"\n=== Fold {fold_i+1} Comparison ===")
        print(f"Time: {dates['test_start']} -> {dates['test_end']}")

        # --- Train Loop ---
        # Determine Epoch split based on config
        total_epochs = config_spatial.epochs 
        
        if hasattr(config_spatial, 'warmup_epochs'):
            n_warmup = config_spatial.warmup_epochs
        else:
            # Heuristic: 20% warmup (max 5), rest joint
            n_warmup = min(5, max(1, int(total_epochs * 0.2)))
            
        n_joint = total_epochs - n_warmup
        
        print(f"Schedule: {n_warmup} Warmup + {n_joint} Joint Epochs")

        
        # Training
        print("Training Models...", end="")
        t0 = time.time()
        
        n_s7_past = len(target_cols)
        n_s7_fut = len(fut_target)
        
        # Warmup Phase
        for ep in range(n_warmup):
            # Spatial
            train_one_epoch_v03(model_spatial, loader_spatial_train, opt_spatial, device, warmup=True)
            
            # Single
            model_single.train()
            # Freeze Mamba, Train Physics
            for p in model_single.mamba_model.parameters(): p.requires_grad=False
            for p in model_single.physics_layer.parameters(): p.requires_grad=True
            
            for batch in loader_spatial_train:
                x_past_all = batch[0].to(device)
                y = batch[1].to(device)
                x_fut_all = batch[2].to(device)
                x_clr = batch[4].to(device)
                x_past_s7 = x_past_all[:, :, :n_s7_past]
                x_fut_s7 = x_fut_all[:, :, :n_s7_fut]
                
                opt_single.zero_grad()
                p_pred, _ = model_single(x_past_s7, x_fut_s7, x_clr)
                p_target = y * (x_clr + 1e-6)
                loss = F.mse_loss(p_pred, p_target)
                loss.backward()
                opt_single.step()
        
        # Joint Phase
        for ep in range(n_joint):
            # Spatial
            train_one_epoch_v03(model_spatial, loader_spatial_train, opt_spatial, device, warmup=False)
            
            # Single
            model_single.train()
            # Train All
            for p in model_single.mamba_model.parameters(): p.requires_grad=True
            for p in model_single.physics_layer.parameters(): p.requires_grad=True
            
            for batch in loader_spatial_train:
                x_past_all = batch[0].to(device)
                y = batch[1].to(device)
                x_fut_all = batch[2].to(device)
                x_clr = batch[4].to(device)
                x_past_s7 = x_past_all[:, :, :n_s7_past]
                x_fut_s7 = x_fut_all[:, :, :n_s7_fut]
                
                opt_single.zero_grad()
                p_pred, _ = model_single(x_past_s7, x_fut_s7, x_clr)
                p_target = y * (x_clr + 1e-6)
                loss = F.mse_loss(p_pred, p_target)
                loss.backward()
                opt_single.step()
                
        print(f" Done ({time.time()-t0:.1f}s)")
        
        # --- Evaluation ---
        # Spatial Eval
        met_spatial, preds_spatial, acts = evaluate_v03(model_spatial, loader_spatial_test, device)
        
        # Single Eval
        model_single.eval()
        preds_s7_list = []
        with torch.no_grad():
            for batch in loader_spatial_test:
                x_past_all = batch[0].to(device)
                x_fut_all = batch[2].to(device)
                x_clr = batch[4].to(device)
                x_past_s7 = x_past_all[:, :, :n_s7_past]
                x_fut_s7 = x_fut_all[:, :, :n_s7_fut]
                
                p, _ = model_single(x_past_s7, x_fut_s7, x_clr)
                preds_s7_list.append(p.cpu().numpy())
        preds_single = np.concatenate(preds_s7_list, axis=0).flatten()
        
        # Calc Single Metrics
        rmse_single = np.sqrt(np.mean((preds_single - acts)**2))
        nrmse_single = (rmse_single/20.0)*100
        
        metrics['single']['rmse'].append(rmse_single)
        metrics['single']['nrmse'].append(nrmse_single)
        metrics['spatial']['rmse'].append(met_spatial['rmse'])
        metrics['spatial']['nrmse'].append(met_spatial['nrmse'])
        
        print(f"  Single (V02) RMSE: {rmse_single:.3f} MW | nRMSE: {nrmse_single:.2f}%")
        print(f"  Spatial(V03) RMSE: {met_spatial['rmse']:.3f} MW | nRMSE: {met_spatial['nrmse']:.2f}%")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=acts, name='Actual', line=dict(color='black', width=2), opacity=0.5))
        fig.add_trace(go.Scatter(y=preds_single, name='Single (S7)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=preds_spatial, name='Spatial (S7+S8)', line=dict(color='red')))
        fig.update_layout(title=f"Spatial Gain Fold {fold_i+1}: Single vs Spatial")
        fig.write_html(f"spatial_comparison_fold_{fold_i+1}.html")
        
    # Summary
    print("\n>>> FINAL SPATIAL COMPARISON <<<")
    avg_s = np.mean(metrics['single']['rmse'])
    avg_m = np.mean(metrics['spatial']['rmse'])
    gain = ((avg_s - avg_m) / avg_s) * 100
    print(f"Avg Single RMSE: {avg_s:.3f} MW")
    print(f"Avg Spatial RMSE: {avg_m:.3f} MW")
    print(f"Spatial Gain: {gain:.2f}% improvement")
    return metrics
