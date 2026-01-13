"""
Base models for solar power forecasting.

Provides:
- MMPISSM_Model_01: Mamba-based model for residual learning
- VanillaLSTM: Standard LSTM baseline
- VanillaMamba: Simplified Mamba baseline
"""

import torch
import torch.nn as nn
from typing import Optional

# Import Mamba if available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using placeholder.")

from .components import RevIN, series_decomp


class MMPISSM_Model_01(nn.Module):
    """
    Power Mamba Model (Base model for residual learning).
    
    This is the original Mamba-based model that serves as the
    residual learner in the Physics-Residual architecture.
    
    In the hybrid architecture, this model's output is interpreted as
    the residual: ε = P_true - P_physics
    
    Architecture:
        - Input construction with shadow features (future NWP)
        - RevIN normalization
        - Series decomposition (trend + seasonal)
        - Dual Mamba branches (main + interaction)
        - Output projection
    
    Args:
        configs: Configuration object with model hyperparameters
        output_residual: If True, output is zero-mean residual (default False)
        
    Example:
        >>> model = MMPISSM_Model_01(configs, output_residual=True)
        >>> residual = model(x_past, x_future)
    """
    
    def __init__(self, configs, output_residual: bool = False):
        super(MMPISSM_Model_01, self).__init__()
        self.configs = configs
        self.output_residual = output_residual
        
        # --- Dimensions ---
        self.num_past_feats = len(self.configs.PAST_INPUT_COLS)
        self.num_shadows = len(self.configs.common_features)
        self.enc_in = self.num_past_feats + self.num_shadows
        
        # --- Pre-calculate Indices ---
        self.fut_indices = torch.tensor(
            [self.configs.FUTURE_INPUT_COLS.index(f) for f in self.configs.common_features],
            dtype=torch.long
        )
        self.past_target_indices = torch.tensor(
            [self.configs.PAST_INPUT_COLS.index(f) for f in self.configs.common_features],
            dtype=torch.long
        )
        
        self.shadow_start_idx = self.num_past_feats
        
        # Length projection
        self.lin1 = nn.Linear(2 * (self.configs.seq_len + self.configs.pred_len), self.configs.seq_len)
        
        # Target indices
        self.target_indices = [self.configs.PAST_INPUT_COLS.index(col) for col in self.configs.TARGET_COL]
        
        # Decomposition
        self.decompsition = series_decomp(self.configs.kernel_size)
        
        # RevIN layers
        self.revin_layer = RevIN(self.enc_in)
        self.revin_layer_enc = RevIN(self.enc_in)
        
        # Projection layers
        self.lin2 = nn.Linear(2 * self.configs.seq_len, self.configs.n_embed)
        self.lin3 = nn.Linear(4 * self.configs.n_embed, self.configs.pred_len)
        
        self.dropout1 = nn.Dropout(self.configs.dropout)
        self.dropout2 = nn.Dropout(self.configs.dropout)
        
        # Mamba layers
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
            # Placeholder layers if Mamba not available
            self.mamba1 = nn.Identity()
            self.mamba2 = nn.Identity()
    
    def forward(self, x: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for residual prediction.
        
        Args:
            x: Past input features [B, L, num_past_feats]
            x_f: Future input features [B, H, num_future_feats]
            
        Returns:
            Residual prediction [B, H, num_targets]
        """
        B = x.size(0)
        device = x.device
        L = self.configs.seq_len
        H = self.configs.pred_len
        
        # Ensure indices are on correct device
        if self.fut_indices.device != device:
            self.fut_indices = self.fut_indices.to(device)
            self.past_target_indices = self.past_target_indices.to(device)
        
        # 1. Input construction with shadow features
        total_len = L + H
        x_pred = torch.zeros(B, total_len, self.enc_in, device=device)
        
        # A) Fill original past data
        x_pred[:, :L, :self.num_past_feats] = x
        
        # B) Extract horizon forecasts for common/shadow features
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        
        # C) Write forecasts into FUTURE part
        for i, past_idx in enumerate(self.past_target_indices):
            x_pred[:, L:, past_idx] = shadow_forecasts[:, :, i]
        
        # D) Shadow columns (future part)
        x_pred[:, L:, self.shadow_start_idx:] = shadow_forecasts
        
        # E) Shadow columns (past part)
        for i, past_idx in enumerate(self.past_target_indices):
            shadow_col = self.shadow_start_idx + i
            x_pred[:, :L, shadow_col] = x_pred[:, -L:, past_idx]
        
        # 2. Standard PowerMamba Processing
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
        x = self.revin_layer(x, 'denorm')
        
        # Zero-Mean Residual Correction
        if self.output_residual:
            # Subtract the mean to center residuals at 0 instead of Mean Power
            x = x - self.revin_layer.mean
        
        # Return only target columns
        x = x[:, :, self.target_indices]
        return x


class VanillaLSTM(nn.Module):
    """
    Standard LSTM Model (Encoder-Decoder style via concatenated input).
    
    Consumes Past + Future NWP (aligned) -> LSTM -> Output.
    Serves as a baseline to compare against more complex models.
    
    Args:
        configs: Configuration object with model hyperparameters
        
    Example:
        >>> model = VanillaLSTM(configs)
        >>> preds = model(x_past, x_future)
    """
    
    def __init__(self, configs):
        super(VanillaLSTM, self).__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # Helper for input construction
        self.num_past_feats = len(configs.PAST_INPUT_COLS)
        self.shadow_start_idx = self.num_past_feats
        self.fut_indices = torch.tensor(
            [configs.FUTURE_INPUT_COLS.index(f) for f in configs.common_features], dtype=torch.long
        )
        self.past_target_indices = torch.tensor(
            [configs.PAST_INPUT_COLS.index(f) for f in configs.common_features], dtype=torch.long
        )
        
        # Architecture
        self.hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.projection = nn.Linear(self.hidden_size, configs.c_out)
    
    def forward(self, x: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Past input features [B, L, F_past]
            x_f: Future input features [B, H, F_future]
            
        Returns:
            Power prediction [B, H, 1]
        """
        B = x.size(0)
        device = x.device
        L = self.seq_len
        H = self.pred_len
        
        if self.fut_indices.device != device:
            self.fut_indices = self.fut_indices.to(device)
            self.past_target_indices = self.past_target_indices.to(device)
        
        # 1. Input Construction (Same as Mamba to be fair with NWP access)
        total_len = L + H
        x_in = torch.zeros(B, total_len, self.enc_in, device=device)
        x_in[:, :L, :self.num_past_feats] = x
        
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        
        # Fill Future part
        for i, past_idx in enumerate(self.past_target_indices):
            x_in[:, L:, past_idx] = shadow_forecasts[:, :, i]
        x_in[:, L:, self.shadow_start_idx:] = shadow_forecasts
        
        # Fill Past Shadow
        for i, past_idx in enumerate(self.past_target_indices):
            x_in[:, :L, self.shadow_start_idx+i] = x_in[:, -L:, past_idx]
        
        # 2. LSTM Forward
        out, _ = self.lstm(x_in)  # [B, L+H, Hidden]
        
        # Take the last H steps as prediction
        pred = self.projection(out[:, -H:, :])  # [B, H, 1]
        
        return pred


class VanillaMamba(nn.Module):
    """
    Vanilla Mamba (No RevIN, No Decomposition).
    
    Just LinearEmbedding -> Mamba -> Projection.
    Serves as a simplified baseline to understand contributions
    of RevIN and series decomposition.
    
    Args:
        configs: Configuration object with model hyperparameters
        
    Example:
        >>> model = VanillaMamba(configs)
        >>> preds = model(x_past, x_future)
    """
    
    def __init__(self, configs):
        super(VanillaMamba, self).__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # Input construction helpers
        self.num_past_feats = len(configs.PAST_INPUT_COLS)
        self.shadow_start_idx = self.num_past_feats
        self.fut_indices = torch.tensor(
            [configs.FUTURE_INPUT_COLS.index(f) for f in configs.common_features], dtype=torch.long
        )
        self.past_target_indices = torch.tensor(
            [configs.PAST_INPUT_COLS.index(f) for f in configs.common_features], dtype=torch.long
        )
        
        # Architecture
        self.d_model = 128
        self.embedding = nn.Linear(self.enc_in, self.d_model)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.mamba = nn.Identity()
        
        self.projection = nn.Linear(self.d_model, configs.c_out)
    
    def forward(self, x: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Vanilla Mamba.
        
        Args:
            x: Past input features [B, L, F_past]
            x_f: Future input features [B, H, F_future]
            
        Returns:
            Power prediction [B, H, 1]
        """
        B = x.size(0)
        device = x.device
        L = self.seq_len
        H = self.pred_len
        
        if self.fut_indices.device != device:
            self.fut_indices = self.fut_indices.to(device)
            self.past_target_indices = self.past_target_indices.to(device)
        
        # 1. Input Construction
        total_len = L + H
        x_in = torch.zeros(B, total_len, self.enc_in, device=device)
        x_in[:, :L, :self.num_past_feats] = x
        shadow_forecasts = torch.index_select(x_f, 2, self.fut_indices)
        for i, past_idx in enumerate(self.past_target_indices):
            x_in[:, L:, past_idx] = shadow_forecasts[:, :, i]
        x_in[:, L:, self.shadow_start_idx:] = shadow_forecasts
        
        # 2. Mamba Forward
        x_emb = self.embedding(x_in)
        x_out = self.mamba(x_emb)
        pred = self.projection(x_out[:, -H:, :])
        
        return pred
