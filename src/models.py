"""
Purpose: To define the neural network architecture.

"""

import torch
import torch.nn as nn

import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model Definitions

import torch
import torch.nn as nn

class ImprovedLSTM(nn.Module):
    """
    LSTM forecaster for fixed-grid data:
    x: (B, D, S, F) → y: (B, H, S)
    Optional self-attention applied over the D (history_days) dimension.
    """
    def __init__(
        self,
        input_size: int,
        steps_per_day: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        horizon_days: int = 1,
        attn_heads: int = 8,
        use_attention: bool = False,   # <-- NEW FLAG (default disabled)
    ):
        super().__init__()
        self.input_size = input_size
        self.steps_per_day = steps_per_day
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.horizon_days = horizon_days
        self.use_attention = use_attention    # <--- store flag

        flat_in = steps_per_day * input_size

        # Input projection
        self.input_projection = nn.Linear(flat_in, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Attention (created only if configured)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * self.num_directions,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.dropout = nn.Dropout(dropout)

        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.horizon_days * self.steps_per_day),
        )

    def forward(self, x):
        """
        x: (B, D, S, F)
        """

        B, D, S, F = x.shape
        assert S == self.steps_per_day, f"steps_per_day mismatch: input S={S}, model {self.steps_per_day}"
        assert F == self.input_size,    f"input_size mismatch: input F={F}, model {self.input_size}"

        # Flatten per-day sequence
        x = x.reshape(B, D, S * F)

        # Project to LSTM input width
        x = self.input_projection(x)

        # Encode temporal history
        lstm_out, _ = self.lstm(x)  # (B, D, H*num_dir)

        # Optionally apply attention over D
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the final timestep representation
        last_hidden = lstm_out[:, -1, :]

        # Predict future day(s)
        y = self.output_projection(self.dropout(last_hidden))  # (B, H*S)
        y = y.view(B, self.horizon_days, self.steps_per_day)   # (B, H, S)
        return y
    

class MATNet(nn.Module):
    """Multi-Attention Transformer Network for solar forecasting"""
    
    def __init__(self, input_size: int, d_model: int = 256,
                 num_heads: int = 8, num_encoder_layers: int = 4, horizon_days: int = 1,
                 dropout: float = 0.2, K_bins: int = 60):
        super().__init__()
        
        self.input_size = input_size
        self.K_bins = K_bins
        self.d_model = d_model
        self.horizon_days = horizon_days
        
        # Flatten size for each time step
        self.flatten_size = K_bins * input_size
        
        # Input embedding
        self.input_embedding = nn.Linear(self.flatten_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 7, d_model))  # 7 days history
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, K_bins * horizon_days)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, history_days, K_bins, features)
        Returns:
            output: (batch, 1, K_bins)
        """
        batch_size, history_days, K_bins, features = x.shape
        
        # Flatten bins and features
        x = x.view(batch_size, history_days, -1)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :history_days, :]
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Use last time step
        last_hidden = x[:, -1, :]
        
        # Output projection
        output = self.output_projection(last_hidden)
        
        # Reshape
        output.view(batch_size, self.horizon_days, self.K_bins)
        
        return output

