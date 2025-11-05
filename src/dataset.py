# """
# Purpose: To bridge the gap between your NumPy arrays and the PyTorch framework. Its only job is to handle PyTorch-specific data loading.

# - class KBinsDataset

# """

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from typing import Dict, List, Tuple, Optional, Any
# import logging
# import torch

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class KBinsDataset(Dataset):
#     """
#     Dataset for K-bins discretized solar data.
#     Handles (N, history_days, K_bins, features) -> (N, horizon_days, K_bins) mapping.
#     Act as a bridge between our pre-processed NumPy arrays and the PyTorch deep learning framework.
#     It handles two critical tasks: data scaling and serving samples to the model.
#     """
    
#     def __init__(self, X: np.ndarray, Y: np.ndarray, 
#                  feature_scaler: Optional[StandardScaler] = None,
#                  target_scaler: Optional[StandardScaler] = None,
#                  fit_scalers: bool = False):
#         """
#         Args:
#             X: Input array of shape (N, history_days, K_bins, features)
#             Y: Target array of shape (N, horizon_days, K_bins)
#             feature_scaler: Pre-fitted feature scaler
#             target_scaler: Pre-fitted target scaler
#             fit_scalers: Whether to fit new scalers on this data
#         """
#         self.X = X
#         self.Y = Y
        
#         # Initialize or fit scalers
#         if fit_scalers:
#             # Reshape for scaling: (N*days*bins, features)
#             X_reshaped = X.reshape(-1, X.shape[-1])
#             Y_reshaped = Y.reshape(-1)
            
#             self.feature_scaler = StandardScaler(with_mean=True, with_std=True)
#             self.target_scaler = StandardScaler()
            
#             self.feature_scaler.fit(X_reshaped)
#             self.target_scaler.fit(Y_reshaped.reshape(-1, 1))
#         else:
#             self.feature_scaler = feature_scaler
#             self.target_scaler = target_scaler
        
#         # Apply scaling if scalers are available
#         if self.feature_scaler is not None:
#             X_reshaped = X.reshape(-1, X.shape[-1])
#             X_scaled = self.feature_scaler.transform(X_reshaped)
#             self.X_scaled = X_scaled.reshape(X.shape)
#         else:
#             self.X_scaled = X
            
#         if self.target_scaler is not None:
#             Y_reshaped = Y.reshape(-1)
#             Y_scaled = self.target_scaler.transform(Y_reshaped.reshape(-1, 1)).flatten()
#             self.Y_scaled = Y_scaled.reshape(Y.shape)
#         else:
#             self.Y_scaled = Y
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return torch.FloatTensor(self.X_scaled[idx]), torch.FloatTensor(self.Y_scaled[idx])
    
#     def get_scalers(self):
#         """Return the scalers for later use"""
#         return self.feature_scaler, self.target_scaler

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class KBinsDataset(Dataset):
    """
    Dataset for fixed-grid or K-bins data:
      X: (N, history_days, T_or_K, F)
      Y: (N, horizon_days, T_or_K)
    Handles robust scaling and returns finite tensors.
    """

    def __init__(self, X, Y, feature_scaler=None, target_scaler=None, fit_scalers=False):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        # 0) Upfront sanitize to avoid scaler.fit crashes
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.Y = np.nan_to_num(self.Y, nan=0.0, posinf=0.0, neginf=0.0)

        # 1) Fit or reuse scalers
        if fit_scalers:
            X_2d = self.X.reshape(-1, self.X.shape[-1])     # (N*H*T, F)
            Y_2d = self.Y.reshape(-1, 1)                    # (N*D*T, 1)

            fs = StandardScaler(with_mean=True, with_std=True)
            fs.fit(X_2d)

            # Guard: zero-variance columns -> set scale to 1.0
            scale = fs.scale_.copy()
            scale[scale == 0] = 1.0
            fs.scale_ = scale

            # For targets: mean-only centering, no std scaling (avoids /0)
            ts = StandardScaler(with_mean=True, with_std=False)
            ts.fit(Y_2d)

            self.feature_scaler = fs
            self.target_scaler  = ts
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler  = target_scaler

        # 2) Transform and sanitize again
        if self.feature_scaler is not None:
            X_2d = self.X.reshape(-1, self.X.shape[-1])
            Xs   = self.feature_scaler.transform(X_2d)
            Xs   = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            self.X_scaled = Xs.reshape(self.X.shape)
        else:
            self.X_scaled = self.X

        if self.target_scaler is not None:
            Y_2d = self.Y.reshape(-1, 1)
            Ys   = self.target_scaler.transform(Y_2d)
            Ys   = np.nan_to_num(Ys, nan=0.0, posinf=0.0, neginf=0.0)
            self.Y_scaled = Ys.reshape(self.Y.shape)
        else:
            self.Y_scaled = self.Y

        # 3) Final assertion (fail fast if anything is off)
        if not np.isfinite(self.X_scaled).all():
            raise ValueError("Non-finite values in X after scaling.")
        if not np.isfinite(self.Y_scaled).all():
            raise ValueError("Non-finite values in Y after scaling.")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_scaled[idx]).float(),
            torch.from_numpy(self.Y_scaled[idx]).float(),
        )

    def get_scalers(self):
        return self.feature_scaler, self.target_scaler