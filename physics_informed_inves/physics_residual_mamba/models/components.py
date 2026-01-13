"""
Helper components for Mamba-based models.

Provides:
- RevIN: Reversible Instance Normalization
- moving_avg: Moving average block
- series_decomp: Series decomposition block
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Removes the need for the model to learn different "rules" for Summer vs. Winter.
    In solar forecasting, a clear sunny day in winter has a much lower peak than a
    clear sunny day in summer due to the sun's angle. Standard models struggle
    because they see these as two different patterns.
    
    RevIN normalizes each input window individually, then denormalizes the output.
    
    Args:
        num_features: Number of features or channels
        eps: Value added for numerical stability (default 1e-3)
        affine: If True, RevIN has learnable affine parameters (default True)
        
    Example:
        >>> revin = RevIN(num_features=10, eps=1e-3, affine=True)
        >>> x_norm = revin(x, mode='norm')
        >>> x_denorm = revin(x_norm, mode='denorm')
    """
    
    def __init__(self, num_features: int, eps: float = 1e-3, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply RevIN normalization or denormalization.
        
        Args:
            x: Input tensor [B, L, F] or [B, F]
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode must be 'norm' or 'denorm', got {mode}")
        return x
    
    def _init_params(self):
        """Initialize learnable affine parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x: torch.Tensor):
        """Compute mean and standard deviation for normalization."""
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(variance + self.eps).detach()
        self.stdev = torch.clamp(self.stdev, min=self.eps)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply denormalization."""
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    
    Computes moving average with zero-padding at boundaries to maintain
    the same sequence length.
    
    Args:
        kernel_size: Size of the moving average window
        stride: Stride for the moving average (default 1)
        
    Example:
        >>> ma = moving_avg(kernel_size=25, stride=1)
        >>> trend = ma(x)
    """
    
    def __init__(self, kernel_size: int, stride: int = 1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply moving average.
        
        Args:
            x: Input tensor [B, L, F]
            
        Returns:
            Moving averaged tensor [B, L, F]
        """
        # Pad at boundaries to maintain sequence length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block.
    
    Decomposes time series into trend and seasonal components
    using moving average for trend extraction.
    
    Args:
        kernel_size: Kernel size for moving average (default 25)
        
    Example:
        >>> decomp = series_decomp(kernel_size=25)
        >>> seasonal, trend = decomp(x)
    """
    
    def __init__(self, kernel_size: int = 25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose series into seasonal and trend components.
        
        Args:
            x: Input tensor [B, L, F]
            
        Returns:
            (seasonal, trend): Seasonal and trend components [B, L, F]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
