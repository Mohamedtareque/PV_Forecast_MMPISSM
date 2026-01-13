"""
Configuration management for Physics-Residual Power Mamba models.

Provides configuration classes for:
- Base Mamba model (no physics)
- Physics-Residual Mamba model (hybrid architecture)
- Hyperparameter definitions and search grids
"""

from typing import List, Dict, Optional


class BaseMambaConfigs:
    """
    Configuration for baseline Power Mamba model (no physics).
    
    Standard Mamba architecture with RevIN normalization and
    series decomposition, but without physics-informed components.
    
    Attributes:
        seq_len: Past context length (timesteps)
        pred_len: Prediction horizon (timesteps)
        PAST_INPUT_COLS: List of past feature column names
        FUTURE_INPUT_COLS: List of future feature column names
        TARGET_COL: List of target column names
        common_features: Features common to past and future
        num_shadows: Number of shadow features
        enc_in: Total input dimension (past + shadows)
        c_out: Output dimension (number of targets)
        
        # Model Architecture
        kernel_size: Moving average kernel size
        n_embed: Mamba embedding dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel size
        e_fact: Mamba expansion factor
        dropout: Dropout rate
        
        # Training Configuration
        n_splits: Number of cross-validation folds
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: PyTorch device (cuda/cpu)
    """
    
    def __init__(self):
        # Sequence Configuration
        self.seq_len = 672  # ~7 days at 15-min intervals
        self.pred_len = 96  # 24 hours at 15-min intervals
        
        # Target
        self.TARGET_COL = ['power']
        
        # Past input columns (all available features)
        self.PAST_INPUT_COLS = [
            'lmd_totalirrad',
            'lmd_diffuseirrad',
            'lmd_temperature',
            'lmd_pressure',
            'nwp_globalirrad',
            'nwp_temperature',
            'nwp_windspeed',
            'power',
            'P_CLR',
            'K_PV',
            'NWP_Power_MW',
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'month_sin',
            'month_cos',
            'season_sin',
            'season_cos'
        ]
        
        # Future input columns (available at inference time)
        self.FUTURE_INPUT_COLS = [
            'nwp_globalirrad',
            'nwp_temperature',
            'nwp_windspeed'
        ]
        
        # Common features (shadow features) - recalculate derived attributes
        self._recalculate_derived_attrs()
        
        # Model Architecture
        self.kernel_size = 25
        self.n_embed = 128
        self.d_state = 64
        self.dconv = 2
        self.e_fact = 2
        self.dropout = 0.2
        
        # Training Configuration
        self.n_splits = 4
        self.batch_size = 64
        self.epochs = 50
        self.device = None  # Will be set to cuda if available

    def _recalculate_derived_attrs(self):
        """Recalculate derived attributes based on current feature lists."""
        self.common_features = [f for f in self.FUTURE_INPUT_COLS if f in self.PAST_INPUT_COLS]
        self.num_shadows = len(self.common_features)
        self.include_pred = 1 if len(self.FUTURE_INPUT_COLS) > 0 else 0
        self.enc_in = len(self.PAST_INPUT_COLS) + self.num_shadows
        self.c_out = len(self.TARGET_COL)

    def update_feature_columns(self, past_cols: List[str], future_cols: List[str]):
        """
        Update feature columns and recalculate derived attributes.
        
        This method should be called after modifying PAST_INPUT_COLS or FUTURE_INPUT_COLS
        to ensure that derived attributes like common_features, num_shadows, and enc_in
        are correctly recalculated.
        
        Args:
            past_cols: New list of past input column names
            future_cols: New list of future input column names
        """
        self.PAST_INPUT_COLS = past_cols
        self.FUTURE_INPUT_COLS = future_cols
        self._recalculate_derived_attrs()


class PhysicsResidualMambaConfigs:
    """
    Configuration for Physics-Residual Mamba hybrid model.
    
    Combines differentiable physics layer with Mamba-based residual learner.
    
    Inherits all BaseMambaConfigs attributes and adds physics-specific
    configuration.
    
    Attributes:
        seq_len: Past context length (timesteps)
        pred_len: Prediction horizon (timesteps)
        PAST_INPUT_COLS: List of past feature column names
        FUTURE_INPUT_COLS: List of future feature column names
        TARGET_COL: List of target column names
        common_features: Features common to past and future
        num_shadows: Number of shadow features
        enc_in: Total input dimension (past + shadows)
        c_out: Output dimension (number of targets)
        
        # Model Architecture (inherited)
        kernel_size: Moving average kernel size
        n_embed: Mamba embedding dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel size
        e_fact: Mamba expansion factor
        dropout: Dropout rate
        
        # Training Configuration (inherited)
        n_splits: Number of cross-validation folds
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: PyTorch device (cuda/cpu)
        
        # Physics Configuration
        T_ref: Reference temperature for PV model (°C)
        G_night_thr: Irradiance threshold for night detection (W/m²)
        
        # Physics Loss Weights
        lambda_data: Weight for data fit loss (default 1.0)
        lambda_night: Weight for night-time penalty (default 0.2)
        lambda_mono: Weight for monotonicity regularization (default 0.1)
        
        # Physics Layer Parameters (initial values)
        eta_init: Initial efficiency (default 0.15)
        P_stc_init: Initial STC power in MW (default 18.0)
        gamma_init: Initial temperature coefficient (default -5.4)
        U0_init: Initial heat transfer coefficient (default 3.2)
        U1_init: Initial wind cooling coefficient (default 1.9)
    """
    
    def __init__(self, n_splits: int = 4):
        # Inherit all BaseMambaConfigs attributes
        self.seq_len = 672
        self.pred_len = 96
        self.TARGET_COL = ['power']
        self.PAST_INPUT_COLS = [
            'lmd_totalirrad',
            'lmd_diffuseirrad',
            'lmd_temperature',
            'lmd_pressure',
            'nwp_globalirrad',
            'nwp_temperature',
            'nwp_windspeed',
            'power',
            'P_CLR',
            'K_PV',
            'NWP_Power_MW',
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'month_sin',
            'month_cos',
            'season_sin',
            'season_cos'
        ]
        self.FUTURE_INPUT_COLS = [
            'nwp_globalirrad',
            'nwp_temperature',
            'nwp_windspeed'
        ]
        self._recalculate_derived_attrs()
        
        # Model Architecture
        self.kernel_size = 25
        self.n_embed = 128
        self.d_state = 64
        self.dconv = 2
        self.e_fact = 2
        self.dropout = 0.2
        
        # Training Configuration
        self.n_splits = n_splits
        self.batch_size = 64
        self.epochs = 30
        self.device = None  # Will be set to cuda if available
        
        # Physics Configuration
        self.T_ref = 25.0  # Reference temperature (°C)
        self.G_night_thr = 10.0  # Night irradiance threshold (W/m²)
        
        # Physics Loss Weights
        self.lambda_data = 1.0
        self.lambda_night = 0.2
        self.lambda_mono = 0.1
    
    def _recalculate_derived_attrs(self):
        """Recalculate derived attributes based on current feature lists."""
        self.common_features = [f for f in self.FUTURE_INPUT_COLS if f in self.PAST_INPUT_COLS]
        self.num_shadows = len(self.common_features)
        self.include_pred = 1 if len(self.FUTURE_INPUT_COLS) > 0 else 0
        self.enc_in = len(self.PAST_INPUT_COLS) + self.num_shadows
        self.c_out = len(self.TARGET_COL)

    def update_feature_columns(self, past_cols: List[str], future_cols: List[str]):
        """
        Update feature columns and recalculate derived attributes.
        
        This method should be called after modifying PAST_INPUT_COLS or FUTURE_INPUT_COLS
        to ensure that derived attributes like common_features, num_shadows, and enc_in
        are correctly recalculated.
        
        Args:
            past_cols: New list of past input column names
            future_cols: New list of future input column names
        """
        self.PAST_INPUT_COLS = past_cols
        self.FUTURE_INPUT_COLS = future_cols
        self._recalculate_derived_attrs()

    def to_dict(self) -> Dict[str, any]:
        """Convert configuration to dictionary for logging."""
        return {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_splits': self.n_splits,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'kernel_size': self.kernel_size,
            'n_embed': self.n_embed,
            'd_state': self.d_state,
            'dconv': self.dconv,
            'e_fact': self.e_fact,
            'dropout': self.dropout,
            'T_ref': self.T_ref,
            'G_night_thr': self.G_night_thr,
            'lambda_data': self.lambda_data,
            'lambda_night': self.lambda_night,
            'lambda_mono': self.lambda_mono,
            'device': str(self.device)
        }
