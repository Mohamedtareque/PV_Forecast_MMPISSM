"""
Model architectures for solar power forecasting.

Provides:
- MMPISSM_Model_01: Mamba-based model for residual learning
- VanillaLSTM: Standard LSTM baseline
- VanillaMamba: Simplified Mamba baseline
- PhysicsResidualMamba: Hybrid physics-informed architecture
- DifferentiablePVLayer: Learnable physics model
- RevIN: Reversible Instance Normalization
- moving_avg: Moving average block
- series_decomp: Series decomposition block
- FeatureScaler: Normalization statistics management
"""

from .components import RevIN, moving_avg, series_decomp
from .physics_layer import DifferentiablePVLayer, FeatureScaler
from .base_models import MMPISSM_Model_01, VanillaLSTM, VanillaMamba
from .hybrid_model import PhysicsResidualMamba

__all__ = [
    # Helper components
    'RevIN',
    'moving_avg',
    'series_decomp',
    
    # Physics layer
    'DifferentiablePVLayer',
    'FeatureScaler',
    
    # Base models
    'MMPISSM_Model_01',
    'VanillaLSTM',
    'VanillaMamba',
    
    # Hybrid model
    'PhysicsResidualMamba',
]
