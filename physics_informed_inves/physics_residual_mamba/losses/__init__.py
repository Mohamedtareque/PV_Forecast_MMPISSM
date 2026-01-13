"""
Loss functions for solar power forecasting models.

Provides:
- PhysicsPVLoss: Triple-constraint physics-informed loss
"""

from .physics_loss import PhysicsPVLoss

__all__ = [
    'PhysicsPVLoss',
]
