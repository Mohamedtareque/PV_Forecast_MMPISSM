"""
Random seed management for reproducibility.

Provides:
- set_random_seed: Set random seeds for all libraries
"""

import random
import numpy as np
import torch


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Ensures that experiments can be reproduced by setting the same seed.
    This is critical for scientific research and debugging.
    
    Args:
        seed: Random seed value (default 42)
        
    Example:
        >>> set_random_seed(42)
        >>> # Now all random operations will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_seed() -> int:
    """
    Get a random seed value.
    
    Useful for generating random seeds for multiple runs.
    
    Returns:
        Random integer seed
        
    Example:
        >>> seed = get_random_seed()
        >>> set_random_seed(seed)
    """
    return random.randint(0, 2**31 - 1)
