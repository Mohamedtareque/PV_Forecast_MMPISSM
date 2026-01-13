"""
Device management utilities for PyTorch.

Provides:
- get_device: Get appropriate device (CUDA if available, else CPU)
- get_device_info: Print device information
"""

import torch


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get appropriate PyTorch device.
    
    Args:
        use_cuda: Whether to use CUDA if available (default True)
        
    Returns:
        torch.device object
        
    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information:
            - cuda_available: Whether CUDA is available
            - cuda_device_count: Number of CUDA devices
            - cuda_device_name: Name of first CUDA device
            - torch_version: PyTorch version
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'torch_version': torch.__version__
    }
    
    return info


def print_device_info():
    """
    Print device information to console.
    
    Useful for debugging and logging.
    """
    info = get_device_info()
    print("\n=== Device Information ===")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"CUDA Device Name: {info['cuda_device_name']}")
    print("========================\n")


def to_device(data, device: torch.device):
    """
    Move data to specified device.
    
    Args:
        data: Tensor or model to move
        device: Target device
        
    Returns:
        Data moved to device
        
    Example:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> x = to_device(x, device)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    else:
        return data
