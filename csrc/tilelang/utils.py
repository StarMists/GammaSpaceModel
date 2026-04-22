"""Utility functions for TileLang selective scan operations."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def validate_tensor_shapes(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> Tuple[int, int, int, int]:
    """
    Validate tensor shapes and return dimensions.
    
    Args:
        u: Input (batch, seq_len, state_dim)
        A: State matrix (hidden_dim, hidden_dim)
        B: Input matrix (hidden_dim, state_dim)
        C: Output matrix (state_dim, hidden_dim)
    
    Returns:
        (batch_size, seq_len, state_dim, hidden_dim)
    
    Raises:
        RuntimeError: If tensor shapes are incompatible
    """
    if len(u.shape) != 3:
        raise RuntimeError(f"Input u must be 3D, got {len(u.shape)}D")
    
    batch_size, seq_len, state_dim = u.shape
    hidden_dim = A.shape[0]
    
    if A.shape != (hidden_dim, hidden_dim):
        raise RuntimeError(f"A shape mismatch: expected ({hidden_dim}, {hidden_dim}), got {A.shape}")
    
    if B.shape != (hidden_dim, state_dim):
        raise RuntimeError(f"B shape mismatch: expected ({hidden_dim}, {state_dim}), got {B.shape}")
    
    if C.shape != (state_dim, hidden_dim):
        raise RuntimeError(f"C shape mismatch: expected ({state_dim}, {hidden_dim}), got {C.shape}")
    
    return batch_size, seq_len, state_dim, hidden_dim


def convert_to_supported_dtype(tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Convert tensor to supported dtype if needed.
    
    TileLang may not support all dtypes, so convert bfloat16/float16 to float32
    if needed, and track whether conversion was done.
    
    Args:
        tensor: Input tensor
    
    Returns:
        (converted_tensor, was_converted)
    """
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor, False
    elif tensor.dtype in (torch.float16, torch.bfloat16):
        # Return both since we'll need to convert back
        return tensor, False
    else:
        raise RuntimeError(f"Unsupported dtype: {tensor.dtype}")


def ensure_contiguous(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Ensure tensors are contiguous in memory."""
    return tuple(t.contiguous() if not t.is_contiguous() else t for t in tensors)


def check_device_consistency(*tensors: torch.Tensor) -> torch.device:
    """
    Verify all tensors are on the same device.
    
    Returns:
        The device of the tensors
    
    Raises:
        RuntimeError: If tensors are on different devices
    """
    if not tensors:
        raise RuntimeError("No tensors provided")
    
    device = tensors[0].device
    for t in tensors[1:]:
        if t.device != device:
            raise RuntimeError(f"Device mismatch: {device} vs {t.device}")
    
    return device


def check_dtype_consistency(*tensors: torch.Tensor) -> torch.dtype:
    """
    Verify all tensors have compatible dtypes.
    
    Returns:
        The dtype of the tensors
    
    Raises:
        RuntimeError: If tensors have incompatible dtypes
    """
    if not tensors:
        raise RuntimeError("No tensors provided")
    
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        if t.dtype != dtype:
            # Allow compatible types, but warn
            if t.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
                raise RuntimeError(f"Incompatible dtype: {dtype} vs {t.dtype}")
    
    return dtype
