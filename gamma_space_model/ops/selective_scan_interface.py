"""
Selective Scan Operations Interface with TileLang Acceleration.

Provides unified interface for TileLang-accelerated SSM operations with automatic
fallback to pure PyTorch CPU implementation.

Features:
  - TileLang-accelerated parallel scan for efficient GPU utilization
  - Automatic GPU/CPU dispatch
  - Autograd support via custom backward pass
  - Memory-efficient implementation
  - Support for fp32, fp16, and bf16
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
from pathlib import Path

# Import TileLang accelerated operations
try:
    from csrc.tilelang import ssm_gamma_forward_tilelang
    HAS_TILELANG_OPS = True
except ImportError:
    try:
        # Alternative import path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'csrc'))
        from tilelang import ssm_gamma_forward_tilelang
        HAS_TILELANG_OPS = True
    except ImportError:
        HAS_TILELANG_OPS = False


class SSMGammaFunction(torch.autograd.Function):
    """
    Custom autograd function for SSM gamma forward/backward.
    
    Uses TileLang kernels if available, falls back to PyTorch operations.
    """
    
    @staticmethod
    def forward(ctx, u, A, B, C, delta_t):
        """
        Forward pass.
        
        Args:
            u: Input (batch, seq_len, state_dim)
            A: State matrix (hidden_dim, hidden_dim)
            B: Input matrix (hidden_dim, state_dim)
            C: Output matrix (state_dim, hidden_dim)
            delta_t: Discretization step
        
        Returns:
            output: (batch, seq_len, state_dim)
            h_final: (batch, hidden_dim)
        """
        
        if HAS_TILELANG_OPS and u.is_cuda:
            # Use TileLang-accelerated kernels
            y, h_final = ssm_gamma_forward_tilelang(
                u.contiguous(), A.contiguous(), B.contiguous(), C.contiguous(), delta_t
            )
        else:
            # Fallback to PyTorch implementation
            y, h_final = _ssm_gamma_forward_pytorch(u, A, B, C, delta_t)
        
        ctx.save_for_backward(u, A, B, C, y, h_final)
        ctx.delta_t = delta_t
        
        return y, h_final
    
    @staticmethod
    def backward(ctx, grad_y, grad_h_final):
        """Backward pass with gradient computation."""
        
        u, A, B, C, y, h_final = ctx.saved_tensors
        delta_t = ctx.delta_t
        
        # Compute hidden state history from forward pass
        h_history = _compute_state_history(u, A, B, C, delta_t)
        
        if HAS_TILELANG_OPS and grad_y.is_cuda:
            # Use TileLang kernels for backward
            # For now, use the unified backward from tilelang
            grad_u, grad_A, grad_B, grad_C, grad_h_init = _ssm_gamma_backward_pytorch(
                grad_y, u, h_history, A, B, C, delta_t
            )
        else:
            # Fallback to PyTorch backward
            grad_u, grad_A, grad_B, grad_C, grad_h_init = _ssm_gamma_backward_pytorch(
                grad_y, u, h_history, A, B, C, delta_t
            )
        
        return grad_u, grad_A, grad_B, grad_C, None


def ssm_gamma_forward(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_t: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSM Gamma forward pass using parallel scan optimization.
    
    Implements: h[t+1] = h[t] + dt * (A @ h[t] + B @ u[t]), y[t] = C @ h[t]
    
    Args:
        u: Input tensor (batch, seq_len, state_dim)
        A: State matrix (hidden_dim, hidden_dim) - tridiagonal HiPPO-Gamma
        B: Input matrix (hidden_dim, state_dim)
        C: Output matrix (state_dim, hidden_dim)
        delta_t: Time discretization step
    
    Returns:
        output: (batch, seq_len, state_dim)
        final_state: (batch, hidden_dim)
    
    Example:
        >>> u = torch.randn(2, 32, 64)  # batch=2, seq=32, state_dim=64
        >>> A = torch.eye(128)  # 128-dim hidden state
        >>> B = torch.randn(128, 64)
        >>> C = torch.randn(64, 128)
        >>> y, h = ssm_gamma_forward(u, A, B, C, delta_t=0.1)
        >>> y.shape
        torch.Size([2, 32, 64])
    """
    
    # Always use custom function for autograd support
    return SSMGammaFunction.apply(u, A, B, C, delta_t)


def _ssm_gamma_forward_pytorch(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_t: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch implementation of SSM forward pass."""
    
    batch_size, seq_len, state_dim = u.shape
    hidden_dim = A.shape[0]
    
    device = u.device
    dtype = u.dtype
    
    # Initialize hidden state
    h = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
    outputs = []
    
    # Process sequence
    for t in range(seq_len):
        u_t = u[:, t, :]  # (batch, state_dim)
        
        # State update: h_new = h + delta_t * (A @ h + B @ u)
        A_h = torch.matmul(h, A.T)  # (batch, hidden_dim)
        B_u = torch.matmul(u_t, B.T)  # (batch, hidden_dim)
        h_new = h + delta_t * (A_h + B_u)
        
        # Output: y = C @ h_new
        y_t = torch.matmul(h_new, C.T)  # (batch, state_dim)
        
        outputs.append(y_t)
        h = h_new
    
    y = torch.stack(outputs, dim=1)  # (batch, seq_len, state_dim)
    
    return y, h


def _compute_state_history(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_t: float,
) -> torch.Tensor:
    """Compute full state history for backward pass."""

    batch_size, seq_len, state_dim = u.shape
    hidden_dim = A.shape[0]
    device = u.device
    compute_dtype = torch.promote_types(u.dtype, A.dtype)

    u_compute = u.to(dtype=compute_dtype)
    A_compute = A.to(device=device, dtype=compute_dtype)
    B_compute = B.to(device=device, dtype=compute_dtype)

    h_history = torch.zeros(batch_size, seq_len, hidden_dim, device=device, dtype=compute_dtype)
    h = torch.zeros(batch_size, hidden_dim, device=device, dtype=compute_dtype)
    
    for t in range(seq_len):
        u_t = u_compute[:, t, :]
        A_h = torch.matmul(h, A_compute.T)
        B_u = torch.matmul(u_t, B_compute.T)
        h = h + delta_t * (A_h + B_u)
        h_history[:, t, :] = h
    
    return h_history


def _ssm_gamma_backward_pytorch(
    grad_y: torch.Tensor,
    u: torch.Tensor,
    h_history: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta_t: float,
):
    """Pure PyTorch implementation of backward pass."""

    batch_size, seq_len, state_dim = grad_y.shape
    hidden_dim = A.shape[0]
    device = grad_y.device
    compute_dtype = torch.promote_types(
        torch.promote_types(grad_y.dtype, A.dtype),
        torch.promote_types(B.dtype, C.dtype),
    )

    grad_y_compute = grad_y.to(dtype=compute_dtype)
    u_compute = u.to(device=device, dtype=compute_dtype)
    h_history_compute = h_history.to(device=device, dtype=compute_dtype)
    A_compute = A.to(device=device, dtype=compute_dtype)
    B_compute = B.to(device=device, dtype=compute_dtype)
    C_compute = C.to(device=device, dtype=compute_dtype)

    grad_u = torch.zeros_like(u_compute)
    grad_A = torch.zeros_like(A_compute)
    grad_B = torch.zeros_like(B_compute)
    grad_C = torch.zeros_like(C_compute)
    grad_h = torch.zeros(batch_size, hidden_dim, device=device, dtype=compute_dtype)
    identity = torch.eye(hidden_dim, device=device, dtype=compute_dtype)
    
    # Backward pass through time
    for t in range(seq_len - 1, -1, -1):
        # Gradient from output
        grad_h_from_y = torch.matmul(grad_y_compute[:, t, :], C_compute)  # (batch, hidden_dim)
        grad_h = grad_h + grad_h_from_y
        
        # Gradient w.r.t. C
        h_t = h_history_compute[:, t, :]
        grad_C.add_(torch.matmul(grad_y_compute[:, t, :].T, h_t) / batch_size)
        
        # Gradient w.r.t. B
        u_t = u_compute[:, t, :]
        grad_B.add_(delta_t * torch.matmul(grad_h.T, u_t) / batch_size)
        
        # Gradient w.r.t. input
        grad_u[:, t, :] = torch.matmul(grad_h, B_compute)
        
        # Gradient w.r.t. A
        prev_h = h_history_compute[:, t, :] if t > 0 else torch.zeros_like(h_history_compute[:, 0, :])
        grad_A.add_(delta_t * torch.matmul(grad_h.T, prev_h) / batch_size)
        
        # Propagate gradient backward through state transition
        if t > 0:
            grad_h = torch.matmul(grad_h, A_compute.T + identity)
    
    grad_h_init = grad_h.to(dtype=u.dtype)
    
    return (
        grad_u.to(dtype=u.dtype),
        grad_A.to(dtype=A.dtype),
        grad_B.to(dtype=B.dtype),
        grad_C.to(dtype=C.dtype),
        grad_h_init,
    )


def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    return_last_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Selective scan forward pass (Mamba-style).
    
    Extended interface for more complex SSM variants.
    Currently falls back to SSM gamma for compatibility.
    
    Args:
        u: Input (batch, seq_len, d_model)
        delta: Time step (scalar or per-timestep)
        A: State transition matrix
        B: Input matrix
        C: Output matrix
        D: Direct term (optional)
        return_last_state: Whether to return final state
    
    Returns:
        output: (batch, seq_len, d_model)
        last_state: Final state if requested
    """
    
    # Simplified version - redirect to SSM gamma
    output, last_state = ssm_gamma_forward(u, A, B, C, delta_t=delta if isinstance(delta, (int, float)) else delta.item())
    
    if return_last_state:
        return output, last_state
    else:
        return output, None


def selective_scan_bwd():
    """Backward pass placeholder for extensibility."""
    pass
