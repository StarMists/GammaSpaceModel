"""Normalization layers for Gamma Space Model."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Simpler normalization often used in modern language models.
    Inspired by T5's implementation.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm: x * (weight / RMS(x))"""
        # (batch, seq_len, d_model) or (batch, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * self.weight / rms


class LayerNorm(nn.Module):
    """Standard Layer Normalization (wrapper around torch.nn.LayerNorm)."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
