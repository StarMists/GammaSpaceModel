"""
TileLang-accelerated SSM operations.

This module provides GPU-accelerated selective scan operations using TileLang principles,
with Triton optimization and PyTorch fallback.
"""

from .selective_scan import ssm_gamma_forward_tilelang

__all__ = [
    'ssm_gamma_forward_tilelang',
]
