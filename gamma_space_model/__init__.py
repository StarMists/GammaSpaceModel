"""GammaSpaceModel public API."""

__version__ = "0.1.0"

from gamma_space_model.modules import (
    GammaSpaceBlock,
    MinimalGammaSpaceBlock,
    LayerNorm,
    RMSNorm,
    GammaSpaceLayer,
)

try:
    from gamma_space_model.ops import HAS_TILELANG_OPS
except ImportError:
    HAS_TILELANG_OPS = False

__all__ = [
    "GammaSpaceLayer",
    "GammaSpaceBlock",
    "MinimalGammaSpaceBlock",
    "LayerNorm",
    "RMSNorm",
    "HAS_TILELANG_OPS",
]
