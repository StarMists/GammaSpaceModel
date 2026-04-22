"""Minimal Gamma Space Model block example."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gamma_space_model import GammaSpaceBlock


def main() -> None:
    block = GammaSpaceBlock(d_model=8, hidden_dim=16)

    x = torch.randn(2, 32, 8)
    y, state = block(x)

    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("state:", tuple(state.shape))


if __name__ == "__main__":
    main()
