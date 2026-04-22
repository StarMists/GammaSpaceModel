"""Small synthetic forecasting demo for Gamma Space Model."""

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gamma_space_model import GammaSpaceBlock


class TinyForecaster(nn.Module):
    def __init__(self, features: int = 2, width: int = 16, hidden_dim: int = 16) -> None:
        super().__init__()
        self.in_proj = nn.Linear(features, width)
        self.block = GammaSpaceBlock(d_model=width, hidden_dim=hidden_dim)
        self.out_proj = nn.Linear(width, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x, _ = self.block(x, return_state=False)
        return self.out_proj(x)


def make_batch(batch: int = 4, seq_len: int = 32, features: int = 2) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, seq_len + 1)
    waves = []
    for idx in range(features):
        freq = 1.0 + idx
        wave = torch.sin(2.0 * math.pi * freq * t) + 0.25 * torch.cos(6.0 * math.pi * t)
        waves.append(wave)
    signal = torch.stack(waves, dim=-1)
    return signal.unsqueeze(0).repeat(batch, 1, 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyForecaster().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    data = make_batch().to(device)
    x, target = data[:, :-1], data[:, 1:]

    for step in range(3):
        pred = model(x)
        loss = loss_fn(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1:02d} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
