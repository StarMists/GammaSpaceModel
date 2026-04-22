# GammaSpaceModel

[![Smoke](https://github.com/StarMists/GammaSpaceModel/actions/workflows/smoke.yml/badge.svg)](https://github.com/StarMists/GammaSpaceModel/actions/workflows/smoke.yml)

GammaSpaceModel is a lightweight PyTorch package for Gamma-structured state
space models with structured stability and full-sequence execution.

The core design keeps a fixed sparse lower-bidiagonal Gamma transition matrix:

```text
A[n, n] = -1
A[n, n-1] = 1
```

This structure uses only `-1`, `0`, and `1`, which makes it friendly to
deployment settings that benefit from sparse or ternary transitions. Instead of
replacing the transition with a dense transition parameterization, this package keeps the
Gamma structure and adds practical structured improvements around it.

## Features

- Fixed ternary-friendly Gamma transition matrix
- Learned positive timestep `dt`
- Euler, bilinear, and ZOH discretization
- Optional direct skip term `D`
- Recurrent token-by-token stepping
- Full-sequence convolutional path for longer sequences
- Inference cache allocation for recurrent use
- Exportable discretized inference matrices
- Residual `GammaSpaceBlock` with input gate, output gate, activation, layer scale,
  dropout, and optional output projection

## Install

Install from GitHub:

```bash
pip install "git+https://github.com/StarMists/GammaSpaceModel.git"
```

For local development:

```bash
git clone https://github.com/StarMists/GammaSpaceModel.git
cd GammaSpaceModel
pip install -e ".[dev]"
```

Optional notebook and performance dependencies:

```bash
pip install -e ".[notebook,performance]"
```

## Quick Start

```python
import torch
from gamma_space_model import GammaSpaceBlock

block = GammaSpaceBlock(d_model=8, hidden_dim=16)

x = torch.randn(2, 32, 8)
y, state = block(x, return_state=False)
print(y.shape)
```

## Core API

```python
from gamma_space_model import (
    GammaSpaceLayer,
    GammaSpaceBlock,
    MinimalGammaSpaceBlock,
    LayerNorm,
    RMSNorm,
)
```

`GammaSpaceLayer` is the core state space layer. `GammaSpaceBlock` wraps it in a
residual sequence block. `MinimalGammaSpaceBlock` keeps the structured
discretization path while removing the richer gating/output pathway.

## Execution Modes

`GammaSpaceLayer` supports three forward modes:

- `kernel_mode="recurrent"`: always run recurrently through the sequence.
- `kernel_mode="conv"`: use the full-sequence causal convolution view.
- `kernel_mode="auto"`: switch to convolution when sequence length reaches
  `kernel_threshold`.

For streaming or autoregressive use, call `step(...)` with a recurrent state:

```python
from gamma_space_model import GammaSpaceLayer

ssm = GammaSpaceLayer(state_dim=8, hidden_dim=16)
state = ssm.init_state(batch_size=2, device=torch.device("cpu"))
cache = ssm.allocate_inference_cache(
    batch_size=2,
    seq_len=1,
    device=torch.device("cpu"),
)

token = torch.randn(2, 8)
out, state = ssm.step(token, state, cache=cache)
```

## Public Release Scope

This repository is focused on the public Gamma Space Model algorithm, implementation,
tests, and small usage examples. Benchmark notebooks, experiment records,
profiling scripts, tuned training recipes, and internal comparison results are
not part of this release.

## Public Validation

The public validation surface is intentionally small and reproducible:

```bash
python -m pytest tests -q
python examples/gamma_space_quickstart.py
python examples/gamma_space_forecasting_demo.py
```

The tests cover core behavior such as recurrent/full-sequence consistency,
cached stepping, exported inference matrices, and public API boundaries. The
examples are tiny generic usage demos, not tuned benchmark recipes.

## Public Benchmarks

See [PUBLIC_BENCHMARKS.md](PUBLIC_BENCHMARKS.md) for a fixed public benchmark
run on complex sinusoid forecasting, permuted sequential MNIST, and copying
memory. The benchmark page includes the full public protocol, hyperparameters,
task configurations, result tables, and a held-out forecasting visualization.

## Repository Layout

```text
gamma_space_model/
|-- modules/
|   |-- gamma_space.py
|   |-- block.py
|   `-- normalization.py
|-- ops/
|   `-- selective_scan_interface.py
examples/
tests/
PUBLIC_BENCHMARKS.md
```

## References

GammaSpaceModel is inspired by the state space model line of work, especially
HiPPO-style sequence memory and selective state-space models:

- Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Re. "HiPPO:
  Recurrent Memory with Optimal Polynomial Projections." NeurIPS 2020.
- Albert Gu, Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces." 2023.

## License

MIT. See [LICENSE](LICENSE).
