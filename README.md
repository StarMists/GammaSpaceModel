# GammaSpaceModel

[![Smoke](https://github.com/StarMists/GammaSpaceModel/actions/workflows/smoke.yml/badge.svg)](https://github.com/StarMists/GammaSpaceModel/actions/workflows/smoke.yml)

GammaSpaceModel is a lightweight PyTorch package for sequence modeling with a
stable diagonal-plus-low-rank state space core. The public API stays compact:
use `GammaSpaceLayer` for the core SSM layer or `GammaSpaceBlock` for a residual
sequence block.

The current implementation uses:

- learned stable diagonal dynamics
- fixed ternary sign masks for the low-rank factors
- learned low-rank magnitudes
- learned positive timestep `dt`
- optional direct skip term `D`
- recurrent token-by-token stepping
- full-sequence FFT execution
- inference cache allocation for streaming use
- exportable inference matrices

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

Optional notebook dependencies:

```bash
pip install -e ".[notebook]"
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

`GammaSpaceLayer` is the DPLR-backed state space layer. `GammaSpaceBlock` wraps
it in a residual sequence block with normalization, gating, activation, dropout,
and output projection. `MinimalGammaSpaceBlock` keeps the same state transition
while removing the richer gating/output pathway.

## Execution Modes

`GammaSpaceLayer` supports three forward modes:

- `kernel_mode="recurrent"`: always run recurrently through the sequence.
- `kernel_mode="conv"`: use the full-sequence FFT path.
- `kernel_mode="auto"`: switch to FFT execution when sequence length reaches
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

This repository is focused on the public Gamma Space Model implementation,
tests, and small usage examples. Internal ablations, tuned training recipes,
private experiment records, and private comparison notebooks are not part of
this release.

## Public Validation

The public validation surface is intentionally small and reproducible:

```bash
python -m pytest tests -q
python examples/gamma_space_quickstart.py
python examples/gamma_space_forecasting_demo.py
```

The tests cover recurrent/full-sequence consistency, cached stepping, exported
inference matrices, and public API boundaries. The examples are tiny generic
usage demos, not tuned benchmark recipes.

## Benchmarks

The previous public benchmark page was generated for an older Gamma Space Model
core. It has been archived to avoid mixing old results with the current DPLR
implementation. See [PUBLIC_BENCHMARKS.md](PUBLIC_BENCHMARKS.md).

## Repository Layout

```text
gamma_space_model/
|-- modules/
|   |-- gamma_space.py
|   |-- block.py
|   `-- normalization.py
examples/
tests/
PUBLIC_BENCHMARKS.md
```

## References

GammaSpaceModel is inspired by the broader state space model line of work,
including diagonal and low-rank sequence models.

## License

MIT. See [LICENSE](LICENSE).
