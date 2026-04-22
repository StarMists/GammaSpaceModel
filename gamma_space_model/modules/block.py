"""structured Gamma block with residual mixing and optional gating."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import LayerNorm
from .gamma_space import GammaSpaceLayer


class GammaSpaceBlock(nn.Module):
    """
    Residual Gamma block that keeps the gamma transition structure but adopts a
    more expressive post-SSM pathway inspired by state-space blocks.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: float = 1e-2,
        discretization: str = "bilinear",
        prenorm: bool = True,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        gate: bool = True,
        use_D: bool = True,
        kernel_mode: str = "auto",
        kernel_threshold: int = 64,
        use_output_linear: bool = True,
        gate_bias: float = 2.0,
        input_gate: bool = True,
        input_gate_bias: float = 2.0,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.prenorm = prenorm
        self.residual_scale = residual_scale
        self.gate = gate
        self.input_gate = input_gate

        self.norm = LayerNorm(d_model)
        self.ssm = GammaSpaceLayer(
            state_dim=d_model,
            hidden_dim=hidden_dim,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            discretization=discretization,
            use_D=use_D,
            kernel_mode=kernel_mode,
            kernel_threshold=kernel_threshold,
        )

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation in {"identity", "linear", None}:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

        self.output_linear = nn.Linear(d_model, d_model) if use_output_linear else nn.Identity()
        self.input_gate_linear = nn.Linear(d_model, d_model) if input_gate else None
        self.gate_linear = nn.Linear(d_model, d_model) if gate else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self._reset_parameters(gate_bias=gate_bias, input_gate_bias=input_gate_bias)

    def _reset_parameters(self, gate_bias: float, input_gate_bias: float) -> None:
        if isinstance(self.output_linear, nn.Linear):
            nn.init.eye_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)
        if self.input_gate_linear is not None:
            nn.init.zeros_(self.input_gate_linear.weight)
            nn.init.constant_(self.input_gate_linear.bias, input_gate_bias)
        if self.gate_linear is not None:
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.constant_(self.gate_linear.bias, gate_bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        rate: float = 1.0,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        core_in = self.norm(x) if self.prenorm else x
        ssm_in = core_in
        if self.input_gate_linear is not None:
            ssm_in = ssm_in * torch.sigmoid(self.input_gate_linear(core_in))
        ssm_out, final_state = self.ssm(ssm_in, mask=mask, state=state, rate=rate, return_state=return_state)
        ssm_out = self.activation(ssm_out)
        if self.gate_linear is not None:
            ssm_out = ssm_out * torch.sigmoid(self.gate_linear(core_in))
        ssm_out = self.output_linear(ssm_out)
        ssm_out = ssm_out * self.layer_scale
        ssm_out = self.dropout(ssm_out)
        output = residual * self.residual_scale + ssm_out
        if not self.prenorm:
            output = self.norm(output)
        return output, final_state

    def step(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        rate: float = 1.0,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = u
        core_in = self.norm(u) if self.prenorm else u
        lightweight = cache is not None and cache.get("lightweight", False)
        ssm_in = core_in
        if self.input_gate_linear is not None:
            if cache is not None and "static_input_gate" in cache:
                ssm_in = ssm_in * cache["static_input_gate"]
            elif cache is not None and "input_gate_weight" in cache:
                input_gate = F.linear(
                    core_in,
                    cache["input_gate_weight"],
                    cache["input_gate_bias"],
                )
                ssm_in = ssm_in * torch.sigmoid(input_gate)
            elif not lightweight:
                ssm_in = ssm_in * torch.sigmoid(self.input_gate_linear(core_in))
        ssm_out, h_new = self.ssm.step(ssm_in, h, rate=rate, cache=cache)
        ssm_out = self.activation(ssm_out)
        if cache is not None and "layer_scale" in cache:
            layer_scale = cache["layer_scale"]
        else:
            layer_scale = self.layer_scale.to(device=u.device, dtype=u.dtype)
        if self.gate_linear is not None and not lightweight:
            if cache is not None and "static_gate" in cache:
                ssm_out = ssm_out * cache["static_gate"]
            elif cache is not None and "gate_weight" in cache:
                gate = F.linear(
                    core_in,
                    cache["gate_weight"],
                    cache["gate_bias"],
                )
                ssm_out = ssm_out * torch.sigmoid(gate)
            else:
                gate = self.gate_linear(core_in)
                ssm_out = ssm_out * torch.sigmoid(gate)
        if cache is not None and "output_weight" in cache:
            ssm_out = F.linear(
                ssm_out,
                cache["output_weight"],
                cache["output_bias"],
            )
        elif lightweight:
            pass
        else:
            ssm_out = self.output_linear(ssm_out)
        ssm_out = ssm_out * layer_scale
        ssm_out = self.dropout(ssm_out)
        output = residual * self.residual_scale + ssm_out
        if not self.prenorm:
            output = self.norm(output)
        return output, h_new

    def allocate_inference_cache(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        rate: float = 1.0,
        lightweight: bool = False,
    ) -> dict:
        cache = self.ssm.allocate_inference_cache(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
            rate=rate,
        )
        if dtype is None:
            dtype = self.layer_scale.dtype
        cache["layer_scale"] = self.layer_scale.to(device=device, dtype=dtype)
        cache["lightweight"] = lightweight
        cache["deployment_mode"] = "lite" if lightweight else "full"
        if self.input_gate_linear is not None and not lightweight:
            cache["input_gate_weight"] = self.input_gate_linear.weight.to(device=device, dtype=dtype)
            cache["input_gate_bias"] = self.input_gate_linear.bias.to(device=device, dtype=dtype)
        if self.gate_linear is not None and not lightweight:
            cache["gate_weight"] = self.gate_linear.weight.to(device=device, dtype=dtype)
            cache["gate_bias"] = self.gate_linear.bias.to(device=device, dtype=dtype)
        if isinstance(self.output_linear, nn.Linear) and not lightweight:
            cache["output_weight"] = self.output_linear.weight.to(device=device, dtype=dtype)
            cache["output_bias"] = self.output_linear.bias.to(device=device, dtype=dtype)
        return cache

    def allocate_deployment_cache(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        rate: float = 1.0,
    ) -> dict:
        return self.allocate_inference_cache(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
            rate=rate,
            lightweight=True,
        )

    def allocate_balanced_deployment_cache(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        rate: float = 1.0,
    ) -> dict:
        """
        Allocate a middle-ground recurrent cache for deployment experiments.

        This keeps the trained output projection but replaces the input-dependent
        gate with a static gate from the learned bias. It is meant to test a
        speed/fidelity point between full recurrent inference and the aggressive
        deployment-lite path.
        """

        cache = self.allocate_inference_cache(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
            rate=rate,
            lightweight=False,
        )
        if dtype is None:
            dtype = self.layer_scale.dtype
        cache["deployment_mode"] = "balanced"
        if self.gate_linear is not None:
            cache.pop("gate_weight", None)
            gate_bias = self.gate_linear.bias.to(device=device, dtype=dtype)
            cache["static_gate"] = torch.sigmoid(gate_bias).view(1, -1)
        if self.input_gate_linear is not None:
            cache.pop("input_gate_weight", None)
            input_gate_bias = self.input_gate_linear.bias.to(device=device, dtype=dtype)
            cache["static_input_gate"] = torch.sigmoid(input_gate_bias).view(1, -1)
        return cache


class MinimalGammaSpaceBlock(GammaSpaceBlock):
    """A lighter enhanced gamma block focused on optimization stability."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: float = 1e-2,
        discretization: str = "bilinear",
        prenorm: bool = True,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
        use_D: bool = True,
        kernel_mode: str = "auto",
        kernel_threshold: int = 64,
    ) -> None:
        super().__init__(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            discretization=discretization,
            prenorm=prenorm,
            residual_scale=residual_scale,
            dropout=dropout,
            activation="identity",
            gate=False,
            use_D=use_D,
            kernel_mode=kernel_mode,
            kernel_threshold=kernel_threshold,
            use_output_linear=False,
            input_gate=False,
            layer_scale_init=1.0,
        )
