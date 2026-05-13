"""DPLR-backed Gamma Space Model layer."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaSpaceLayer(nn.Module):
    """
    Core Gamma Space Model layer using a stable diagonal-plus-low-rank state
    transition.

    The transition has a learned negative diagonal component plus fixed ternary
    sign masks for the low-rank factors. This keeps the public Gamma Space Model
    API compact while moving the implementation to the current DPLR SSM core.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        rank: int = 1,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: float = 1e-2,
        learn_dt: bool = True,
        use_D: bool = True,
        kernel_mode: str = "auto",
        kernel_threshold: int = 64,
        max_low_rank_scale: float = 0.1,
        discretization: Optional[str] = None,
    ) -> None:
        super().__init__()
        del discretization
        if rank < 1:
            raise ValueError("Expected rank >= 1.")
        if kernel_mode not in {"auto", "recurrent", "conv"}:
            raise ValueError(
                f"Unsupported kernel_mode '{kernel_mode}'. "
                "Expected one of {'auto', 'recurrent', 'conv'}."
            )
        if not (0.0 < dt_min <= dt_init <= dt_max):
            raise ValueError("Expected 0 < dt_min <= dt_init <= dt_max.")

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.learn_dt = learn_dt
        self.use_D = use_D
        self.kernel_mode = kernel_mode
        self.kernel_threshold = kernel_threshold
        self.max_low_rank_scale = max_low_rank_scale

        real_init = torch.linspace(0.25, 1.25, hidden_dim, dtype=torch.float32)
        self.log_lambda_real = nn.Parameter(torch.log(real_init))

        self.register_buffer("ternary_u_mask", self._build_ternary_mask(hidden_dim, rank, phase=0))
        self.register_buffer("ternary_v_mask", self._build_ternary_mask(hidden_dim, rank, phase=1))
        self.log_u_amp = nn.Parameter(torch.full((rank, hidden_dim), -2.0, dtype=torch.float32))
        self.log_v_amp = nn.Parameter(torch.full((rank, hidden_dim), -2.0, dtype=torch.float32))
        self.low_rank_logit = nn.Parameter(torch.full((rank,), -2.0, dtype=torch.float32))

        scale = hidden_dim ** -0.5
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim) * scale)
        self.C = nn.Parameter(torch.randn(state_dim, hidden_dim) * scale)
        if use_D:
            self.D = nn.Parameter(torch.ones(state_dim))
        else:
            self.register_buffer("D", torch.zeros(state_dim, dtype=torch.float32))

        dt_init_tensor = torch.tensor(float(dt_init), dtype=torch.float32)
        inv_softplus_dt = torch.log(torch.expm1(dt_init_tensor))
        if learn_dt:
            self.log_dt = nn.Parameter(inv_softplus_dt)
        else:
            self.register_buffer("log_dt", inv_softplus_dt)

        self._kernel_cache: Dict[Tuple[object, ...], torch.Tensor] = {}

    @staticmethod
    def _build_ternary_mask(hidden_dim: int, rank: int, phase: int) -> torch.Tensor:
        base = torch.arange(hidden_dim, dtype=torch.int64)
        masks = []
        for r in range(rank):
            shifted = (base + phase + r) % 3
            mask = torch.zeros(hidden_dim, dtype=torch.float32)
            mask[shifted == 0] = 1.0
            mask[shifted == 2] = -1.0
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def clear_kernel_cache(self) -> None:
        self._kernel_cache.clear()

    def _kernel_cache_key(
        self,
        seq_len: int,
        fft_len: int,
        rate: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[object, ...]:
        return (
            seq_len,
            fft_len,
            float(rate),
            dtype,
            device.type,
            device.index,
            self.log_lambda_real._version,
            self.log_u_amp._version,
            self.log_v_amp._version,
            self.low_rank_logit._version,
            self.B._version,
            self.C._version,
            self.D._version,
            self.log_dt._version,
        )

    def _get_dt(self, rate: float = 1.0, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        dt = F.softplus(self.log_dt)
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max) * rate
        if dtype is not None:
            dt = dt.to(dtype=dtype)
        return dt

    def _discrete_params(
        self,
        rate: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = dtype or self.B.dtype
        target_device = device or self.B.device

        dt = self._get_dt(rate=rate, dtype=target_dtype).to(device=target_device, dtype=target_dtype)
        lambda_cont = -F.softplus(self.log_lambda_real).to(device=target_device, dtype=target_dtype)
        diag = torch.exp(dt * lambda_cont)

        numerator = diag - 1.0
        safe_denom = torch.where(lambda_cont.abs() > 1e-6, lambda_cont, torch.full_like(lambda_cont, -1.0))
        b_scale = torch.where(lambda_cont.abs() > 1e-6, numerator / safe_denom, dt * torch.ones_like(lambda_cont))
        B_disc = b_scale.unsqueeze(-1) * self.B.to(device=target_device, dtype=target_dtype)

        scale = self.max_low_rank_scale * torch.sigmoid(self.low_rank_logit).to(
            device=target_device,
            dtype=target_dtype,
        )
        u_amp = F.softplus(self.log_u_amp).to(device=target_device, dtype=target_dtype)
        v_amp = F.softplus(self.log_v_amp).to(device=target_device, dtype=target_dtype)
        U = (
            self.ternary_u_mask.to(device=target_device, dtype=target_dtype)
            * u_amp
            * scale.unsqueeze(-1)
        ).transpose(0, 1)
        V = (
            self.ternary_v_mask.to(device=target_device, dtype=target_dtype) * v_amp
        ).transpose(0, 1)

        return diag, U, V, B_disc

    def _dense_discrete_A(
        self,
        rate: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        diag, U, V, _ = self._discrete_params(rate=rate, dtype=dtype, device=device)
        return torch.diag(diag) - torch.matmul(U, V.transpose(0, 1))

    def _compute_frequency_response(
        self,
        seq_len: int,
        fft_len: int,
        rate: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        target_dtype = dtype or self.B.dtype
        target_device = device or self.B.device
        cache_key = self._kernel_cache_key(seq_len, fft_len, rate, target_dtype, target_device)
        if use_cache and cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        diag, U, V, B_disc = self._discrete_params(rate=rate, dtype=target_dtype, device=target_device)
        A_dense = self._dense_discrete_A(rate=rate, dtype=target_dtype, device=target_device)
        C = self.C.to(device=target_device, dtype=target_dtype)
        D = self.D.to(device=target_device, dtype=target_dtype)
        identity = torch.eye(self.hidden_dim, device=target_device, dtype=target_dtype)
        A_power = torch.linalg.matrix_power(A_dense, seq_len)

        complex_dtype = torch.complex64 if target_dtype != torch.float64 else torch.complex128
        freq_count = fft_len // 2 + 1
        theta = 2.0 * math.pi * torch.arange(freq_count, device=target_device, dtype=target_dtype) / float(fft_len)
        roots = torch.polar(torch.ones_like(theta), -theta).to(dtype=complex_dtype)

        diag_complex = diag.to(dtype=complex_dtype)
        U_complex = U.to(dtype=complex_dtype)
        V_complex = V.to(dtype=complex_dtype)
        B_complex = B_disc.to(dtype=complex_dtype)
        denom = 1.0 - roots[:, None] * diag_complex[None, :]
        inv_diag = denom.reciprocal()

        inv_b = inv_diag[:, :, None] * B_complex[None, :, :]
        omega_u = roots[:, None, None] * U_complex[None, :, :]
        inv_u = inv_diag[:, :, None] * omega_u

        vt_inv_u = torch.einsum("nr,fns->frs", V_complex, inv_u)
        vt_inv_b = torch.einsum("nr,fnd->frd", V_complex, inv_b)
        rank_eye = torch.eye(self.rank, device=target_device, dtype=complex_dtype).expand(freq_count, -1, -1)
        middle = torch.linalg.inv(rank_eye + vt_inv_u)
        correction = torch.einsum("fns,frs,frd->fnd", inv_u, middle, vt_inv_b)
        response = inv_b - correction

        finite_correction = identity.to(dtype=complex_dtype).unsqueeze(0) - (
            roots.pow(seq_len).view(freq_count, 1, 1) * A_power.to(dtype=complex_dtype).unsqueeze(0)
        )
        c_term = torch.matmul(C.to(dtype=complex_dtype).unsqueeze(0), finite_correction)
        transfer = torch.einsum("fon,fnd->fod", c_term, response)
        diag_idx = torch.arange(self.state_dim, device=target_device)
        transfer[:, diag_idx, diag_idx] += D.to(dtype=complex_dtype)

        if use_cache:
            self._kernel_cache[cache_key] = transfer.detach()
        return transfer

    def _forward_convolutional(
        self,
        u: torch.Tensor,
        rate: float = 1.0,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, _ = u.shape
        original_dtype = u.dtype
        fft_dtype = torch.float32 if u.dtype in {torch.float16, torch.bfloat16} else u.dtype
        fft_len = 1 << max(1, (2 * seq_len - 1).bit_length())
        use_kernel_cache = not self.training and not torch.is_grad_enabled()

        with torch.autocast(device_type=u.device.type, enabled=False):
            transfer = self._compute_frequency_response(
                seq_len=seq_len,
                fft_len=fft_len,
                rate=rate,
                dtype=fft_dtype,
                device=u.device,
                use_cache=use_kernel_cache,
            )
            u_channels = u.transpose(1, 2).to(dtype=fft_dtype)
            u_f = torch.fft.rfft(u_channels, n=fft_len)
            y_f = torch.einsum("foi,bif->bof", transfer, u_f)
            y = torch.fft.irfft(y_f, n=fft_len)[..., :seq_len]
            y = y.transpose(1, 2).to(dtype=original_dtype)

        if not return_state:
            return y, None

        cache = self.allocate_inference_cache(
            batch_size=batch,
            seq_len=seq_len,
            device=u.device,
            dtype=u.dtype,
            rate=rate,
        )
        h = self.init_state(batch, u.device, u.dtype)
        for t in range(seq_len):
            _, h = self.step(u[:, t, :], h, cache=cache)
        return y, h

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = self.B.dtype
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def allocate_inference_cache(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        rate: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        del batch_size, seq_len
        if dtype is None:
            dtype = self.B.dtype
        A_disc = self._dense_discrete_A(rate=rate, dtype=dtype, device=device)
        _, _, _, B_disc = self._discrete_params(rate=rate, dtype=dtype, device=device)
        C = self.C.to(device=device, dtype=dtype)
        D = self.D.to(device=device, dtype=dtype)
        return {
            "A_T": A_disc.transpose(0, 1).contiguous(),
            "B_T": B_disc.transpose(0, 1).contiguous(),
            "C_T": C.transpose(0, 1).contiguous(),
            "D": D,
        }

    def export_inference_matrices(self, rate: float = 1.0) -> Dict[str, torch.Tensor]:
        diag, U, V, B_disc = self._discrete_params(rate=rate, dtype=self.B.dtype, device=self.B.device)
        A_disc = self._dense_discrete_A(rate=rate, dtype=self.B.dtype, device=self.B.device)
        lambda_cont = -F.softplus(self.log_lambda_real).to(device=self.B.device, dtype=self.B.dtype)
        return {
            "A_continuous_diag": lambda_cont.detach().clone(),
            "A_discrete": A_disc.detach().clone(),
            "low_rank_U": U.detach().clone(),
            "low_rank_V": V.detach().clone(),
            "B": B_disc.detach().clone(),
            "C": self.C.detach().clone(),
            "D": self.D.detach().clone(),
            "dt": self._get_dt(rate=rate, dtype=self.B.dtype).detach().clone(),
            "ternary_u_mask": self.ternary_u_mask.detach().clone(),
            "ternary_v_mask": self.ternary_v_mask.detach().clone(),
            "diag": diag.detach().clone(),
        }

    def step(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        rate: float = 1.0,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            cache = self.allocate_inference_cache(
                batch_size=u.size(0),
                seq_len=1,
                device=u.device,
                dtype=u.dtype,
                rate=rate,
            )
        h_new = torch.matmul(h, cache["A_T"]) + torch.matmul(u, cache["B_T"])
        y = torch.matmul(h_new, cache["C_T"]) + u * cache["D"]
        return y, h_new

    def forward(
        self,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        rate: float = 1.0,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if state is None and mask is None:
            use_conv = self.kernel_mode == "conv" or (
                self.kernel_mode == "auto" and u.size(1) >= self.kernel_threshold
            )
            if use_conv:
                return self._forward_convolutional(u, rate=rate, return_state=return_state)

        batch, seq_len, _ = u.shape
        if state is None:
            h = self.init_state(batch, u.device, u.dtype)
        else:
            h = state.to(device=u.device, dtype=u.dtype)

        cache = self.allocate_inference_cache(
            batch_size=batch,
            seq_len=seq_len,
            device=u.device,
            dtype=u.dtype,
            rate=rate,
        )

        outputs = []
        for t in range(seq_len):
            y_t, h = self.step(u[:, t, :], h, cache=cache)
            if mask is not None:
                mask_t = mask[:, t].unsqueeze(-1).to(dtype=u.dtype)
                y_t = y_t * mask_t
                h = h * mask_t
            outputs.append(y_t)

        final_state = h if return_state else None
        return torch.stack(outputs, dim=1), final_state
