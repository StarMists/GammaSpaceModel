"""structured Gamma SSM with stable discretization and learned timescales."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaSpaceLayer(nn.Module):
    """
    Gamma SSM variant that preserves the fixed ternary-friendly gamma transition
    while adding practical stability features:

    - learned positive timestep `dt`
    - bilinear or ZOH discretization
    - direct skip term `D`
    - stateful recurrent stepping with reusable discretized matrices

    The continuous-time transition matrix A is fixed to the lower-bidiagonal
    gamma structure:

        A[n, n] = -1
        A[n, n-1] = 1

    This keeps inference deployment compatible with hardware that benefits from
    sparse ternary transition structure.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: float = 1e-2,
        discretization: str = "bilinear",
        learn_dt: bool = True,
        use_D: bool = True,
        kernel_mode: str = "auto",
        kernel_threshold: int = 64,
    ) -> None:
        super().__init__()
        if discretization not in {"bilinear", "zoh", "euler"}:
            raise ValueError(
                f"Unsupported discretization '{discretization}'. "
                "Expected one of {'bilinear', 'zoh', 'euler'}."
            )
        if kernel_mode not in {"auto", "recurrent", "conv"}:
            raise ValueError(
                f"Unsupported kernel_mode '{kernel_mode}'. "
                "Expected one of {'auto', 'recurrent', 'conv'}."
            )
        if not (0.0 < dt_min <= dt_init <= dt_max):
            raise ValueError("Expected 0 < dt_min <= dt_init <= dt_max.")

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.discretization = discretization
        self.learn_dt = learn_dt
        self.use_D = use_D
        self.kernel_mode = kernel_mode
        self.kernel_threshold = kernel_threshold

        A = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32)
        indices = torch.arange(hidden_dim)
        A[indices, indices] = -1.0
        if hidden_dim > 1:
            A[indices[1:], indices[:-1]] = 1.0
        self.register_buffer("A", A)
        self.register_buffer("I", torch.eye(hidden_dim, dtype=torch.float32))

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

    def clear_kernel_cache(self) -> None:
        """Drop cached convolution kernels used by eval/no-grad full forwards."""

        self._kernel_cache.clear()

    def _kernel_cache_key(
        self,
        seq_len: int,
        rate: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[object, ...]:
        return (
            seq_len,
            float(rate),
            dtype,
            device.type,
            device.index,
            self.discretization,
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

    def _discretize(self, rate: float = 1.0, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = dtype or self.B.dtype
        A = self.A.to(dtype=target_dtype)
        I = self.I.to(dtype=target_dtype)
        B = self.B.to(dtype=target_dtype)
        dt = self._get_dt(rate=rate, dtype=target_dtype)

        if self.discretization == "euler":
            dA = I + dt * A
            dB = dt * B
        elif self.discretization == "bilinear":
            backward = I - 0.5 * dt * A
            forward = I + 0.5 * dt * A
            dA = torch.linalg.solve(backward, forward)
            dB = torch.linalg.solve(backward, dt * B)
        else:
            dA = torch.matrix_exp(dt * A)
            dB = torch.linalg.solve(A, (dA - I) @ B)

        return dA, dB

    def export_inference_matrices(self, rate: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Return the discretized matrices used during inference. This is helpful for
        deployment flows that want to serialize the learned model into a separate
        runtime or hardware-specific executor.
        """

        dA, dB = self._discretize(rate=rate, dtype=self.B.dtype)
        return {
            "A_continuous": self.A.detach().clone(),
            "dA": dA.detach().clone(),
            "dB": dB.detach().clone(),
            "C": self.C.detach().clone(),
            "D": self.D.detach().clone(),
            "dt": self._get_dt(rate=rate, dtype=self.B.dtype).detach().clone(),
        }

    def _compute_kernel(
        self,
        seq_len: int,
        rate: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Compute the causal convolution kernel of shape (D_out, D_in, L)."""

        target_dtype = dtype or self.B.dtype
        target_device = device or self.B.device
        cache_key = self._kernel_cache_key(seq_len, rate, target_dtype, target_device)
        if use_cache and cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        dA, dB = self._discretize(rate=rate, dtype=dtype)
        dA = dA.to(device=target_device, dtype=target_dtype)
        dB = dB.to(device=target_device, dtype=target_dtype)
        C = self.C.to(dtype=dA.dtype)
        C = C.to(device=target_device, dtype=target_dtype)

        state = dB
        kernel_terms = []
        for _ in range(seq_len):
            kernel_terms.append(torch.matmul(C, state))
            state = torch.matmul(dA, state)
        kernel = torch.stack(kernel_terms, dim=-1)
        if use_cache:
            self._kernel_cache[cache_key] = kernel.detach()
        return kernel

    def _apply_dA_to_state(
        self,
        h: torch.Tensor,
        rate: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        method = None if cache is None else cache.get("structured_method")
        if method == "euler":
            dt = cache["dt"]
            shifted = torch.cat([torch.zeros_like(h[..., :1]), h[..., :-1]], dim=-1)
            return (1.0 - dt) * h + dt * shifted
        if method == "bilinear":
            if "forward_matrix" in cache and "backward_matrix" in cache:
                rhs = torch.matmul(h, cache["forward_matrix"].transpose(0, 1))
                solved = torch.linalg.solve_triangular(
                    cache["backward_matrix"],
                    rhs.transpose(0, 1),
                    upper=False,
                )
                return solved.transpose(0, 1)
            a = cache["a"]
            b = cache["b"]
            c = cache["c"]
            rhs0 = (c * h)[..., :1]
            rhs_rest = (c * h)[..., 1:] + a * h[..., :-1]
            rhs = torch.cat([rhs0, rhs_rest], dim=-1)
            outputs = [rhs[..., :1] / b]
            for i in range(1, rhs.size(-1)):
                next_col = (rhs[..., i : i + 1] + a * outputs[-1]) / b
                outputs.append(next_col)
            return torch.cat(outputs, dim=-1)

        dA, _ = self._discretize(rate=rate, dtype=dtype or h.dtype)
        return torch.matmul(h, dA.transpose(0, 1))

    def _apply_dA_to_matrix(
        self,
        matrix: torch.Tensor,
        rate: float = 1.0,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if self.discretization == "euler":
            dt = self._get_dt(rate=rate, dtype=dtype or matrix.dtype).to(device=matrix.device, dtype=matrix.dtype)
            shifted = torch.cat([torch.zeros_like(matrix[:1]), matrix[:-1]], dim=0)
            return (1.0 - dt) * matrix + dt * shifted
        if self.discretization == "bilinear":
            dt = self._get_dt(rate=rate, dtype=dtype or matrix.dtype).to(device=matrix.device, dtype=matrix.dtype)
            a = 0.5 * dt
            b = 1.0 + a
            c = 1.0 - a
            rhs0 = (c * matrix)[:1]
            rhs_rest = (c * matrix)[1:] + a * matrix[:-1]
            rhs = torch.cat([rhs0, rhs_rest], dim=0)
            outputs = [rhs[:1] / b]
            for i in range(1, rhs.size(0)):
                next_row = (rhs[i : i + 1] + a * outputs[-1]) / b
                outputs.append(next_row)
            return torch.cat(outputs, dim=0)

        dA, _ = self._discretize(rate=rate, dtype=dtype or matrix.dtype)
        return torch.matmul(dA, matrix)

    def _forward_convolutional(
        self,
        u: torch.Tensor,
        rate: float = 1.0,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, state_dim = u.shape
        original_dtype = u.dtype
        fft_dtype = torch.float32 if u.dtype in {torch.float16, torch.bfloat16} else u.dtype
        use_kernel_cache = not self.training and not torch.is_grad_enabled()
        with torch.autocast(device_type=u.device.type, enabled=False):
            kernel = self._compute_kernel(
                seq_len=seq_len,
                rate=rate,
                dtype=fft_dtype,
                device=u.device,
                use_cache=use_kernel_cache,
            ).to(dtype=fft_dtype)
            D = self.D.to(device=u.device, dtype=fft_dtype)

            u_channels = u.transpose(1, 2).to(dtype=fft_dtype)
            fft_len = 1 << max(1, (2 * seq_len - 1).bit_length())
            kernel_f = torch.fft.rfft(kernel, n=fft_len)
            u_f = torch.fft.rfft(u_channels, n=fft_len)
            y_f = torch.einsum("bif,oif->bof", u_f, kernel_f)
            y = torch.fft.irfft(y_f, n=fft_len)[..., :seq_len]
            y = y + u_channels * D.view(1, state_dim, 1)
            y = y.transpose(1, 2).to(dtype=original_dtype)

        if not return_state:
            return y, None

        dA, dB = self._discretize(rate=rate, dtype=u.dtype)
        dA_T = dA.transpose(0, 1).contiguous()
        dB_T = dB.transpose(0, 1).contiguous()
        h = self.init_state(batch, u.device, u.dtype)
        for t in range(seq_len):
            h = torch.matmul(h, dA_T) + torch.matmul(u[:, t, :], dB_T)
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
        dA, dB = self._discretize(rate=rate, dtype=dtype)
        cache = {
            "dA": dA.to(device=device),
            "dB": dB.to(device=device),
            "C": self.C.to(device=device, dtype=dtype),
            "D": self.D.to(device=device, dtype=dtype),
        }
        cache["dB_T"] = cache["dB"].transpose(0, 1).contiguous()
        cache["C_T"] = cache["C"].transpose(0, 1).contiguous()
        if self.discretization == "euler":
            dt = self._get_dt(rate=rate, dtype=dtype).to(device=device, dtype=dtype)
            cache["structured_method"] = "euler"
            cache["dt"] = dt
        elif self.discretization == "bilinear":
            dt = self._get_dt(rate=rate, dtype=dtype).to(device=device, dtype=dtype)
            A = self.A.to(device=device, dtype=dtype)
            I = self.I.to(device=device, dtype=dtype)
            cache["forward_matrix"] = I + 0.5 * dt * A
            cache["backward_matrix"] = I - 0.5 * dt * A
            cache["structured_method"] = "bilinear"
            cache["a"] = 0.5 * dt
            cache["b"] = 1.0 + 0.5 * dt
            cache["c"] = 1.0 - 0.5 * dt
        return cache

    def _step_with_matrices(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        dA_T: torch.Tensor,
        dB_T: torch.Tensor,
        C_T: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_new = torch.matmul(h, dA_T) + torch.matmul(u, dB_T)
        y = torch.matmul(h_new, C_T) + u * D
        return y, h_new

    def step(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        rate: float = 1.0,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            dB = self._discretize(rate=rate, dtype=u.dtype)[1]
            C = self.C.to(device=u.device, dtype=u.dtype)
            D = self.D.to(device=u.device, dtype=u.dtype)
            h_next = self._apply_dA_to_state(h, rate=rate, dtype=u.dtype)
            dB_T = dB.transpose(0, 1).contiguous()
            C_T = C.transpose(0, 1).contiguous()
        else:
            D = cache["D"]
            h_next = self._apply_dA_to_state(h, cache=cache)
            dB_T = cache.get("dB_T")
            C_T = cache.get("C_T")
            if dB_T is None:
                dB_T = cache["dB"].transpose(0, 1).contiguous()
            if C_T is None:
                C_T = cache["C"].transpose(0, 1).contiguous()
        h_new = h_next + torch.matmul(u, dB_T)
        y = torch.matmul(h_new, C_T) + u * D
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

        dA, dB = self._discretize(rate=rate, dtype=u.dtype)
        C = self.C.to(device=u.device, dtype=u.dtype)
        D = self.D.to(device=u.device, dtype=u.dtype)
        dA_T = dA.transpose(0, 1).contiguous()
        dB_T = dB.transpose(0, 1).contiguous()
        C_T = C.transpose(0, 1).contiguous()

        outputs = []
        for t in range(seq_len):
            y_t, h = self._step_with_matrices(u[:, t, :], h, dA_T, dB_T, C_T, D)
            if mask is not None:
                mask_t = mask[:, t].unsqueeze(-1).to(dtype=u.dtype)
                y_t = y_t * mask_t
                h = h * mask_t
            outputs.append(y_t)

        final_state = h if return_state else None
        return torch.stack(outputs, dim=1), final_state
