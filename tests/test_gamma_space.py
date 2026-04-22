"""Tests for the structured Gamma SSM modules."""

import torch

from gamma_space_model import GammaSpaceBlock, MinimalGammaSpaceBlock, GammaSpaceLayer


def test_gamma_space_forward_shapes():
    model = GammaSpaceLayer(state_dim=4, hidden_dim=8)
    x = torch.randn(2, 6, 4)
    y, h = model(x)

    assert y.shape == (2, 6, 4)
    assert h.shape == (2, 8)


def test_gamma_space_step_matches_forward():
    model = GammaSpaceLayer(
        state_dim=4,
        hidden_dim=8,
        discretization="bilinear",
        kernel_mode="recurrent",
    )
    x = torch.randn(2, 5, 4)

    y_forward, _ = model(x)
    state = model.init_state(batch_size=2, device=x.device, dtype=x.dtype)
    cache = model.allocate_inference_cache(
        batch_size=2,
        seq_len=x.size(1),
        device=x.device,
        dtype=x.dtype,
    )
    outputs = []
    for t in range(x.size(1)):
        y_t, state = model.step(x[:, t, :], state, cache=cache)
        outputs.append(y_t)
    y_step = torch.stack(outputs, dim=1)

    assert torch.allclose(y_forward, y_step, atol=1e-5)


def test_gamma_space_conv_matches_recurrent():
    torch.manual_seed(7)
    recurrent = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="recurrent",
    )
    conv = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="conv",
    )
    conv.load_state_dict(recurrent.state_dict())

    x = torch.randn(2, 32, 3)
    y_recurrent, h_recurrent = recurrent(x)
    y_conv, h_conv = conv(x)

    assert torch.allclose(y_recurrent, y_conv, atol=1e-5)
    assert torch.allclose(h_recurrent, h_conv, atol=1e-5)


def test_gamma_space_eval_conv_reuses_kernel_cache():
    torch.manual_seed(7)
    model = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="conv",
    )
    model.eval()
    x = torch.randn(2, 32, 3)

    with torch.no_grad():
        y_first, _ = model(x, return_state=False)
        assert len(model._kernel_cache) == 1
        y_second, _ = model(x, return_state=False)

    assert torch.allclose(y_first, y_second, atol=1e-6)
    assert len(model._kernel_cache) == 1
    model.clear_kernel_cache()
    assert len(model._kernel_cache) == 0


def test_gamma_space_conv_accepts_half_non_power_two_fft_length():
    torch.manual_seed(7)
    model = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="conv",
    )
    x = torch.randn(2, 24, 3).half()

    y, h = model(x, return_state=False)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert h is None


def test_gamma_space_conv_autocast_keeps_fft_dtypes_compatible():
    torch.manual_seed(7)
    model = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="conv",
    )
    x = torch.randn(2, 24, 3)

    autocast_dtype = torch.float16 if x.device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=x.device.type, dtype=autocast_dtype):
        y, h = model(x, return_state=False)

    assert y.shape == x.shape
    assert h is None


def test_gamma_space_structured_step_matches_dense():
    torch.manual_seed(7)
    dense = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="recurrent",
    )
    structured = GammaSpaceLayer(
        state_dim=3,
        hidden_dim=6,
        discretization="bilinear",
        kernel_mode="recurrent",
    )
    structured.load_state_dict(dense.state_dict())

    x = torch.randn(2, 11, 3)
    state_dense = dense.init_state(batch_size=2, device=x.device, dtype=x.dtype)
    state_structured = structured.init_state(batch_size=2, device=x.device, dtype=x.dtype)
    dense_cache = {
        "dA": dense._discretize(dtype=x.dtype)[0],
        "dB": dense._discretize(dtype=x.dtype)[1],
        "C": dense.C.to(dtype=x.dtype),
        "D": dense.D.to(dtype=x.dtype),
        "structured_method": None,
    }
    structured_cache = structured.allocate_inference_cache(
        batch_size=2,
        seq_len=x.size(1),
        device=x.device,
        dtype=x.dtype,
    )

    dense_outputs = []
    structured_outputs = []
    for t in range(x.size(1)):
        y_dense, state_dense = dense.step(x[:, t, :], state_dense, cache=dense_cache)
        y_structured, state_structured = structured.step(x[:, t, :], state_structured, cache=structured_cache)
        dense_outputs.append(y_dense)
        structured_outputs.append(y_structured)

    assert torch.allclose(torch.stack(dense_outputs, dim=1), torch.stack(structured_outputs, dim=1), atol=1e-5)
    assert torch.allclose(state_dense, state_structured, atol=1e-5)


def test_gamma_space_export_shapes():
    model = GammaSpaceLayer(state_dim=3, hidden_dim=6)
    mats = model.export_inference_matrices()

    assert mats["A_continuous"].shape == (6, 6)
    assert mats["dA"].shape == (6, 6)
    assert mats["dB"].shape == (6, 3)
    assert mats["C"].shape == (3, 6)
    assert mats["D"].shape == (3,)


def test_gamma_space_block_forward_shapes():
    block = GammaSpaceBlock(d_model=4, hidden_dim=8)
    x = torch.randn(2, 7, 4)
    y, h = block(x)

    assert y.shape == (2, 7, 4)
    assert h.shape == (2, 8)


def test_gamma_space_block_cached_step_matches_forward_with_input_gate():
    torch.manual_seed(7)
    block = GammaSpaceBlock(d_model=4, hidden_dim=8, kernel_mode="recurrent")
    x = torch.randn(2, 7, 4)

    y_forward, _ = block(x)
    state = block.ssm.init_state(batch_size=2, device=x.device, dtype=x.dtype)
    cache = block.allocate_inference_cache(
        batch_size=2,
        seq_len=x.size(1),
        device=x.device,
        dtype=x.dtype,
    )
    outputs = []
    for t in range(x.size(1)):
        y_t, state = block.step(x[:, t, :], state, cache=cache)
        outputs.append(y_t)

    y_step = torch.stack(outputs, dim=1)
    assert torch.allclose(y_forward, y_step, atol=1e-5)


def test_gamma_space_minimal_block_forward_shapes():
    block = MinimalGammaSpaceBlock(d_model=4, hidden_dim=8)
    x = torch.randn(2, 7, 4)
    y, h = block(x)

    assert y.shape == (2, 7, 4)
    assert h.shape == (2, 8)


def test_gamma_space_block_deployment_cache_step_shapes():
    block = GammaSpaceBlock(d_model=4, hidden_dim=8)
    x = torch.randn(2, 5, 4)
    state = block.ssm.init_state(batch_size=2, device=x.device, dtype=x.dtype)
    cache = block.allocate_deployment_cache(
        batch_size=2,
        seq_len=x.size(1),
        device=x.device,
        dtype=x.dtype,
    )

    outputs = []
    for t in range(x.size(1)):
        y_t, state = block.step(x[:, t, :], state, cache=cache)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    assert y.shape == (2, 5, 4)
    assert state.shape == (2, 8)


def test_gamma_space_block_balanced_deployment_cache_step_shapes():
    block = GammaSpaceBlock(d_model=4, hidden_dim=8)
    x = torch.randn(2, 5, 4)
    state = block.ssm.init_state(batch_size=2, device=x.device, dtype=x.dtype)
    cache = block.allocate_balanced_deployment_cache(
        batch_size=2,
        seq_len=x.size(1),
        device=x.device,
        dtype=x.dtype,
    )

    outputs = []
    for t in range(x.size(1)):
        y_t, state = block.step(x[:, t, :], state, cache=cache)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    assert y.shape == (2, 5, 4)
    assert state.shape == (2, 8)
    assert cache["deployment_mode"] == "balanced"
