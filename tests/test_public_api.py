import gamma_space_model


def test_public_api_is_gamma_space_model_only():
    assert set(gamma_space_model.__all__) == {
        "GammaSpaceLayer",
        "GammaSpaceBlock",
        "MinimalGammaSpaceBlock",
        "LayerNorm",
        "RMSNorm",
        "HAS_TILELANG_OPS",
    }
