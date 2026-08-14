"""
Shared fixtures for the kvquant test suite.

The GQA fixture is not optional.  `repeat_kv` handling is the single most
likely place a block-wise attention implementation breaks, and an MHA model
hides the bug completely: with `num_key_value_groups == 1`, `repeat_kv` is
the identity, so a missing or misplaced call passes every MHA test and then
produces garbage on any real GQA model (Qwen2.5, Llama-3, Mistral).  Both
fixtures are built from a local config with random weights — no download, and
deterministic under a fixed seed.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def torch_seed() -> int:
    return 1234


@pytest.fixture(scope="session")
def tiny_model():
    """
    Small MHA causal LM (GPT-2 family).  `n_head == n_kv_head`, so this
    exercises the simple path where `repeat_kv` is a no-op.
    """
    from transformers import AutoModelForCausalLM

    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2"
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_gqa_model():
    """
    Small GQA causal LM: 8 query heads over 2 KV heads, so g == 4 and
    `repeat_kv` genuinely expands.  Built locally from a Qwen2Config with
    random weights so the test suite stays offline and deterministic.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1234)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def gqa_factor(tiny_gqa_model) -> int:
    cfg = tiny_gqa_model.config
    return cfg.num_attention_heads // cfg.num_key_value_heads


def pytest_runtest_setup(item):
    """Skip tests marked `cuda` when no device is present."""
    if any(mark.name == "cuda" for mark in item.iter_markers()):
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
