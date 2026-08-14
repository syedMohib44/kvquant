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


# Function-scoped, not session-scoped, on purpose.  Tests legitimately need to
# move a model to CUDA or cast it to fp16, and a shared instance would carry
# that state into every later test — producing device-mismatch and dtype errors
# far from their cause.  Loading these tiny models is cheap; the weights come
# from a cached snapshot or a local config, so per-test construction costs
# little and buys full isolation.


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def gqa_factor(tiny_gqa_model) -> int:
    cfg = tiny_gqa_model.config
    return cfg.num_attention_heads // cfg.num_key_value_heads


def make_ids(model, length: int, device=None) -> torch.Tensor:
    """
    Build a valid token-id sequence of ``length`` for ``model``.

    Both limits here are easy to blow past by accident and fail in unhelpful
    ways.  Token ids at or above ``vocab_size`` trip an embedding bounds assert
    *on the device*, which poisons the CUDA context — every later test in the
    session then fails during setup with an unrelated-looking error, so the
    real cause is far from the reported failure.  Exceeding the model's position
    limit fails just as obscurely.  Clamping both here keeps a test's intent
    ("a long context") from silently becoming an invalid input.
    """
    cfg = model.config
    vocab = getattr(cfg, "vocab_size")
    max_pos = (
        getattr(cfg, "n_positions", None)
        or getattr(cfg, "max_position_embeddings", None)
        or length
    )
    if length > max_pos:
        raise ValueError(
            f"{type(model).__name__} supports at most {max_pos} positions, "
            f"asked for {length}"
        )
    ids = torch.arange(length) % (vocab - 1) + 1
    ids = ids.unsqueeze(0)
    return ids.to(device) if device is not None else ids


@pytest.fixture(scope="session")
def warm_cuda_codec():
    """
    Pay the codec's one-time GPU costs before any VRAM measurement.

    The first compact-cache forward on a device allocates roughly 9 MB more
    than every subsequent one: Lloyd-Max codebooks are solved and cached
    per ``(dim, num_bits)`` in a module-global dict, rotation and boundary
    buffers migrate to the device, and cuBLAS picks its workspace.  All of it
    is process-global and paid once.

    A measurement that includes it attributes those 9 MB to the cache and can
    report the compact path as several times *worse* than float — the opposite
    of the truth (steady state is ~1.2 MB vs ~11.5 MB here).  Because the
    caches are global, warming inside a single test is not enough: whichever
    CUDA test runs first would absorb the cost and the rest would look fine,
    making results depend on test order.  Warming once per session removes
    that dependency.
    """
    if not torch.cuda.is_available():
        return
    from src.compact_cache import CompactKVCache
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2"
    ).cuda()
    model.eval()
    ids = torch.arange(1, 65).unsqueeze(0).cuda()
    for bits in (2, 3, 4, 8):
        cache = CompactKVCache(
            n_layers=model.config.n_layer, bits=bits, block_size=16, codec="paper"
        )
        with cache.attached(model) as c:
            with torch.no_grad():
                model(ids, past_key_values=c, use_cache=True)
    model.cpu()
    del model
    torch.cuda.empty_cache()


def pytest_runtest_setup(item):
    """Skip tests marked `cuda` when no device is present."""
    if any(mark.name == "cuda" for mark in item.iter_markers()):
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
