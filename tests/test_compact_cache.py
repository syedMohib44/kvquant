"""
Phase 2: the compact cache must be correct before it is measured.

A cache that saves VRAM and produces garbage is worthless, so correctness
comes first and gets the most tests.

Testing strategy — two separate gates, because conflating them makes failures
uninterpretable:

  * **Plumbing** is tested with an *identity* codec (``_IdentityLayer``).  Any
    error there is a real bug in block bookkeeping, causal masking, position
    arithmetic, or GQA expansion.  The bound is 1e-6, and it passes at 1.2e-07.
  * **Quantization** is tested separately.  The paper codec has ~0.5% relative
    error even at 8 bits, so a logits bound tight enough to catch plumbing bugs
    would fail for entirely legitimate reasons.

Without that split, the only way to make an end-to-end logits test pass is to
loosen its threshold until it no longer detects the bugs it exists to catch.
"""

from __future__ import annotations

import pytest
import torch

from src.compact_cache import (
    ATTN_IMPL_NAME,
    CompactKVCache,
    CompactKVLayer,
    _causal_block_mask,
    _repeat_kv,
    _resolve_scaling,
)


class _IdentityLayer(CompactKVLayer):
    """Stores blocks verbatim — isolates plumbing from quantization error."""

    def _compress(self, t, is_value):
        return t.clone()

    def _dequantize(self, codes, is_value):
        return codes


def _identity_cache(n_layers: int, block_size: int) -> CompactKVCache:
    cache = CompactKVCache(n_layers=n_layers, bits=8, block_size=block_size)
    cache.layers = [
        _IdentityLayer(bits=8, block_size=block_size, codec="paper")
        for _ in range(n_layers)
    ]
    return cache


def _n_layers(model) -> int:
    cfg = model.config
    return getattr(cfg, "num_hidden_layers", None) or cfg.n_layer


# ---------------------------------------------------------------------------
# Plumbing correctness (identity codec)
# ---------------------------------------------------------------------------


class TestPlumbingMatchesStockModel:
    def test_prefill_logits_match(self, tiny_model):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        with torch.no_grad():
            ref = tiny_model(ids, use_cache=True)

        cache = _identity_cache(_n_layers(tiny_model), block_size=4)
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                out = tiny_model(ids, past_key_values=c, use_cache=True)
        assert (out.logits - ref.logits).abs().max().item() < 1e-6

    def test_decode_logits_match_over_many_steps(self, tiny_model):
        """
        Multi-step decode catches stale `_cum_len` and `get_mask_sizes`
        off-by-ones that a single step hides.
        """
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        cache = _identity_cache(_n_layers(tiny_model), block_size=4)

        with torch.no_grad():
            ref = tiny_model(ids, use_cache=True)
            ref_past = ref.past_key_values

        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(ids, past_key_values=c, use_cache=True)
                for step in range(8):
                    tok = torch.tensor([[10 + step]])
                    got = tiny_model(tok, past_key_values=c, use_cache=True)
                    want = tiny_model(tok, past_key_values=ref_past, use_cache=True)
                    err = (got.logits - want.logits).abs().max().item()
                    assert err < 1e-6, f"step {step}: err {err}"

    def test_gqa_logits_match(self, tiny_gqa_model, gqa_factor):
        """
        The important one.  With g == 1 a missing or misplaced `repeat_kv` is
        invisible; here g == 4, so the expansion is actually exercised.
        """
        assert gqa_factor > 1, "fixture must be genuinely grouped-query"
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            ref = tiny_gqa_model(ids, use_cache=True)

        cache = _identity_cache(_n_layers(tiny_gqa_model), block_size=4)
        with cache.attached(tiny_gqa_model) as c:
            with torch.no_grad():
                out = tiny_gqa_model(ids, past_key_values=c, use_cache=True)
        assert (out.logits - ref.logits).abs().max().item() < 1e-6

    def test_gqa_decode_logits_match(self, tiny_gqa_model):
        ids = torch.tensor([[2, 3, 4, 5]])
        cache = _identity_cache(_n_layers(tiny_gqa_model), block_size=3)
        with torch.no_grad():
            ref_past = tiny_gqa_model(ids, use_cache=True).past_key_values

        with cache.attached(tiny_gqa_model) as c:
            with torch.no_grad():
                tiny_gqa_model(ids, past_key_values=c, use_cache=True)
                for step in range(6):
                    tok = torch.tensor([[7 + step]])
                    got = tiny_gqa_model(tok, past_key_values=c, use_cache=True)
                    want = tiny_gqa_model(
                        tok, past_key_values=ref_past, use_cache=True
                    )
                    assert (got.logits - want.logits).abs().max().item() < 1e-6

    @pytest.mark.parametrize("block_size", [1, 3, 4, 7, 128])
    @pytest.mark.parametrize("extra", [-1, 0, 1, 2])
    def test_block_boundary_alignment(self, tiny_model, block_size, extra):
        """
        Prompt lengths straddling the flush boundary — the likeliest place for
        an off-by-one between the pending window and a completed block.
        """
        length = max(block_size + extra, 1)
        ids = torch.arange(1, length + 1).unsqueeze(0)
        with torch.no_grad():
            ref = tiny_model(ids, use_cache=True)

        cache = _identity_cache(_n_layers(tiny_model), block_size=block_size)
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                out = tiny_model(ids, past_key_values=c, use_cache=True)
            assert c.get_seq_length() == length
        assert (out.logits - ref.logits).abs().max().item() < 1e-6

    def test_chunked_prefill_matches_single_shot(self, tiny_model):
        """Feeding the prompt in chunks must equal feeding it whole."""
        ids = torch.arange(1, 11).unsqueeze(0)
        with torch.no_grad():
            ref = tiny_model(ids, use_cache=True)

        cache = _identity_cache(_n_layers(tiny_model), block_size=4)
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                for start in range(0, 10, 3):
                    out = tiny_model(
                        ids[:, start : start + 3], past_key_values=c, use_cache=True
                    )
            assert c.get_seq_length() == 10
        assert (out.logits[:, -1] - ref.logits[:, -1]).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Quantization quality (real codec)
# ---------------------------------------------------------------------------


class TestRealCodecQuality:
    @pytest.mark.parametrize("codec", ["paper", "paper-outlier"])
    def test_generates_finite_logits(self, tiny_model, codec):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        cache = CompactKVCache(
            n_layers=_n_layers(tiny_model), bits=4, block_size=4, codec=codec
        )
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                out = tiny_model(ids, past_key_values=c, use_cache=True)
        assert torch.isfinite(out.logits).all()

    def test_higher_bits_track_stock_more_closely(self, tiny_model):
        """
        Quality must improve monotonically with bit-width.  This is the real
        end-to-end quality signal; an absolute logits bound would only measure
        the codec's known ~0.5% error at 8 bits.
        """
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            ref = tiny_model(ids, use_cache=True)

        errs = {}
        for bits in (2, 4, 8):
            cache = CompactKVCache(
                n_layers=_n_layers(tiny_model),
                bits=bits,
                block_size=4,
                codec="paper",
            )
            with cache.attached(tiny_model) as c:
                with torch.no_grad():
                    out = tiny_model(ids, past_key_values=c, use_cache=True)
            errs[bits] = (out.logits - ref.logits).abs().max().item()

        assert errs[8] < errs[4] < errs[2]

    def test_gqa_real_codec_finite_and_reasonable(self, tiny_gqa_model):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            ref = tiny_gqa_model(ids, use_cache=True)
        cache = CompactKVCache(
            n_layers=_n_layers(tiny_gqa_model),
            bits=8,
            block_size=4,
            codec="paper",
            gqa_factor=4,
        )
        with cache.attached(tiny_gqa_model) as c:
            with torch.no_grad():
                out = tiny_gqa_model(ids, past_key_values=c, use_cache=True)
        assert torch.isfinite(out.logits).all()
        rel = (out.logits - ref.logits).norm() / ref.logits.norm()
        assert rel.item() < 0.2


# ---------------------------------------------------------------------------
# Append-only invariant
# ---------------------------------------------------------------------------


class TestAppendOnly:
    def test_frozen_blocks_are_byte_identical_after_decoding(self, tiny_model):
        """
        A stored block must never be re-encoded.  Lloyd-Max is lossy, so
        recompressing a reconstruction compounds error — the drift that turns
        long generations into gibberish.
        """
        ids = torch.arange(1, 10).unsqueeze(0)
        cache = CompactKVCache(
            n_layers=_n_layers(tiny_model), bits=4, block_size=4, codec="paper"
        )
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(ids, past_key_values=c, use_cache=True)
            first = c.layers[0]._k_blocks[0]
            snapshot = first.packed.clone()

            with torch.no_grad():
                for step in range(20):
                    tiny_model(
                        torch.tensor([[20 + step]]), past_key_values=c, use_cache=True
                    )

            after = c.layers[0]._k_blocks[0]
            assert after is first, "block object was replaced"
            assert torch.equal(after.packed, snapshot), "block was re-encoded"

    def test_block_count_grows_monotonically(self, tiny_model):
        cache = CompactKVCache(
            n_layers=_n_layers(tiny_model), bits=4, block_size=2, codec="paper"
        )
        counts = []
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(torch.tensor([[1, 2]]), past_key_values=c, use_cache=True)
                for step in range(6):
                    tiny_model(
                        torch.tensor([[5 + step]]), past_key_values=c, use_cache=True
                    )
                    counts.append(len(c.layers[0]._k_blocks))
        assert counts == sorted(counts)
        assert counts[-1] > counts[0]


# ---------------------------------------------------------------------------
# Storage accounting
# ---------------------------------------------------------------------------


class TestStorage:
    def test_cache_smaller_than_float16_including_sidecars(self, tiny_model):
        """
        The headline storage claim, measured honestly — norms included, and
        with enough tokens that the pending float window is not the bulk.
        """
        ids = torch.arange(1, 65).unsqueeze(0)
        cache = CompactKVCache(
            n_layers=_n_layers(tiny_model), bits=3, block_size=16, codec="paper"
        )
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(ids, past_key_values=c, use_cache=True)
            b = c.byte_breakdown()

        assert b.code_bytes > 0
        assert b.sidecar_bytes > 0, "norms must be counted"
        assert b.total < b.float16_bytes, "must beat float16"
        assert c.compression_ratio > 1.0

    def test_reported_bits_exceed_nominal(self, tiny_model):
        """Measured bits/coord must be worse than nominal — sidecars are real."""
        ids = torch.arange(1, 33).unsqueeze(0)
        cache = CompactKVCache(
            n_layers=_n_layers(tiny_model), bits=4, block_size=8, codec="paper"
        )
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(ids, past_key_values=c, use_cache=True)
            assert c.avg_bits_per_dim > 4.0

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_resident_cache_bytes_beat_float_cache(self, tiny_model, device):
        """
        The headline claim, measured the right way: bytes *held by the cache*,
        compact vs float, same model and prompt.

        Deliberately not peak VRAM.  Peak allocation during a forward pass is
        dominated by the attention kernel's workspace, not by the cache — the
        SDPA path this model uses by default peaks at ~10.5 MB where an eager
        path peaks at ~2.8 MB on identical inputs.  A peak comparison between
        the float path (SDPA) and the compact path (our own kernel) therefore
        measures which *kernel* allocates less scratch, and would report a
        storage win or loss that has nothing to do with storage.  Resident
        cache bytes are what the compression claim is actually about.
        """
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        from src.cache_bytes import cache_nbytes

        model = tiny_model.to(device)
        ids = torch.arange(1, 257).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                float_past = model(ids, use_cache=True).past_key_values
            float_bytes = cache_nbytes(float_past).total
            del float_past

            cache = CompactKVCache(
                n_layers=_n_layers(model), bits=3, block_size=64, codec="paper"
            )
            with cache.attached(model) as c:
                with torch.no_grad():
                    model(ids, past_key_values=c, use_cache=True)
                compact_bytes = c.byte_breakdown().total
        finally:
            model.cpu()

        ratio = float_bytes / compact_bytes
        assert ratio > 2.0, (
            f"only {ratio:.2f}x smaller (float {float_bytes}, "
            f"compact {compact_bytes})"
        )

    @pytest.mark.cuda
    def test_resident_gpu_tensors_are_codes_not_floats(
        self, tiny_model, warm_cuda_codec
    ):
        """
        Proves the saving is structural rather than incidental: what stays on
        the device between forward passes must be uint8 code payload, with
        float bytes limited to the small pending window.
        """
        model = tiny_model.cuda()
        ids = torch.arange(1, 257).unsqueeze(0).cuda()
        try:
            cache = CompactKVCache(
                n_layers=_n_layers(model), bits=3, block_size=64, codec="paper"
            )
            with cache.attached(model) as c:
                with torch.no_grad():
                    model(ids, past_key_values=c, use_cache=True)
                b = c.byte_breakdown()
                for layer in c.layers:
                    assert layer.keys.numel() == 0, "no full float K/V may persist"
                    for blk in layer._k_blocks:
                        assert blk.packed.dtype == torch.uint8
                        assert blk.packed.is_cuda
        finally:
            model.cpu()

        assert b.float_bytes < b.code_bytes, (
            f"float residue {b.float_bytes} should be under code payload "
            f"{b.code_bytes}"
        )


# ---------------------------------------------------------------------------
# Attachment lifecycle
# ---------------------------------------------------------------------------


class TestAttachment:
    def test_restores_attn_implementation(self, tiny_model):
        before = tiny_model.config._attn_implementation
        cache = CompactKVCache(n_layers=_n_layers(tiny_model))
        with cache.attached(tiny_model):
            assert tiny_model.config._attn_implementation == ATTN_IMPL_NAME
        assert tiny_model.config._attn_implementation == before

    def test_restores_on_exception(self, tiny_model):
        """A raised exception must not leave the model on compact attention."""
        before = tiny_model.config._attn_implementation
        cache = CompactKVCache(n_layers=_n_layers(tiny_model))
        with pytest.raises(ValueError):
            with cache.attached(tiny_model):
                raise ValueError("boom")
        assert tiny_model.config._attn_implementation == before

    def test_attention_without_attached_cache_raises(self, tiny_model):
        """
        Never silently attend to the empty tensors `update()` returns — that
        would produce plausible-looking output from no context at all.
        """
        from src.compact_cache import _ACTIVE_CACHES, compact_attention_forward

        assert not _ACTIVE_CACHES
        with pytest.raises(RuntimeError, match="without an attached"):
            compact_attention_forward(
                object(), torch.zeros(1, 1, 1, 4), None, None
            )


# ---------------------------------------------------------------------------
# Unit tests for the attention helpers
# ---------------------------------------------------------------------------


class TestAttentionHelpers:
    def test_scaling_from_kwarg(self):
        q = torch.zeros(1, 1, 1, 16)
        assert _resolve_scaling(object(), q, (), {"scaling": 0.5}) == 0.5

    def test_scaling_from_positional(self):
        q = torch.zeros(1, 1, 1, 16)
        assert _resolve_scaling(object(), q, (0.25,), {}) == 0.25

    def test_scaling_from_module_attribute(self):
        class M:
            scaling = 0.125

        q = torch.zeros(1, 1, 1, 16)
        assert _resolve_scaling(M(), q, (), {}) == 0.125

    def test_scaling_falls_back_to_inverse_sqrt_d(self):
        q = torch.zeros(1, 1, 1, 16)
        assert _resolve_scaling(object(), q, (), {}) == pytest.approx(0.25)

    @pytest.mark.parametrize("n_rep", [1, 2, 4])
    def test_repeat_kv_shape_and_content(self, n_rep):
        t = torch.randn(1, 2, 5, 8)
        out = _repeat_kv(t, n_rep)
        assert out.shape == (1, 2 * n_rep, 5, 8)
        # Each source head must be replicated contiguously.
        for h in range(2):
            for r in range(n_rep):
                assert torch.equal(out[:, h * n_rep + r], t[:, h])

    def test_repeat_kv_identity_when_n_rep_one(self):
        t = torch.randn(1, 4, 3, 8)
        assert _repeat_kv(t, 1) is t

    def test_causal_mask_skips_fully_past_blocks(self):
        """A block entirely before the query needs no mask — and pays nothing."""
        scores = torch.zeros(1, 1, 1, 4)
        out = _causal_block_mask(scores, q_abs_start=10, k_abs_start=0)
        assert out is scores

    def test_causal_mask_blocks_future_positions(self):
        scores = torch.zeros(1, 1, 2, 4)
        # Queries at absolute 2,3; keys at absolute 0..3.
        out = _causal_block_mask(scores, q_abs_start=2, k_abs_start=0)
        assert torch.isinf(out[0, 0, 0, 3]) and out[0, 0, 0, 3] < 0
        assert out[0, 0, 0, 2] == 0
        assert out[0, 0, 1, 3] == 0


# ---------------------------------------------------------------------------
# Layer mechanics
# ---------------------------------------------------------------------------


class TestLayerMechanics:
    def test_update_returns_zero_length(self):
        layer = CompactKVLayer(bits=4, block_size=4, codec="paper")
        k = torch.randn(1, 2, 3, 16)
        rk, rv = layer.update(k, k.clone())
        assert rk.shape[-2] == 0 and rv.shape[-2] == 0
        assert layer.get_seq_length() == 3

    def test_keys_are_empty_tensors_not_none(self):
        """`kvs_from_cache` and `Cache.__getitem__` read these directly."""
        layer = CompactKVLayer(bits=4, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 2, 16), torch.randn(1, 2, 2, 16))
        assert isinstance(layer.keys, torch.Tensor)
        assert layer.keys.numel() == 0

    def test_prefill_larger_than_block_makes_many_blocks(self):
        """The `while` loop, not `if` — one call can complete several blocks."""
        layer = CompactKVLayer(bits=4, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 14, 16), torch.randn(1, 2, 14, 16))
        assert len(layer._k_blocks) == 3
        assert layer._pending_k.shape[-2] == 2
        assert layer.get_seq_length() == 14

    def test_iter_blocks_covers_every_position(self):
        layer = CompactKVLayer(bits=8, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16))
        total = sum(k.shape[-2] for k, _ in layer.iter_kv_blocks())
        assert total == layer.get_seq_length() == 10

    def test_reset_clears_everything(self):
        layer = CompactKVLayer(bits=4, block_size=2, codec="paper")
        layer.update(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16))
        layer.reset()
        assert layer.get_seq_length() == 0
        assert not layer._k_blocks
        assert layer._pending_k is None

    def test_crop_to_block_boundary(self):
        layer = CompactKVLayer(bits=8, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16))
        layer.crop(8)
        assert layer.get_seq_length() == 8
        assert len(layer._k_blocks) == 2

    def test_crop_into_pending_window(self):
        layer = CompactKVLayer(bits=8, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16))
        layer.crop(9)
        assert layer.get_seq_length() == 9

    def test_crop_beyond_length_is_noop(self):
        layer = CompactKVLayer(bits=8, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16))
        layer.crop(99)
        assert layer.get_seq_length() == 6

    def test_get_mask_sizes_includes_query_length(self):
        layer = CompactKVLayer(bits=8, block_size=4, codec="paper")
        layer.update(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16))
        kv_len, offset = layer.get_mask_sizes(torch.arange(3))
        assert kv_len == 8 and offset == 0

    def test_bad_codec_rejected(self):
        with pytest.raises(ValueError, match="codec must be"):
            CompactKVLayer(codec="int8")

    def test_calibration_is_per_layer(self):
        """
        Outlier channels are layer-specific; pooling across layers
        mis-identifies them.  Our finding — the paper prescribes no calibration.
        """
        cache = CompactKVCache(n_layers=2, bits=4, block_size=4)
        torch.manual_seed(0)
        kvs = [
            (torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16)),
            (torch.randn(1, 2, 8, 16) * 5, torch.randn(1, 2, 8, 16)),
        ]
        cache.calibrate_from(kvs)
        q0, q1 = cache.layers[0]._k_quant, cache.layers[1]._k_quant
        assert q0 is not None and q1 is not None
        assert q0 is not q1

    def test_calibrate_from_wrong_length_raises(self):
        cache = CompactKVCache(n_layers=3, bits=4)
        with pytest.raises(ValueError, match="expected 3"):
            cache.calibrate_from([(torch.randn(1, 1, 2, 8),) * 2])
