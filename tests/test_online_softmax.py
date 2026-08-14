"""
Phase 3: streaming attention must be numerically exact and actually streaming.

Two separate claims, each needing its own evidence:

  * **Exactness** — online softmax must equal full softmax. This is tested as a
    pure tensor computation, with no model involved, so a regression localizes
    to the kernel rather than to cache plumbing.
  * **Streaming** — the ``(B, H, Tq, Tk)`` score matrix must never be built.
    Correctness tests cannot detect this: the concat-then-softmax formulation
    from Phase 2 is *also* exact, and passes every accuracy test while
    allocating memory linear in context length.  The memory-scaling test below
    is the only thing that distinguishes them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.compact_cache import CompactKVCache, CompactKVLayer, _repeat_kv


class _IdentityLayer(CompactKVLayer):
    def _compress(self, t, is_value):
        return t.clone()

    def _dequantize(self, codes, is_value):
        return codes


def _reference_attention(q, k, v, scaling, causal=True):
    """Plain full-matrix attention, as the thing to match."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
    if causal:
        t_q, t_k = q.shape[-2], k.shape[-2]
        q_pos = torch.arange(t_k - t_q, t_k).view(-1, 1)
        k_pos = torch.arange(t_k).view(1, -1)
        scores = scores.masked_fill(
            (k_pos > q_pos).view(1, 1, t_q, t_k), float("-inf")
        )
    return torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32), v)


def _streaming_attention(q, k, v, scaling, block_size, causal=True):
    """
    The same recurrence the attention function uses, extracted so it can be
    tested without a model.
    """
    b, h, t_q, d = q.shape
    t_k = k.shape[-2]
    running_max = torch.full((b, h, t_q, 1), float("-inf"))
    running_sum = torch.zeros_like(running_max)
    acc = torch.zeros((b, h, t_q, d))
    q_abs_start = t_k - t_q

    for start in range(0, t_k, block_size):
        k_blk = k[..., start : start + block_size, :]
        v_blk = v[..., start : start + block_size, :]
        blk_len = k_blk.shape[-2]
        scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scaling
        if causal:
            q_pos = torch.arange(q_abs_start, q_abs_start + t_q).view(-1, 1)
            k_pos = torch.arange(start, start + blk_len).view(1, -1)
            scores = scores.masked_fill(
                (k_pos > q_pos).view(1, 1, t_q, blk_len), float("-inf")
            )
        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        probs = torch.exp(scores - new_max)
        rescale = torch.exp(running_max - new_max)
        rescale = torch.where(
            torch.isinf(running_max), torch.zeros_like(rescale), rescale
        )
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        running_sum = running_sum * rescale + probs.sum(dim=-1, keepdim=True)
        acc = acc * rescale + torch.matmul(probs, v_blk)
        running_max = new_max

    return acc / running_sum.clamp(min=torch.finfo(torch.float32).tiny)


class TestOnlineSoftmaxExactness:
    @pytest.mark.parametrize("t_k", [1, 2, 127, 128, 129, 517])
    @pytest.mark.parametrize("block_size", [1, 8, 128])
    def test_matches_full_softmax_single_query(self, t_k, block_size):
        """
        Ragged tails (t_k not a multiple of block_size) are where a block loop
        goes wrong, so they are parametrized explicitly.
        """
        torch.manual_seed(0)
        b, h, d = 1, 4, 16
        q = torch.randn(b, h, 1, d)
        k = torch.randn(b, h, t_k, d)
        v = torch.randn(b, h, t_k, d)
        scaling = d**-0.5

        ref = _reference_attention(q, k, v, scaling)
        got = _streaming_attention(q, k, v, scaling, block_size)
        assert (got - ref).abs().max().item() < 1e-6

    @pytest.mark.parametrize("t_q,t_k", [(4, 4), (3, 10), (16, 16), (5, 130)])
    def test_matches_full_softmax_multi_query_causal(self, t_q, t_k):
        torch.manual_seed(0)
        b, h, d = 1, 4, 16
        q = torch.randn(b, h, t_q, d)
        k = torch.randn(b, h, t_k, d)
        v = torch.randn(b, h, t_k, d)
        scaling = d**-0.5

        ref = _reference_attention(q, k, v, scaling)
        got = _streaming_attention(q, k, v, scaling, block_size=4)
        assert (got - ref).abs().max().item() < 1e-6

    def test_survives_extreme_logits(self):
        """
        The running-max rescaling exists to prevent overflow.  Large scores
        would make a naive exp() saturate to inf.
        """
        torch.manual_seed(0)
        b, h, t_k, d = 1, 2, 200, 16
        q = torch.randn(b, h, 1, d) * 50
        k = torch.randn(b, h, t_k, d) * 50
        v = torch.randn(b, h, t_k, d)
        scaling = d**-0.5

        got = _streaming_attention(q, k, v, scaling, block_size=32)
        ref = _reference_attention(q, k, v, scaling)
        assert torch.isfinite(got).all()
        assert (got - ref).abs().max().item() < 1e-4

    def test_block_size_does_not_change_result(self):
        """Block size is a memory knob, not a numerical one."""
        torch.manual_seed(0)
        b, h, t_k, d = 1, 4, 100, 16
        q = torch.randn(b, h, 1, d)
        k = torch.randn(b, h, t_k, d)
        v = torch.randn(b, h, t_k, d)
        scaling = d**-0.5

        results = [
            _streaming_attention(q, k, v, scaling, bs) for bs in (1, 7, 32, 100, 128)
        ]
        for other in results[1:]:
            assert (other - results[0]).abs().max().item() < 1e-6


class TestStreamingMemoryProperty:
    @pytest.mark.cuda
    def test_peak_grows_with_block_size_not_context_length(
        self, tiny_model, warm_cuda_codec
    ):
        """
        The only test that proves attention is *streaming* rather than merely
        correct.  Phase 2's concat-then-softmax was equally exact but allocated
        O(Tk); if this regresses, accuracy tests would stay green while memory
        silently became linear again.

        Measured by decoding one token against caches of very different
        lengths at a fixed block size: peak must stay roughly flat.
        """
        from tests.conftest import make_ids
        from tests.vram import measure_peak_vram, reset_vram

        model = tiny_model.cuda()
        n_layers = model.config.n_layer
        # Leave headroom below the model's position limit (512 here) so the
        # extra decode step below still has a valid position to occupy.
        short, long = 60, 480
        peaks = {}
        try:
            for context in (short, long):
                cache = CompactKVCache(
                    n_layers=n_layers, bits=8, block_size=32, codec="paper"
                )
                cache.layers = [
                    _IdentityLayer(bits=8, block_size=32, codec="paper")
                    for _ in range(n_layers)
                ]
                ids = make_ids(model, context, device="cuda")
                with cache.attached(model) as c:
                    with torch.no_grad():
                        model(ids, past_key_values=c, use_cache=True)
                    reset_vram()

                    def one_step():
                        with torch.no_grad():
                            return model(
                                make_ids(model, 1, device="cuda"),
                                past_key_values=c,
                                use_cache=True,
                            ).logits

                    _, peak = measure_peak_vram(one_step)
                    peaks[context] = peak
        finally:
            model.cpu()

        # An 8x longer context must not cost 8x the transient memory.  A
        # generous 3x ceiling keeps this robust while still failing loudly if
        # the score matrix comes back.
        growth = peaks[long] / max(peaks[short], 1)
        assert growth < 3.0, (
            f"peak grew {growth:.2f}x for {long // short}x context "
            f"({short}: {peaks[short]}, {long}: {peaks[long]}) — "
            f"score matrix is likely being materialized"
        )


class TestFp16Numerics:
    def test_long_context_fp16_stays_finite(self, tiny_model):
        """
        The softmax denominator accumulates over the whole context.  Kept in
        fp16 it would lose precision and eventually overflow, so the kernel
        accumulates in float32 even for a half-precision model.
        """
        from tests.conftest import make_ids

        model = tiny_model.half()
        n_layers = model.config.n_layer
        cache = CompactKVCache(
            n_layers=n_layers, bits=8, block_size=32, codec="paper"
        )
        ids = make_ids(model, 480)  # headroom below the 512 position limit
        with cache.attached(model) as c:
            with torch.no_grad():
                out = model(ids, past_key_values=c, use_cache=True)
                step = model(make_ids(model, 1), past_key_values=c, use_cache=True)

        assert torch.isfinite(out.logits).all()
        assert torch.isfinite(step.logits).all()

    def test_fp16_matches_fp32_closely(self, tiny_model):
        from tests.conftest import make_ids

        ids = make_ids(tiny_model, 64)
        n_layers = tiny_model.config.n_layer

        cache32 = CompactKVCache(n_layers=n_layers, bits=8, block_size=16)
        cache32.layers = [
            _IdentityLayer(bits=8, block_size=16, codec="paper")
            for _ in range(n_layers)
        ]
        with cache32.attached(tiny_model) as c:
            with torch.no_grad():
                ref = tiny_model(ids, past_key_values=c, use_cache=True).logits

        model = tiny_model.half()
        cache16 = CompactKVCache(n_layers=n_layers, bits=8, block_size=16)
        cache16.layers = [
            _IdentityLayer(bits=8, block_size=16, codec="paper")
            for _ in range(n_layers)
        ]
        with cache16.attached(model) as c:
            with torch.no_grad():
                got = model(ids, past_key_values=c, use_cache=True).logits

        assert (got.float() - ref).abs().max().item() < 0.05


class TestAttentionWeightCapture:
    def test_captured_weights_sum_to_one(self, tiny_model):
        """
        Online softmax never builds the weight matrix, so capture reconstructs
        it from the per-block partials.  Rows summing to 1 is the check that the
        reconstruction is normalized correctly.
        """
        from tests.conftest import make_ids

        ids = make_ids(tiny_model, 20)
        n_layers = tiny_model.config.n_layer
        cache = CompactKVCache(n_layers=n_layers, bits=8, block_size=8)
        cache.layers = [
            _IdentityLayer(bits=8, block_size=8, codec="paper")
            for _ in range(n_layers)
        ]
        cache.capture_attn_weights = True
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(ids, past_key_values=c, use_cache=True)

        assert cache.captured_attn, "no weights captured"
        for layer_idx, w in cache.captured_attn.items():
            sums = w.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"layer {layer_idx} rows do not sum to 1"
            )

    def test_captured_weights_match_reference_softmax(self, tiny_model):
        """
        Validates the instrument itself.  Phase 8 re-measures paper claims with
        these weights, so an unvalidated capture would silently invalidate every
        one of those numbers.
        """
        from tests.conftest import make_ids

        ids = make_ids(tiny_model, 12)
        n_layers = tiny_model.config.n_layer
        cache = CompactKVCache(n_layers=n_layers, bits=8, block_size=4)
        cache.layers = [
            _IdentityLayer(bits=8, block_size=4, codec="paper")
            for _ in range(n_layers)
        ]
        cache.capture_attn_weights = True

        # Intercept layer 0's attention output so the captured weights can be
        # checked against the value they actually produced.
        import src.compact_cache as cc

        real_fn = cc.compact_attention_forward
        seen = {}

        def spy(module, query, key, value, attention_mask=None, *args, **kwargs):
            out, weights = real_fn(
                module, query, key, value, attention_mask, *args, **kwargs
            )
            if getattr(module, "layer_idx", 0) == 0 and 0 not in seen:
                seen[0] = out.detach()
            return out, weights

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        with cache.attached(tiny_model) as c:
            ALL_ATTENTION_FUNCTIONS[cc.ATTN_IMPL_NAME] = spy
            try:
                with torch.no_grad():
                    tiny_model(ids, past_key_values=c, use_cache=True)
            finally:
                ALL_ATTENTION_FUNCTIONS[cc.ATTN_IMPL_NAME] = real_fn
            layer0 = c.layers[0]
            layer0_blocks = list(layer0.iter_kv_blocks())
            k = torch.cat([blk for blk, _ in layer0_blocks], dim=-2)

        got = cache.captured_attn[0]
        assert got.shape[-1] == k.shape[-2] == 12

        # attention returns (B, Tq, H, d); the weights are (B, H, Tq, Tk), so
        # put the output back into (B, H, Tq, d) to compare.
        attn_out = seen[0].transpose(1, 2).to(torch.float32)

        # Causality: no weight on future positions.  The final row has no
        # future to check, hence the guard.
        for i in range(12):
            future = got[0, 0, i, i + 1 :]
            if future.numel():
                assert future.abs().max().item() < 1e-6

        # The real validation: reconstructing the output from the captured
        # weights must reproduce what attention actually returned.  Comparing
        # against a freshly-invented reference would only test the reference.
        v = torch.cat([blk for _, blk in layer0_blocks], dim=-2)
        rebuilt = torch.matmul(got.to(v.dtype), v)
        assert torch.allclose(rebuilt, attn_out, atol=1e-4), (
            "captured weights do not reproduce the attention output they "
            "were captured from"
        )

    def test_capture_is_off_by_default(self, tiny_model):
        """
        Capture rebuilds the full (B,H,Tq,Tk) matrix, which is exactly what
        streaming avoids — so it must never be on unless asked for.
        """
        cache = CompactKVCache(n_layers=tiny_model.config.n_layer, bits=8)
        assert cache.capture_attn_weights is False
        from tests.conftest import make_ids

        ids = make_ids(tiny_model, 8)
        with cache.attached(tiny_model) as c:
            with torch.no_grad():
                tiny_model(ids, past_key_values=c, use_cache=True)
        assert not cache.captured_attn
