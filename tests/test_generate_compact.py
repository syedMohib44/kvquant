"""
End-to-end tests for generate() and stream() over the compact cache (Phase 4).

These are the first tests in the repo that exercise the whole public path —
prefill, calibration, ingest, decode loop, detokenization — rather than a
component in isolation.  That matters because the defect this phase fixes was
invisible to component tests: every piece worked, and the assembled pipeline
still saved no memory, because `_build_quantized_cache` compressed the prefill
and immediately expanded it back to float while the decode loop appended
uncompressed tokens to a plain DynamicCache.

Two deliberate choices about what is asserted:

*Quality is judged on logits, not on generated text.*  The fixture is a
randomly-initialized model whose output collapses to one token repeated, and
that makes text comparison an insensitive instrument: removing the causal mask
outright leaves the generated string byte-identical, while it moves max logit
error from 0.068 to 2.07.  Text is still compared in one place — with the
repetition penalty explicitly disabled, because the penalty's divide-by-1.3
threshold turns a 2e-2 codec difference into a permanently divergent sequence
in both directions, which measures chaos rather than correctness.

*The reported ratio must equal the measured one.*  The old
`compression_ratio` was `16 / avg_bits`, a nominal figure that described
neither the sidecar norms nor the generated tokens.  The tests here pin the
number in the result object to the bytes actually held, so the two cannot
drift apart again.
"""

from __future__ import annotations

import importlib

import pytest
import torch

# `import src.generate as G` would bind the re-exported generate() *function*
# (src/__init__.py does `from .generate import generate`), not the module, so
# every G.<internal> lookup below would fail with AttributeError on a function.
G = importlib.import_module("src.generate")

from src.compact_cache import CompactKVCache  # noqa: E402
from tests.cache_bytes import cache_nbytes  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — a real generate() call needs a tokenizer, so we register a locally
# built model into the module cache under a fake name rather than downloading.
# ---------------------------------------------------------------------------


@pytest.fixture
def gqa_lm():
    """
    A small GQA causal LM registered into generate()'s model cache.

    GQA (8 query heads over 2 KV heads) rather than MHA because
    `num_key_value_groups == 1` makes `repeat_kv` the identity, which hides
    every mistake in the block-wise attention path.  head_dim is 64 — a
    realistic width.  At the tiny models' head_dim of 8, the codec's two
    float32 norms per vector cost 8 bits/coord of sidecar and swamp the
    payload, so a ratio measured there says more about the fixture than the
    codec.

    The model's vocab is the *tokenizer's* size, not a larger round number.
    An oversized vocab lets sampling pick ids the tokenizer cannot decode, and
    those come back as U+FFFD replacement characters — which stream() strips as
    incomplete multi-byte sequences, so it yields nothing at all while
    generate() still returns a (garbage) string.  That looks exactly like a
    broken stream() and is really a broken fixture.

    Yields `(name, model, tokenizer)`; the cache entry is removed afterwards so
    tests do not leak state into each other.
    """
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1234)
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()

    name = "test-local-gqa"
    key = f"{name}::None::None::None::None"
    G._MODEL_CACHE[key] = (model, tok)
    try:
        yield name, model, tok
    finally:
        G._MODEL_CACHE.pop(key, None)


PROMPT = "The quick brown fox jumps over the lazy dog. " * 12


def _gen(name, **kw):
    kw.setdefault("raw", True)
    kw.setdefault("device", "cpu")
    kw.setdefault("max_new_tokens", 12)
    return G.generate(PROMPT, model=name, **kw)


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------


class TestCompactIsDefault:
    def test_default_uses_compact_cache(self, gqa_lm, monkeypatch):
        """generate() with no cache argument must take the compact path."""
        name, _, _ = gqa_lm
        called = {}
        real = G._build_compact_cache

        def spy(*a, **k):
            called["yes"] = True
            return real(*a, **k)

        monkeypatch.setattr(G, "_build_compact_cache", spy)
        out = _gen(name, bits=4)
        assert called.get("yes"), "compact cache was not used by default"
        assert out.measured is True

    def test_escape_hatch_restores_float_path(self, gqa_lm):
        """compact_cache=False must reach the old path and say so."""
        name, _, _ = gqa_lm
        out = _gen(name, bits=4, compact_cache=False)
        assert out.measured is False
        assert out.cache_bytes == 0
        assert out.text

    def test_offload_combination_raises(self, gqa_lm):
        """
        The one combination that would compress twice is refused.

        Running the disk codec over the compact codec's dequantized output
        compounds Lloyd-Max error (PAPER.md §3.3).  Choosing one silently
        would hide that, so this must raise rather than pick a winner.
        """
        name, _, _ = gqa_lm
        with pytest.raises(ValueError, match="cannot be combined"):
            _gen(name, bits=4, offload_to_disk=True)

        # ...and remains reachable with the escape hatch.
        with pytest.raises(ValueError, match="cannot be combined"):
            list(G.stream(PROMPT, model=name, raw=True, device="cpu",
                          offload_to_disk=True))


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------


class TestOutputQuality:
    def test_produces_nonempty_text(self, gqa_lm):
        name, _, _ = gqa_lm
        out = _gen(name, bits=4, max_new_tokens=16)
        assert isinstance(out.text, str) and out.text.strip()

    def test_identical_to_float_path_without_repetition_penalty(self, gqa_lm):
        """
        With sampling disabled *and* the repetition penalty off, the two paths
        must produce the same tokens at 8 bits.

        The penalty has to be off for this to mean anything.  It divides or
        multiplies the logits of already-seen tokens by 1.3, which is a
        discrete threshold: the ~2e-2 logit difference between the two codecs
        is enough to flip which token crosses it, and greedy decoding then
        amplifies that into a permanently different sequence.  Both paths
        remain individually correct — each matches an unquantized forward on
        argmax — so a text mismatch under penalty measures chaos, not quality.
        Leaving the penalty on and loosening the assertion to a shared prefix
        would just be tuning until it passes.
        """
        name, _, _ = gqa_lm
        kw = dict(bits=8, max_new_tokens=8, temperature=0.0,
                  repetition_penalty=1.0)
        compact = _gen(name, **kw)
        float_path = _gen(name, compact_cache=False, **kw)
        assert compact.text == float_path.text, (
            f"8-bit compact and float paths disagree with sampling and "
            f"penalty disabled — this is a real plumbing difference:\n"
            f"  compact: {compact.text!r}\n  float:   {float_path.text!r}"
        )

    def test_prefill_logits_match_unquantized_model(self, gqa_lm):
        """
        Compact prefill logits must track a plain unquantized forward.

        This, not text comparison, is the quality instrument for the
        randomly-initialized fixture.  Its output collapses to one repeated
        token, so *text* agreement is insensitive: deleting the causal mask
        entirely leaves the generated string byte-identical.  Logits over all
        positions do detect it — measured, that mutation moves max error from
        0.068 to 2.07, a 30x jump — so the bound below sits far under the
        broken value and comfortably above the healthy one.

        All positions rather than the last: with a single query token there is
        nothing for a causal mask to hide, so a decode-step comparison cannot
        see masking bugs at all.
        """
        name, model, tok = gqa_lm
        ids = tok(PROMPT, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            reference = model(ids, use_cache=True).logits

        cache = CompactKVCache(
            n_layers=model.config.num_hidden_layers, bits=8, block_size=128,
            gqa_factor=4,
        )
        with cache.attached(model):
            with torch.no_grad():
                got = model(ids, past_key_values=cache, use_cache=True).logits

        err = (got - reference).abs().max().item()
        assert err < 0.5, (
            f"compact prefill logits deviate by {err:.3e} from the "
            f"unquantized model; healthy is ~7e-2 at 8 bits"
        )

    def test_lower_bits_costs_accuracy_monotonically(self, gqa_lm):
        """
        Error must grow as bits shrink.

        A codec that ignored its bit-width — or a cache silently falling back
        to storing floats — would show flat error across widths while still
        reporting compact ratios.  This is cheap insurance against exactly
        that: the ordering is the assertion, not any particular value.
        """
        name, model, tok = gqa_lm
        ids = tok(PROMPT, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            reference = model(ids, use_cache=True).logits

        errors = {}
        for bits in (2, 4, 8):
            cache = CompactKVCache(
                n_layers=model.config.num_hidden_layers, bits=bits,
                block_size=128, gqa_factor=4,
            )
            with cache.attached(model):
                with torch.no_grad():
                    got = model(ids, past_key_values=cache, use_cache=True).logits
            errors[bits] = (got - reference).abs().max().item()

        assert errors[8] < errors[4] < errors[2], (
            f"error is not monotone in bit-width: {errors}"
        )


# ---------------------------------------------------------------------------
# The number in the result object
# ---------------------------------------------------------------------------


class TestReportedRatioIsMeasured:
    def test_ratio_equals_measured_bytes(self, gqa_lm, monkeypatch):
        """
        The ratio in GenerateResult must be the one an independent count of
        the surviving cache produces — not `16 / bits`.

        Captures the cache on its way out of the decode loop and re-measures
        it with `cache_nbytes`, a different entry point from the
        `byte_breakdown` the result was built from.
        """
        name, _, _ = gqa_lm
        seen = {}
        real = G._build_compact_cache

        def spy(*a, **k):
            cache, ab, tp, ids = real(*a, **k)
            seen["cache"] = cache
            return cache, ab, tp, ids

        monkeypatch.setattr(G, "_build_compact_cache", spy)
        out = _gen(name, bits=3, max_new_tokens=12)

        measured = cache_nbytes(seen["cache"])
        assert out.cache_bytes == measured.total
        assert out.sidecar_bytes == measured.sidecar_bytes
        assert out.compression_ratio == pytest.approx(measured.compression_ratio)
        assert out.avg_bits_per_dim == pytest.approx(measured.bits_per_coord)

    def test_ratio_is_not_the_nominal_figure(self, gqa_lm):
        """
        The measured ratio must differ from the old nominal `16 / bits`.

        The nominal number ignores the per-vector float32 norms the codec
        stores, so a measured figure that happens to equal it would mean the
        sidecar went uncounted — the exact dishonesty this phase removes.
        """
        name, _, _ = gqa_lm
        out = _gen(name, bits=3, max_new_tokens=12)
        assert out.sidecar_bytes > 0, "sidecar norms were not counted"
        assert out.avg_bits_per_dim > 3.0, (
            "measured bits/coord is at or below the nominal width, so the "
            "sidecar is being ignored"
        )

    @pytest.mark.parametrize("n_new", [1, 16, 63, 64, 127, 128])
    def test_ratio_does_not_depend_on_where_generation_stopped(
        self, gqa_lm, n_new
    ):
        """
        The achieved ratio must be the codec's, not an artifact of the token
        budget.

        Generated tokens land in the same float pending window as prefill, so
        a run ending mid-block leaves up to `block_size - 1` positions
        uncompressed.  Before the end-of-generation flush this sawtoothed
        across `max_new_tokens` on one fixed prompt — 3.56x at 128 generated
        tokens, 1.17x at 127, 1.59x at 64 — which is a property of the stopping
        point rather than of the compression.

        The threshold is deliberately tight.  An earlier version of this test
        asserted `> 1.5` and passed against every one of those broken values,
        which is the failure mode a loose bound produces: green while the
        thing it exists to protect is broken.
        """
        name, _, _ = gqa_lm
        out = _gen(name, bits=3, max_new_tokens=n_new)
        assert out.compression_ratio > 3.0, (
            f"only {out.compression_ratio:.2f}x after generating {n_new} "
            f"tokens; the tail was probably left uncompressed"
        )


# ---------------------------------------------------------------------------
# stream() parity
# ---------------------------------------------------------------------------


class TestStreamMatchesGenerate:
    def test_same_text_greedy(self, gqa_lm):
        """
        Streaming must assemble to what generate() returns.

        generate() runs `_clean_text` (whitespace collapse, think-block
        stripping) and stream() deliberately does not, so compare on
        whitespace-normalized tokens rather than raw strings.
        """
        name, _, _ = gqa_lm
        out = _gen(name, bits=4, max_new_tokens=12, temperature=0.0)
        chunks = list(
            G.stream(PROMPT, model=name, bits=4, max_new_tokens=12,
                     temperature=0.0, raw=True, device="cpu")
        )
        assert chunks, "stream() yielded nothing"
        assert "".join(chunks).split() == out.text.split()

    def test_yields_progressively(self, gqa_lm):
        """More than one fragment, i.e. it streams rather than batching."""
        name, _, _ = gqa_lm
        chunks = list(
            G.stream(PROMPT, model=name, bits=4, max_new_tokens=12,
                     temperature=0.0, raw=True, device="cpu")
        )
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Attachment hygiene
# ---------------------------------------------------------------------------


class TestAttachmentIsRestored:
    def test_attn_impl_restored_after_generate(self, gqa_lm):
        name, model, _ = gqa_lm
        before = model.config._attn_implementation
        _gen(name, bits=4)
        assert model.config._attn_implementation == before

    def test_attn_impl_restored_after_exception(self, gqa_lm, monkeypatch):
        """
        A crash mid-generation must not leave the model wired to the compact
        attention function with no cache behind it.

        That state is worse than the crash: the next unrelated forward pass
        would raise from inside attention, far from the real cause.
        """
        name, model, _ = gqa_lm
        before = model.config._attn_implementation

        real = G._sample_next
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("synthetic failure mid-generation")
            return real(*a, **k)

        monkeypatch.setattr(G, "_sample_next", boom)
        with pytest.raises(RuntimeError, match="synthetic failure"):
            _gen(name, bits=4, max_new_tokens=8)

        assert model.config._attn_implementation == before

    def test_partial_stream_consumption_detaches(self, gqa_lm):
        """
        Abandoning a stream() part-way must still restore the model.

        A generator that is garbage-collected before exhaustion has
        GeneratorExit thrown into it at the current yield; the `finally` in
        `_compact_decode` is what turns that into a clean detach.  Without it,
        `for chunk in stream(...): break` would leave the model unusable.
        """
        name, model, _ = gqa_lm
        before = model.config._attn_implementation

        gen = G.stream(PROMPT, model=name, bits=4, max_new_tokens=16,
                       raw=True, device="cpu")
        next(gen)
        gen.close()

        assert model.config._attn_implementation == before
        # And the model still works afterwards.
        with torch.no_grad():
            model(torch.tensor([[1, 2, 3]]))


# ---------------------------------------------------------------------------
# Memory, on CUDA
# ---------------------------------------------------------------------------


class TestEndToEndMemory:
    @pytest.mark.cuda
    def test_resident_cache_smaller_than_float_path(self, gqa_lm, warm_cuda_codec,
                                                    monkeypatch):
        """
        The end-to-end claim: after generating, the compact path is holding
        fewer cache bytes than the float path is.

        Resident bytes rather than peak VRAM, deliberately.  Peak during a
        forward is dominated by the attention kernel's scratch space — SDPA
        and our block loop have very different workspaces — so a peak
        comparison between the two paths measures kernels, not storage, and
        earlier reported the compact path as 4.6x *worse*.  Storage is what
        the compression claim is about, so storage is what this measures.
        """
        name, model, _ = gqa_lm
        model.cuda()
        try:
            caches = {}
            real_compact = G._build_compact_cache
            real_float = G._build_quantized_cache

            def spy_c(*a, **k):
                r = real_compact(*a, **k)
                caches["compact"] = r[0]
                return r

            def spy_f(*a, **k):
                r = real_float(*a, **k)
                caches["float"] = r[0]
                return r

            monkeypatch.setattr(G, "_build_compact_cache", spy_c)
            monkeypatch.setattr(G, "_build_quantized_cache", spy_f)

            _gen(name, bits=3, max_new_tokens=12, device="cuda")
            _gen(name, bits=3, max_new_tokens=12, device="cuda",
                 compact_cache=False)

            compact_bytes = cache_nbytes(caches["compact"]).total
            float_bytes = cache_nbytes(caches["float"]).total
            assert compact_bytes < float_bytes, (
                f"compact cache held {compact_bytes} bytes vs float "
                f"{float_bytes}"
            )
        finally:
            model.cpu()
            torch.cuda.empty_cache()
