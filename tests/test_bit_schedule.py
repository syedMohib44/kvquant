"""
The outlier bit schedule: monotone, premium-preserving, and the same everywhere.

Three code paths derive outlier/regular bit-widths independently — the compact
cache, the disk-offload tier, and KVCacheQuantizer — and they had drifted.  Two
capped ``outlier_bits`` at 4 before adding the GQA allowance; the third did not.
The cap came with a citation to a "paper §5.1" [non-existent-section] — the
paper ends at §4.4 and prescribes no schedule — and it quietly broke the thing
the codec is for:

    outlier premium (ob - rb)   b=2  b=3  b=4  b=5
      capped at 4:                2    2    1    0
      capped at 8:                2    2    2    2

At ``bits=5`` outliers got no extra bits at all — two quantizers, two rotations
and two sets of norms, all to reproduce uniform quantization.  These tests pin
the properties that make the schedule worth having, rather than pinning the
specific numbers, so a future retune stays free as long as it stays sane.
"""

from __future__ import annotations

import math

import pytest

from src.compact_cache import _make_outlier_quantizer

# 8 is the codec's real ceiling: _pack_indices asserts num_bits <= 8.
MAX_BITS = 8


def _offload_schedule(bits: int, gqa_factor: int, dim: int) -> tuple[int, int]:
    """The disk-offload tier's arithmetic, mirrored from _get_outlier_q."""
    ge = math.ceil(math.log(gqa_factor, 4)) if gqa_factor > 1 else 0
    return min(bits + 1 + ge, MAX_BITS), min(max(max(bits - 1, 1) + ge, 1), MAX_BITS)


def _compact_schedule(bits: int, gqa_factor: int, dim: int) -> tuple[int, int]:
    q = _make_outlier_quantizer(dim, bits, gqa_factor)
    return q.outlier_bits, q.regular_bits


# Real head dims and real GQA factors: Qwen2.5-0.5B is g=7, Llama-3 g=4, MHA g=1.
_SHAPES = [(64, 1), (64, 4), (64, 7), (128, 1), (128, 4), (128, 7), (256, 8)]
_BITS = [2, 3, 4, 5, 6]


@pytest.mark.parametrize("dim,gqa", _SHAPES)
@pytest.mark.parametrize("bits", _BITS)
def test_outliers_always_get_more_bits(dim: int, gqa: int, bits: int):
    """
    The premium must survive every (bits, dim, g) combination.

    This is the test the old cap-at-4 schedule fails, at bits >= 4.  Without a
    premium the outlier split is pure overhead: it costs an extra quantizer, an
    extra rotation and an extra norm per vector to produce what a single
    uniform quantizer would.
    """
    ob, rb = _compact_schedule(bits, gqa, dim)
    if rb >= MAX_BITS:
        pytest.skip("regular group already at the 8-bit ceiling; no room above it")
    assert ob > rb, (
        f"dim={dim} g={gqa} bits={bits}: outlier={ob} regular={rb} — outliers "
        f"gain nothing, so the split is pure overhead"
    )


@pytest.mark.parametrize("dim,gqa", _SHAPES)
def test_schedule_is_monotone_in_bits(dim: int, gqa: int):
    """
    Asking for more bits must never store fewer.

    Non-monotonicity is what makes a quality regression impossible to diagnose:
    a user raising `bits` to fix output quality would see it get worse, and
    would reasonably blame the codec rather than the schedule.
    """
    prev_ob = prev_rb = -1
    for bits in _BITS:
        ob, rb = _compact_schedule(bits, gqa, dim)
        assert ob >= prev_ob, f"dim={dim} g={gqa}: outlier_bits fell at bits={bits}"
        assert rb >= prev_rb, f"dim={dim} g={gqa}: regular_bits fell at bits={bits}"
        prev_ob, prev_rb = ob, rb


@pytest.mark.parametrize("dim,gqa", _SHAPES)
@pytest.mark.parametrize("bits", _BITS)
def test_schedules_agree_across_code_paths(dim: int, gqa: int, bits: int):
    """
    The compact cache and the disk-offload tier must derive identical widths.

    They are separate implementations of one schedule, and they had already
    drifted once.  Disagreement is not merely untidy: `avg_bits` is reported
    from one and the bytes are written by the other, so a mismatch makes the
    published compression figures describe a configuration nothing ran.
    """
    assert _compact_schedule(bits, gqa, dim) == _offload_schedule(bits, gqa, dim), (
        f"dim={dim} g={gqa} bits={bits}: compact_cache and kv_cache disagree"
    )


@pytest.mark.parametrize("dim,gqa", _SHAPES)
@pytest.mark.parametrize("bits", _BITS)
def test_widths_stay_within_the_codec_limit(dim: int, gqa: int, bits: int):
    """Both groups must be packable: 1 <= width <= 8, since _pack_indices asserts it."""
    ob, rb = _compact_schedule(bits, gqa, dim)
    assert 1 <= rb <= MAX_BITS, f"regular_bits={rb} out of range"
    assert 1 <= ob <= MAX_BITS, f"outlier_bits={ob} out of range"


@pytest.mark.parametrize("dim,gqa", _SHAPES)
@pytest.mark.parametrize("bits", _BITS)
def test_avg_bits_matches_what_is_stored(dim: int, gqa: int, bits: int):
    """
    The reported ``avg_bits`` must be the weighted mean of the widths actually
    used, not a nominal figure.

    Recomputed here from the group sizes rather than trusting the property, so
    a change to either the split or the widths that forgets the other is caught.
    """
    q = _make_outlier_quantizer(dim, bits, gqa)
    expected = (
        q.n_outlier * q.outlier_bits + q.n_regular * q.regular_bits
    ) / q.dim
    assert q.avg_bits == pytest.approx(expected)
    assert q.n_outlier + q.n_regular == dim


def test_gqa_allowance_grows_with_group_size():
    """
    More query heads per KV head must not mean fewer bits.

    The allowance is ceil(log4(g)) — ours, not the paper's, which never mentions
    GQA.  Its justification is that an error in one KV head is amplified across
    all g query heads reading it, so the schedule must be non-decreasing in g.
    """
    prev = -1
    for g in (1, 2, 4, 7, 8, 16, 64):
        ob, _ = _compact_schedule(bits=3, gqa_factor=g, dim=64)
        assert ob >= prev, f"g={g}: outlier_bits fell to {ob} as g grew"
        prev = ob
