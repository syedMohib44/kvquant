"""
``max_vram_tokens`` bounds resident floats, and ``memory_summary`` says so.

These two defects were a pair.  ``max_vram_tokens`` was stored in ``__init__``
and never read by anything, so the documented bound did not exist; and
``memory_summary()`` hard-coded ``"vram_layers": 0`` with a comment claiming
"nothing is held dequantized between calls".  The comment was false — staged
floats persist between calls by design, since that is what makes the
incremental decode O(T) instead of O(T^2) — and because the reported number was
a constant, no test could have caught the missing bound.  A false instrument
hides the thing it is supposed to measure.

So the ordering matters here: ``test_summary_reports_resident_floats`` must fail
against the hard-coded 0 before ``test_budget_bounds_resident_tokens`` means
anything.  Both were verified to fail against the pre-fix code.

Eviction correctness is the third leg: dropping a staged float must be a pure
recomputation, never a fidelity loss, because the codes it decodes from are
immutable.  ``test_eviction_is_bit_identical`` pins exact equality rather than a
tolerance — anything looser would accept a re-encode, which is precisely the
append-only violation this cache exists to avoid.
"""

from __future__ import annotations

import torch

from src.kv_cache import KVCacheDiskOffload

D = 16
SEED = 1234


def _fake_cache(n_layers=6, B=1, H=2, T=8, d=D):
    torch.manual_seed(SEED)
    return tuple(
        (torch.randn(B, H, T, d), torch.randn(B, H, T, d)) for _ in range(n_layers)
    )


def _staged_pairs(staged):
    if isinstance(staged, tuple):
        return list(staged)
    return [(l.keys, l.values) for l in staged.layers]


def _offload(**kw):
    kw.setdefault("warm_size", 0)
    kw.setdefault("disk_dir", None)
    kw.setdefault("codec", "paper")
    kw.setdefault("bits", 4)
    kw.setdefault("device", "cpu")
    return KVCacheDiskOffload(**kw)


class TestSummaryIsTruthful:
    def test_summary_reports_resident_floats(self):
        """
        After staging, the summary must report the floats actually held.

        This is the test that fails against the old hard-coded
        ``"vram_layers": 0`` — the entire point of writing it.
        """
        cache = _fake_cache(n_layers=6)
        off = _offload(max_vram_tokens=0)  # unbounded: nothing may be evicted
        try:
            off.store(cache)
            off.stage_for_forward()
            s = off.memory_summary()

            assert s["vram_layers"] == 6, (
                f"6 layers staged, summary says {s['vram_layers']}"
            )
            assert s["vram_tokens"] == 8 * 6, (
                f"6 layers x 8 positions, summary says {s['vram_tokens']}"
            )
            # 6 layers x 2 tensors x (1*2*8*16) elements x 4 bytes.
            assert s["vram_float_bytes"] == 6 * 2 * (1 * 2 * 8 * D) * 4
        finally:
            off.close()

    def test_summary_is_zero_before_staging(self):
        """Nothing is resident until stage_for_forward runs.

        Guards the opposite error from the one above: a summary that always
        reports a positive number would pass the previous test while being just
        as uninformative as the constant 0 it replaced.
        """
        off = _offload(max_vram_tokens=0)
        try:
            off.store(_fake_cache())
            s = off.memory_summary()
            assert s["vram_layers"] == 0
            assert s["vram_tokens"] == 0
            assert s["vram_float_bytes"] == 0
        finally:
            off.close()


class TestBudgetBoundsResidency:
    def test_budget_bounds_resident_tokens(self):
        """
        Staging layers one at a time, residency never exceeds the budget.

        Layers are staged individually because ``stage_for_forward`` exempts the
        layers of the current call from eviction — they are about to be read.
        Staging all six at once would legitimately exceed a 24-token budget.
        """
        cache = _fake_cache(n_layers=6)  # 8 positions per layer
        off = _offload(max_vram_tokens=24)  # room for 3 layers
        try:
            off.store(cache)
            for l_idx in range(6):
                off.stage_for_forward(layers=[l_idx])
                s = off.memory_summary()
                assert s["vram_tokens"] <= 24, (
                    f"after staging layer {l_idx}: {s['vram_tokens']} tokens "
                    f"resident, budget 24"
                )
        finally:
            off.close()

    def test_zero_budget_means_unbounded(self):
        """0 is documented as "unlimited, but you'll OOM" — not "evict everything"."""
        cache = _fake_cache(n_layers=6)
        off = _offload(max_vram_tokens=0)
        try:
            off.store(cache)
            for l_idx in range(6):
                off.stage_for_forward(layers=[l_idx])
            assert off.memory_summary()["vram_layers"] == 6
        finally:
            off.close()

    def test_current_layers_are_never_evicted(self):
        """
        A budget smaller than one pass must not evict what the caller is about
        to read.  Returning a freed tensor would be a correctness bug, not just
        a performance one, so this asserts the returned data is intact rather
        than merely that the call succeeded.
        """
        cache = _fake_cache(n_layers=4)
        off = _offload(max_vram_tokens=1)  # far below one pass
        try:
            off.store(cache)
            pairs = _staged_pairs(off.stage_for_forward())
            assert len(pairs) == 4
            for k_hat, v_hat in pairs:
                assert k_hat is not None and v_hat is not None
                assert k_hat.shape[-2] == 8
                assert torch.isfinite(k_hat).all()
        finally:
            off.close()

    def test_eviction_order_is_least_recently_staged(self):
        """The layer staged longest ago goes first, not an arbitrary dict entry."""
        cache = _fake_cache(n_layers=4)
        off = _offload(max_vram_tokens=16)  # room for 2 layers
        try:
            off.store(cache)
            off.stage_for_forward(layers=[0])
            off.stage_for_forward(layers=[1])
            off.stage_for_forward(layers=[2])  # must evict layer 0, not 1
            resident = set(off._staged)
            assert 0 not in resident, f"layer 0 was oldest but survived: {resident}"
            assert 2 in resident, "the just-staged layer must be resident"
        finally:
            off.close()


class TestEvictionIsLossless:
    def test_eviction_is_bit_identical(self):
        """
        Re-staging an evicted layer reproduces it exactly.

        Exact equality, not a tolerance: eviction discards decoded floats only,
        and the codes it re-decodes from are immutable.  Any difference would
        mean something re-encoded, which breaks compress-exactly-once and would
        let the lossy codec drift across a long generation.
        """
        cache = _fake_cache(n_layers=4)
        off = _offload(max_vram_tokens=0)
        try:
            off.store(cache)
            before = _staged_pairs(off.stage_for_forward(layers=[0]))[0]
            k_before = before[0].clone()
            v_before = before[1].clone()

            # Force layer 0 out, then bring it back.
            off._staged.clear()
            off._stage_order.clear()
            after = _staged_pairs(off.stage_for_forward(layers=[0]))[0]

            assert torch.equal(k_before, after[0]), "K changed across eviction"
            assert torch.equal(v_before, after[1]), "V changed across eviction"
        finally:
            off.close()

    def test_generation_survives_a_tight_budget(self):
        """
        A store/stage/append cycle under a budget too small to hold everything
        still reconstructs every layer at full length.

        This is the integration check: the unit tests above could all pass while
        eviction quietly desynchronised the per-layer chunk counters, which
        would show up here as a short or missing layer.
        """
        cache = _fake_cache(n_layers=4, T=8)
        off = _offload(max_vram_tokens=8)
        try:
            off.store(cache)
            # Append two more positions to every layer, as a decode step would.
            for step in range(2):
                grown = tuple(
                    (
                        torch.cat([k, torch.randn(1, 2, 1, D)], dim=-2),
                        torch.cat([v, torch.randn(1, 2, 1, D)], dim=-2),
                    )
                    for k, v in _staged_pairs(off.stage_for_forward())
                )
                off.replace(grown)

            pairs = _staged_pairs(off.stage_for_forward())
            assert len(pairs) == 4
            for k_hat, v_hat in pairs:
                assert k_hat.shape[-2] == 10, f"expected 10 positions, got {k_hat.shape[-2]}"
                assert v_hat.shape[-2] == 10
                assert torch.isfinite(k_hat).all() and torch.isfinite(v_hat).all()
        finally:
            off.close()
