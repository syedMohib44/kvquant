"""
Tests for the measurement instruments themselves (Phase 0).

Every memory and storage claim downstream is read off these two tools, so
they get their own tests.  A broken ruler does not fail loudly — it silently
reports whatever we hoped to see.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.kv_cache import _paper_compress, _paper_outlier_compress
from src.outlier import OutlierKVQuant
from src.quantizer import KVQuantMSE
from tests.cache_bytes import ByteBreakdown, cache_nbytes, codec_bytes
from tests.vram import measure_peak_vram

D = 64


def _mse_factory(dim: int, bits: int) -> KVQuantMSE:
    return KVQuantMSE(dim=dim, num_bits=bits, seed=0)


# ---------------------------------------------------------------------------
# VRAM harness
# ---------------------------------------------------------------------------


class TestVramHarness:
    @pytest.mark.cuda
    def test_measures_known_allocation(self):
        """
        Allocate a known 64 MiB inside the measured callable and assert the
        reported peak lands within 10%.  This is the calibration test: if it
        fails, no other VRAM number in the suite can be trusted.
        """
        want = 64 * 1024 * 1024

        def alloc():
            t = torch.empty(want // 4, dtype=torch.float32, device="cuda")
            t.fill_(1.0)
            return t.sum()

        _, peak = measure_peak_vram(alloc)
        assert peak == pytest.approx(want, rel=0.10)

    @pytest.mark.cuda
    def test_baseline_is_subtracted(self):
        """
        Memory allocated *before* the call must not be attributed to it.
        Without baseline subtraction, model weights would swamp every cache
        measurement.
        """
        resident = torch.empty(32 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        resident.fill_(2.0)

        want = 16 * 1024 * 1024

        def alloc():
            t = torch.empty(want // 4, dtype=torch.float32, device="cuda")
            t.fill_(1.0)
            return t.sum()

        _, peak = measure_peak_vram(alloc)
        del resident
        assert peak == pytest.approx(want, rel=0.15)

    def test_returns_zero_on_cpu_without_crashing(self):
        result, peak = measure_peak_vram(lambda: torch.zeros(8).sum())
        assert peak == 0
        assert result is not None


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_paper_kv_counts_norms_as_sidecar(self):
        """
        `_PaperKV.norms` is a real float32 cost the nominal `16/avg_bits`
        ratio ignores.  It must show up as sidecar, not vanish.
        """
        torch.manual_seed(0)
        t = torch.randn(1, 4, 32, D)
        q = _paper_compress(t, 4, _mse_factory)

        b = codec_bytes(q)
        assert b.code_bytes > 0
        assert b.sidecar_bytes > 0, "norms must be counted"
        assert b.detail["norms"] == b.sidecar_bytes
        assert b.n_elements == t.numel()

    def test_norms_overhead_is_half_a_bit_per_coord_at_d64(self):
        """
        Pin the magnitude of the overhead the old ratio hid: one float32 norm
        per d-coordinate vector is 32/d bits per coordinate = 0.5 at d=64.
        """
        torch.manual_seed(0)
        t = torch.randn(1, 2, 16, D)
        q = _paper_compress(t, 4, _mse_factory)
        b = codec_bytes(q)

        sidecar_bits_per_coord = b.sidecar_bytes * 8.0 / b.n_elements
        assert sidecar_bits_per_coord == pytest.approx(32.0 / D, rel=0.05)

    def test_measured_bits_exceed_nominal(self):
        """
        The honest figure must be strictly worse than the nominal one — that
        gap is the whole reason this module exists.
        """
        torch.manual_seed(0)
        t = torch.randn(1, 4, 32, D)
        nominal_bits = 4
        b = codec_bytes(_paper_compress(t, nominal_bits, _mse_factory))
        assert b.bits_per_coord > nominal_bits

    def test_outlier_kv_counts_both_norm_tensors(self):
        torch.manual_seed(0)
        t = torch.randn(1, 4, 32, D)
        oq = OutlierKVQuant(D, D // 4, 4, 2, seed=0, quantizer_cls=KVQuantMSE)
        oq.calibrate(t.reshape(-1, D))
        q = _paper_outlier_compress(t, oq)

        b = codec_bytes(q)
        assert b.code_bytes > 0
        assert b.sidecar_bytes > 0
        assert b.n_elements == t.numel()

    def test_packed_payload_smaller_than_float16(self):
        """Sanity: the payload alone must beat float16, or nothing works."""
        torch.manual_seed(0)
        t = torch.randn(1, 4, 64, D)
        b = codec_bytes(_paper_compress(t, 3, _mse_factory))
        assert b.code_bytes < b.float16_bytes

    def test_compression_ratio_is_measured_not_nominal(self):
        torch.manual_seed(0)
        t = torch.randn(1, 4, 64, D)
        b = codec_bytes(_paper_compress(t, 3, _mse_factory))
        # Nominal would be 16/3 = 5.33x; measured must be lower once norms
        # are counted.
        assert 1.0 < b.compression_ratio < 16.0 / 3.0

    def test_float_cache_classified_as_float(self):
        """A plain float cache must report float bytes, no codes."""
        torch.manual_seed(0)
        pairs = tuple(
            (torch.randn(1, 2, 8, D), torch.randn(1, 2, 8, D)) for _ in range(3)
        )
        b = cache_nbytes(pairs)
        assert b.float_bytes > 0
        assert b.code_bytes == 0
        assert b.compression_ratio == pytest.approx(0.5, rel=0.01)  # fp32 vs fp16

    def test_breakdown_addition_merges_detail(self):
        a = ByteBreakdown(code_bytes=10, sidecar_bytes=2, n_elements=8)
        a.detail["norms"] = 2
        c = ByteBreakdown(code_bytes=5, sidecar_bytes=3, n_elements=4)
        c.detail["norms"] = 3
        m = a + c
        assert m.code_bytes == 15
        assert m.sidecar_bytes == 5
        assert m.n_elements == 12
        assert m.detail["norms"] == 5
        assert m.total == 20

    def test_unrecognised_cache_type_raises(self):
        with pytest.raises(TypeError, match="unrecognised cache type"):
            cache_nbytes(object())
