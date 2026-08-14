"""
Honest byte accounting for quantized KV caches.

The reported `compression_ratio` has historically been the *nominal*
`16 / avg_bits_per_dim`, which counts only the packed indices.  Real
representations carry side information that the nominal figure ignores:

  * `_PaperKV.norms` — one float32 per vector.  At `d=64` that is 32 bits
    spread over 64 coordinates, i.e. **0.5 bits/coord of pure overhead**, or
    a sixth of a 3-bit budget.
  * `_PaperOutlierKV` carries *two* norm tensors, one per sub-quantizer.
  * Low-rank correction adds `r * (T + d)` floats per matrix.
  * Attention-weighted quantization adds an `(N, T)` bool mask.
  * Product quantization adds an `(M, K, sub_dim)` codebook.

This module measures what is actually stored, split into payload vs
sidecar, so a claim can be stated in measured bytes rather than nominal
bits.  Anything it cannot classify is counted in `total` and surfaced in
`unknown_bytes` rather than silently dropped — an accounting tool that
quietly ignores what it does not recognise is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class ByteBreakdown:
    """Measured storage for one cache, or one layer of one."""

    code_bytes: int = 0      # bit-packed indices — the actual payload
    float_bytes: int = 0     # uncompressed float K/V still held
    sidecar_bytes: int = 0   # norms, masks, codebooks, correction factors
    unknown_bytes: int = 0   # anything unclassified (never silently dropped)
    n_elements: int = 0      # logical KV coordinates represented
    detail: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return (
            self.code_bytes
            + self.float_bytes
            + self.sidecar_bytes
            + self.unknown_bytes
        )

    @property
    def bits_per_coord(self) -> float:
        """Measured bits per logical KV coordinate, sidecars included."""
        if self.n_elements == 0:
            return 0.0
        return self.total * 8.0 / self.n_elements

    @property
    def float16_bytes(self) -> int:
        """What the same logical KV would cost stored as float16."""
        return self.n_elements * 2

    @property
    def compression_ratio(self) -> float:
        """Measured float16-equivalent bytes / measured actual bytes."""
        if self.total == 0:
            return 0.0
        return self.float16_bytes / self.total

    def __add__(self, other: "ByteBreakdown") -> "ByteBreakdown":
        merged = ByteBreakdown(
            code_bytes=self.code_bytes + other.code_bytes,
            float_bytes=self.float_bytes + other.float_bytes,
            sidecar_bytes=self.sidecar_bytes + other.sidecar_bytes,
            unknown_bytes=self.unknown_bytes + other.unknown_bytes,
            n_elements=self.n_elements + other.n_elements,
        )
        for key in set(self.detail) | set(other.detail):
            merged.detail[key] = self.detail.get(key, 0) + other.detail.get(key, 0)
        return merged


def _nbytes(t: Tensor) -> int:
    return t.numel() * t.element_size()


def _logical_elements(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def codec_bytes(q) -> ByteBreakdown:
    """
    Measure one compressed K or V object.

    Handles `_PaperKV`, `_PaperOutlierKV`, `QuantizedMSE`, `QuantizedIP`,
    `QuantizedPQ`, `CorrectedQuantized`, and raw float tensors.  Unrecognised
    objects contribute their reachable tensors to `unknown_bytes`.
    """
    b = ByteBreakdown()

    if isinstance(q, Tensor):
        if q.dtype.is_floating_point:
            b.float_bytes += _nbytes(q)
        else:
            b.code_bytes += _nbytes(q)
        b.n_elements += q.numel()
        return b

    name = type(q).__name__

    if name == "_PaperKV":
        b.code_bytes += _nbytes(q.packed)
        b.sidecar_bytes += _nbytes(q.norms)
        b.detail["norms"] = _nbytes(q.norms)
        b.n_elements += _logical_elements(q.shape)
        return b

    if name == "_PaperOutlierKV":
        b.code_bytes += _nbytes(q.out_packed) + _nbytes(q.reg_packed)
        sidecar = _nbytes(q.out_norms) + _nbytes(q.reg_norms)
        b.sidecar_bytes += sidecar
        b.detail["norms"] = sidecar
        b.n_elements += _logical_elements(q.shape)
        return b

    if name == "QuantizedMSE":
        # int64 indices — deliberately counted as-is, not as the packed
        # width they *could* occupy.  Measuring the aspiration rather than
        # the allocation is exactly the error this module exists to avoid.
        b.code_bytes += _nbytes(q.indices)
        b.sidecar_bytes += _nbytes(q.norms)
        b.detail["norms"] = _nbytes(q.norms)
        b.n_elements += _logical_elements(q.shape)
        return b

    if name == "QuantizedIP":
        b.code_bytes += _nbytes(q.indices) + _nbytes(q.qjl_bits)
        sidecar = _nbytes(q.r_norm) + _nbytes(q.vec_norms)
        b.sidecar_bytes += sidecar
        b.detail["norms"] = sidecar
        b.n_elements += _logical_elements(q.shape)
        return b

    if name == "QuantizedPQ":
        b.code_bytes += _nbytes(q.codes)
        b.sidecar_bytes += _nbytes(q.norms)
        b.detail["norms"] = _nbytes(q.norms)
        b.n_elements += _logical_elements(q.shape)
        return b

    if name == "OutlierQuantized":
        inner = codec_bytes(q.outlier_q) + codec_bytes(q.regular_q)
        # The sub-objects' n_elements double-count the logical coordinates
        # (each covers a channel subset); use the outer shape instead.
        inner.n_elements = _logical_elements(q.shape)
        return inner

    if name == "CorrectedQuantized":
        inner = codec_bytes(q.base_q)
        factors = _nbytes(q.U) + _nbytes(q.V)
        inner.sidecar_bytes += factors
        inner.detail["lowrank_UV"] = factors
        inner.n_elements = _logical_elements(q.shape)
        return inner

    if name == "AttentionWeightedQuantized":
        inner = codec_bytes(q.hi_q) + codec_bytes(q.lo_q)
        mask = _nbytes(q.top_mask)
        inner.sidecar_bytes += mask
        inner.detail["top_mask"] = mask
        inner.n_elements = _logical_elements(q.shape)
        return inner

    # Unrecognised: sum whatever tensors we can reach so the total stays
    # honest, and record that classification failed.
    for attr in getattr(q, "_fields", ()) or ():
        val = getattr(q, attr, None)
        if isinstance(val, Tensor):
            b.unknown_bytes += _nbytes(val)
    b.detail[f"unclassified:{name}"] = b.unknown_bytes
    return b


def cache_nbytes(past) -> ByteBreakdown:
    """
    Measure a whole cache: a `CompactKVCache`, a `DynamicCache`, or a
    legacy tuple-of-tuples.  Returns the summed breakdown across layers.
    """
    total = ByteBreakdown()

    # Compact cache: layers expose code blocks plus a float pending window.
    layers = getattr(past, "layers", None)
    if layers is not None:
        for layer in layers:
            if hasattr(layer, "byte_breakdown"):
                total = total + layer.byte_breakdown()
                continue
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if isinstance(t, Tensor) and t.numel():
                    total = total + codec_bytes(t)
        return total

    if hasattr(past, "key_cache"):
        for k, v in zip(past.key_cache, past.value_cache):
            if isinstance(k, Tensor):
                total = total + codec_bytes(k) + codec_bytes(v)
        return total

    if isinstance(past, (tuple, list)):
        for k, v in past:
            total = total + codec_bytes(k) + codec_bytes(v)
        return total

    raise TypeError(f"cache_nbytes: unrecognised cache type {type(past).__name__}")
