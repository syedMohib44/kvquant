"""
KV-cache quantizer built on top of KVQuantIP / OutlierKVQuant.

KVCacheQuantizer wraps either KVQuantIP or OutlierKVQuant and
provides a simple .compress() / .decompress() interface that matches
the (batch, heads, seq, dim) layout used by most transformer KV caches.

Usage
-----
    quant = KVCacheQuantizer(head_dim=128, num_bits=3, use_outlier=True)
    quant.calibrate(k_calib, v_calib)   # one-shot calibration

    # During generation:
    k_c = quant.compress(k, is_value=False)   # store compressed keys
    v_c = quant.compress(v, is_value=True)    # store compressed values
    k_hat = quant.decompress(k_c)             # recover for attention

The compressed representation is stored as Python objects (NamedTuples)
rather than packed bit-strings - suitable for research / profiling.  A
production system would add entropy coding on top.
"""

from __future__ import annotations

import copy
import math
import shutil as _shutil
import tempfile
import threading
import weakref
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple

from .quantizer import KVQuantIP, KVQuantMSE, QuantizedIP, QuantizedMSE
from .outlier import OutlierKVQuant, OutlierQuantized


# Accept any quantized representation (K uses IP, V uses MSE)
CompressedKV = QuantizedIP | QuantizedMSE | OutlierQuantized


# ---------------------------------------------------------------------------
# Model-cache utilities
# These live here because they operate on KVCacheQuantizer objects and on the
# native cache format returned by HuggingFace models.  Keeping them in one
# place prevents the same logic from being duplicated across generate.py,
# demo scripts, and user code.
# ---------------------------------------------------------------------------


def kvs_from_cache(past_key_values) -> list[tuple[Tensor, Tensor]]:
    """
    Extract a flat list of (k, v) tensors from any HuggingFace cache format.

    Handles:
      - new DynamicCache  (transformers >= 4.50): .layers[i].keys / .values
      - old DynamicCache  (transformers 4.38-4.49): .key_cache[i] / .value_cache[i]
      - tuple-of-tuples   (transformers < 4.38): ((k0,v0), (k1,v1), …)

    Skips None entries (linear-attn / sliding-window / Mamba layers in hybrid
    models).  Returns only transformer-attention layers that have real KV tensors.

    Args:
        past_key_values: The cache object returned by model(..., use_cache=True).

    Returns:
        List of (k, v) tensor pairs, one per transformer attention layer.
    """
    if hasattr(past_key_values, "layers"):
        return [
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if isinstance(getattr(layer, "keys", None), Tensor)
        ]
    if hasattr(past_key_values, "key_cache"):
        return [
            (past_key_values.key_cache[i], past_key_values.value_cache[i])
            for i in range(len(past_key_values.key_cache))
            if past_key_values.key_cache[i] is not None
        ]
    if isinstance(past_key_values, (tuple, list)):
        return [(k, v) for k, v in past_key_values if isinstance(k, Tensor)]
    return []


def quantize_model_cache(
    past_key_values,
    kvc_or_list: "KVCacheQuantizer | list[KVCacheQuantizer]",
    correction_rank: int = 0,
):
    """
    Return a deep copy of ``past_key_values`` with every transformer-attention
    K/V pair compressed then decompressed via ``kvc_or_list``.

    Non-KV state (Mamba hidden state, linear-attention state) is left untouched
    so hybrid models (Qwen3.5, Jamba, …) continue to work correctly.

    Args:
        past_key_values:  Cache from model(..., use_cache=True).
        kvc_or_list:      A single KVCacheQuantizer applied to all layers, or a
                          list of per-layer KVCacheQuantizers (one per attention
                          layer in order).  Per-layer gives better quality because
                          each layer has its own outlier-channel profile.
        correction_rank:  If > 0, apply a rank-r SVD correction to the residual
                          (K - K_hat) before storing.  Reduces quantization error
                          at the cost of storing r*(T+d) extra floats per layer.
                          0 = disabled.

    Returns:
        A deep-copied cache with quantized K/V tensors.
    """
    cache_q = copy.deepcopy(past_key_values)
    _is_list = isinstance(kvc_or_list, list)

    def _compress_pair(k: Tensor, v: Tensor, kvc: "KVCacheQuantizer"):
        dt = k.dtype
        k_hat, v_hat = kvc.decompress_kv(*kvc.compress_kv(k.float(), v.float()))
        if correction_rank > 0:
            k_hat = _low_rank_correct(k.float(), k_hat, correction_rank)
            v_hat = _low_rank_correct(v.float(), v_hat, correction_rank)
        return k_hat.to(dt), v_hat.to(dt)

    if hasattr(cache_q, "layers"):
        idx = 0
        for layer in cache_q.layers:
            k = getattr(layer, "keys", None)
            if isinstance(k, Tensor):
                kvc = kvc_or_list[idx] if _is_list else kvc_or_list
                layer.keys, layer.values = _compress_pair(k, layer.values, kvc)
                idx += 1
        return cache_q

    if hasattr(cache_q, "key_cache"):
        idx = 0
        for i in range(len(cache_q.key_cache)):
            k = cache_q.key_cache[i]
            if isinstance(k, Tensor):
                kvc = kvc_or_list[idx] if _is_list else kvc_or_list
                cache_q.key_cache[i], cache_q.value_cache[i] = _compress_pair(
                    k, cache_q.value_cache[i], kvc
                )
                idx += 1
        return cache_q

    if isinstance(past_key_values, (tuple, list)):
        result, idx = [], 0
        for k, v in past_key_values:
            if isinstance(k, Tensor):
                kvc = kvc_or_list[idx] if _is_list else kvc_or_list
                result.append(_compress_pair(k, v, kvc))
                idx += 1
            else:
                result.append((k, v))
        return tuple(result)

    return cache_q


def crop_model_cache(past_key_values, seq_len: int):
    """
    Return a deep copy of ``past_key_values`` with K/V tensors truncated to
    ``seq_len`` positions along the sequence dimension.

    Used to obtain a T-1 cache for getting accurate first-token logits from the
    compressed cache (see generate.py).  Non-KV state is preserved unchanged.

    Args:
        past_key_values: Cache from model(..., use_cache=True).
        seq_len:         Number of token positions to keep.

    Returns:
        Deep-copied cache truncated to seq_len.
    """
    cache = copy.deepcopy(past_key_values)
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            if isinstance(k, Tensor) and k.shape[-2] > seq_len:
                layer.keys = k[..., :seq_len, :]
                layer.values = layer.values[..., :seq_len, :]
    elif hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i]
            if isinstance(k, Tensor) and k.shape[-2] > seq_len:
                cache.key_cache[i] = k[..., :seq_len, :]
                cache.value_cache[i] = cache.value_cache[i][..., :seq_len, :]
    return cache


def _low_rank_correct(x: Tensor, x_hat: Tensor, rank: int) -> Tensor:
    """Add a rank-r SVD correction of the residual (x - x_hat) to x_hat."""
    orig = x_hat.shape
    N = x_hat.reshape(-1, orig[-2], orig[-1]).shape[0]
    R = (x - x_hat).reshape(N, orig[-2], orig[-1])
    T_len = R.shape[1]
    r = min(rank, T_len - 1, orig[-1] - 1)
    if r < 1:
        return x_hat
    if T_len >= 64:
        try:
            from .correction import _randomized_svd
            U, S, Vh = _randomized_svd(R, rank=r)
        except Exception:
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            U, S, Vh = U[..., :r], S[..., :r], Vh[..., :r, :]
    else:
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        U, S, Vh = U[..., :r], S[..., :r], Vh[..., :r, :]
    correction = (U * S.unsqueeze(-2)) @ Vh
    return (x_hat + correction.reshape(orig))


# ---------------------------------------------------------------------------
# Tiered memory offload: VRAM  ->  CPU-RAM  ->  disk (memmap / pickle)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paper codec: rotate + Lloyd-Max (the paper's core algorithm), stored at the
# true 2–4 bit width via bit-packing.
#
# This is the paper's MSE-optimal path (KVQuantMSE) applied to BOTH K and V.
# We deliberately do NOT use KVQuant-IP for K here: IP is an inner-product
# *estimator* (it preserves Q·K scores, not the K vectors themselves), so its
# per-coordinate reconstruction is unusable as an actual KV cache — feeding it
# back produces garbled generation.  The MSE path reconstructs faithfully and
# is exactly the "rotate → Lloyd-Max quantize" scheme the paper validates.
#
# The Lloyd-Max indices are integers in [0, 2^bits): KVQuantMSE returns them as
# int64 (8 bytes each), which on disk would be *larger* than int8.  We bit-pack
# them to their true `bits` width so the SSD footprint is genuinely ~bits/coord
# (e.g. 3-bit ≈ 3/8 the bytes of int8, 1/5 of float16).
# ---------------------------------------------------------------------------

class _PaperKV(NamedTuple):
    """Rotate + Lloyd-Max representation of one K or V tensor, bit-packed."""
    packed: Tensor      # uint8   bit-packed Lloyd-Max indices (num_bits per coord)
    norms: Tensor       # float32 (..., 1)  per-vector L2 norm
    shape: tuple        # original tensor shape (B, H, T, d)
    num_bits: int       # bits per coordinate (also the pack width)
    dtype: torch.dtype  # original dtype to restore on dequant


def _pack_indices(idx: Tensor, num_bits: int) -> Tensor:
    """
    Bit-pack integer indices (values in [0, 2^num_bits)) into a uint8 stream.

    Torch-native and device-preserving: codes packed on GPU stay on GPU.  The
    previous numpy implementation forced a D2H/H2D round-trip on every call,
    which costs ~1.9 ms per 128-token block — at 28 layers x 2 tensors that is
    ~106 ms per decode token before any model compute, i.e. it dominated
    generation.  The torch version measures ~0.16 ms for the same block.

    Layout: each value contributes `num_bits` bits MSB-first, the stream is
    zero-padded to a byte boundary, and bytes are filled MSB-first.  This
    matches numpy's `packbits` default, so streams written by the old
    implementation remain readable.
    """
    assert 1 <= num_bits <= 8, f"num_bits must be in [1, 8], got {num_bits}"
    flat = idx.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros(0, dtype=torch.uint8, device=idx.device)

    # uint8 is the widest we need (num_bits <= 8) and keeps the (N, num_bits)
    # intermediate as small as possible — it is the peak allocation here.
    flat = flat.to(torch.uint8)
    shifts = torch.arange(
        num_bits - 1, -1, -1, device=idx.device, dtype=torch.uint8
    )
    bits = ((flat.unsqueeze(1) >> shifts) & 1).reshape(-1)

    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat(
            [bits, torch.zeros(pad, dtype=torch.uint8, device=bits.device)]
        )
    weights = 1 << torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    return (bits.reshape(-1, 8) * weights).sum(dim=1, dtype=torch.uint8)


def _unpack_indices(packed: Tensor, num_bits: int, count: int) -> Tensor:
    """
    Inverse of :func:`_pack_indices`: recover `count` integer indices.

    Device-preserving — the returned int64 tensor lives wherever `packed`
    lives, so a GPU-resident cache never round-trips through host memory.
    """
    assert 1 <= num_bits <= 8, f"num_bits must be in [1, 8], got {num_bits}"
    if count == 0:
        return torch.zeros(0, dtype=torch.int64, device=packed.device)

    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = ((packed.unsqueeze(1) >> shifts) & 1).reshape(-1)[: count * num_bits]
    bits = bits.reshape(count, num_bits).to(torch.int64)
    weights = 1 << torch.arange(
        num_bits - 1, -1, -1, device=packed.device, dtype=torch.int64
    )
    return (bits * weights).sum(dim=1)


def _paper_compress(t: Tensor, num_bits: int, get_mse) -> _PaperKV:
    """
    Rotate + Lloyd-Max quantize `t`, returning bit-packed indices + norms.

    Operates on `t`'s own device.  The quantizers are already device-agnostic
    (rotation and boundary buffers all `.to(x.device)` internally), so keeping
    the tensor where it is avoids a host round-trip on the generation path.
    Callers that specifically want codes in host RAM — the disk-offload tier —
    move them explicitly rather than having it forced here.
    """
    dt = t.dtype
    dim = t.shape[-1]
    mse = get_mse(dim, num_bits)
    q = mse.quantize(t.float())                # QuantizedMSE: int64 indices + norms
    packed = _pack_indices(q.indices, num_bits)
    return _PaperKV(
        packed=packed,
        norms=q.norms.reshape(*t.shape[:-1], 1).float(),
        shape=tuple(t.shape),
        num_bits=num_bits,
        dtype=dt,
    )


def _paper_dequantize(q: _PaperKV, get_mse, device=None) -> Tensor:
    """Reconstruct a float tensor from its _PaperKV representation."""
    dim = q.shape[-1]
    mse = get_mse(dim, q.num_bits)
    # math.prod is a single C-level call (and clearer than a loop)
    count = math.prod(q.shape)
    idx = _unpack_indices(q.packed, q.num_bits, count).reshape(q.shape)
    rebuilt = QuantizedMSE(indices=idx, norms=q.norms, shape=q.shape)
    out = mse.dequantize(rebuilt).to(q.dtype)
    return out if device is None else out.to(device)


# ---------------------------------------------------------------------------
# Outlier-aware codec: the paper's §3.1 MSE quantizer applied separately to the
# outlier and regular channel groups described in its §4.3 experimental aside.
#
# Real attention K/V tensors have a few "outlier" channels with much larger
# magnitude than the rest.  Plain Lloyd-Max spends equal precision on every
# channel, so the big channels dominate the error and generation degrades.
# The paper's fix (OutlierKVQuant) splits channels into outlier vs regular and
# quantizes each group at its own bit-width.  This reaches int8-level fidelity
# (cosine ~0.998-1.000) at 3-4 bits/dim on outlier-heavy data.
#
# Unlike the plain paper codec, this one is NOT parameter-free: it must be
# calibrated once on the prefill KVs to identify the outlier channels.  The
# calibrated OutlierKVQuant is supplied by the caller (KVCacheDiskOffload keeps
# one per (layer, is_value)).  We store the packed MSE indices for both the
# outlier and regular sub-quantizers plus their norms; codebooks live in the
# calibrated module, not on disk.
# ---------------------------------------------------------------------------

class _PaperOutlierKV(NamedTuple):
    """Outlier-aware Lloyd-Max representation of one K or V tensor."""
    out_packed: Tensor   # uint8  bit-packed outlier-channel indices
    out_norms: Tensor    # float32 per-vector norms for the outlier sub-quantizer
    out_bits: int
    out_dim: int         # n_outlier (channels in the outlier group)
    reg_packed: Tensor   # uint8  bit-packed regular-channel indices
    reg_norms: Tensor    # float32 per-vector norms for the regular sub-quantizer
    reg_bits: int
    reg_dim: int         # n_regular
    shape: tuple         # original tensor shape (B, H, T, d)
    dtype: torch.dtype


def _paper_outlier_compress(t: Tensor, oq: "OutlierKVQuant") -> _PaperOutlierKV:
    """
    Quantize `t` with a calibrated OutlierKVQuant, bit-packing both groups.

    Device-preserving, for the same reason as :func:`_paper_compress`.
    """
    dt = t.dtype
    q = oq.quantize(t.float())                # OutlierQuantized(outlier_q, regular_q, shape)
    oqn, rqn = q.outlier_q, q.regular_q       # each a QuantizedMSE
    return _PaperOutlierKV(
        out_packed=_pack_indices(oqn.indices, oq.outlier_bits),
        out_norms=oqn.norms.float(),
        out_bits=oq.outlier_bits,
        out_dim=oq.n_outlier,
        reg_packed=_pack_indices(rqn.indices, oq.regular_bits),
        reg_norms=rqn.norms.float(),
        reg_bits=oq.regular_bits,
        reg_dim=oq.n_regular,
        shape=tuple(t.shape),
        dtype=dt,
    )


def _paper_outlier_dequantize(q: _PaperOutlierKV, oq: "OutlierKVQuant", device=None) -> Tensor:
    """Reconstruct a float tensor from its _PaperOutlierKV representation."""
    # math.prod is a single C-level call (and clearer than a loop)
    N = math.prod(q.shape[:-1])
    out_idx = _unpack_indices(q.out_packed, q.out_bits, N * q.out_dim).reshape(N, q.out_dim)
    reg_idx = _unpack_indices(q.reg_packed, q.reg_bits, N * q.reg_dim).reshape(N, q.reg_dim)
    rebuilt = OutlierQuantized(
        outlier_q=QuantizedMSE(indices=out_idx, norms=q.out_norms.reshape(N, 1),
                               shape=(N, q.out_dim)),
        regular_q=QuantizedMSE(indices=reg_idx, norms=q.reg_norms.reshape(N, 1),
                               shape=(N, q.reg_dim)),
        shape=q.shape,
    )
    out = oq.dequantize(rebuilt).to(q.dtype)
    return out if device is None else out.to(device)


class _DiskEntry(NamedTuple):
    """One token-position's K/V data persisted to disk."""
    k_path: str
    v_path: str
    shape: tuple
    dtype: str
    layer_idx: int


class KVCacheDiskOffload:
    """
    Memory-efficient KV cache manager that keeps quantized cache across three tiers:

        Tier 0 (hot):   VRAM – dequantized float K/V for the layers being computed
        Tier 1 (warm):  CPU-RAM – quantized representations as Python / numpy objects
        Tier 2 (cold):  Disk – numpy memmap files (.npy) for very long contexts

    Only the layers needed for the next forward pass are staged to VRAM; all
    other layers remain in Tier 1 or Tier 2 until their layer is accessed again.

    Reconstruction fidelity
    ------------------------
    The cache is stored with the paper's rotate + Lloyd-Max codec ("paper", or
    the outlier-aware "paper-outlier"), which uses KVQuantMSE for both
    K and V.  MSE reconstructs faithfully per coordinate, so the dequantized
    cache drives real generation correctly.  It deliberately does NOT use the
    research KVQuant-IP quantizer: that returns an inner-product *estimator*
    whose per-coordinate values are wrong, so feeding it back as an actual KV
    cache produces garbled output.  The paper codec cuts cache bytes to the true
    2-4 bits/coord (bit-packed) and enables the RAM -> SSD spill for very long
    contexts.

    Usage
    -----
        offload = KVCacheDiskOffload(max_vram_tokens=512, disk_dir="./kv_tmp")
        offload.store(native_cache)           # paper-compress + offload
        past = offload.stage_for_forward()    # dequantize to VRAM (LRU-aware)
        out = model(ids, past_key_values=past, use_cache=True)
        offload.replace(out.past_key_values)  # freeze new token(s), offload again

    Args:
        max_vram_tokens: Max number of token positions to keep dequantized in VRAM
                         simultaneously (0 = unlimited, but you'll OOM).
        disk_dir:        Directory for spill files (auto-cleaned on close).
        warm_size:       Max number of full-layer entries to keep in CPU
                         RAM before spilling to disk (0 = unlimited).
        pin_memory:      Pin CPU tensors for faster host->device transfer.
        cleanup_on_del:  Delete disk files when this object is garbage-collected.
        quantizer:       Deprecated/ignored — kept for backwards-compatible call
                         sites.  The paper-outlier codec calibrates its own
                         quantizer from the prefill; "paper" needs none.
    """

    def __init__(
        self,
        quantizer=None,
        max_vram_tokens: int = 512,
        disk_dir: str | None = None,
        warm_size: int = 16,
        pin_memory: bool = True,
        cleanup_on_del: bool = True,
        codec: str = "paper",
        bits: int = 3,
        device: "str | torch.device | None" = None,
        gqa_factor: int = 1,
    ):
        # Codecs (both are the paper's method — no non-paper fallback):
        #   "paper"         plain rotate + Lloyd-Max, bit-packed (parameter-free)
        #   "paper-outlier" outlier-aware Lloyd-Max (paper §3.1 + §4.3) — calibrated
        #                   once on the prefill; best fidelity on real KV tensors.
        if codec not in ("paper", "paper-outlier"):
            raise ValueError(
                f"codec must be 'paper' or 'paper-outlier', got {codec!r}"
            )
        self._codec = codec
        self._bits = bits
        self._gqa_factor = max(int(gqa_factor), 1)
        self._max_vram_tokens = max_vram_tokens
        self._warm_size = warm_size
        # Target device for staged (dequantized) tensors.  Defaults to CUDA when
        # available, else CPU — but the caller should pass the model's actual
        # device so offload works on CPU / MPS / a specific GPU, not just CUDA.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        # Pinning host memory only helps (and is only valid) for CUDA transfers.
        self._pin_memory = pin_memory and self._device.type == "cuda"

        self._disk_dir = Path(disk_dir) if disk_dir else Path(tempfile.mkdtemp(prefix="kv_offload_"))
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_on_del = cleanup_on_del
        self._disk_files: list[str] = []

        # ------------------------------------------------------------------
        # APPEND-ONLY storage.  Each token position is compressed exactly ONCE
        # (as the paper does) and then frozen — never re-quantized.  This is
        # essential for the lossy Lloyd-Max codec: re-compressing an already-
        # quantized token every generation step makes it drift (cosine
        # 0.91 -> 0.60 over 40 steps) and produces gibberish.  Compressing each
        # token once and freezing it keeps the paper codec faithful over long
        # generations.
        #
        # A "chunk" is one contiguous block of already-compressed tokens for one
        # layer: chunk 0 = the prefill block, chunks 1.. = one per generation step.
        # Chunks are immutable once written, so old ones spill to disk freely.
        # ------------------------------------------------------------------
        self._n_layers = 0
        self._stored_len = 0                       # token positions already frozen
        self._num_chunks: dict[int, int] = {}      # l_idx -> number of chunks
        # Tier 1 (warm): LRU of (l_idx, chunk_idx) -> [k_c, v_c] compressed pairs
        self._ram: "OrderedDict[tuple[int, int], list]" = OrderedDict()
        # Tier 2 (cold): set of (l_idx, chunk_idx) currently on disk
        self._disk: set[tuple[int, int]] = set()
        # Tier 0 (hot): resident dequantized float cache per layer for incremental
        # staging: l_idx -> (k_hat, v_hat, n_chunks_included).  Each chunk is
        # decoded once and appended; see stage_for_forward().
        self._staged: dict[int, tuple] = {}
        self._lock = threading.Lock()

        # Cache of KVQuantMSE modules by (dim, bits) so the paper codec doesn't
        # rebuild the rotation matrix + Lloyd-Max codebook on every K/V tensor.
        self._mse_cache: dict[tuple[int, int], KVQuantMSE] = {}
        # For "paper-outlier": one calibrated OutlierKVQuant per (l_idx, is_value),
        # trained once on the prefill and reused for every appended token.
        self._outlier_q: dict[tuple[int, bool], "OutlierKVQuant"] = {}

        # Register cleanup
        if cleanup_on_del:
            weakref.finalize(self, self._cleanup)

    @property
    def _gqa_extra(self) -> int:
        """Extra bits/coordinate to offset GQA error amplification.

        Each KV head is shared by g = n_heads / n_kv_heads query heads, so the
        attention-weighted distortion is ~g x the per-vector MSE.  To hold quality
        constant: g * 4^{-b_eff} = 4^{-b_req} -> b_eff = b_req + log4(g).  This is
        the SAME compensation KVCacheQuantizer applies in-memory (see its __init__);
        the offload codec must match it or a GQA model (e.g. Qwen2.5-7B, g=7)
        stores far too few effective bits and generates gibberish.
        """
        g = self._gqa_factor
        return math.ceil(math.log(g, 4)) if g > 1 else 0

    @property
    def _effective_bits(self) -> int:
        """Plain-codec pack width after GQA compensation (capped at 8-bit)."""
        return min(self._bits + self._gqa_extra, 8)

    def _get_mse(self, dim: int, num_bits: int) -> KVQuantMSE:
        """Return a cached KVQuantMSE for (dim, num_bits) — seed fixed for determinism."""
        key = (dim, num_bits)
        mse = self._mse_cache.get(key)
        if mse is None:
            mse = KVQuantMSE(dim=dim, num_bits=num_bits, seed=0)
            self._mse_cache[key] = mse
        return mse

    def _get_outlier_q(self, l_idx: int, is_value: bool, calib: Tensor) -> "OutlierKVQuant":
        """Return a calibrated OutlierKVQuant for one (layer, K/V), building it once.

        Calibrated on the prefill slice `calib` (the first store() call for this
        layer); reused unchanged for every appended token so the codebook and
        outlier-channel choice stay fixed (append-only).
        """
        key = (l_idx, is_value)
        oq = self._outlier_q.get(key)
        if oq is None:
            dim = calib.shape[-1]
            b = self._bits
            # Our generalisation of the paper's single §4.3 example (32 of 128
            # channels at 3 bits, the remaining 96 at 2):
            #   n_outlier    = head_dim / 4        (32 of 128 — the paper's fraction)
            #   outlier_bits = min(bits + 1, 4)
            #   regular_bits = max(bits - 1, 1)
            # At d=128 that gives b=2 -> 32@3 + 96@1 = 1.50 bits/coord, and
            # b=3 -> 32@4 + 96@2 = 2.50.  The paper supplies no formula, no
            # outlier-selection criterion, and no second configuration, so the
            # schedule above is ours.  Do not label these with the paper's
            # "2.5-bit"/"3.5-bit" names: those refer to a different split, and
            # the paper's own arithmetic for them does not check out (see
            # outlier.py).
            #
            # GQA compensation: bump BOTH groups by gqa_extra so a grouped-query
            # model (Qwen2.5-7B, g=7 -> +2) stores enough effective bits.  This is
            # entirely ours — the paper never mentions grouped-query attention.
            # The base config is capped at min(b+1,4)/max(b-1,1) FIRST, then
            # gqa_extra is added (capped at 8-bit) — the exact same order
            # KVCacheQuantizer uses in-memory, so the reported avg_bits matches
            # what is actually stored.
            ge = self._gqa_extra
            ob = min(min(b + 1, 4) + ge, 8)
            rb = min(max(max(b - 1, 1) + ge, 1), 8)
            # clamp n_outlier to leave >=1 regular channel on odd head dims.
            n_outlier = min(max(dim // 4, 1), dim - 1)
            oq = OutlierKVQuant(
                dim=dim,
                n_outlier=n_outlier,
                outlier_bits=ob,
                regular_bits=rb,
                seed=0,
                quantizer_cls=KVQuantMSE,          # MSE path reconstructs faithfully
            )
            oq.calibrate(calib.float().cpu())
            self._outlier_q[key] = oq
        return oq

    def _compress_one(
        self,
        t: Tensor,
        l_idx: int = 0,
        is_value: bool = False,
        target_device: "torch.device | str | None" = "cpu",
    ):
        """
        Compress one K or V tensor with the selected codec.

        `target_device` says where the *input* should be moved before
        compression, which is also where the resulting codes land.  This tier
        defaults to `"cpu"` because its whole purpose is holding codes in host
        RAM with disk spill; pass `None` to compress in place on the tensor's
        own device (what a VRAM-resident code cache wants).
        """
        if target_device is not None:
            t = t.to(target_device)
        if self._codec == "paper":
            # GQA-compensated pack width (see _effective_bits) so grouped-query
            # models don't under-quantize.
            return _paper_compress(t, self._effective_bits, self._get_mse)
        if self._codec == "paper-outlier":
            oq = self._get_outlier_q(l_idx, is_value, t)
            return _paper_outlier_compress(t, oq)
        raise ValueError(f"unknown codec {self._codec!r}")

    def _dequantize_one(self, q, device, l_idx: int = 0, is_value: bool = False) -> Tensor:
        """Reconstruct one K or V tensor, dispatching on the stored codec type."""
        if isinstance(q, _PaperOutlierKV):
            oq = self._outlier_q[(l_idx, is_value)]
            return _paper_outlier_dequantize(q, oq, device=device)
        if isinstance(q, _PaperKV):
            return _paper_dequantize(q, self._get_mse, device=device)
        raise TypeError(f"unknown compressed type {type(q).__name__}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, past_key_values, token_offset: int = 0) -> None:
        """
        Compress and offload ONLY the token positions that have not been stored
        yet (append-only).  Already-frozen tokens are never re-quantized, so the
        lossy 'paper' codec does not drift across generation steps.

        On the first call this freezes the whole prefill; on each subsequent call
        it freezes just the new token(s) the last forward pass appended.

        Args:
            past_key_values: HuggingFace cache object (DynamicCache, tuple, …).
            token_offset:     Unused (kept for backwards-compatible call sites).
        """
        kvs = kvs_from_cache(past_key_values)
        if not kvs:
            return

        self._n_layers = len(kvs)
        total_len = kvs[0][0].shape[-2]          # sequence length now in the cache
        new_from = self._stored_len
        if total_len <= new_from:
            return                                # nothing new to freeze

        # Compress ONLY the new slice [new_from:total_len] for every layer, on CPU
        # so the float tensors leave VRAM immediately.  This new block becomes one
        # immutable chunk per layer.
        with self._lock:
            for l_idx, (k, v) in enumerate(kvs):
                k_new = k[..., new_from:total_len, :]
                v_new = v[..., new_from:total_len, :]
                k_c = self._compress_one(k_new, l_idx=l_idx, is_value=False)
                v_c = self._compress_one(v_new, l_idx=l_idx, is_value=True)
                chunk_idx = self._num_chunks.get(l_idx, 0)
                self._ram[(l_idx, chunk_idx)] = [k_c, v_c]
                self._ram.move_to_end((l_idx, chunk_idx))
                self._num_chunks[l_idx] = chunk_idx + 1
            self._stored_len = total_len
            self._evict_to_disk()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stage_for_forward(self, past_key_values=None, layers: list[int] | None = None) -> object:
        """
        Return a HuggingFace-compatible cache with the full dequantized K/V for the
        requested layers on the current device.

        INCREMENTAL: each frozen chunk is dequantized exactly ONCE and its float
        result kept resident; subsequent calls only decode the newly-appended
        chunk and concatenate it onto the resident tensor.  This turns a T-token
        generation from O(T^2) decode work (re-inflating the whole cache every
        step) into O(T).  Correctness is unchanged because chunks are immutable
        (append-only), so a decoded chunk can never go stale.

        Peak VRAM is unchanged: the full float cache was already materialized here
        every step and handed to the model; we just stop rebuilding it from scratch.

        Args:
            past_key_values: Optional original cache used for structure reference.
            layers:          Layer indices to stage (None = all).

        Returns:
            A cache-like object with K/V float tensors on the current device.
        """
        if layers is None:
            layers = list(range(self._n_layers))

        device = self._device
        result: list[tuple[Tensor, Tensor]] = []

        with self._lock:
            for l_idx in layers:
                total = self._num_chunks.get(l_idx, 0)
                cached = self._staged.get(l_idx)
                # Resident float cache for this layer: (k_hat, v_hat, n_included).
                if cached is None or cached[2] > total:
                    k_hat = v_hat = None
                    done = 0
                else:
                    k_hat, v_hat, done = cached

                # Decode only the chunks not yet folded into the resident tensor.
                for chunk_idx in range(done, total):
                    k_c, v_c = self._get_chunk(l_idx, chunk_idx)
                    k_new = self._dequantize_one(k_c, device=device,
                                                 l_idx=l_idx, is_value=False)
                    v_new = self._dequantize_one(v_c, device=device,
                                                 l_idx=l_idx, is_value=True)
                    if k_hat is None:
                        k_hat, v_hat = k_new, v_new
                    else:
                        k_hat = torch.cat([k_hat, k_new], dim=-2)
                        v_hat = torch.cat([v_hat, v_new], dim=-2)

                self._staged[l_idx] = (k_hat, v_hat, total)
                result.append((k_hat, v_hat))

        return self._rebuild_cache(result, past_key_values)

    def replace(self, past_key_values) -> None:
        """
        Freeze the new token(s) the forward pass just appended.  Append-only:
        this never re-compresses already-stored tokens.
        """
        self.store(past_key_values)

    def close(self) -> None:
        """Explicitly release all resources and delete disk files."""
        self._cleanup()

    # ------------------------------------------------------------------
    # Chunk access + Tier 1 -> Tier 2 eviction
    # ------------------------------------------------------------------

    def _get_chunk(self, l_idx: int, chunk_idx: int) -> list:
        """Return a compressed [k_c, v_c] chunk from RAM or disk (LRU-promoting)."""
        key = (l_idx, chunk_idx)
        pair = self._ram.get(key)
        if pair is not None:
            self._ram.move_to_end(key)
            return pair
        # Cold: load from disk back into RAM (then re-check eviction).
        pair = self._load_from_disk(l_idx, chunk_idx)
        self._ram[key] = pair
        self._ram.move_to_end(key)
        self._disk.discard(key)
        self._evict_to_disk()
        return pair

    def _evict_to_disk(self) -> None:
        """Spill oldest RAM chunks to disk when RAM holds more than warm_size."""
        limit = max(self._warm_size, 1)
        while len(self._ram) > limit:
            key, pair = self._ram.popitem(last=False)
            self._save_to_disk(key[0], key[1], pair)
            self._disk.add(key)

    def _save_to_disk(self, l_idx: int, chunk_idx: int, q_pair: list) -> None:
        """Persist one immutable chunk's compressed K/V to disk as .pt files.

        The codec type rides inside the stored NamedTuple (_PaperKV /
        _PaperOutlierKV), so dtype and bit-width are recovered on load with no
        side files.
        """
        k_c, v_c = q_pair
        k_path = str(self._disk_dir / f"layer_{l_idx}_c{chunk_idx}_k.pt")
        v_path = str(self._disk_dir / f"layer_{l_idx}_c{chunk_idx}_v.pt")
        torch.save(k_c, k_path)
        torch.save(v_c, v_path)
        for p in (k_path, v_path):
            if p not in self._disk_files:
                self._disk_files.append(p)

    def _load_from_disk(self, l_idx: int, chunk_idx: int) -> list:
        """Load one immutable chunk's compressed K/V from disk."""
        k_path = str(self._disk_dir / f"layer_{l_idx}_c{chunk_idx}_k.pt")
        v_path = str(self._disk_dir / f"layer_{l_idx}_c{chunk_idx}_v.pt")
        k_c = torch.load(k_path, map_location="cpu", weights_only=False)
        v_c = torch.load(v_path, map_location="cpu", weights_only=False)
        return [k_c, v_c]

    # ------------------------------------------------------------------
    # Cache reconstruction
    # ------------------------------------------------------------------

    def _rebuild_cache(self, pairs: list[tuple[Tensor, Tensor]], original) -> object:
        """Rebuild a HuggingFace-compatible cache from (k, v) float pairs."""
        try:
            from transformers import DynamicCache
            cache = DynamicCache()
            for i, (k_hat, v_hat) in enumerate(pairs):
                cache.update(k_hat, v_hat, layer_idx=i)
            return cache
        except (ImportError, AttributeError, TypeError):
            pass

        if original is not None and hasattr(original, "key_cache"):
            try:
                import copy
                cache = copy.copy(original)
                for i, (k_hat, v_hat) in enumerate(pairs):
                    cache.key_cache[i] = k_hat
                    cache.value_cache[i] = v_hat
                return cache
            except Exception:
                pass

        return tuple(pairs)

    # ------------------------------------------------------------------
    # Memory stats & cleanup
    # ------------------------------------------------------------------

    def memory_summary(self) -> dict:
        """Return a dict with current tier occupancy.

        Counts are per-layer (chunks collapsed to layers) so the numbers stay
        comparable to the pre-chunk API: 'warm_layers' = layers with >=1 chunk in
        RAM, 'disk_layers' = layers with all chunks spilled to disk.
        """
        with self._lock:
            warm_layers = {k[0] for k in self._ram}
            disk_layers = {k[0] for k in self._disk}
            # A layer counts as "disk" only if none of its chunks are in RAM.
            disk_only = disk_layers - warm_layers
            return {
                "vram_layers": 0,   # nothing is held dequantized between calls
                "warm_layers": len(warm_layers),
                "disk_layers": len(disk_only),
                "ram_chunks": len(self._ram),
                "disk_chunks": len(self._disk),
                "stored_len": self._stored_len,
                "disk_dir": str(self._disk_dir),
            }

    def _cleanup(self) -> None:
        """Delete all disk spill files."""
        try:
            if self._disk_dir.exists():
                _shutil.rmtree(self._disk_dir, ignore_errors=True)
        except Exception:
            # Interpreter shutdown can null out imports before the finalizer
            # runs; nothing we can do then, and the OS reclaims the temp dir.
            pass
        # Guarded: __del__ may fire before __init__ finished (bad codec).
        if hasattr(self, "_ram"):
            self._ram.clear()
        if hasattr(self, "_disk"):
            self._disk.clear()
        if hasattr(self, "_staged"):
            self._staged.clear()
        if hasattr(self, "_outlier_q"):
            self._outlier_q.clear()
        self._disk_files.clear()

    def __del__(self):
        # getattr guard: if __init__ raised before setting attributes (e.g. a
        # bad codec value), __del__ must not itself raise.
        if getattr(self, "_cleanup_on_del", False):
            self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class KVCacheQuantizer(nn.Module):
    """
    Compress and decompress transformer KV cache tensors with KVQuant.

    Args:
        head_dim:     Dimension per attention head (d in the paper).
        num_bits:     Average bits per coordinate.
        use_outlier:  If True, uses OutlierKVQuant with automatic
                      outlier detection (recommended for LLM KV caches).
        n_outlier:    Number of outlier channels (only if use_outlier=True).
        outlier_bits: Bit-width for outlier channels.
        regular_bits: Bit-width for regular channels.
        use_hadamard: Use structured Hadamard rotation (O(d log d)) instead of
                      dense QR (O(d^2)).  Requires head_dim to be a power of 2.
                      Faster for large d on GPU; negligible difference for d<=64.
        seed:         Base RNG seed.

    K-V asymmetry
    -------------
    K uses KVQuantIP (inner-product optimal) - minimises attention score error.
    V uses KVQuantMSE (MSE optimal)          - minimises output reconstruction error.

    This IP-for-K choice is optimal for PPL *scoring* (the paper's evaluation
    setting) but KVQuantIP is an inner-product *estimator* — its per-coordinate
    reconstruction is wrong, so a cache built with it is unusable for real
    autoregressive generation.  Pass ``k_quantizer_cls=KVQuantMSE`` to make K use
    the MSE-optimal quantizer instead: reconstruction is then faithful per
    coordinate (at a small inner-product cost), which is what generation needs.
    The default (None -> KVQuantIP) preserves the scoring behaviour unchanged.
    """

    def __init__(
        self,
        head_dim: int,
        num_bits: int = 3,
        use_outlier: bool = True,
        n_outlier: int = 32,
        outlier_bits: int | None = None,
        regular_bits: int | None = None,
        use_hadamard: bool = False,
        seed: int = 0,
        gqa_factor: int = 1,
        k_quantizer_cls: type | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.use_outlier = use_outlier
        self.gqa_factor = gqa_factor

        # GQA amplification: each KV head is shared by g = n_heads/n_kv_heads query
        # heads, so the effective attention-weighted distortion is g × D_mse.
        # To keep quality constant: g × 4^{-b_eff} = 4^{-b_req}
        # → b_eff = b_req + log4(g) = b_req + log(g)/log(4).
        # For g=1 (MHA) +0; g=4 +1; g=6/7 (Qwen2.5) +2; g=32 (Llama-3) +3.
        # Capped at 8 (Lloyd-Max codebook now supports 1–8 bits; 8-bit = 256 centroids).
        # We apply gqa_extra to BOTH effective_bits AND any caller-supplied
        # outlier/regular bits so the compensation is never bypassed.
        _MAX_BITS = 8
        gqa_extra = math.ceil(math.log(max(gqa_factor, 1), 4)) if gqa_factor > 1 else 0
        effective_bits = min(num_bits + gqa_extra, _MAX_BITS)

        if use_outlier:
            ob_base = outlier_bits if outlier_bits is not None else num_bits + 1
            rb_base = regular_bits if regular_bits is not None else num_bits - 1
            ob = min(ob_base + gqa_extra, _MAX_BITS)
            rb = min(max(rb_base + gqa_extra, 1), _MAX_BITS)
            # K: KVQuantIP inner-product optimal (attention scores use Q @ K^T)
            # for PPL scoring; caller may pass k_quantizer_cls=KVQuantMSE for a
            # faithfully-reconstructing cache (required for generation).
            self.k_quant = OutlierKVQuant(
                head_dim, n_outlier, ob, rb, seed=seed,
                quantizer_cls=(k_quantizer_cls or KVQuantIP),
                use_hadamard=use_hadamard,
            )
            # V: KVQuantMSE MSE optimal (output is weighted sum of V)
            self.v_quant = OutlierKVQuant(
                head_dim, n_outlier, ob, rb, seed=seed + 100,
                quantizer_cls=KVQuantMSE, use_hadamard=use_hadamard,
            )
        else:
            # K: KVQuantIP preserves inner products for attention scores
            self.k_quant = KVQuantIP(
                head_dim, effective_bits, seed=seed, qjl_seed=seed + 1,
                use_hadamard=use_hadamard,
            )
            # V: KVQuantMSE minimises reconstruction MSE for output values
            self.v_quant = KVQuantMSE(
                head_dim, effective_bits, seed=seed + 2,
                use_hadamard=use_hadamard,
            )

        self._calibrated = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, k: Tensor, v: Tensor) -> None:
        """
        Identify outlier channels from a representative KV sample.

        Must be called before compress() when use_outlier=True.

        Args:
            k: Key tensor   of shape (B, H, T, d) or (N, d).
            v: Value tensor of shape (B, H, T, d) or (N, d).
        """
        if self.use_outlier:
            k_flat = k.reshape(-1, self.head_dim)
            v_flat = v.reshape(-1, self.head_dim)
            self.k_quant.calibrate(k_flat)
            self.v_quant.calibrate(v_flat)
        self._calibrated = True

    # ------------------------------------------------------------------
    # Compress / Decompress
    # ------------------------------------------------------------------

    def compress(self, x: Tensor, is_value: bool = False) -> CompressedKV:
        """
        Quantize a K or V cache tensor.

        Args:
            x:        Float tensor of shape (B, H, T, d).
            is_value: If True, uses the V quantizer; otherwise K quantizer.

        Returns:
            K: QuantizedIP or OutlierQuantized (KVQuantIP).
            V: QuantizedMSE or OutlierQuantized (KVQuantMSE).
        """
        if self.use_outlier and not self._calibrated:
            raise RuntimeError("Call KVCacheQuantizer.calibrate(k, v) first.")
        quant = self.v_quant if is_value else self.k_quant
        return quant.quantize(x)

    def decompress(self, q: CompressedKV, is_value: bool = False) -> Tensor:
        """
        Reconstruct a K or V tensor from its compressed form.

        Args:
            q:        Compressed representation from compress().
            is_value: Must match the flag used in compress().

        Returns:
            Float tensor of the original shape.
        """
        quant = self.v_quant if is_value else self.k_quant
        return quant.dequantize(q)

    def compress_kv(self, k: Tensor, v: Tensor) -> tuple[CompressedKV, CompressedKV]:
        """Convenience: compress K and V together."""
        return self.compress(k, is_value=False), self.compress(v, is_value=True)

    def decompress_kv(
        self,
        k_c: CompressedKV,
        v_c: CompressedKV,
    ) -> tuple[Tensor, Tensor]:
        """Convenience: decompress K and V together."""
        return self.decompress(k_c, is_value=False), self.decompress(v_c, is_value=True)

    # ------------------------------------------------------------------
    @property
    def avg_bits(self) -> float:
        """Average bits/coordinate (K and V share the same budget)."""
        if self.use_outlier:
            return self.k_quant.avg_bits
        return float(self.num_bits)

    def extra_repr(self) -> str:
        gqa = f", gqa_factor={self.gqa_factor}" if self.gqa_factor > 1 else ""
        return (
            f"head_dim={self.head_dim}, num_bits={self.num_bits}, "
            f"use_outlier={self.use_outlier}, avg_bits={self.avg_bits:.2f}{gqa}"
        )
