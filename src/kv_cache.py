"""
KV-cache quantizer built on top of KVQuantIP / OutlierKVQuant.

KVCacheQuantizer wraps either KVQuantIP or OutlierKVQuant and
provides a simple .compress() / .decompress() interface that matches
the (batch, heads, seq, dim) layout used by most transformer KV caches.

Tiered memory offload
---------------------
KVCacheDiskOffload stores quantized KV cache across VRAM / CPU-RAM / disk.
Only the layers needed for the next forward pass are staged to GPU; the rest
live in cheaper tiers and are loaded on demand (LRU eviction).

Usage
-----
    quant = KVCacheQuantizer(head_dim=128, num_bits=3, use_outlier=True)
    quant.calibrate(k_calib, v_calib)

    # Basic compress/decompress
    k_c = quant.compress(k, is_value=False)
    v_c = quant.compress(v, is_value=True)
    k_hat = quant.decompress(k_c)

    # Memory-efficient generation with disk offload
    offload = KVCacheDiskOffload(quant, max_vram_tokens=512, disk_dir="./kv_tmp")
    offload.store(native_cache)          # quantize + offload to RAM/disk
    past = offload.stage_for_forward()   # dequantize only what fits in VRAM
    out = model(ids, past_key_values=past, use_cache=True)
    offload.replace(out.past_key_values)  # re-quantize and offload again
"""

from __future__ import annotations

import copy
import gc
import math
import os
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

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

    Usage
    -----
        offload = KVCacheDiskOffload(kvc, max_vram_tokens=512, disk_dir="./kv_tmp")
        offload.store(native_cache)           # quantize + offload
        past = offload.stage_for_forward()    # dequantize to VRAM (LRU-aware)
        out = model(ids, past_key_values=past, use_cache=True)
        offload.replace(out.past_key_values)  # re-quantize and offload again

    Args:
        quantizer:       KVCacheQuantizer (or list of per-layer quantizers).
        max_vram_tokens: Max number of token positions to keep dequantized in VRAM
                         simultaneously (0 = unlimited, but you'll OOM).
        disk_dir:        Directory for memmap spill files (auto-cleaned on close).
        warm_size:       Max number of full-layer quantized entries to keep in CPU
                         RAM before spilling to disk (0 = unlimited).
        pin_memory:      Pin CPU tensors for faster host->device transfer.
        cleanup_on_del:  Delete disk files when this object is garbage-collected.
    """

    def __init__(
        self,
        quantizer,
        max_vram_tokens: int = 512,
        disk_dir: str | None = None,
        warm_size: int = 16,
        pin_memory: bool = True,
        cleanup_on_del: bool = True,
    ):
        self._quantizer = quantizer
        self._is_list = isinstance(quantizer, list)
        self._max_vram_tokens = max_vram_tokens
        self._warm_size = warm_size
        self._pin_memory = pin_memory and torch.cuda.is_available()

        self._disk_dir = Path(disk_dir) if disk_dir else Path(tempfile.mkdtemp(prefix="kv_offload_"))
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_on_del = cleanup_on_del
        self._disk_files: list[str] = []

        # Tier 1 (warm): OrderedDict acts as LRU cache for quantized layer entries
        self._warm_cache: OrderedDict[int, list[CompressedKV]] = OrderedDict()
        # Tier 0 (hot): list of layer_idx currently resident in VRAM
        self._vram_layers: set[int] = set()
        # Original dtype of each layer's KV tensors (for restoring after dequantize)
        self._dtype_cache: OrderedDict[int, torch.dtype] = OrderedDict()
        self._lock = threading.Lock()

        # Register cleanup
        if cleanup_on_del:
            weakref.finalize(self, self._cleanup)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, past_key_values, token_offset: int = 0) -> None:
        """
        Quantize and offload the entire cache.  After this call the original
        ``past_key_values`` is no longer needed and can be freed.

        Args:
            past_key_values: HuggingFace cache object (DynamicCache, tuple, …).
            token_offset:     Current sequence length offset (for LRU ordering).
        """
        kvs = kvs_from_cache(past_key_values)
        if not kvs:
            return

        quantized_layers: list[list[CompressedKV]] = []
        for l_idx, (k, v) in enumerate(kvs):
            kvc = self._quantizer[l_idx] if self._is_list else self._quantizer
            k_c = kvc.compress(k.float().cpu() if k.is_cuda else k)
            v_c = kvc.compress(v.float().cpu() if v.is_cuda else v)
            quantized_layers.append([k_c, v_c])

        with self._lock:
            for l_idx, q_pair in enumerate(quantized_layers):
                self._warm_cache[l_idx] = q_pair
                self._warm_cache.move_to_end(l_idx)
                # Record original dtype from the first tensor we see for this layer
                if l_idx not in self._dtype_cache:
                    orig_k = kvs[l_idx][0]
                    self._dtype_cache[l_idx] = orig_k.dtype

            # Evict excess warm entries to disk
            self._evict_warm_to_disk()

        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stage_for_forward(self, past_key_values=None, layers: list[int] | None = None) -> object:
        """
        Dequantize the requested layers (or all layers) to VRAM for a forward pass.

        Args:
            past_key_values: Optional original cache used for structure reference.
            layers:          Layer indices to stage (None = all).

        Returns:
            A cache-like object with K/V float tensors on the current CUDA device.
        """
        kvs = kvs_from_cache(past_key_values) if past_key_values is not None else None
        n_layers = len(self._warm_cache)

        if layers is None:
            layers = list(range(n_layers))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result: list[tuple[Tensor, Tensor]] = []

        with self._lock:
            for l_idx in layers:
                q_pair = self._warm_cache.get(l_idx)
                dtype = None
                if q_pair is None:
                    # Load from disk
                    q_pair, dtype = self._load_from_disk(l_idx)
                    if dtype is not None:
                        self._dtype_cache[l_idx] = dtype

                k_c, v_c = q_pair
                kvc = self._quantizer[l_idx] if self._is_list else self._quantizer
                k_hat = kvc.decompress(k_c).to(device, non_blocking=True)
                v_hat = kvc.decompress(v_c).to(device, non_blocking=True)
                # Restore original dtype (bfloat16 / float16 / float32) for model compatibility
                orig_dtype = dtype or self._dtype_cache.get(l_idx)
                if orig_dtype is not None and k_hat.dtype != orig_dtype:
                    k_hat = k_hat.to(orig_dtype)
                    v_hat = v_hat.to(orig_dtype)
                result.append((k_hat, v_hat))
                self._vram_layers.add(l_idx)

        # Rebuild a cache object compatible with HuggingFace models
        return self._rebuild_cache(result, past_key_values)

    def replace(self, past_key_values) -> None:
        """
        Re-quantize the cache returned by a forward pass and offload it again.
        This keeps VRAM usage bounded across the entire generation loop.
        """
        self.store(past_key_values)

    def close(self) -> None:
        """Explicitly release all resources and delete disk files."""
        self._cleanup()

    # ------------------------------------------------------------------
    # Tier 1 -> Tier 2 eviction
    # ------------------------------------------------------------------

    def _evict_warm_to_disk(self) -> None:
        """Move oldest warm entries to disk when warm cache exceeds warm_size."""
        while len(self._warm_cache) > max(self._warm_size, 1):
            l_idx, q_pair = self._warm_cache.popitem(last=False)
            dtype = self._dtype_cache.get(l_idx)
            self._save_to_disk(l_idx, q_pair, dtype)
            self._dtype_cache.pop(l_idx, None)

    def _save_to_disk(self, l_idx: int, q_pair: list[CompressedKV], dtype) -> None:
        """Persist quantized K/V for one layer to disk as .pt files."""
        k_c, v_c = q_pair
        k_path = str(self._disk_dir / f"layer_{l_idx}_k.pt")
        v_path = str(self._disk_dir / f"layer_{l_idx}_v.pt")
        torch.save(k_c, k_path)
        torch.save(v_c, v_path)
        dtype_path = str(self._disk_dir / f"layer_{l_idx}_dtype.pt")
        torch.save(dtype, dtype_path)
        self._disk_files.extend([k_path, v_path, dtype_path])

    def _load_from_disk(self, l_idx: int) -> tuple[list[CompressedKV], torch.dtype | None]:
        """Load quantized K/V for one layer from disk."""
        k_path = str(self._disk_dir / f"layer_{l_idx}_k.pt")
        v_path = str(self._disk_dir / f"layer_{l_idx}_v.pt")
        dtype_path = str(self._disk_dir / f"layer_{l_idx}_dtype.pt")
        k_c = torch.load(k_path, map_location="cpu", weights_only=False)
        v_c = torch.load(v_path, map_location="cpu", weights_only=False)
        dtype = None
        if os.path.exists(dtype_path):
            try:
                dtype = torch.load(dtype_path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        return [k_c, v_c], dtype

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
        """Return a dict with current tier occupancy."""
        with self._lock:
            warm_count = len(self._warm_cache)
            vram_count = len(self._vram_layers)
        disk_count = len(self._disk_files) // 2
        return {
            "vram_layers": vram_count,
            "warm_layers": warm_count,
            "disk_layers": disk_count,
            "disk_dir": str(self._disk_dir),
        }

    def _cleanup(self) -> None:
        """Delete all disk spill files."""
        import shutil
        try:
            if self._disk_dir.exists():
                shutil.rmtree(self._disk_dir, ignore_errors=True)
        except Exception:
            pass
        self._warm_cache.clear()
        self._dtype_cache.clear()
        self._disk_files.clear()
        self._vram_layers.clear()

    def __del__(self):
        if self._cleanup_on_del:
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
            # K: KVQuantIP  inner-product optimal (attention scores use Q @ K^T)
            self.k_quant = OutlierKVQuant(
                head_dim, n_outlier, ob, rb, seed=seed,
                quantizer_cls=KVQuantIP, use_hadamard=use_hadamard,
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
