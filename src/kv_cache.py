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
