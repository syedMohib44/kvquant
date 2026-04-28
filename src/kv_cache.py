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

import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple

from .quantizer import KVQuantIP, KVQuantMSE, QuantizedIP, QuantizedMSE
from .outlier import OutlierKVQuant, OutlierQuantized


# Accept any quantized representation (K uses IP, V uses MSE)
CompressedKV = QuantizedIP | QuantizedMSE | OutlierQuantized


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
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.use_outlier = use_outlier

        if use_outlier:
            ob = outlier_bits if outlier_bits is not None else min(num_bits + 1, 4)
            rb = regular_bits if regular_bits is not None else max(num_bits - 1, 1)
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
                head_dim, num_bits, seed=seed, qjl_seed=seed + 1,
                use_hadamard=use_hadamard,
            )
            # V: KVQuantMSE minimises reconstruction MSE for output values
            self.v_quant = KVQuantMSE(
                head_dim, num_bits, seed=seed + 2,
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
        return (
            f"head_dim={self.head_dim}, num_bits={self.num_bits}, "
            f"use_outlier={self.use_outlier}, avg_bits={self.avg_bits:.2f}"
        )
