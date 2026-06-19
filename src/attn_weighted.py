"""
Attention-weighted quantization (novel extension of KVQuant).

Standard KVQuant minimises uniform MSE across all tokens:
    L = E[ ||k_i - k(cap)_i||^2 ]

This module minimises attention-weighted MSE instead:
    L = E[ a_i · ||k_i - k(cap)_i||^2 ]

where a_i = softmax(q · k_i / sqrt(d)) is the attention weight that query q
assigns to token i. Tokens the model actually attends to are quantized more
accurately; low-attention tokens tolerate more error.

Two mechanisms are implemented:

  AttentionWeightedQuantizer
    Given a query vector q and keys K, computes attention weights and assigns
    each key a bit-width from a budget such that high-weight keys get more bits.
    Uses KVQuantMSE internally per bit-width group.

  weighted_distortion(q, K, K_hat)
    Evaluates the attention-weighted reconstruction error for analysis.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .quantizer import KVQuantMSE, QuantizedMSE
from .csrc import softmax_triton


class AttentionWeightedQuantizer(nn.Module):
    """
    Quantize a KV cache using attention weights to guide bit allocation.

    High-attention tokens get `hi_bits`, low-attention tokens get `lo_bits`.
    The threshold between the two groups is the median attention weight by
    default, giving a 50/50 split. Adjust `top_fraction` to change the ratio.

    Args:
        dim:          Head dimension d.
        hi_bits:      Bits for high-attention tokens (default 4).
        lo_bits:      Bits for low-attention tokens (default 2).
        top_fraction: Fraction of tokens treated as high-attention (default 0.5).
        seed:         RNG seed for the rotation matrices.
    """

    def __init__(
        self,
        dim: int,
        hi_bits: int = 4,
        lo_bits: int = 2,
        top_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        assert 0 < top_fraction < 1
        self.dim = dim
        self.hi_bits = hi_bits
        self.lo_bits = lo_bits
        self.top_fraction = top_fraction

        self.hi_quantizer = KVQuantMSE(dim, hi_bits, seed=seed)
        self.lo_quantizer = KVQuantMSE(dim, lo_bits, seed=seed + 1)

    # ------------------------------------------------------------------
    def quantize(self, keys: Tensor, query: Tensor) -> "AttentionWeightedQuantized":
        """
        Quantize keys guided by attention weights from query.

        Args:
            keys:  (B, H, T, d) or (T, d) - key cache to compress.
            query: (B, H, d) or (d,)      - current query vector.

        Returns:
            AttentionWeightedQuantized
        """
        shape = keys.shape
        T = shape[-2]

        # Flatten to (N, T, d) and query to (N, 1, d)
        keys_flat = keys.reshape(-1, T, self.dim)  # (N, T, d)
        query_flat = query.reshape(-1, 1, self.dim)  # (N, 1, d)
        N = keys_flat.shape[0]

        # Attention weights: softmax(q @ K^T / sqrt(d))  -> (N, T)
        scores = (query_flat @ keys_flat.transpose(-2, -1)).squeeze(1)  # (N, T)
        scores = scores / math.sqrt(self.dim)
        weights = softmax_triton(scores, dim=-1)  # (N, T)

        # Split tokens by attention weight - top fraction -> hi_bits
        k_hi = max(1, int(T * self.top_fraction))
        _, top_idx = weights.topk(k_hi, dim=-1)  # (N, k_hi)
        top_mask = torch.zeros(N, T, dtype=torch.bool, device=keys.device)
        top_mask.scatter_(1, top_idx, True)

        # Quantize each group
        hi_keys = keys_flat[top_mask].reshape(N, k_hi, self.dim)
        lo_keys = keys_flat[~top_mask].reshape(N, T - k_hi, self.dim)

        hi_q = self.hi_quantizer.quantize(hi_keys)
        lo_q = self.lo_quantizer.quantize(lo_keys)

        return AttentionWeightedQuantized(
            hi_q=hi_q,
            lo_q=lo_q,
            top_mask=top_mask,
            shape=shape,
            dim=self.dim,
        )

    def dequantize(self, q: "AttentionWeightedQuantized") -> Tensor:
        """Reconstruct keys from AttentionWeightedQuantized."""
        shape = q.shape
        T = shape[-2]
        top_mask = q.top_mask  # (N, T)
        N = top_mask.shape[0]

        hi_keys = self.hi_quantizer.dequantize(q.hi_q)  # (N, k_hi, d)
        lo_keys = self.lo_quantizer.dequantize(q.lo_q)  # (N, T-k_hi, d)

        # Reconstruct in original token order
        out = torch.empty(N, T, self.dim, device=hi_keys.device, dtype=hi_keys.dtype)
        out[top_mask] = hi_keys.reshape(-1, self.dim)
        out[~top_mask] = lo_keys.reshape(-1, self.dim)

        return out.reshape(shape)

    def forward(self, keys: Tensor, query: Tensor) -> Tensor:
        return self.dequantize(self.quantize(keys, query))

    @property
    def avg_bits(self) -> float:
        return self.top_fraction * self.hi_bits + (1 - self.top_fraction) * self.lo_bits

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hi_bits={self.hi_bits}, lo_bits={self.lo_bits}, "
            f"top_fraction={self.top_fraction}, avg_bits={self.avg_bits:.2f}"
        )


# ---------------------------------------------------------------------------
# Named tuple
# ---------------------------------------------------------------------------

from typing import NamedTuple


class AttentionWeightedQuantized(NamedTuple):
    hi_q: QuantizedMSE  # high-attention group
    lo_q: QuantizedMSE  # low-attention group
    top_mask: Tensor  # (N, T) bool - which tokens are hi-attention
    shape: tuple  # original keys shape
    dim: int


# ---------------------------------------------------------------------------
# Analysis helper
# ---------------------------------------------------------------------------


def weighted_distortion(q: Tensor, K: Tensor, K_hat: Tensor) -> Tensor:
    """
    Compute attention-weighted reconstruction error.

    L = sum_i [ softmax(q·k_i/sqrt(d))_i · ||k_i - k(cap)_i||^2 ]

    Args:
        q:     (..., d) query vectors.
        K:     (..., T, d) true keys.
        K_hat: (..., T, d) reconstructed keys.

    Returns:
        Scalar mean weighted distortion.
    """
    d = K.shape[-1]
    scores = (q.unsqueeze(-2) @ K.transpose(-2, -1)).squeeze(-2) / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)  # (..., T)
    per_tok = ((K - K_hat) ** 2).mean(-1)  # (..., T)
    return (weights * per_tok).sum(-1).mean()
