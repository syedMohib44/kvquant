"""
Delta compression for streaming KV caches (novel extension of KVQuant).

During autoregressive generation the KV cache grows one token at a time.
Consecutive key/value vectors are highly correlated - the delta between
adjacent tokens is much smaller in magnitude than the token itself.

Standard KVQuant compresses each token independently:
    compress(k_t)

Delta KVQuant compresses the residual instead:
    compress(k_t - k_{t-1})   for t > 0
    compress(k_0)              for t == 0  (anchor token, full precision)

Since ||k_t - k_{t-1}|| << ||k_t||, the same bit-width achieves lower
distortion, or equivalently the same distortion can be achieved with fewer
bits.

DeltaKVCache manages the anchor + delta stream for a single generation
sequence.  It integrates directly with KVQuantIP so inner products
(attention scores) remain unbiased.

Optimisations vs naive implementation:
  - Incremental reconstruction: reconstructed vectors are accumulated in
    push() so get() is O(1) (just stack) instead of O(T) per call.
    Trade-off: stores T float32 reconstructions permanently alongside
    the compressed deltas faster but uses more RAM at long contexts.
  - _anchors is a set for O(1) membership test instead of O(T) list scan.
  - Adaptive anchoring: re-anchor when delta/vector ratio exceeds
    anchor_threshold, limiting error accumulation at sequence change-points.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .quantizer import KVQuantIP, QuantizedIP


class DeltaKVCache(nn.Module):
    """
    Streaming KV cache with delta compression.

    Usage::

        cache = DeltaKVCache(head_dim=64, num_bits=3)

        # During generation - call once per new token
        for t, (k_new, v_new) in enumerate(token_stream):
            cache.push(k_new, v_new)

        # Retrieve full reconstructed cache for attention
        K_hat, V_hat = cache.get()

    Args:
        head_dim:         Dimension per attention head.
        num_bits:         Bits per coordinate for delta quantization.
        anchor_every:     Re-anchor every N tokens (limits error accumulation).
                          Default 0 = only anchor at t=0.
        anchor_threshold: Adaptive re-anchor when ||delta|| / ||k|| exceeds
                          this ratio. 0.0 = disabled (default). Combines with
                          anchor_every whichever triggers first wins.
        seed:             RNG seed.
    """

    def __init__(
        self,
        head_dim: int,
        num_bits: int = 3,
        anchor_every: int = 0,
        anchor_threshold: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.anchor_every = anchor_every
        self.anchor_threshold = anchor_threshold

        self.k_quantizer = KVQuantIP(head_dim, num_bits, seed=seed, qjl_seed=seed + 1)
        self.v_quantizer = KVQuantIP(
            head_dim, num_bits, seed=seed + 2, qjl_seed=seed + 3
        )

        # Compressed delta storage
        self._k_store: list[QuantizedIP | Tensor] = []
        self._v_store: list[QuantizedIP | Tensor] = []

        # Incrementally maintained reconstructions  makes get() O(1)
        self._k_reconstructed: list[Tensor] = []
        self._v_reconstructed: list[Tensor] = []

        self._k_prev: Tensor | None = None  # last reconstructed k (for delta)
        self._v_prev: Tensor | None = None
        self._anchors: set[int] = set()  # O(1) membership test

    # ------------------------------------------------------------------
    def push(self, k: Tensor, v: Tensor) -> None:
        """
        Compress and store one new token's KV pair.

        Args:
            k: (..., head_dim) key for the new token.
            v: (..., head_dim) value for the new token.
        """
        t = len(self._k_store)
        is_anchor = (t == 0) or (self.anchor_every > 0 and t % self.anchor_every == 0)

        # Adaptive anchoring: re-anchor when delta is large relative to vector
        # magnitude  limits error accumulation at rapid sequence change-points.
        if not is_anchor and self.anchor_threshold > 0.0 and self._k_prev is not None:
            dk_probe = k - self._k_prev
            if dk_probe.norm() / k.norm().clamp(min=1e-8) > self.anchor_threshold:
                is_anchor = True

        if is_anchor:
            # Store anchor at full float32 (one vector per head  small)
            self._k_store.append(k.detach().clone())
            self._v_store.append(v.detach().clone())
            self._anchors.add(t)
            # _k_prev is the exact anchor  new tensor each time, safe to store
            self._k_prev = k.detach().clone()
            self._v_prev = v.detach().clone()
        else:
            # Compress delta relative to previous reconstructed token
            dk = k - self._k_prev
            dv = v - self._v_prev

            qk = self.k_quantizer.quantize(dk)
            qv = self.v_quantizer.quantize(dv)
            self._k_store.append(qk)
            self._v_store.append(qv)

            # Update prev with reconstructed delta (propagate error correctly).
            # Addition returns a new tensor each time  reference stored in
            # _k_reconstructed is not aliased to _k_prev after reassignment.
            self._k_prev = self._k_prev + self.k_quantizer.dequantize(qk)
            self._v_prev = self._v_prev + self.v_quantizer.dequantize(qv)

        # Accumulate reconstruction incrementally so get() is O(1)
        self._k_reconstructed.append(self._k_prev)
        self._v_reconstructed.append(self._v_prev)

    def get(self) -> tuple[Tensor, Tensor]:
        """
        Return the full reconstructed KV cache.

        O(1)  reconstructions are maintained incrementally in push().

        Returns:
            K_hat: Tensor of shape (T, ..., head_dim)
            V_hat: Tensor of shape (T, ..., head_dim)
        """
        if not self._k_reconstructed:
            raise RuntimeError("Cache is empty - call push() first.")
        return (
            torch.stack(self._k_reconstructed, dim=0),
            torch.stack(self._v_reconstructed, dim=0),
        )

    def reset(self) -> None:
        """Clear the cache (start of a new sequence)."""
        self._k_store.clear()
        self._v_store.clear()
        self._k_reconstructed.clear()
        self._v_reconstructed.clear()
        self._k_prev = None
        self._v_prev = None
        self._anchors.clear()

    @property
    def length(self) -> int:
        """Number of tokens currently in the cache."""
        return len(self._k_store)

    def delta_norms(self) -> Tensor:
        """
        Return the L2 norm of each stored delta (excluding anchors).
        Useful for verifying that deltas are smaller than raw keys.
        """
        norms = []
        for t, item in enumerate(self._k_store):
            if t not in self._anchors and isinstance(item, QuantizedIP):
                norms.append(item.vec_norms.mean().item())
        return torch.tensor(norms) if norms else torch.tensor([])

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_bits={self.num_bits}, "
            f"anchor_every={self.anchor_every}, "
            f"anchor_threshold={self.anchor_threshold}, "
            f"length={self.length}"
        )
