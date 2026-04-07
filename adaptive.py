"""
Importance-based adaptive bit allocation (novel extension of KVQuant).

During generation, not all cached tokens are equally important.  Tokens that
receive high cumulative attention weight matter more for output quality than
tokens that are rarely attended to.

This module tracks a running importance score per token position and
dynamically assigns bit-widths from a fixed budget:

    high importance  -> hi_bits  (e.g. 4)
    mid  importance  -> mid_bits (e.g. 3)
    low  importance  -> lo_bits  (e.g. 2)
    very low        -> evict     (drop from cache entirely)

The importance score is updated every generation step using an exponential
moving average of the attention weights received by each cached token.

AdaptiveKVCache manages the full lifecycle:
  1. push()   - add a new token at full precision temporarily
  2. attend() - provide the attention weights from this step -> update scores
                and (re)compress tokens whose bit-width should change
  3. get()    - return the full reconstructed cache for the next attention op
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from collections import defaultdict
from .quantizer import KVQuantMSE, QuantizedMSE


# Bit-width tiers
_TIERS = [4, 3, 2, 1]   # hi -> lo


class _CacheEntry(NamedTuple):
    q:              QuantizedMSE   # compressed KV
    bits:           int            # current bit-width
    score:          float          # importance score (EMA of attention weights)
    pending_bits:   int            # target tier being waited on (hysteresis)
    pending_steps:  int            # consecutive steps in pending_bits tier


class AdaptiveKVCache(nn.Module):
    """
    KV cache that adapts per-token bit allocation based on attention scores.

    Args:
        head_dim:      Dimension per attention head.
        hi_bits:       Bits for high-importance tokens (default 4).
        mid_bits:      Bits for mid-importance tokens (default 3).
        lo_bits:       Bits for low-importance tokens (default 2).
        evict_bits:    Tokens below evict_threshold get compressed to 1 bit.
        hi_threshold:  Importance score above which tokens get hi_bits.
        lo_threshold:  Importance score below which tokens get lo_bits.
        evict_threshold: Importance score below which tokens get evict_bits.
        ema_decay:        EMA decay factor for importance scores (default 0.9).
        hysteresis_steps: A token is recompressed only after its score has been
                          in the new tier for this many consecutive attend() calls
                          (default 2).  Prevents expensive recompression from
                          transient attention fluctuations.
        seed:             RNG seed.
    """

    def __init__(
        self,
        head_dim: int,
        hi_bits: int = 4,
        mid_bits: int = 3,
        lo_bits: int = 2,
        evict_bits: int = 1,
        hi_threshold: float = 0.1,
        lo_threshold: float = 0.01,
        evict_threshold: float = 0.001,
        ema_decay: float = 0.9,
        hysteresis_steps: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.head_dim  = head_dim
        self.hi_bits   = hi_bits
        self.mid_bits  = mid_bits
        self.lo_bits   = lo_bits
        self.evict_bits = evict_bits
        self.hi_threshold    = hi_threshold
        self.lo_threshold    = lo_threshold
        self.evict_threshold = evict_threshold
        self.ema_decay        = ema_decay
        self.hysteresis_steps = hysteresis_steps

        # One quantizer per bit-width tier
        self._quantizers = nn.ModuleDict({
            str(b): KVQuantMSE(head_dim, b, seed=seed + b)
            for b in (hi_bits, mid_bits, lo_bits, evict_bits)
            if b >= 1
        })

        self._k_entries: list[_CacheEntry] = []
        self._v_entries: list[_CacheEntry] = []

    # ------------------------------------------------------------------
    def push(self, k: Tensor, v: Tensor) -> None:
        """
        Add a new token to the cache at hi_bits initially.

        Args:
            k: (..., head_dim)
            v: (..., head_dim)
        """
        qk = self._quantize(k, self.hi_bits)
        qv = self._quantize(v, self.hi_bits)
        self._k_entries.append(_CacheEntry(q=qk, bits=self.hi_bits, score=1.0,
                                           pending_bits=self.hi_bits, pending_steps=0))
        self._v_entries.append(_CacheEntry(q=qv, bits=self.hi_bits, score=1.0,
                                           pending_bits=self.hi_bits, pending_steps=0))

    def attend(self, attn_weights: Tensor) -> None:
        """
        Update importance scores and recompress tokens whose tier changed.

        Args:
            attn_weights: (..., T) attention weights from the current step,
                          already softmax-normalised.  T must equal cache length.
        """
        T = len(self._k_entries)
        if T == 0:
            return

        # Average weights across batch/head dims -> (T,) Python list
        w = attn_weights.reshape(-1, T).mean(0).tolist()

        new_k, new_v = [], []
        for t in range(T):
            ek = self._k_entries[t]
            new_score  = self.ema_decay * ek.score + (1 - self.ema_decay) * w[t]
            want_bits  = self._score_to_bits(new_score)

            # Hysteresis: only commit to a new tier after hysteresis_steps
            # consecutive steps in that tier.  Avoids expensive recompression
            # triggered by transient attention spikes.
            if want_bits == ek.bits:
                # Staying in the same tier — reset pending counter
                new_pending_bits  = ek.bits
                new_pending_steps = 0
                qk = ek.q
                qv = self._v_entries[t].q
                actual_bits = ek.bits
            elif want_bits == ek.pending_bits:
                # Continuing towards a pending tier change
                new_pending_steps = ek.pending_steps + 1
                if new_pending_steps >= self.hysteresis_steps:
                    # Confirmed: recompress now
                    k_hat = self._dequantize(ek)
                    v_hat = self._dequantize(self._v_entries[t])
                    qk = self._quantize(k_hat, want_bits)
                    qv = self._quantize(v_hat, want_bits)
                    actual_bits       = want_bits
                    new_pending_bits  = want_bits
                    new_pending_steps = 0
                else:
                    # Not yet confirmed — keep current compression
                    qk = ek.q
                    qv = self._v_entries[t].q
                    actual_bits      = ek.bits
                    new_pending_bits = want_bits
            else:
                # New tier being requested — start fresh hysteresis counter
                new_pending_bits  = want_bits
                new_pending_steps = 1
                qk = ek.q
                qv = self._v_entries[t].q
                actual_bits = ek.bits

            new_k.append(_CacheEntry(q=qk, bits=actual_bits, score=new_score,
                                     pending_bits=new_pending_bits,
                                     pending_steps=new_pending_steps))
            new_v.append(_CacheEntry(q=qv, bits=actual_bits, score=new_score,
                                     pending_bits=new_pending_bits,
                                     pending_steps=new_pending_steps))

        self._k_entries = new_k
        self._v_entries = new_v

    def get(self) -> tuple[Tensor, Tensor]:
        """
        Reconstruct the full KV cache.

        Tokens are grouped by bit-width and dequantized in a single batched
        call per tier, instead of T individual calls.  For T=128 with 4 tiers
        this is ~4 calls instead of 128, giving a large reduction in Python
        dispatch overhead and better utilisation of BLAS.

        Returns:
            K_hat: (T, ..., head_dim)
            V_hat: (T, ..., head_dim)
        """
        if not self._k_entries:
            raise RuntimeError("Cache is empty.")

        T = len(self._k_entries)

        # Group token indices by bit-width
        tier_idx: dict[int, list[int]] = defaultdict(list)
        for t, e in enumerate(self._k_entries):
            tier_idx[e.bits].append(t)

        k_out: list[Tensor | None] = [None] * T
        v_out: list[Tensor | None] = [None] * T

        for bits, idxs in tier_idx.items():
            quantizer = self._quantizers[str(bits)]
            k_batch = self._batch_dequantize(
                quantizer, [self._k_entries[t].q for t in idxs]
            )
            v_batch = self._batch_dequantize(
                quantizer, [self._v_entries[t].q for t in idxs]
            )
            for i, t in enumerate(idxs):
                k_out[t] = k_batch[i]
                v_out[t] = v_batch[i]

        return torch.stack(k_out), torch.stack(v_out)  # type: ignore[arg-type]

    def _batch_dequantize(
        self, quantizer: KVQuantMSE, qs: list[QuantizedMSE]
    ) -> Tensor:
        """
        Dequantize a list of QuantizedMSE entries in a single fused call.

        Each entry has indices/norms of shape (..., head_dim) / (..., 1).
        Returns a tensor of shape (len(qs), ..., head_dim).
        """
        n = len(qs)
        BH = qs[0].indices.reshape(-1, self.head_dim).shape[0]

        all_idx   = torch.cat([q.indices.reshape(-1, self.head_dim) for q in qs])
        all_norms = torch.cat([q.norms.reshape(-1, 1)               for q in qs])
        combined  = QuantizedMSE(all_idx, all_norms, (n * BH, self.head_dim))
        result    = quantizer.dequantize(combined)      # (n*BH, head_dim)
        return result.reshape(n, BH, self.head_dim)

    def reset(self) -> None:
        self._k_entries.clear()
        self._v_entries.clear()

    # ------------------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self._k_entries)

    def bit_allocation(self) -> dict[int, int]:
        """Return count of tokens at each bit-width tier."""
        counts: dict[int, int] = {}
        for e in self._k_entries:
            counts[e.bits] = counts.get(e.bits, 0) + 1
        return counts

    def avg_bits(self) -> float:
        if not self._k_entries:
            return 0.0
        return sum(e.bits for e in self._k_entries) / len(self._k_entries)

    # ------------------------------------------------------------------
    def _score_to_bits(self, score: float) -> int:
        if score >= self.hi_threshold:
            return self.hi_bits
        if score >= self.lo_threshold:
            return self.mid_bits
        if score >= self.evict_threshold:
            return self.lo_bits
        return self.evict_bits

    def _quantize(self, x: Tensor, bits: int) -> QuantizedMSE:
        return self._quantizers[str(bits)].quantize(x)

    def _dequantize(self, entry: _CacheEntry) -> Tensor:
        return self._quantizers[str(entry.bits)].dequantize(entry.q)

    def extra_repr(self) -> str:
        return (f"head_dim={self.head_dim}, tiers=({self.hi_bits},{self.mid_bits},"
                f"{self.lo_bits},{self.evict_bits}), length={self.length}, "
                f"avg_bits={self.avg_bits():.2f}")
