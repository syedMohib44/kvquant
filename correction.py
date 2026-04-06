"""
Low-rank quantization error correction (novel extension of KVQuant).

After KVQuant compresses a key matrix K (belongs to) R^{T×d}, the residual is:

    R = K - K_hat         R (belongs to) R^{T×d}

Storing R exactly would require as much memory as K itself.  But R is
low-rank in practice - quantization error is structured, not random.
We approximate R with a rank-r truncated SVD:

    R ≈ U @ S @ V^T       U (belongs to) R^{T×r}, S (belongs to) R^r, V (belongs to) R^{d×r}

The corrected key matrix is:

    K_corrected = K_hat + U @ diag(S) @ V^T

For attention score computation:
    Q @ K_corrected^T = Q @ K_hat^T + (Q @ V) @ (U @ diag(S))^T

This adds an O(T·r·d) correction at query time, which is negligible for
small r (e.g. r=4 or r=8).

Memory cost of correction:  T·r + r + d·r  vs  T·d  for full residual.
Break-even rank:  r = T·d / (T + d + 1) ≈ d  for long sequences.
Effective for r << d, i.e. capturing the top few error directions.

LowRankCorrection wraps any KVQuantMSE / KVQuantIP quantizer and
adds the SVD correction transparently.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .quantizer import KVQuantMSE, KVQuantIP, QuantizedMSE, QuantizedIP


class CorrectedQuantized(NamedTuple):
    base_q:  QuantizedMSE | QuantizedIP   # compressed base
    U:       Tensor                        # (T, r)  left singular vectors * singular values
    V:       Tensor                        # (d, r)  right singular vectors
    shape:   tuple


class LowRankCorrection(nn.Module):
    """
    Wraps a KVQuant quantizer and adds low-rank residual correction.

    The correction captures the structured part of the quantization error
    using a rank-r SVD of the residual matrix R = K - K_hat.

    Args:
        quantizer:   A KVQuantMSE or KVQuantIP instance.
        rank:        Rank of the SVD correction (default 4).
                     Higher rank -> better quality, more storage.
                     Rule of thumb: rank = num_bits works well.
    """

    def __init__(
        self,
        quantizer: KVQuantMSE | KVQuantIP,
        rank: int = 4,
    ) -> None:
        super().__init__()
        assert rank >= 1
        self.quantizer = quantizer
        self.rank = rank
        self.dim = quantizer.dim

    # ------------------------------------------------------------------
    def quantize(self, x: Tensor) -> CorrectedQuantized:
        """
        Quantize x and compute rank-r correction of the residual.

        Args:
            x: Float tensor of shape (..., T, d).
               The last-but-one dimension is the sequence length T.

        Returns:
            CorrectedQuantized.
        """
        shape = x.shape
        # Flatten to (N*T, d) for the quantizer, then reshape residual to (N, T, d)
        x_2d = x.reshape(-1, self.dim)                    # (NT, d)

        base_q  = self.quantizer.quantize(x_2d)
        x_hat   = self.quantizer.dequantize(base_q)       # (NT, d)
        residual = (x_2d - x_hat).reshape(*shape)          # (..., T, d)

        # SVD of residual matrix: shape (..., T, d)
        # We apply per-sample (last 2 dims)
        residual_flat = residual.reshape(-1, shape[-2], self.dim)  # (N, T, d)
        U, S, Vh = torch.linalg.svd(residual_flat, full_matrices=False)
        # Truncate to rank r
        r = min(self.rank, S.shape[-1])
        U  = U[..., :r]        # (N, T, r)
        S  = S[..., :r]        # (N, r)
        Vh = Vh[..., :r, :]    # (N, r, d)

        # Absorb S into U:  U_scaled = U * S  so correction = U_scaled @ Vh
        U_scaled = U * S.unsqueeze(-2)                    # (N, T, r)
        V = Vh.transpose(-2, -1)                          # (N, d, r)

        return CorrectedQuantized(
            base_q=base_q,
            U=U_scaled.reshape(*shape[:-1], r),           # (..., T, r)
            V=V.reshape(*shape[:-2], self.dim, r),        # (..., d, r)
            shape=shape,
        )

    def dequantize(self, q: CorrectedQuantized) -> Tensor:
        """
        Reconstruct with low-rank correction applied.

        K_corrected = K_hat + U @ V^T
        """
        shape = q.shape
        x_hat = self.quantizer.dequantize(q.base_q).reshape(shape)  # (..., T, d)

        # Correction: (..., T, r) @ (..., r, d) = (..., T, d)
        correction = q.U @ q.V.transpose(-2, -1)
        return x_hat + correction

    def forward(self, x: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x))

    def residual_rank_analysis(self, x: Tensor, max_rank: int = 16) -> Tensor:
        """
        Return the fraction of residual energy captured by each rank 1..max_rank.
        Useful for choosing the right rank for a given model.

        Returns:
            Tensor of shape (max_rank,) with cumulative energy fractions.
        """
        x_2d = x.reshape(-1, self.dim)
        base_q = self.quantizer.quantize(x_2d)
        x_hat  = self.quantizer.dequantize(base_q)
        R = (x_2d - x_hat).reshape(1, -1, self.dim)   # (1, N, d)

        _, S, _ = torch.linalg.svd(R, full_matrices=False)
        S = S.squeeze(0)
        energy = S ** 2
        total  = energy.sum()
        cumulative = energy[:max_rank].cumsum(0) / total
        return cumulative

    def storage_ratio(self, T: int) -> float:
        """
        Ratio of correction storage to full residual storage.

        correction stores: T*r + d*r  floats
        full residual:     T*d        floats
        """
        return (T * self.rank + self.dim * self.rank) / (T * self.dim)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, dim={self.dim}"
