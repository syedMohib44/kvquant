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


def _randomized_svd(
    A: Tensor,
    rank: int,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Randomized SVD of a batch of matrices A of shape (N, m, n).

    Returns (U, S, Vh) truncated to `rank`, where:
        U  : (N, m, rank)
        S  : (N, rank)
        Vh : (N, rank, n)

    Algorithm: randomized range finder + power iteration (Halko et al., 2011,
    Algorithm 4.4).  Power iteration refines the sketch so that accuracy
    converges to the full SVD as n_power_iter increases, with only
    2*n_power_iter extra (cheap) matrix multiplies.

    Args:
        A:              Batch of matrices (N, m, n).
        rank:           Number of singular vectors to compute.
        n_oversampling: Extra sketch dimensions (default 10).
        n_power_iter:   Power-iteration steps (default 2).  Even 1–2 steps
                        bring the error very close to the full SVD for typical
                        KV-cache residuals.
    """
    N, m, n = A.shape
    k = min(rank + n_oversampling, min(m, n))

    # Random Gaussian sketch: Y = A @ Omega -> (N, m, k)
    Omega = torch.randn(N, n, k, device=A.device, dtype=A.dtype)
    Y = A @ Omega

    # Power iteration: Y ← (A @ A^T)^q @ Y  improves range approximation
    # QR at each step keeps the columns orthonormal and avoids numerical drift.
    for _ in range(n_power_iter):
        Q, _ = torch.linalg.qr(Y)
        Z, _ = torch.linalg.qr(A.transpose(-2, -1) @ Q)
        Y = A @ Z

    # Final orthonormal basis for range(A)
    Q, _ = torch.linalg.qr(Y)  # (N, m, k)
    # Project into small space and run exact SVD there - cheap: O(k²·n)
    B = Q.transpose(-2, -1) @ A  # (N, k, n)
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat  # (N, m, k)

    return U[..., :rank], S[..., :rank], Vh[..., :rank, :]


class CorrectedQuantized(NamedTuple):
    base_q: QuantizedMSE | QuantizedIP  # compressed base
    U: Tensor  # (T, r)  left singular vectors * singular values
    V: Tensor  # (d, r)  right singular vectors
    shape: tuple


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
        x_2d = x.reshape(-1, self.dim)  # (NT, d)

        base_q = self.quantizer.quantize(x_2d)
        x_hat = self.quantizer.dequantize(base_q)  # (NT, d)
        residual = (x_2d - x_hat).reshape(*shape)  # (..., T, d)

        residual_flat = residual.reshape(-1, shape[-2], self.dim)  # (N, T, d)
        r = min(self.rank, min(shape[-2], self.dim))
        T_seq = shape[-2]
        if T_seq >= 64:
            # Randomized SVD: faster for long sequences (O(T·d·r) sketch).
            U, S, Vh = _randomized_svd(residual_flat, rank=r)
        else:
            # Full SVD: lower fixed overhead wins for short sequences.
            U, S, Vh = torch.linalg.svd(residual_flat, full_matrices=False)
            U, S, Vh = U[..., :r], S[..., :r], Vh[..., :r, :]
        # U: (N, T, r)  S: (N, r)  Vh: (N, r, d)

        # Absorb S into U:  U_scaled = U * S  so correction = U_scaled @ Vh
        U_scaled = U * S.unsqueeze(-2)  # (N, T, r)
        V = Vh.transpose(-2, -1)  # (N, d, r)

        return CorrectedQuantized(
            base_q=base_q,
            U=U_scaled.reshape(*shape[:-1], r),  # (..., T, r)
            V=V.reshape(*shape[:-2], self.dim, r),  # (..., d, r)
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
        x_hat = self.quantizer.dequantize(base_q)
        R = (x_2d - x_hat).reshape(1, -1, self.dim)  # (1, N, d)

        rank = min(max_rank, min(R.shape[-2], self.dim))
        _, S, _ = _randomized_svd(R, rank=rank)
        S = S.squeeze(0)
        energy = S**2
        total = energy.sum()
        cumulative = energy.cumsum(0) / total
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
