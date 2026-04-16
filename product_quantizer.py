"""
Product Quantization (PQ) for KV cache vectors.

Splits each d-dimensional vector into M subspaces of size d/M, trains a
separate k-means codebook per subspace, and encodes each vector as M small
integer codes.

Storage:  M * b  bits/vector   (b = bits per subspace code)
vs scalar: d * b_scalar bits/vector

For head_dim=64, M=16, b=4:  16*4 = 64 bits/vector  = 1 bit/dim effective
vs 4-bit scalar:              64*4 = 256 bits/vector = 4 bits/dim
→ 4x compression at the same subspace bit-width.

The rotation is applied before splitting into subspaces so that information
is spread evenly across dimensions before partitioning.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .rotation import HadamardRotation


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class QuantizedPQ(NamedTuple):
    codes: Tensor   # (N, M)  int64 — one subspace code per vector per subspace
    norms: Tensor   # (N, 1)  float — original L2 norms, restored at dequantize
    shape: tuple    # original input shape (..., d)


# ---------------------------------------------------------------------------
# K-means helper
# ---------------------------------------------------------------------------

def _kmeans_plusplus_init(x: Tensor, K: int) -> Tensor:
    """
    K-means++ initialisation. Selects K seed centroids with probability
    proportional to squared distance from the nearest already-chosen centroid.
    Provably gives O(log K) approximation vs random init (Arthur & Vassilvitskii, 2007).
    """
    N = x.shape[0]
    # Pick first centroid uniformly at random
    centroids = [x[torch.randint(N, (1,), device=x.device)].squeeze(0)]

    for _ in range(1, K):
        # Squared distance from each point to its nearest chosen centroid
        c_stack = torch.stack(centroids)                      # (c, d)
        sq_dists = torch.cdist(x, c_stack) ** 2              # (N, c)
        min_sq_dists = sq_dists.min(dim=-1).values            # (N,)
        # Sample next centroid with probability ∝ squared distance
        probs = min_sq_dists / min_sq_dists.sum().clamp(min=1e-10)
        idx = torch.multinomial(probs, 1).item()
        centroids.append(x[idx])

    return torch.stack(centroids)                             # (K, d)


def _kmeans(
    x: Tensor,
    K: int,
    num_iters: int = 25,
    tol: float = 1e-4,
) -> Tensor:
    """
    K-means on (N, d) data. Returns (K, d) centroids.

    Uses k-means++ initialisation (Arthur & Vassilvitskii, 2007) for better
    starting codebooks, squared-L2 distances for assignment (standard in PQ
    literature — same assignments as L2 since sqrt is monotonic, but consistent
    with the mean-update step which minimises squared L2 loss), and early
    stopping when the objective change falls below tol.

    Empty clusters keep their previous centroid so the codebook never shrinks.
    """
    N, d = x.shape
    x = x.float()

    # k-means++ initialisation
    centroids = _kmeans_plusplus_init(x, K)

    prev_obj = float("inf")
    for _ in range(num_iters):
        # Assignment: squared L2 → argmin (same as L2 argmin, consistent with mean update)
        sq_dists = torch.cdist(x, centroids) ** 2            # (N, K)
        assignments = sq_dists.argmin(dim=-1)                 # (N,)

        # Objective: mean within-cluster squared distance (for convergence check)
        obj = sq_dists.gather(1, assignments.unsqueeze(1)).mean().item()
        if abs(prev_obj - obj) / max(prev_obj, 1e-10) < tol:
            break
        prev_obj = obj

        # Update: mean of assigned points (optimal centroid under squared L2 loss)
        new_c = torch.zeros_like(centroids)                   # (K, d)
        counts = torch.zeros(K, device=x.device)
        new_c.scatter_add_(0, assignments.unsqueeze(1).expand(-1, d), x)
        counts.scatter_add_(0, assignments, torch.ones(N, device=x.device))

        # Keep previous centroid for empty clusters
        mask = counts > 0
        new_c[mask] /= counts[mask].unsqueeze(1)
        new_c[~mask] = centroids[~mask]

        centroids = new_c

    return centroids


# ---------------------------------------------------------------------------
# ProductQuantizer
# ---------------------------------------------------------------------------

class ProductQuantizer(nn.Module):
    """
    Product Quantizer for unit-sphere vectors.

    Args:
        dim:               Input dimensionality (must be divisible by num_subspaces).
        num_subspaces:     Number of subspaces M.  Default 16.
        bits_per_subspace: Bits per subspace code b.  Default 4  (K=16 entries).
        seed:              Random seed for HadamardRotation sign mask.

    After construction, call calibrate(x) with a representative sample of
    vectors before calling quantize/dequantize.
    """

    def __init__(
        self,
        dim: int,
        num_subspaces: int = 16,
        bits_per_subspace: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        assert dim % num_subspaces == 0, (
            f"dim ({dim}) must be divisible by num_subspaces ({num_subspaces})"
        )
        self.dim = dim
        self.M = num_subspaces
        self.b = bits_per_subspace
        self.K = 2 ** bits_per_subspace       # codebook entries per subspace
        self.sub_dim = dim // num_subspaces    # dimensions per subspace

        # Hadamard rotation spreads information evenly before subspace split
        self.rotation = HadamardRotation(dim, seed=seed)

        # Codebooks are set by calibrate(); shape (M, K, sub_dim)
        self.codebooks: Tensor | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, x: Tensor) -> None:
        """
        Train M k-means codebooks from calibration vectors.

        Args:
            x: Float tensor of shape (N, dim) or (..., dim).
               Should be a representative sample — the actual prefill KV
               vectors work well.
        """
        flat = x.reshape(-1, self.dim).float()

        # Unit-normalise then rotate (same preprocessing as quantize)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        y = self.rotation(flat / norms)                     # (N, dim)
        y_split = y.reshape(-1, self.M, self.sub_dim)       # (N, M, sub_dim)

        books = []
        for m in range(self.M):
            sub = y_split[:, m, :].contiguous()             # (N, sub_dim)
            books.append(_kmeans(sub, self.K))              # (K, sub_dim)

        self.codebooks = torch.stack(books)                 # (M, K, sub_dim)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def quantize(self, x: Tensor) -> QuantizedPQ:
        """
        Encode x to PQ codes.

        Args:
            x: Float tensor of shape (..., dim).

        Returns:
            QuantizedPQ with codes (N, M), norms (N, 1), and original shape.
        """
        if self.codebooks is None:
            raise RuntimeError("Call calibrate() before quantize().")

        shape = x.shape
        flat = x.reshape(-1, self.dim).float()               # (N, dim)
        N = flat.shape[0]

        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        y = self.rotation(flat / norms)                      # (N, dim)
        y_split = y.reshape(N, self.M, self.sub_dim)         # (N, M, sub_dim)

        codes = torch.zeros(N, self.M, dtype=torch.long, device=x.device)
        books = self.codebooks.to(x.device)                  # (M, K, sub_dim)

        for m in range(self.M):
            # Pairwise distances between N sub-vectors and K centroids
            dists = torch.cdist(
                y_split[:, m, :].contiguous(),               # (N, sub_dim)
                books[m],                                    # (K, sub_dim)
            )                                                # (N, K)
            codes[:, m] = dists.argmin(dim=-1)               # (N,)

        return QuantizedPQ(codes=codes, norms=norms, shape=shape)

    def dequantize(self, q: QuantizedPQ) -> Tensor:
        """
        Reconstruct vectors from PQ codes.

        Args:
            q: QuantizedPQ from quantize().

        Returns:
            Float tensor of shape q.shape.
        """
        N = q.codes.shape[0]
        books = self.codebooks.to(q.codes.device)            # (M, K, sub_dim)

        # Gather centroids for each subspace
        y_hat = torch.zeros(N, self.M, self.sub_dim, device=q.codes.device)
        for m in range(self.M):
            y_hat[:, m, :] = books[m][q.codes[:, m]]        # (N, sub_dim)

        y_hat = y_hat.reshape(N, self.dim)

        # Inverse rotation then restore original norms.
        # Note: ||y_hat|| is approximately 1 but not exact due to PQ error
        # across subspaces. We intentionally do NOT re-normalise here —
        # attention scores depend on the DIRECTION of K, not magnitude, so
        # preserving direction is more important than exact norm reconstruction.
        x_hat = self.rotation.inverse(y_hat) * q.norms.to(q.codes.device)

        return x_hat.reshape(q.shape)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def bits_per_vector(self) -> int:
        """Total bits to store one quantized vector."""
        return self.M * self.b

    @property
    def effective_bits_per_dim(self) -> float:
        """Average bits per dimension (bits_per_vector / dim)."""
        return self.bits_per_vector / self.dim

    def compression_ratio(self, scalar_bits: int = 4) -> float:
        """
        Ratio of scalar storage to PQ storage.
        Values > 1 mean PQ uses fewer bits.
        """
        return (self.dim * scalar_bits) / self.bits_per_vector

    def __repr__(self) -> str:
        calibrated = self.codebooks is not None
        return (
            f"ProductQuantizer(dim={self.dim}, M={self.M}, b={self.b}, "
            f"K={self.K}, sub_dim={self.sub_dim}, "
            f"effective={self.effective_bits_per_dim:.2f} bits/dim, "
            f"calibrated={calibrated})"
        )


# ---------------------------------------------------------------------------
# ProductKVCache — drop-in replacement for KVCacheQuantizer
# ---------------------------------------------------------------------------

class ProductKVCache(nn.Module):
    """
    Wraps two ProductQuantizers (one for K, one for V) with the same
    compress_kv / decompress_kv interface as KVCacheQuantizer.

    Args:
        head_dim:          KV head dimension.
        num_subspaces:     PQ subspaces M.  Default 16.
        bits_per_subspace: Bits per code b.  Default 4.
        seed:              Rotation seed.
    """

    def __init__(
        self,
        head_dim: int,
        num_subspaces: int = 16,
        bits_per_subspace: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.k_quant = ProductQuantizer(head_dim, num_subspaces, bits_per_subspace, seed)
        self.v_quant = ProductQuantizer(head_dim, num_subspaces, bits_per_subspace, seed + 1)

    def calibrate(self, k: Tensor, v: Tensor) -> None:
        """Train codebooks from prefill K, V tensors."""
        self.k_quant.calibrate(k.reshape(-1, k.shape[-1]))
        self.v_quant.calibrate(v.reshape(-1, v.shape[-1]))

    def compress_kv(
        self, k: Tensor, v: Tensor
    ) -> tuple[QuantizedPQ, QuantizedPQ]:
        return self.k_quant.quantize(k), self.v_quant.quantize(v)

    def decompress_kv(
        self, k_c: QuantizedPQ, v_c: QuantizedPQ
    ) -> tuple[Tensor, Tensor]:
        return self.k_quant.dequantize(k_c), self.v_quant.dequantize(v_c)

    @property
    def effective_bits_per_dim(self) -> float:
        return self.k_quant.effective_bits_per_dim

    @property
    def compression_ratio(self) -> float:
        return self.k_quant.compression_ratio()

    def __repr__(self) -> str:
        return (
            f"ProductKVCache(M={self.k_quant.M}, b={self.k_quant.b}, "
            f"effective={self.effective_bits_per_dim:.2f} bits/dim, "
            f"compression={self.compression_ratio:.1f}x vs 4-bit scalar)"
        )
