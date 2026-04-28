"""
TurboQuant quantizers (paper: arxiv 2504.19874).

Two classes are provided:

  KVQuantMSE  - MSE-optimal quantization (Theorem 1).
                   Random rotation + per-coordinate Lloyd-Max quantization.
                   Distortion ≤ (sqrt3 pi/2) · 4^{-b}.

  KVQuantIP   - Inner-product-optimal quantization (Theorem 2).
                   Two-stage: (b-1)-bit MSE on x, then 1-bit QJL on residual.
                   Unbiased: E[<y, x'>] = <y, x>.
                   Distortion ≤ (sqrt3 pi^2 ‖y‖^2/d) · 4^{-b}.

Normalisation note
------------------
TurboQuant is designed for vectors on the unit sphere S^{d-1}.  After a
random rotation each coordinate has distribution ~ N(0, 1/d).  The
Lloyd-Max codebook is therefore built for N(0, 1) and the centroids are
rescaled by 1/sqrtd at quantize time.

For general (non-unit) vectors we first save the per-vector norm, project
onto S^{d-1}, quantize, then restore the norm at dequantize time.

Quantized representation
------------------------
KVQuantMSE.quantize() returns a QuantizedMSE named-tuple:
    indices  : LongTensor   (..., d)  - codebook index per coordinate
    norms    : FloatTensor  (..., 1)  - per-vector L2 norm
    shape    : tuple                  - original input shape

KVQuantIP.quantize() returns a QuantizedIP named-tuple:
    indices   : LongTensor  (..., d)  - MSE indices (b-1 bits)
    qjl_bits  : BoolTensor  (..., d)  - sign bits from QJL step
    r_norm    : FloatTensor (..., 1)  - residual L2 norm
    vec_norms : FloatTensor (..., 1)  - input vector L2 norms
    shape     : tuple
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .codebook import build_codebook
from .rotation import RandomRotation, HadamardRotation


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class QuantizedMSE(NamedTuple):
    indices: Tensor  # LongTensor  (..., d)
    norms: Tensor  # FloatTensor (..., 1)  - original vector norms
    shape: tuple  # original shape before flattening to (..., d)


class CompressedMSE(NamedTuple):
    bits: list  # Huffman-encoded bit stream
    norms: Tensor  # FloatTensor (..., 1)
    shape: tuple
    codec: object  # HuffmanCodec instance
    indices_len: int  # total number of indices encoded


class QuantizedIP(NamedTuple):
    indices: Tensor  # LongTensor  (..., d)  - (b-1)-bit MSE indices
    qjl_bits: Tensor  # BoolTensor  (..., d)  - QJL sign bits
    r_norm: Tensor  # FloatTensor (..., 1)  - residual norm (normalised space)
    vec_norms: Tensor  # FloatTensor (..., 1)  - input vector norms
    shape: tuple


# ---------------------------------------------------------------------------
# MSE-optimal quantizer
# ---------------------------------------------------------------------------


class KVQuantMSE(nn.Module):
    """
    MSE-optimal KVQuant (Section 3 of the paper).

    Vectors are projected onto the unit sphere before quantization and the
    original norm is stored separately.

    Args:
        dim:       Dimensionality d of vectors.
        num_bits:  Bits per coordinate b (belongs to) {1, 2, 3, 4}.
        seed:      Seed for the random rotation matrix.
    """

    def __init__(
        self,
        dim: int,
        num_bits: int = 2,
        seed: int = 0,
        use_hadamard: bool | None = None,
    ) -> None:
        super().__init__()
        assert 1 <= num_bits <= 4, "num_bits must be 1–4"
        self.dim = dim
        self.num_bits = num_bits

        # Hadamard is O(d log d) vs O(d²) for QR in theory, but the FWHT uses
        # a Python loop that loses to a single BLAS matmul on CPU for d<=256.
        # Default stays QR; pass use_hadamard=True explicitly for GPU workloads
        # or larger d where the structured transform pays off.
        self.use_hadamard = False if use_hadamard is None else use_hadamard

        if self.use_hadamard:
            assert (
                dim & (dim - 1)
            ) == 0, "use_hadamard=True requires dim to be a power of 2"
            self.rotation = HadamardRotation(dim, seed)
        else:
            self.rotation = RandomRotation(dim, seed)

        # Centroids and pre-computed boundaries for fast bucketize
        centroids, boundaries = build_codebook(num_bits, dim)
        self.register_buffer("centroids", centroids)  # (k,)
        self.register_buffer("boundaries", boundaries)  # (k-1,)

    # ------------------------------------------------------------------
    def quantize(self, x: Tensor) -> QuantizedMSE:
        """
        Quantize vectors x.

        Args:
            x: Float tensor of shape (..., d).

        Returns:
            QuantizedMSE with indices of shape (..., d).
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).to(self.centroids.dtype)

        # Save norms, project onto unit sphere
        norms = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        x_unit = x_flat / norms

        # Rotate
        y = self.rotation(x_unit)  # (N, d)

        # Nearest centroid via binary search - boundaries cached from build_codebook
        indices = torch.bucketize(y, self.boundaries)  # (N, d)

        return QuantizedMSE(
            indices=indices.reshape(*shape[:-1], self.dim),
            norms=norms.reshape(*shape[:-1], 1),
            shape=shape,
        )

    def dequantize(self, q: QuantizedMSE) -> Tensor:
        """
        Reconstruct vectors from quantized representation.

        Args:
            q: QuantizedMSE from quantize().

        Returns:
            Float tensor of shape q.shape.
        """
        N = q.indices.reshape(-1, self.dim).shape[0]
        idx_flat = q.indices.reshape(N, self.dim)
        norms_flat = q.norms.reshape(N, 1)

        y_tilde = self.centroids[idx_flat]  # (N, d)
        x_unit_hat = self.rotation.inverse(y_tilde)  # (N, d)

        # Restore scale
        x_hat = x_unit_hat * norms_flat
        return x_hat.reshape(q.shape)

    def _quantize_unit(self, x_unit: Tensor) -> Tensor:
        """
        Fast path: quantize pre-normalised (unit-norm) vectors, return raw indices.
        Skips norm computation and QuantizedMSE allocation.
        Used by KVQuantIP.quantize() to avoid double-normalising.
        """
        return torch.bucketize(self.rotation(x_unit), self.boundaries)  # (N, d)

    def _dequantize_unit(self, idx_flat: Tensor) -> Tensor:
        """
        Fast path: dequantize pre-flattened indices assuming unit-norm vectors.
        Skips the norm-restore multiply and the QuantizedMSE allocation.
        Used internally by KVQuantIP.dequantize() where norms are always 1.
        """
        y_tilde = self.centroids[idx_flat]  # (N, d)
        return self.rotation.inverse(y_tilde)  # (N, d)

    def forward(self, x: Tensor) -> Tensor:
        """Quantize then immediately dequantize."""
        return self.dequantize(self.quantize(x))

    def distortion_mse(self, x: Tensor) -> Tensor:
        """Compute actual MSE distortion on a batch."""
        return ((x - self.forward(x)) ** 2).mean()

    def compress(self, x: Tensor) -> "CompressedMSE":
        """
        Quantize x and apply Huffman entropy coding to the indices.

        Returns a CompressedMSE with encoded bit-streams instead of raw indices.
        Useful for measuring actual storage cost after entropy coding.
        """
        from .entropy import HuffmanCodec

        q = self.quantize(x)
        codec = HuffmanCodec(self.num_bits, self.dim)
        bits = codec.encode(q.indices)
        return CompressedMSE(
            bits=bits,
            norms=q.norms,
            shape=q.shape,
            codec=codec,
            indices_len=q.indices.numel(),
        )

    def extra_repr(self) -> str:
        rot = "hadamard" if self.use_hadamard else "qr"
        return f"dim={self.dim}, num_bits={self.num_bits}, rotation={rot}"


# ---------------------------------------------------------------------------
# Inner-product-optimal quantizer (two-stage)
# ---------------------------------------------------------------------------


class KVQuantIP(nn.Module):
    """
    Inner-product-optimal KVQuant (Section 4 of the paper).

    Uses (b-1) bits for MSE quantization of the normalised vector, then
    1-bit QJL on the residual, giving an unbiased estimator of inner products.

    At b=1 the MSE stage uses 0 bits (x_hat = 0) and the entire budget is
    spent on QJL.

    Args:
        dim:       Dimensionality d.
        num_bits:  Total bits per coordinate b (belongs to) {1, 2, 3, 4}.
        seed:      Seed for the rotation matrix.
        qjl_seed:  Separate seed for the QJL random matrix S.
    """

    def __init__(
        self,
        dim: int,
        num_bits: int = 2,
        seed: int = 0,
        qjl_seed: int = 1,
        use_hadamard: bool | None = None,
    ) -> None:
        super().__init__()
        assert 1 <= num_bits <= 4, "num_bits must be 1–4"
        self.dim = dim
        self.num_bits = num_bits
        self.mse_bits = max(0, num_bits - 1)

        self.use_hadamard = False if use_hadamard is None else use_hadamard

        if self.mse_bits > 0:
            self.mse_quantizer: KVQuantMSE | None = KVQuantMSE(
                dim, self.mse_bits, seed, use_hadamard=use_hadamard
            )
        else:
            self.mse_quantizer = None

        # QJL random matrix S ~ N(0,1)^{d×d}
        gen = torch.Generator()
        gen.manual_seed(qjl_seed)
        S = torch.randn(dim, dim, generator=gen)
        self.register_buffer("S", S)  # (d, d)

    # ------------------------------------------------------------------
    def quantize(self, x: Tensor) -> QuantizedIP:
        """
        Quantize vectors x for inner-product preservation.

        Args:
            x: Float tensor of shape (..., d).

        Returns:
            QuantizedIP.
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).to(self.S.dtype)
        N = x_flat.shape[0]

        # Save and normalise
        vec_norms = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        x_unit = x_flat / vec_norms

        # --- Stage 1: MSE quantize with (b-1) bits (on unit-norm vector) ---
        if self.mse_quantizer is not None:
            # x_unit is already normalised - use the fast path that skips the
            # internal norm() + clamp() + div() and avoids allocating QuantizedMSE.
            indices = self.mse_quantizer._quantize_unit(x_unit)  # (N, d)
            x_hat_unit = self.mse_quantizer._dequantize_unit(indices)  # (N, d)
        else:
            x_hat_unit = torch.zeros_like(x_unit)
            indices = torch.zeros(N, self.dim, dtype=torch.long, device=x_flat.device)

        # --- Stage 2: QJL on unit-norm residual ---
        r_unit = x_unit - x_hat_unit  # (N, d)
        r_norm = r_unit.norm(dim=-1, keepdim=True)  # (N, 1)

        # sign(S @ r_unit), S is (d, d) - computed as r_unit @ S.T
        qjl_proj = r_unit @ self.S.T  # (N, d)
        qjl_bits = qjl_proj > 0  # (N, d) bool

        return QuantizedIP(
            indices=indices.reshape(*shape[:-1], self.dim),
            qjl_bits=qjl_bits.reshape(*shape[:-1], self.dim),
            r_norm=r_norm.reshape(*shape[:-1], 1),
            vec_norms=vec_norms.reshape(*shape[:-1], 1),
            shape=shape,
        )

    def dequantize(self, q: QuantizedIP) -> Tensor:
        """
        Reconstruct vectors from QuantizedIP for use in inner products.

        Returns an unbiased estimator x' such that E[<y, x'>] = <y, x>.
        """
        N = math.prod(q.shape[:-1])

        idx_flat = q.indices.reshape(N, self.dim)
        bits_flat = q.qjl_bits.reshape(N, self.dim).to(self.S.dtype)
        r_norm_flat = q.r_norm.reshape(N, 1)
        vec_norms_flat = q.vec_norms.reshape(N, 1)

        # --- Recover MSE component (unit scale) ---
        if self.mse_quantizer is not None:
            # Use the internal unit-scale path to skip the norm multiply entirely
            # and avoid allocating a (N, 1) ones tensor on every call.
            x_hat_unit = self.mse_quantizer._dequantize_unit(idx_flat)
        else:
            x_hat_unit = torch.zeros(
                N, self.dim, device=self.S.device, dtype=self.S.dtype
            )

        # --- QJL residual correction ---
        # Unbiasedness of QJL (Lemma 4):
        #   E[<y, (sqrt(pi/2)/d) * r_norm * S.T @ sign(S @ r)>] = <y, r>
        signs = 2.0 * bits_flat - 1.0  # {-1, +1}
        # signs @ S  ≡ per-row  S.T @ sign(S @ r)
        correction = (
            (math.sqrt(math.pi / 2.0) / self.dim) * r_norm_flat * (signs @ self.S)
        )

        # Combine and restore original scale
        x_tilde_unit = x_hat_unit + correction  # (N, d)
        x_tilde = x_tilde_unit * vec_norms_flat

        return x_tilde.reshape(q.shape)

    def forward(self, x: Tensor) -> Tensor:
        """Quantize then dequantize."""
        return self.dequantize(self.quantize(x))

    def distortion_ip(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute inner-product distortion E[(<y,x> - <y,x'>)^2] on a batch.

        Args:
            x: (..., d) vectors to quantize.
            y: (..., d) query vectors.
        """
        x_tilde = self.forward(x)
        true_ip = (x * y).sum(-1)
        approx_ip = (x_tilde * y).sum(-1)
        return ((true_ip - approx_ip) ** 2).mean()

    def extra_repr(self) -> str:
        rot = "hadamard" if self.use_hadamard else "qr"
        return f"dim={self.dim}, num_bits={self.num_bits}, mse_bits={self.mse_bits}, rotation={rot}"
