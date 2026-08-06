"""
Outlier-channel-aware KV quantization using KVQuantIP (Section 5 of the paper).

Certain channels in attention K/V tensors have disproportionately large
magnitudes ("outlier channels").  Quantizing them at the same bit-width as
regular channels wastes precision.  This module:

  1. Identifies outlier channels by their empirical variance (calibrated once).
  2. Partitions channels into outlier and regular groups.
  3. Applies KVQuantIP independently to each group at different bit-widths.

Typical configurations from the paper
--------------------------------------
  "2.5-bit":  32 outlier channels @ 3 bits + 96 regular @ 2 bits  (d=128)
  "3.5-bit":  32 outlier channels @ 4 bits + 96 regular @ 3 bits  (d=128)

The average bit-width across all channels is the weighted mean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple

from .quantizer import KVQuantIP, KVQuantMSE, QuantizedIP, QuantizedMSE


class OutlierQuantized(NamedTuple):
    outlier_q: QuantizedIP | QuantizedMSE  # quantized outlier channels  (first n_outlier cols)
    regular_q: QuantizedIP | QuantizedMSE  # quantized regular channels  (remaining cols)
    shape: tuple  # original shape (..., d)


class OutlierKVQuant(nn.Module):
    """
    KVQuant with per-channel outlier handling.

    Calibrate once with a representative batch, then quantize/dequantize.

    Args:
        dim:            Total channel dimension d.
        n_outlier:      Number of channels treated as outliers.
        outlier_bits:   Bits per coordinate for outlier channels.
        regular_bits:   Bits per coordinate for regular channels.
        seed:           Base RNG seed (outlier and regular use seed and seed+1).
        quantizer_cls:  KVQuantIP (default, inner-product optimal) or KVQuantMSE
                        (MSE optimal).  Use KVQuantIP for K tensors and KVQuantMSE
                        for V tensors to match each tensor's role in attention.
        use_hadamard:   Use structured Hadamard rotation instead of dense QR.
                        Requires dim to be a power of 2.  Faster for large d.
    """

    def __init__(
        self,
        dim: int,
        n_outlier: int = 32,
        outlier_bits: int = 3,
        regular_bits: int = 2,
        seed: int = 0,
        quantizer_cls: type = KVQuantIP,
        use_hadamard: bool = False,
    ) -> None:
        super().__init__()
        assert n_outlier < dim, "n_outlier must be less than dim"

        self.dim = dim
        self.n_outlier = n_outlier
        self.n_regular = dim - n_outlier
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self._quantizer_cls = quantizer_cls
        self._use_hadamard = use_hadamard

        # Quantizers - created lazily after calibration sets their dims
        self._outlier_q: KVQuantIP | KVQuantMSE | None = None
        self._regular_q: KVQuantIP | KVQuantMSE | None = None
        self._seed = seed

        # perm:     original -> contiguous order  (outliers first, then regular)
        # inv_perm: contiguous -> original order  (used in dequantize)
        # Both are set after calibrate().
        self.register_buffer("perm", torch.empty(0, dtype=torch.long))
        self.register_buffer("inv_perm", torch.empty(0, dtype=torch.long))
        self._calibrated = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, x: Tensor) -> None:
        """
        Identify outlier channels from a representative data sample.

        Args:
            x: Float tensor of shape (N, d) or (B, T, d).
               All tokens are used to estimate per-channel variance.
        """
        flat = x.reshape(-1, self.dim).float()
        var = flat.var(dim=0)  # (d,)

        # Top-n_outlier channels by variance
        _, top_idx = var.topk(self.n_outlier)
        outlier_idx = top_idx.sort().values  # sorted for determinism

        all_idx = torch.arange(self.dim, device=x.device)
        mask = torch.ones(self.dim, dtype=torch.bool, device=x.device)
        mask[outlier_idx] = False
        regular_idx = all_idx[mask]

        # Build a contiguous permutation: outliers first, then regular channels.
        # quantize() can then slice with [:, :n_outlier] and [:, n_outlier:] instead
        # of fancy-index scatter/gather, which is faster and more cache-friendly.
        perm = torch.cat([outlier_idx, regular_idx])  # (d,)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = all_idx

        self.perm = perm
        self.inv_perm = inv_perm

        # Build sub-quantizers with correct sub-dimensions.
        # KVQuantIP preserves inner products (optimal for K tensors).
        # KVQuantMSE minimises reconstruction error (optimal for V tensors).
        h = self._use_hadamard
        if self._quantizer_cls is KVQuantMSE:
            self._outlier_q = KVQuantMSE(
                dim=self.n_outlier, num_bits=self.outlier_bits,
                seed=self._seed, use_hadamard=h,
            ).to(x.device)
            self._regular_q = KVQuantMSE(
                dim=self.n_regular, num_bits=self.regular_bits,
                seed=self._seed + 2, use_hadamard=h,
            ).to(x.device)
        else:  # KVQuantIP (default)
            self._outlier_q = KVQuantIP(
                dim=self.n_outlier, num_bits=self.outlier_bits,
                seed=self._seed, qjl_seed=self._seed + 1, use_hadamard=h,
            ).to(x.device)
            self._regular_q = KVQuantIP(
                dim=self.n_regular, num_bits=self.regular_bits,
                seed=self._seed + 2, qjl_seed=self._seed + 3, use_hadamard=h,
            ).to(x.device)

        self._calibrated = True

    # ------------------------------------------------------------------
    # Quantize / Dequantize
    # ------------------------------------------------------------------

    def quantize(self, x: Tensor) -> OutlierQuantized:
        """
        Quantize x with outlier-aware bit allocation.

        Args:
            x: Float tensor of shape (..., d).

        Returns:
            OutlierQuantized.
        """
        self._check_calibrated()
        shape = x.shape
        # Permute channels to contiguous layout: outliers first, then regular.
        # Slicing [:, :n_outlier] / [:, n_outlier:] is a simple strided view -
        # no scatter/gather overhead compared to fancy indexing.
        # Move perm to x's device so this works when x is on CPU (e.g. the
        # disk-offload path stores KV on CPU) while the buffer was calibrated on GPU.
        perm = self.perm.to(x.device)
        flat = x.reshape(-1, self.dim)[:, perm]  # (N, d) contiguous

        x_out = flat[:, : self.n_outlier]  # (N, n_outlier)
        x_reg = flat[:, self.n_outlier :]  # (N, n_regular)

        return OutlierQuantized(
            outlier_q=self._outlier_q.quantize(x_out),
            regular_q=self._regular_q.quantize(x_reg),
            shape=shape,
        )

    def dequantize(self, q: OutlierQuantized) -> Tensor:
        """
        Reconstruct vectors from OutlierQuantized.

        Returns a Float tensor of shape q.shape.
        """
        self._check_calibrated()
        import math

        N = math.prod(q.shape[:-1])

        x_out = self._outlier_q.dequantize(q.outlier_q).reshape(N, self.n_outlier)
        x_reg = self._regular_q.dequantize(q.regular_q).reshape(N, self.n_regular)

        # Concatenate in permuted order, then invert permutation back to original
        # channel order.  inv_perm indexing is a single gather - no scatter needed.
        permuted = torch.cat([x_out, x_reg], dim=1)  # (N, d) in perm order
        out = permuted[:, self.inv_perm.to(permuted.device)]  # (N, d) original order

        return out.reshape(q.shape)

    def forward(self, x: Tensor) -> Tensor:
        """Quantize then dequantize."""
        return self.dequantize(self.quantize(x))

    # ------------------------------------------------------------------
    @property
    def outlier_idx(self) -> Tensor:
        """Channel indices of outlier channels in the original ordering."""
        return self.perm[: self.n_outlier]

    @property
    def regular_idx(self) -> Tensor:
        """Channel indices of regular channels in the original ordering."""
        return self.perm[self.n_outlier :]

    @property
    def avg_bits(self) -> float:
        """Weighted average bits/coordinate."""
        return (
            self.n_outlier * self.outlier_bits + self.n_regular * self.regular_bits
        ) / self.dim

    def _check_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError("Call OutlierKVQuant.calibrate(x) before quantizing.")

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_outlier={self.n_outlier}, "
            f"outlier_bits={self.outlier_bits}, regular_bits={self.regular_bits}, "
            f"avg_bits={self.avg_bits:.2f}"
        )
