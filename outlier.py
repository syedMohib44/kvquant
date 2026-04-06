"""
Outlier-channel-aware KVQuant (Section 5 of the paper).

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

from .quantizer import KVQuantIP, QuantizedIP


class OutlierQuantized(NamedTuple):
    outlier_q:  QuantizedIP     # quantized outlier channels
    regular_q:  QuantizedIP     # quantized regular channels
    outlier_idx: Tensor         # LongTensor (n_outlier,) - channel indices
    regular_idx: Tensor         # LongTensor (n_regular,) - channel indices
    shape:      tuple           # original shape (..., d)


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
    """

    def __init__(
        self,
        dim: int,
        n_outlier: int = 32,
        outlier_bits: int = 3,
        regular_bits: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        assert n_outlier < dim, "n_outlier must be less than dim"

        self.dim = dim
        self.n_outlier = n_outlier
        self.n_regular = dim - n_outlier
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits

        # Quantizers - created lazily after calibration sets their dims
        self._outlier_q: KVQuantIP | None = None
        self._regular_q: KVQuantIP | None = None
        self._seed = seed

        # Channel indices (set after calibrate())
        self.register_buffer("outlier_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("regular_idx", torch.empty(0, dtype=torch.long))
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
        var = flat.var(dim=0)                             # (d,)

        # Top-n_outlier channels by variance
        _, top_idx = var.topk(self.n_outlier)
        outlier_idx = top_idx.sort().values               # keep sorted for consistency

        all_idx = torch.arange(self.dim, device=x.device)
        mask = torch.ones(self.dim, dtype=torch.bool, device=x.device)
        mask[outlier_idx] = False
        regular_idx = all_idx[mask]

        self.outlier_idx = outlier_idx
        self.regular_idx = regular_idx

        # Build sub-quantizers with correct sub-dimensions
        self._outlier_q = KVQuantIP(
            dim=self.n_outlier,
            num_bits=self.outlier_bits,
            seed=self._seed,
            qjl_seed=self._seed + 1,
        ).to(x.device)

        self._regular_q = KVQuantIP(
            dim=self.n_regular,
            num_bits=self.regular_bits,
            seed=self._seed + 2,
            qjl_seed=self._seed + 3,
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
        flat = x.reshape(-1, self.dim)

        x_out = flat[:, self.outlier_idx]     # (N, n_outlier)
        x_reg = flat[:, self.regular_idx]     # (N, n_regular)

        return OutlierQuantized(
            outlier_q=self._outlier_q.quantize(x_out),
            regular_q=self._regular_q.quantize(x_reg),
            outlier_idx=self.outlier_idx,
            regular_idx=self.regular_idx,
            shape=shape,
        )

    def dequantize(self, q: OutlierQuantized) -> Tensor:
        """
        Reconstruct vectors from OutlierQuantized.

        Returns a Float tensor of shape q.shape.
        """
        self._check_calibrated()
        N = 1
        for s in q.shape[:-1]:
            N *= s

        x_out = self._outlier_q.dequantize(q.outlier_q).reshape(N, self.n_outlier)
        x_reg = self._regular_q.dequantize(q.regular_q).reshape(N, self.n_regular)

        # Reconstruct full tensor in original channel order
        out = torch.empty(N, self.dim, device=x_out.device, dtype=x_out.dtype)
        out[:, q.outlier_idx] = x_out
        out[:, q.regular_idx] = x_reg

        return out.reshape(q.shape)

    def forward(self, x: Tensor) -> Tensor:
        """Quantize then dequantize."""
        return self.dequantize(self.quantize(x))

    # ------------------------------------------------------------------
    @property
    def avg_bits(self) -> float:
        """Weighted average bits/coordinate."""
        return (self.n_outlier * self.outlier_bits + self.n_regular * self.regular_bits) / self.dim

    def _check_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError("Call OutlierKVQuant.calibrate(x) before quantizing.")

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_outlier={self.n_outlier}, "
            f"outlier_bits={self.outlier_bits}, regular_bits={self.regular_bits}, "
            f"avg_bits={self.avg_bits:.2f}"
        )
