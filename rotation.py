"""
Random orthogonal rotation for KVQuant.

Two implementations are provided:

  RandomRotation    - dense QR-based rotation, O(d^2) per vector.
                      Stores the full d×d matrix as a buffer.
                      Use for small d or when exact Haar-uniform rotation matters.

  HadamardRotation  - structured rotation: random sign flip + Walsh-Hadamard
                      transform, O(d log d) per vector, O(d) storage.
                      Requires d to be a power of 2.
                      Achieves the same randomisation guarantee as QR in high d
                      (see: QuIP, QuaRot papers) and is much faster for large d.

Both share the same .forward() / .inverse() interface so they are drop-in
replacements for each other inside KVQuantMSE / KVQuantIP.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Dense QR rotation  (original)
# ---------------------------------------------------------------------------


class RandomRotation(nn.Module):
    """
    Fixed random orthogonal rotation Pi (belongs to) R^{d×d} via QR decomposition.

    O(d^2) multiply per vector.  Stores d×d floats.

    Args:
        dim:  Dimensionality of the vectors to rotate.
        seed: RNG seed - must match between quantize and dequantize.
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.seed = seed
        self.register_buffer("Pi", _make_qr_rotation(dim, seed))  # (d, d)

    def forward(self, x: Tensor) -> Tensor:
        """y = x @ Pi.T"""
        return x @ self.Pi.T

    def inverse(self, y: Tensor) -> Tensor:
        """x = y @ Pi  (Pi^{-1} = Pi^T for orthogonal matrices)"""
        return y @ self.Pi

    def extra_repr(self) -> str:
        return f"dim={self.dim}, seed={self.seed}"


# ---------------------------------------------------------------------------
# Structured Hadamard rotation  (paper Section 3, practical variant)
# ---------------------------------------------------------------------------


class HadamardRotation(nn.Module):
    """
    Structured random rotation via randomised Walsh-Hadamard transform.

    Rotation:  y = H(D * x) / sqrt(d)
    Inverse:   x = D * H(y * sqrt(d))   (D is its own inverse since D^2 = I)

    where H is the normalised Walsh-Hadamard matrix and D is a diagonal
    matrix of i.i.d. uniform (+/-)1 random signs.

    Complexity: O(d log d) per vector, O(d) storage - vs O(d^2) for dense QR.
    Requires d to be a power of 2.

    The combination D followed by H produces a uniformly random rotation in
    expectation, giving the same theoretical guarantees as a Haar-random
    orthogonal matrix for the purposes of KVQuant (Ailon & Chazelle 2006).

    Args:
        dim:  Dimensionality (must be a power of 2).
        seed: RNG seed.
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        super().__init__()
        assert (
            dim > 0 and (dim & (dim - 1)) == 0
        ), f"HadamardRotation requires dim to be a power of 2, got {dim}"
        self.dim = dim
        self.seed = seed

        gen = torch.Generator()
        gen.manual_seed(seed)
        # Random (+/-)1 diagonal signs
        signs = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).float()
        self.register_buffer("signs", signs)  # (d,)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """y = WHT(D * x) / sqrt(d)"""
        return _fwht(x * self.signs) / math.sqrt(self.dim)

    def inverse(self, y: Tensor) -> Tensor:
        """x = D * (WHT(y) / sqrt(d))  - WHT is self-inverse up to 1/d factor"""
        # H^{-1} = H / d, so: x = D * H(y) / d = D * (H(y) / sqrt(d)) / sqrt(d)
        return self.signs * (_fwht(y) / math.sqrt(self.dim))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, seed={self.seed}"


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform  (in-place, iterative, no external deps)
# ---------------------------------------------------------------------------


def _fwht(x: Tensor) -> Tensor:
    """
    Normalised Fast Walsh-Hadamard Transform along the last dimension.

    x: (..., d) where d must be a power of 2.
    Returns (..., d) with the same dtype and device.

    The transform is its own inverse (H^{-1} = H / d  - but we return the
    unnormalised H here; callers divide by sqrt(d) as needed).
    """
    x = x.clone()
    d = x.shape[-1]
    h = 1
    while h < d:
        # Butterfly step: reshape to (..., d/2h, 2h), update in-place

        # Concrete example with shape (3, 4) and slice 2:
        #
        # x = tensor([[1, 2, 3, 4],
        #            [5, 6, 7, 8],
        #            [9, 10, 11, 12]])
        #
        # x[:, :2]
        #  [[1, 2],
        #  [5, 6],
        #  [9, 10]]    # all rows, first 2 columns

        x = x.reshape(*x.shape[:-1], d // (2 * h), 2 * h)
        a = x[..., :h] + x[..., h:]
        x[..., h:] = x[..., :h] - x[..., h:]
        x[..., :h] = a
        x = x.reshape(*x.shape[:-2], d)
        h *= 2
    return x


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_qr_rotation(dim: int, seed: int) -> Tensor:
    """Return a Haar-uniform random orthogonal matrix via QR decomposition."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(G)
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)  # (d, d)
    # Enforce det = +1 so Q is a proper rotation (SO(d)) not a reflection
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q
