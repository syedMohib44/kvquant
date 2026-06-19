"""
Triton row-wise softmax kernel.

Adapted from D:/cuda-triton/kernels/softmax.py with two changes:
  1. BLOCK_SIZE is chosen at call time (next power-of-2 >= N_cols) so it
     works for any sequence length without recompilation.
  2. Falls back to F.softmax when Triton is unavailable or tensor is on CPU.

Grid: (num_rows,) — one Triton program per row.
Each program loads a full row, computes max-subtracted exp, divides by sum,
and writes back.  The numerically-stable (max-subtract) formulation matches
PyTorch's own softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_OK:

    @triton.jit
    def _softmax_kernel(
        X_ptr,
        Out_ptr,
        row_stride,        # stride between rows (== N_cols for contiguous)
        N_cols,            # number of columns (runtime)
        BLOCK_SIZE: tl.constexpr,  # next power-of-2 >= N_cols
    ):
        row = tl.program_id(axis=0)
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < N_cols

        # Load row with out-of-bounds masked to -inf (safe for max/exp)
        x = tl.load(X_ptr + row * row_stride + col_offs, mask=mask, other=float("-inf"))

        # Numerically stable softmax: subtract row max before exp
        x_max = tl.max(x, axis=0)
        x_exp = tl.exp(x - x_max)
        x_exp = tl.where(mask, x_exp, 0.0)  # zero padding slots
        x_sum = tl.sum(x_exp, axis=0)

        out = x_exp / x_sum
        tl.store(Out_ptr + row * row_stride + col_offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def softmax_triton(x: Tensor, dim: int = -1) -> Tensor:
    """
    Row-wise softmax using a fused Triton kernel.

    Supports 2-D and higher tensors; softmax is always applied along the
    last dimension (or `dim=-1`).  Falls back to F.softmax for CPU tensors
    or when Triton is not installed.

    Args:
        x:   Input tensor (any shape).
        dim: Dimension to reduce (default -1 / last).

    Returns:
        Tensor of same shape and dtype as x.
    """
    if not _TRITON_OK or not x.is_cuda or dim not in (-1, x.ndim - 1):
        return F.softmax(x, dim=dim)

    # Flatten all leading dims into a single "rows" dimension
    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1]).contiguous().float()
    N_rows, N_cols = x2d.shape

    out = torch.empty_like(x2d)

    # BLOCK_SIZE = next power-of-2 >= N_cols  (Triton requires constexpr)
    BLOCK_SIZE = triton.next_power_of_2(N_cols)

    _softmax_kernel[(N_rows,)](
        x2d, out,
        x2d.stride(0),
        N_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.to(x.dtype).reshape(orig_shape)
