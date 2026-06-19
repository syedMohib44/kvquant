"""
Triton row-wise softmax.

Import priority:
  1. cuda-triton-kernels package (pip install kvquant-plus-plus[cuda])
     — same kernel, kept in sync with the cuda-triton project.
  2. Built-in fallback using F.softmax (CPU or Triton unavailable).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Try the cuda-triton-kernels package first
try:
    from kernels import softmax_triton as _softmax_triton_impl
    _SOURCE = "cuda_triton_kernels"
except ImportError:
    _softmax_triton_impl = None
    _SOURCE = "fallback"

# If cuda-triton-kernels not installed, try local Triton implementation
if _softmax_triton_impl is None:
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _softmax_kernel(
            X_ptr, Out_ptr, row_stride, N_cols, BLOCK_SIZE: tl.constexpr,
        ):
            row = tl.program_id(axis=0)
            col_offs = tl.arange(0, BLOCK_SIZE)
            mask = col_offs < N_cols
            x = tl.load(X_ptr + row * row_stride + col_offs, mask=mask, other=float("-inf"))
            x_max = tl.max(x, axis=0)
            x_exp = tl.exp(x - x_max)
            x_exp = tl.where(mask, x_exp, 0.0)
            x_sum = tl.sum(x_exp, axis=0)
            out = x_exp / x_sum
            tl.store(Out_ptr + row * row_stride + col_offs, out, mask=mask)

        def _softmax_triton_impl(x: Tensor) -> Tensor:
            x2d = x.reshape(-1, x.shape[-1]).contiguous().float()
            N_rows, N_cols = x2d.shape
            out = torch.empty_like(x2d)
            BLOCK_SIZE = triton.next_power_of_2(N_cols)
            _softmax_kernel[(N_rows,)](x2d, out, x2d.stride(0), N_cols, BLOCK_SIZE=BLOCK_SIZE)
            return out.to(x.dtype).reshape(x.shape)

        _SOURCE = "local_triton"
    except ImportError:
        pass


  # Triton launch overhead dominates for small tensors.
# Benchmarks show break-even at ~N_cols >= 8192 on RTX-class GPUs.
_SOFTMAX_TRITON_MIN_COLS = 8192


def softmax_triton(x: Tensor, dim: int = -1) -> Tensor:
    """
    Row-wise softmax. Uses a Triton kernel for large sequences where
    the fused exp+sum+div beats PyTorch's kernel launch overhead,
    falls back to F.softmax for small tensors or CPU.
    """
    N_cols = x.shape[-1]
    if (
        _softmax_triton_impl is None
        or not x.is_cuda
        or dim not in (-1, x.ndim - 1)
        or N_cols < _SOFTMAX_TRITON_MIN_COLS
    ):
        return F.softmax(x, dim=dim)

    if _SOURCE == "cuda_triton_kernels":
        orig_shape = x.shape
        x2d = x.reshape(-1, N_cols).contiguous()
        out = _softmax_triton_impl(x2d)
        return out.reshape(orig_shape)

    return _softmax_triton_impl(x)
