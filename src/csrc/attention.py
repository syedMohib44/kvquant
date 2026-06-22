"""
attention_bhsd — Flash attention wrapper for KVQuant (B, H, T, d) tensors.

Import priority:
  1. cuda-triton-kernels flash_attention_bhsd  (WMMA/CUTLASS/Triton cascade)
  2. torch.nn.functional.scaled_dot_product_attention  (CPU / any dtype fallback)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from kernels import flash_attention_bhsd as _fa_impl
    from kernels import flash_attention_backend as _fa_backend
    _SOURCE = "cuda_triton_kernels"
except ImportError:
    _fa_impl = None
    _fa_backend = None
    _SOURCE = "fallback"


def attention_bhsd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
) -> Tensor:
    """
    Multi-head attention for (B, H, T, d) tensors.

    Uses cuda-triton flash attention when available (WMMA → CUTLASS → Triton),
    otherwise falls back to torch.nn.functional.scaled_dot_product_attention.

    Args:
        q:         Query  — (batch, heads, seqlen_q, head_dim)
        k:         Key    — (batch, heads, seqlen_k, head_dim)
        v:         Value  — (batch, heads, seqlen_k, head_dim)
        is_causal: Apply causal mask (default False).

    Returns:
        Attention output — same shape and dtype as q.
    """
    if _fa_impl is not None and q.is_cuda:
        return _fa_impl(q, k, v, is_causal=is_causal)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def attention_backend() -> str:
    """Return the name of the backend attention_bhsd will use."""
    if _fa_backend is not None:
        return _fa_backend()
    return "scaled_dot_product_attention (PyTorch fallback — cuda-triton not installed)"
