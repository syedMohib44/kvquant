"""
CUDA/Triton kernels for kvquant acceleration.

Provides drop-in replacements for the hot paths in kvquant:
  - pq_encode_triton : batched PQ encode (replaces M sequential torch.cdist calls)
  - softmax_triton   : row-wise numerically-stable softmax

Both fall back silently to pure-PyTorch when Triton is not installed or the
tensor is not on a CUDA device, so the package continues to work on CPU.
"""

from .pq_encode import pq_encode_triton, TRITON_AVAILABLE
from .softmax import softmax_triton

__all__ = ["pq_encode_triton", "softmax_triton", "TRITON_AVAILABLE"]
