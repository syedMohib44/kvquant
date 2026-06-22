"""
CUDA/Triton kernels for kvquant acceleration.

Import priority for each kernel:
  1. cuda-triton-kernels package  (pip install kvquant-plus-plus[cuda])
  2. Local Triton implementation  (pip install triton)
  3. Pure PyTorch fallback        (always available, CPU-safe)

Install the GPU-accelerated path:
    pip install "kvquant-plus-plus[cuda]"

This installs both triton and cuda-triton-kernels from:
    https://github.com/syedMohib44/cuda-triton-multiarch
"""

from .pq_encode import pq_encode_triton, TRITON_AVAILABLE
from .softmax import softmax_triton
from .attention import attention_bhsd, attention_backend

__all__ = ["pq_encode_triton", "softmax_triton", "TRITON_AVAILABLE", "attention_bhsd", "attention_backend"]
