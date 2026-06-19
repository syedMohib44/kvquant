"""
Triton kernel for Product Quantization encode.

Replaces the Python loop of M sequential torch.cdist calls in
ProductQuantizer.quantize() with a single fused GPU kernel that processes
all M subspaces in parallel across the token dimension.

Grid: (M, ceil(N / BLOCK_N))
  axis-0 program handles one subspace
  axis-1 program handles BLOCK_N consecutive tokens

Each program:
  1. Loads BLOCK_N sub-vectors of shape (BLOCK_N, sub_d) from x
  2. Iterates over K centroids (runtime loop — supports K=16 and K=256)
  3. Computes squared L2 distance via tl.sum(diff*diff, axis=1)
  4. Tracks argmin with scalar compare
  5. Writes BLOCK_N codes to the output

Expected speedup: ~10-15x over M sequential torch.cdist calls on CUDA
(eliminates M kernel launches + M allocation/copy cycles).
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _pq_encode_kernel(
        x_ptr,      # (N, d)  float32  rotated, unit-normed sub-vectors (flat)
        books_ptr,  # (M, K, sub_d)  float32  codebook centroids (flat)
        codes_ptr,  # (N, M)  int32  output codes (flat)
        N,          # number of vectors
        K,          # codebook entries per subspace (runtime — not constexpr)
        d: tl.constexpr,      # total dimension
        M: tl.constexpr,      # number of subspaces
        sub_d: tl.constexpr,  # d // M  — sub-vector dimensionality
        BLOCK_N: tl.constexpr,  # vectors per program along axis-1
    ):
        m = tl.program_id(0)          # which subspace
        block_n = tl.program_id(1)    # which block of tokens

        n_start = block_n * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        offs_sub = tl.arange(0, sub_d)             # (sub_d,)

        # Load BLOCK_N sub-vectors: x[offs_n, m*sub_d : (m+1)*sub_d]
        # x is stored as (N, d) row-major, so offset = n * d + m * sub_d + s
        x_offs = offs_n[:, None] * d + m * sub_d + offs_sub[None, :]  # (BLOCK_N, sub_d)
        mask_n = offs_n < N
        x_sub = tl.load(x_ptr + x_offs, mask=mask_n[:, None], other=0.0)  # (BLOCK_N, sub_d)

        # Find nearest centroid (argmin over K entries)
        best_dist = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_k    = tl.zeros((BLOCK_N,), dtype=tl.int32)

        for k in range(K):
            # Load centroid k of subspace m: books[m, k, :]
            c_offs = m * K * sub_d + k * sub_d + offs_sub  # (sub_d,)
            c = tl.load(books_ptr + c_offs)                  # (sub_d,)

            # Squared L2 distance: sum over sub_d dimension
            diff = x_sub - c[None, :]         # (BLOCK_N, sub_d)
            sq_dist = tl.sum(diff * diff, axis=1)  # (BLOCK_N,)

            closer   = sq_dist < best_dist
            best_dist = tl.where(closer, sq_dist, best_dist)
            best_k    = tl.where(closer, k, best_k)

        # Write codes: codes[offs_n, m]  stored as (N, M) row-major
        out_offs = offs_n * M + m  # (BLOCK_N,)
        tl.store(codes_ptr + out_offs, best_k, mask=mask_n)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def pq_encode_triton(
    y: Tensor,       # (N, d)  float32, already rotated + unit-normed
    books: Tensor,   # (M, K, sub_d)  float32, codebook centroids
) -> Tensor:
    """
    Encode N rotated vectors into PQ codes using a fused Triton kernel.

    Args:
        y:     (N, d) float32 tensor (post-rotation, unit-norm).
        books: (M, K, sub_d) float32 codebook centroids.

    Returns:
        codes: (N, M) int64 tensor of subspace indices.

    Falls back to sequential torch.cdist when Triton is unavailable or
    the tensors are not on a CUDA device.
    """
    if not TRITON_AVAILABLE or not y.is_cuda:
        return _pq_encode_pytorch(y, books)

    N, d = y.shape
    M, K, sub_d = books.shape
    assert d == M * sub_d, f"d ({d}) must equal M*sub_d ({M}*{sub_d})"

    # Triton requires contiguous float32
    y = y.contiguous().float()
    books = books.contiguous().float()

    # Output codes (int32 in kernel, cast to int64 for PyTorch compatibility)
    codes_i32 = torch.empty(N, M, dtype=torch.int32, device=y.device)

    BLOCK_N = 64  # tunable: 32-128; 64 is a good default for sub_d <= 8
    grid = (M, triton.cdiv(N, BLOCK_N))

    _pq_encode_kernel[grid](
        y, books, codes_i32,
        N, K,
        d, M, sub_d, BLOCK_N,
    )

    return codes_i32.long()  # (N, M) int64


def _pq_encode_pytorch(y: Tensor, books: Tensor) -> Tensor:
    """Pure-PyTorch fallback (vectorised, single cdist per subspace)."""
    N, d = y.shape
    M, K, sub_d = books.shape
    y_split = y.reshape(N, M, sub_d)
    codes = torch.empty(N, M, dtype=torch.long, device=y.device)
    for m in range(M):
        dists = torch.cdist(y_split[:, m, :].contiguous(), books[m])
        codes[:, m] = dists.argmin(dim=-1)
    return codes
