"""
GPU acceleration benchmark for kvquant.

Measures wall-clock speedup for all four optimized hot paths:
  1. PQ encode        — Triton kernel vs M sequential cdist calls
  2. FWHT             — torch.compile vs plain Python butterfly loop
  3. Softmax          — Triton vs F.softmax
  4. QJL projection   — torch.compile fused vs plain matmul + compare

Run from the kvquant/ directory with the kvquant env active:
    python bench_gpu.py

CUDA is required for the Triton / torch.compile paths.
On CPU the script still runs but reports CPU timings only.
"""

import time
import torch
import torch.nn.functional as F

# ── helpers ────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench(fn, warmup=5, iters=50):
    """Return median wall-clock ms over `iters` calls after `warmup` calls."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]   # median

def row(label, baseline_ms, fast_ms):
    speedup = baseline_ms / fast_ms if fast_ms > 0 else float("inf")
    tag = f"{speedup:.2f}x faster" if speedup >= 1.0 else f"{1/speedup:.2f}x SLOWER"
    print(f"  {label:<40}  baseline={baseline_ms:.3f}ms  optimized={fast_ms:.3f}ms  [{tag}]")

# ── device ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"SM    : {major}{minor}")
else:
    print("WARNING: CUDA not available — timings are CPU only, no Triton speedups expected.")

print()

# ══════════════════════════════════════════════════════════════════════════
# 1. PQ ENCODE
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. PQ ENCODE  (M=16 subspaces, K=16 centroids, d=64, N=2048)")
print("=" * 70)

from kvquant import ProductQuantizer

dim, M, b = 64, 16, 4
pq = ProductQuantizer(dim=dim, num_subspaces=M, bits_per_subspace=b).to(device)
cal = torch.randn(1024, dim, device=device)
pq.calibrate(cal)

x_pq = torch.randn(2048, dim, device=device)

# baseline: the old sequential cdist loop (bypass our new dispatch)
from kvquant.csrc.pq_encode import _pq_encode_pytorch

norms = x_pq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
y_pq  = pq.rotation(x_pq / norms)
books = pq.codebooks.to(device)

baseline_pq = bench(lambda: _pq_encode_pytorch(y_pq, books))

# optimized: Triton kernel (or torch fallback on CPU)
from kvquant.csrc.pq_encode import pq_encode_triton, TRITON_AVAILABLE
print(f"  Triton available: {TRITON_AVAILABLE}")
optimized_pq = bench(lambda: pq_encode_triton(y_pq, books))
if not TRITON_AVAILABLE:
    print("  (Triton not installed — both paths use PyTorch cdist, expect ~1x)")

row("PQ encode (N=2048, M=16, K=16, sub_d=4)", baseline_pq, optimized_pq)
print()

# ══════════════════════════════════════════════════════════════════════════
# 2. FWHT
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("2. FWHT  (d=128, batch=4096 vectors)")
print("=" * 70)

from kvquant.rotation import _fwht_impl, _fwht

x_fwht = torch.randn(4096, 128, device=device)

# warmup torch.compile on first call
_ = _fwht(x_fwht)
_sync()

baseline_fwht  = bench(lambda: _fwht_impl(x_fwht))
optimized_fwht = bench(lambda: _fwht(x_fwht))

row("FWHT d=128, N=4096", baseline_fwht, optimized_fwht)
print()

# ══════════════════════════════════════════════════════════════════════════
# 3. SOFTMAX
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("3. SOFTMAX  (batch=32, various sequence lengths)")
print("=" * 70)

from kvquant.csrc.softmax import softmax_triton, _SOURCE as softmax_source

print(f"  Softmax backend: {softmax_source}")
if softmax_source == "fallback":
    print("  (Triton not installed — softmax falls back to F.softmax, expect ~1x)")

for T in [128, 512, 1024, 2048, 4096]:
    scores = torch.randn(32, T, device=device)
    _sync()
    b_sm = bench(lambda: F.softmax(scores, dim=-1))
    o_sm = bench(lambda: softmax_triton(scores))
    row(f"softmax  batch=32, T={T}", b_sm, o_sm)
print()

# ══════════════════════════════════════════════════════════════════════════
# 4. QJL PROJECTION
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("4. QJL PROJECTION  r @ S.T > 0  (d=64, N=2048)")
print("=" * 70)

from kvquant.quantizer import _qjl_project, _qjl_project_dispatch

d = 64
r = torch.randn(2048, d, device=device)
S = torch.randn(d, d, device=device)

# warmup compile
_ = _qjl_project_dispatch(r, S)
_sync()

baseline_qjl  = bench(lambda: _qjl_project(r, S))
optimized_qjl = bench(lambda: _qjl_project_dispatch(r, S))

row("QJL  r @ S.T > 0  (N=2048, d=64)", baseline_qjl, optimized_qjl)
print()

# ══════════════════════════════════════════════════════════════════════════
# 5. END-TO-END: ProductQuantizer.quantize()
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("5. END-TO-END: ProductQuantizer.quantize()  (N=2048, dim=64, M=16)")
print("=" * 70)

e2e = bench(lambda: pq.quantize(x_pq))
print(f"  ProductQuantizer.quantize()  N=2048  {e2e:.3f}ms  (includes rotation + encode)")
print()

print("=" * 70)
print("Done.")
print("=" * 70)
