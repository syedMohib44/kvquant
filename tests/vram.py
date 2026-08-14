"""
Peak-VRAM measurement for KV-cache paths.

Every memory claim in the paper and README rests on this file, so the
protocol is spelled out rather than left implicit.  Three deliberate
choices, each of which would otherwise let a false claim through:

1. **Subtract the baseline.**  Model weights dominate absolute VRAM (a 7B
   model in 4-bit is ~4 GB) and would swamp a cache delta of a few hundred
   MB.  We report peak *over the already-loaded model*, so the number is
   attributable to the cache.

2. **`memory_allocated`, not `memory_reserved`.**  PyTorch's caching
   allocator holds on to freed blocks, so `reserved` is sticky: once the
   float path has reserved a large pool, a subsequent compact run appears
   to use the same amount.  That would make a real saving invisible.

3. **Same process, back-to-back, `empty_cache()` between.**  Comparing two
   separate processes folds in allocator warm-up and import-order
   differences that have nothing to do with the cache.

The self-test in `test_vram_harness.py` allocates a known size and asserts
the measurement lands within 10% — a broken ruler would otherwise silently
"prove" whatever we hoped for.
"""

from __future__ import annotations

import gc
from typing import Any, Callable

import torch


def cuda_available() -> bool:
    """True when a CUDA device is present and usable."""
    return torch.cuda.is_available()


def reset_vram() -> None:
    """Drop cached blocks and zero the peak counter."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def measure_peak_vram(fn: Callable[..., Any], *args, **kwargs) -> tuple[Any, int]:
    """
    Run ``fn`` and return ``(result, peak_bytes_over_baseline)``.

    ``peak_bytes`` is the high-water mark of *allocated* memory during the
    call, minus whatever was already allocated when the call began.  On CPU
    this returns ``(result, 0)`` — there is no meaningful equivalent, and
    callers are expected to gate on :func:`cuda_available`.
    """
    if not torch.cuda.is_available():
        return fn(*args, **kwargs), 0

    reset_vram()
    baseline = torch.cuda.memory_allocated()
    try:
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
    finally:
        # Peak is read even on failure so a crashing run still reports.
        peak = torch.cuda.max_memory_allocated()
    return result, max(peak - baseline, 0)


def compare_peak_vram(
    fn_a: Callable[[], Any],
    fn_b: Callable[[], Any],
) -> tuple[int, int]:
    """
    Measure two callables back-to-back in one process and return
    ``(peak_a, peak_b)``.

    Use this rather than two separate :func:`measure_peak_vram` calls when
    the comparison itself is the claim: it guarantees both runs see the same
    allocator state, and it drops each result before measuring the next so
    ``fn_a``'s cache cannot inflate ``fn_b``'s baseline.
    """
    result_a, peak_a = measure_peak_vram(fn_a)
    del result_a
    reset_vram()
    result_b, peak_b = measure_peak_vram(fn_b)
    del result_b
    reset_vram()
    return peak_a, peak_b
