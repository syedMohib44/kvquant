"""
A KV cache that stores compact codes and never materializes the full float
tensors — so quantization actually reduces VRAM instead of only simulating
its quality cost.

Why this exists
---------------
The original in-memory path compressed the prefill and then immediately
decompressed it back to float (``quantize_model_cache``), which faithfully
reproduces quantization's *distortion* while banking none of its *savings*.
Worse, only the prefill was ever compressed: the decode loop handed the
model's own ``DynamicCache`` straight back each step, so every generated
token's K/V stayed at full precision forever.

Here, codes are the storage of record.  Floats exist only for one block at a
time, inside the attention computation, and are freed immediately.

How it works
------------
``CompactKVLayer`` holds an append-only list of compressed blocks plus a small
float staging window.  Its ``update()`` returns *zero-length* tensors, so the
model never receives a full K/V tensor at all; a custom attention function
registered into ``ALL_ATTENTION_FUNCTIONS`` reads the blocks off the layer,
dequantizes them one at a time, and accumulates the attention output.

Append-only is load-bearing, not incidental.  Lloyd-Max quantization is lossy,
so re-compressing an already-reconstructed block compounds error (measured
elsewhere in this repo: 4-bit MSE 0.0096, demote to 3-bit 0.0352, promote back
to 4-bit 0.0492 — worse than the state it came from).  Each token is therefore
compressed exactly once and its block is frozen.

Paper alignment
---------------
Blocks use the outlier-aware Lloyd-Max codec — the paper's §3.1 MSE quantizer
(Algorithm 1, Theorem 1) applied separately to the outlier and regular channel
groups of its §4.3 aside — via the same ``_paper_compress`` /
``_paper_outlier_compress`` primitives the disk-offload tier uses.  The codec is
imported, never duplicated.

Both K and V use ``KVQuantMSE``.  ``KVQuantIP`` (§3.2, Theorem 2) is an
inner-product *estimator*: it is unbiased in ``<y, x>`` but its per-coordinate
reconstruction is wrong, so a cache built from it produces garbage.  The paper
never says which quantizer to use for keys versus values — it does not
distinguish them anywhere — so this is our choice, pinned by
``test_mse_key_reconstructs_far_better_than_ip``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from .kv_cache import (
    _PaperKV,
    _PaperOutlierKV,
    _paper_compress,
    _paper_dequantize,
    _paper_outlier_compress,
    _paper_outlier_dequantize,
)
from .outlier import OutlierKVQuant
from .quantizer import KVQuantMSE

ATTN_IMPL_NAME = "kvquant_compact"
DEFAULT_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Codec helpers (shared with the disk-offload tier — imported, not duplicated)
# ---------------------------------------------------------------------------


def _make_outlier_quantizer(
    dim: int, bits: int, gqa_factor: int = 1, seed: int = 0
) -> OutlierKVQuant:
    """
    Build the outlier configuration, GQA-compensated.

    The bit schedule is ours.  The paper gives exactly one configuration — 32 of
    128 channels at 3 bits, the rest at 2 (§4.3) — with no formula to generalise
    it and no criterion for picking the outlier channels.  (Its stated average
    for that split, "2.5", is also arithmetically wrong; the split gives 2.25.
    See ``outlier.py``.)  We generalise to
    ``outlier_bits = min(bits+1, 4)``, ``regular_bits = max(bits-1, 1)``,
    ``n_outlier = dim//4``, which reproduces the paper's channel fraction and
    its "outliers get more bits" intent.

    The GQA allowance has no basis in the paper at all: TurboQuant never
    mentions grouped-query attention.  It is our compensation for the fact that
    g query heads share one KV head, so a quantization error in that head is
    amplified across all g of them.

    Mirrors ``KVCacheDiskOffload._get_outlier_q`` exactly, including the
    cap-then-bump order: ``outlier_bits`` is capped at 4 *first*, then the GQA
    allowance is added.  Doing it the other way round reports a bit-width that
    differs from what is actually stored.
    """
    gqa_extra = math.ceil(math.log(gqa_factor, 4)) if gqa_factor > 1 else 0
    outlier_bits = min(min(bits + 1, 4) + gqa_extra, 8)
    regular_bits = min(max(max(bits - 1, 1) + gqa_extra, 1), 8)
    n_outlier = min(max(dim // 4, 1), dim - 1)
    return OutlierKVQuant(
        dim=dim,
        n_outlier=n_outlier,
        outlier_bits=outlier_bits,
        regular_bits=regular_bits,
        seed=seed,
        quantizer_cls=KVQuantMSE,  # IP reconstructs per-coordinate incorrectly
    )


def _effective_bits(bits: int, gqa_factor: int = 1) -> int:
    """Pack width for the plain `paper` codec, with the GQA allowance."""
    gqa_extra = math.ceil(math.log(gqa_factor, 4)) if gqa_factor > 1 else 0
    return min(bits + gqa_extra, 8)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


class CompactKVLayer:
    """
    One transformer layer's worth of compressed KV, append-only.

    Deliberately duck-typed against ``transformers.cache_utils.DynamicLayer``
    rather than subclassing it: the base class's methods all assume ``keys``
    and ``values`` are the real float tensors, and inheriting them would mean
    overriding nearly every one.  The surface a decode step actually touches is
    small and implemented in full here.

    ``keys``/``values`` are zero-length tensors rather than ``None`` because
    ``kvs_from_cache`` and ``Cache.__getitem__``/``__iter__`` read them
    directly; ``None`` would crash those, while a zero-length tensor keeps the
    structure valid and carries no data.

    **The pending window is real memory.**  Tokens accumulate in float until
    they fill a block, so up to ``block_size - 1`` positions are uncompressed
    at any moment.  That cost is fixed while the compressed part grows, so the
    achieved ratio is context-dependent and approaches the steady state from
    below.  Measured on a 2-layer GQA model at 3 bits, ``block_size=128``, with
    a 64-token tail:

        T=192  1.17x     T=1088  2.62x
        T=320  1.60x     T=2112  3.00x
        T=576  2.12x     T=4160  3.25x     -> 3.56x fully flushed

    At short contexts the tail dominates, and unflushed it can push total bytes
    *above* float16 (0.88x at T=255 before ``flush()`` existed).  Hence
    ``flush()`` at the end of prefill.  A ratio quoted without its context
    length is not a meaningful number for this cache.
    """

    is_sliding = False
    is_compileable = False

    def __init__(
        self,
        bits: int = 4,
        block_size: int = DEFAULT_BLOCK_SIZE,
        codec: str = "paper-outlier",
        gqa_factor: int = 1,
        seed: int = 0,
    ) -> None:
        if codec not in ("paper", "paper-outlier"):
            raise ValueError(
                f"codec must be 'paper' or 'paper-outlier', got {codec!r}"
            )
        self.bits = int(bits)
        self.block_size = max(int(block_size), 1)
        self.codec = codec
        self.gqa_factor = max(int(gqa_factor), 1)
        self.seed = seed

        self._k_blocks: list[Any] = []
        self._v_blocks: list[Any] = []
        self._k_quant: OutlierKVQuant | None = None
        self._v_quant: OutlierKVQuant | None = None
        self._mse_cache: dict[tuple[int, int], KVQuantMSE] = {}

        self._pending_k: Tensor | None = None
        self._pending_v: Tensor | None = None
        self._cum_len = 0

        self.dtype: torch.dtype | None = None
        self.device: torch.device | None = None
        self.is_initialized = False
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    # -- transformers layer contract ------------------------------------

    def lazy_initialization(self, key_states: Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states[..., :0, :]
        self.values = key_states[..., :0, :]
        self.is_initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Absorb new K/V, compressing whole blocks as they complete.

        Returns zero-length tensors: the registered attention function reads
        the stored blocks off this layer instead of receiving them here.  That
        is what keeps the full float cache from ever existing.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self._pending_k = (
            key_states
            if self._pending_k is None
            else torch.cat([self._pending_k, key_states], dim=-2)
        )
        self._pending_v = (
            value_states
            if self._pending_v is None
            else torch.cat([self._pending_v, value_states], dim=-2)
        )
        self._cum_len += key_states.shape[-2]

        # `while`, not `if`: a prefill chunk hands in hundreds of tokens at
        # once and must produce several blocks in one call.
        while self._pending_k.shape[-2] >= self.block_size:
            k_blk = self._pending_k[..., : self.block_size, :]
            v_blk = self._pending_v[..., : self.block_size, :]
            self._k_blocks.append(self._compress(k_blk, is_value=False))
            self._v_blocks.append(self._compress(v_blk, is_value=True))
            # Slice off the consumed prefix; `.contiguous()` so the freed
            # portion is actually released rather than kept alive by a view.
            self._pending_k = self._pending_k[..., self.block_size :, :].contiguous()
            self._pending_v = self._pending_v[..., self.block_size :, :].contiguous()

        return key_states[..., :0, :], value_states[..., :0, :]

    def flush(self) -> None:
        """
        Compress the pending window as a short block.

        Blocks need not be uniform: the attention loop reads each block's
        length from its own shape, and the codec's norms are per-vector, so a
        128-token block and a 7-token block cost the same per token.  The only
        per-block waste is bit-packing padding to a byte boundary — under two
        bytes.

        Worth calling once when a burst of tokens ends (end of prefill), and
        not on every decode step: it leaves each token compressed exactly once
        either way, but per-step flushing pays a codec call per layer per token
        for no storage gain.  Left un-flushed, the tail stays float, which at
        short contexts can be most of the cache — 127 of 255 tokens at the
        default block size, enough to put total bytes *above* float16.
        """
        if self._pending_k is None or self._pending_k.shape[-2] == 0:
            return
        self._k_blocks.append(self._compress(self._pending_k, is_value=False))
        self._v_blocks.append(self._compress(self._pending_v, is_value=True))
        self._pending_k = None
        self._pending_v = None

    def get_seq_length(self, cache_position=None) -> int:
        """
        Total positions represented, blocks plus pending.

        Must not fall back to reading ``keys.shape[-2]`` — that is zero by
        construction here, which would make the model believe the cache is
        permanently empty.
        """
        return self._cum_len

    def get_mask_sizes(self, cache_position: Tensor, layer_idx: int = 0) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        return self._cum_len + query_length, 0

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self._k_blocks.clear()
        self._v_blocks.clear()
        self._pending_k = None
        self._pending_v = None
        self._cum_len = 0

    def crop(self, max_length: int) -> None:
        """
        Truncate to ``max_length`` positions.

        Whole blocks are dropped; a cut landing inside a stored block can only
        be honoured down to that block's boundary, since re-encoding the
        partial block would violate compress-exactly-once.  The result is
        therefore a *lower* bound on ``max_length``, and ``_cum_len`` is set to
        what is genuinely stored rather than what was asked for — reporting a
        length longer than ``iter_kv_blocks`` yields would misalign every
        subsequent causal mask.  The generation path sidesteps the whole
        question by prefilling one token short instead of cropping.
        """
        if max_length >= self._cum_len:
            return
        if max_length < 0:
            max_length = max(self._cum_len + max_length, 0)

        # Walk each block's own length rather than assuming all hold
        # `block_size` tokens: `flush()` seals a short tail block, so blocks
        # are not uniform and `max_length // block_size` would cut in the
        # wrong place after one.
        kept = 0
        n_whole = 0
        for blk in self._k_blocks:
            blk_len = blk.shape[-2]
            if kept + blk_len > max_length:
                break
            kept += blk_len
            n_whole += 1
        if n_whole < len(self._k_blocks):
            del self._k_blocks[n_whole:]
            del self._v_blocks[n_whole:]

        have_pending = 0 if self._pending_k is None else self._pending_k.shape[-2]
        keep_pending = min(max(max_length - kept, 0), have_pending)
        if keep_pending <= 0:
            self._pending_k = None
            self._pending_v = None
        else:
            self._pending_k = self._pending_k[..., :keep_pending, :].contiguous()
            self._pending_v = self._pending_v[..., :keep_pending, :].contiguous()
        self._cum_len = kept + keep_pending

    def offload(self) -> None:
        """Move stored codes (not the empty `keys`) to host memory."""
        self._k_blocks = [_codes_to(b, "cpu") for b in self._k_blocks]
        self._v_blocks = [_codes_to(b, "cpu") for b in self._v_blocks]

    def prefetch(self) -> None:
        if self.device is None:
            return
        self._k_blocks = [_codes_to(b, self.device) for b in self._k_blocks]
        self._v_blocks = [_codes_to(b, self.device) for b in self._v_blocks]

    # -- calibration and codec ------------------------------------------

    def calibrate(self, k: Tensor, v: Tensor) -> None:
        """
        Fit this layer's outlier detector on this layer's own KV.

        Per-layer rather than pooled: outlier channels are layer-specific, and
        pooling across layers mis-identifies them.  Called with real prefill
        activations, never a synthetic corpus.

        Calibrating at all is a departure from the paper, which advertises
        itself as data-oblivious (§1.2) and specifies no calibration step.  The
        outlier split of §4.3 forces the issue — it names no way to choose the
        channels — and per-layer is what our measurements support.
        """
        if self.codec != "paper-outlier":
            return
        dim = k.shape[-1]
        self._k_quant = _make_outlier_quantizer(dim, self.bits, self.gqa_factor, self.seed)
        self._v_quant = _make_outlier_quantizer(dim, self.bits, self.gqa_factor, self.seed)
        self._k_quant.calibrate(k.reshape(-1, dim).float())
        self._v_quant.calibrate(v.reshape(-1, dim).float())

    def _get_mse(self, dim: int, num_bits: int) -> KVQuantMSE:
        key = (dim, num_bits)
        mse = self._mse_cache.get(key)
        if mse is None:
            mse = KVQuantMSE(dim=dim, num_bits=num_bits, seed=self.seed)
            self._mse_cache[key] = mse
        return mse

    def _compress(self, t: Tensor, is_value: bool):
        if self.codec == "paper":
            return _paper_compress(
                t, _effective_bits(self.bits, self.gqa_factor), self._get_mse
            )
        quant = self._v_quant if is_value else self._k_quant
        if quant is None:
            # Calibrate on this first block when no explicit calibration ran.
            # Still real activations, still this layer's own data.
            dim = t.shape[-1]
            quant = _make_outlier_quantizer(
                dim, self.bits, self.gqa_factor, self.seed
            )
            quant.calibrate(t.reshape(-1, dim).float())
            if is_value:
                self._v_quant = quant
            else:
                self._k_quant = quant
        return _paper_outlier_compress(t, quant)

    def _dequantize(self, codes, is_value: bool) -> Tensor:
        if isinstance(codes, _PaperOutlierKV):
            quant = self._v_quant if is_value else self._k_quant
            return _paper_outlier_dequantize(codes, quant, device=self.device)
        if isinstance(codes, _PaperKV):
            return _paper_dequantize(codes, self._get_mse, device=self.device)
        raise TypeError(f"unknown compressed type {type(codes).__name__}")

    # -- read API for the attention function ----------------------------

    def iter_kv_blocks(self) -> Iterator[tuple[Tensor, Tensor]]:
        """
        Yield ``(k_block, v_block)`` float tensors one block at a time,
        oldest first, ending with the uncompressed pending window.

        A generator on purpose: the caller consumes and drops each block, so
        at most one block's worth of float exists at a time.
        """
        for k_codes, v_codes in zip(self._k_blocks, self._v_blocks):
            yield (
                self._dequantize(k_codes, is_value=False),
                self._dequantize(v_codes, is_value=True),
            )
        if self._pending_k is not None and self._pending_k.shape[-2] > 0:
            yield self._pending_k, self._pending_v

    def byte_breakdown(self):
        """Measured storage for this layer, payload vs sidecar."""
        from .cache_bytes import ByteBreakdown, codec_bytes

        total = ByteBreakdown()
        for k_codes, v_codes in zip(self._k_blocks, self._v_blocks):
            total = total + codec_bytes(k_codes) + codec_bytes(v_codes)
        for t in (self._pending_k, self._pending_v):
            if isinstance(t, Tensor) and t.numel():
                total = total + codec_bytes(t)
        return total

    def __repr__(self) -> str:
        return (
            f"CompactKVLayer(bits={self.bits}, block_size={self.block_size}, "
            f"codec={self.codec!r}, blocks={len(self._k_blocks)}, "
            f"len={self._cum_len})"
        )


def _codes_to(codes, device):
    """Move every tensor field of a codec NamedTuple to `device`."""
    moved = {
        name: (val.to(device) if isinstance(val, Tensor) else val)
        for name, val in zip(codes._fields, codes)
    }
    return type(codes)(**moved)


# ---------------------------------------------------------------------------
# The cache container
# ---------------------------------------------------------------------------


class CompactKVCache:
    """
    A whole-model KV cache holding compact codes.

    Duck-typed against the ``transformers`` cache interface: a decode step
    touches only ``layers``, ``update``, ``get_seq_length``, ``get_mask_sizes``,
    ``is_sliding``, ``is_compileable`` and ``offloading``, all of which are
    provided here.

    Use as a context manager (or call ``attach_to``/``detach``) so the model's
    attention implementation is always restored:

        with CompactKVCache(n_layers=28, bits=4).attached(model) as cache:
            out = model(ids, past_key_values=cache, use_cache=True)
    """

    is_compileable = False
    offloading = False

    def __init__(
        self,
        n_layers: int,
        bits: int = 4,
        block_size: int = DEFAULT_BLOCK_SIZE,
        codec: str = "paper-outlier",
        gqa_factor: int = 1,
        seed: int = 0,
    ) -> None:
        self.layers = [
            CompactKVLayer(
                bits=bits,
                block_size=block_size,
                codec=codec,
                gqa_factor=gqa_factor,
                seed=seed,
            )
            for _ in range(n_layers)
        ]
        self.bits = bits
        self.block_size = block_size
        self.codec = codec
        self.gqa_factor = gqa_factor
        self._model = None
        self._prev_attn_impl: str | None = None
        self.capture_attn_weights = False
        self.captured_attn: dict[int, Tensor] = {}

    # -- cache contract --------------------------------------------------

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: Tensor, layer_idx: int = 0) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(cache_position, layer_idx)

    def get_max_cache_shape(self) -> int:
        return -1

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        layer = self.layers[layer_idx]
        return layer.keys, layer.values

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    # -- attachment ------------------------------------------------------

    def attach_to(self, model) -> "CompactKVCache":
        """
        Switch ``model`` onto the compact attention implementation.

        Raises if the switch does not take effect.  Silently falling back to
        standard attention would leave the model reading the zero-length
        tensors ``update()`` returns — i.e. attending to nothing — while this
        class still reported compact storage ratios.  A loud failure with an
        actionable message is the only safe behaviour.
        """
        _register_attention()
        self._prev_attn_impl = getattr(model.config, "_attn_implementation", None)
        try:
            model.set_attn_implementation(ATTN_IMPL_NAME)
        except Exception as exc:  # pragma: no cover - model-specific
            raise RuntimeError(
                f"{type(model).__name__} would not accept the compact attention "
                f"implementation ({exc}). Pass compact_cache=False to use the "
                f"float cache path instead."
            ) from exc

        active = getattr(model.config, "_attn_implementation", None)
        if active != ATTN_IMPL_NAME:
            raise RuntimeError(
                f"{type(model).__name__} did not switch to {ATTN_IMPL_NAME!r} "
                f"(still {active!r}). Refusing to continue: the model would "
                f"attend to an empty cache. Pass compact_cache=False instead."
            )
        _ACTIVE_CACHES[id(model)] = self
        self._model = model
        return self

    def detach(self) -> None:
        """Restore the model's previous attention implementation."""
        model, self._model = self._model, None
        if model is None:
            return
        _ACTIVE_CACHES.pop(id(model), None)
        if self._prev_attn_impl is not None:
            try:
                model.set_attn_implementation(self._prev_attn_impl)
            except Exception:  # pragma: no cover - best-effort restore
                model.config._attn_implementation = self._prev_attn_impl
        self._prev_attn_impl = None

    def attached(self, model):
        """Context manager form of :meth:`attach_to` — always detaches."""
        cache = self

        class _Attached:
            def __enter__(self):
                return cache.attach_to(model)

            def __exit__(self, *exc):
                cache.detach()
                return False

        return _Attached()

    # -- calibration and accounting --------------------------------------

    def calibrate_from(self, kvs: list[tuple[Tensor, Tensor]]) -> None:
        """Calibrate each layer on its own (k, v) prefill activations."""
        if len(kvs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} (k, v) pairs, got {len(kvs)}"
            )
        for layer, (k, v) in zip(self.layers, kvs):
            layer.calibrate(k, v)

    def ingest(self, kvs: list[tuple[Tensor, Tensor]], flush: bool = True) -> None:
        """
        Feed float prefill KV into the compact layers, block by block.

        Flushes the tail by default: prefill is a one-shot burst, so any
        remainder below ``block_size`` would otherwise stay float for the rest
        of the run.  Pass ``flush=False`` to ingest in several calls and seal
        the tail only after the last.
        """
        if len(kvs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} (k, v) pairs, got {len(kvs)}"
            )
        for layer, (k, v) in zip(self.layers, kvs):
            layer.update(k, v)
            if flush:
                layer.flush()

    def flush(self) -> None:
        """Seal every layer's pending window into a short block."""
        for layer in self.layers:
            layer.flush()

    def byte_breakdown(self):
        """Measured storage across all layers, payload vs sidecar."""
        from .cache_bytes import ByteBreakdown

        total = ByteBreakdown()
        for layer in self.layers:
            total = total + layer.byte_breakdown()
        return total

    @property
    def avg_bits_per_dim(self) -> float:
        """Measured bits per KV coordinate, side information included."""
        return self.byte_breakdown().bits_per_coord

    @property
    def compression_ratio(self) -> float:
        """Measured float16-equivalent bytes / measured actual bytes."""
        return self.byte_breakdown().compression_ratio

    def __repr__(self) -> str:
        return (
            f"CompactKVCache(layers={len(self.layers)}, bits={self.bits}, "
            f"block_size={self.block_size}, codec={self.codec!r}, "
            f"len={self.get_seq_length()})"
        )


# Maps id(model) -> active cache, so the registered attention function can find
# the blocks for the layer it is called on.  A module-level registry is needed
# because the attention interface hands us the attention *module*, not the cache.
_ACTIVE_CACHES: dict[int, CompactKVCache] = {}


# ---------------------------------------------------------------------------
# The registered attention function
# ---------------------------------------------------------------------------


def _resolve_scaling(module, query: Tensor, args: tuple, kwargs: dict) -> float:
    """
    Find the attention scaling factor.

    Model families disagree about how they pass it: Qwen2 supplies it
    positionally, GPT-2 supplies it not at all (it folds the scale into the
    module).  Checking every source in order, then falling back to the textbook
    ``1/sqrt(d)``, keeps one implementation working across families.
    """
    scaling = kwargs.get("scaling")
    if scaling is None and args:
        scaling = args[0]
    if scaling is None:
        scaling = getattr(module, "scaling", None)
    if scaling is None:
        scaling = float(query.shape[-1]) ** -0.5
    return scaling


def _repeat_kv(t: Tensor, n_rep: int) -> Tensor:
    """
    Expand KV heads to query heads for grouped-query attention.

    Applied per block, after dequantization, and never to the whole cache:
    expanding first would multiply resident floats by the GQA factor and undo
    the saving this cache exists to deliver.
    """
    if n_rep == 1:
        return t
    b, h, s, d = t.shape
    return t[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _causal_block_mask(
    scores: Tensor, q_abs_start: int, k_abs_start: int
) -> Tensor:
    """
    Apply causality to one block of scores using absolute positions.

    Needed because registering a custom attention implementation means
    transformers skips mask construction entirely (our key is absent from
    ``ALL_MASK_ATTENTION_FUNCTIONS``, so ``attention_mask`` arrives as None).
    Blocks that lie wholly in the past need no mask, so only the diagonal
    block pays for this.
    """
    q_len, k_len = scores.shape[-2], scores.shape[-1]
    if k_abs_start + k_len <= q_abs_start + 1:
        return scores  # entirely in the past: nothing to mask
    q_pos = torch.arange(
        q_abs_start, q_abs_start + q_len, device=scores.device
    ).view(-1, 1)
    k_pos = torch.arange(
        k_abs_start, k_abs_start + k_len, device=scores.device
    ).view(1, -1)
    return scores.masked_fill(
        (k_pos > q_pos).view(1, 1, q_len, k_len), float("-inf")
    )


def compact_attention_forward(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    *args,
    **kwargs,
):
    """
    Attention over a compact cache, reading compressed blocks directly.

    ``key``/``value`` arrive zero-length (the cache's ``update()`` returns
    nothing), so the real K/V come from the layer's stored blocks. Each block
    is dequantized, used, and dropped, so peak float memory is one block rather
    than the whole cache.
    """
    cache = _find_cache(module)
    if cache is None:
        raise RuntimeError(
            "compact attention was invoked without an attached CompactKVCache. "
            "Use `cache.attach_to(model)` (or the `attached()` context manager) "
            "so the cache is discoverable, and always detach afterwards."
        )

    layer_idx = getattr(module, "layer_idx", 0)
    layer = cache.layers[layer_idx]
    scaling = _resolve_scaling(module, query, args, kwargs)
    n_rep = getattr(module, "num_key_value_groups", 1)

    q_len = query.shape[-2]
    total_len = layer.get_seq_length()
    q_abs_start = total_len - q_len  # queries occupy the final q_len positions

    # Running softmax state, in float32 even when the model runs fp16: `l` is a
    # sum of exponentials over the whole context, which loses precision and
    # eventually overflows at half precision.
    q32 = query.to(torch.float32)
    b, h_q, _, head_dim = q32.shape
    running_max = torch.full(
        (b, h_q, q_len, 1), float("-inf"), device=q32.device, dtype=torch.float32
    )
    running_sum = torch.zeros_like(running_max)
    acc = torch.zeros(
        (b, h_q, q_len, head_dim), device=q32.device, dtype=torch.float32
    )

    captured: list[Tensor] | None = [] if cache.capture_attn_weights else None
    k_abs = 0
    saw_block = False

    for k_blk, v_blk in layer.iter_kv_blocks():
        saw_block = True
        blk_len = k_blk.shape[-2]
        k_exp = _repeat_kv(k_blk.to(torch.float32), n_rep)
        v_exp = _repeat_kv(v_blk.to(torch.float32), n_rep)

        scores = torch.matmul(q32, k_exp.transpose(-2, -1)) * scaling
        scores = _causal_block_mask(scores, q_abs_start, k_abs)
        if attention_mask is not None:
            # A provided mask carries padding information that self-built
            # causality cannot know; slice out this block's key range.
            span = attention_mask[..., k_abs : k_abs + blk_len]
            if span.shape[-1] == blk_len:
                scores = scores + span.to(scores.dtype)

        # Online (flash-style) softmax: rescale the running total to the new
        # maximum rather than materializing the full (B, H, Tq, Tk) matrix.
        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        probs = torch.exp(scores - new_max)
        rescale = torch.exp(running_max - new_max)
        # A block that is entirely masked gives -inf - -inf = nan; those rows
        # contribute nothing, so force their correction factor to zero.
        rescale = torch.where(
            torch.isinf(running_max), torch.zeros_like(rescale), rescale
        )
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        running_sum = running_sum * rescale + probs.sum(dim=-1, keepdim=True)
        acc = acc * rescale + torch.matmul(probs, v_exp)
        running_max = new_max

        if captured is not None:
            # Unnormalized, relative to the running max at this point; fixed up
            # after the loop once the final max and denominator are known.
            captured.append((probs, new_max))

        k_abs += blk_len
        del k_blk, v_blk, k_exp, v_exp, scores, probs

    if not saw_block:
        raise RuntimeError(
            f"layer {layer_idx} holds no KV; the cache was never populated"
        )

    # Fully-masked query rows have a zero denominator; clamp so they yield zeros
    # rather than NaN.
    safe_sum = running_sum.clamp(min=torch.finfo(torch.float32).tiny)
    attn_output = acc / safe_sum
    attn_output = attn_output.transpose(1, 2).contiguous().to(query.dtype)

    if captured is not None:
        weights = torch.cat(
            [p * torch.exp(m - running_max) for p, m in captured], dim=-1
        ) / safe_sum
        cache.captured_attn[layer_idx] = weights.detach()
        return attn_output, weights.to(query.dtype)
    return attn_output, None


def _find_cache(module) -> CompactKVCache | None:
    """Locate the cache attached to the model owning `module`."""
    if len(_ACTIVE_CACHES) == 1:
        return next(iter(_ACTIVE_CACHES.values()))
    # Multiple models attached: disambiguate by layer count.
    n_layers = getattr(module, "_kvq_n_layers", None)
    for cache in _ACTIVE_CACHES.values():
        if n_layers is None or len(cache.layers) == n_layers:
            return cache
    return None


_REGISTERED = False


def _register_attention() -> None:
    """Register the compact attention implementation (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[ATTN_IMPL_NAME] = compact_attention_forward

    # Also register a mask builder.  Without this, transformers short-circuits
    # mask creation for unknown attention implementations and hands us None,
    # which is fine for causality (we rebuild it) but loses padding
    # information for left-padded batches.
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

        ALL_MASK_ATTENTION_FUNCTIONS[ATTN_IMPL_NAME] = eager_mask
    except Exception:  # pragma: no cover - older/newer transformers layouts
        pass
    _REGISTERED = True
