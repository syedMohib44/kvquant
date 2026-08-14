"""
High-level generate() and stream() API for kvquant.

    from kvquant import generate, stream

    # Full response
    out = generate("What is machine learning?", model="Qwen/Qwen2.5-1.5B-Instruct")
    print(out.text)
    print(f"{out.compression_ratio:.1f}x smaller than float16")

    # Streaming — print tokens as they arrive
    for token in stream("Explain transformers in simple terms"):
        print(token, end="", flush=True)

Supports any HuggingFace causal LM: instruct models, base models, hybrid
architectures (Qwen2.5, Llama-3, Phi-3, Mistral, Falcon, Gemma, Mamba …).
Works on CPU, single GPU, and multi-GPU via device_map="auto".

The user's own prompt context is used as calibration data — the KV vectors
from the prefill pass are exactly the data being quantized, so they are
the ideal and most accurate calibration source. No separate corpus needed.
"""

from __future__ import annotations

import gc
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Generator, Optional

import torch

from .kv_cache import (
    KVCacheQuantizer,
    kvs_from_cache,
    quantize_model_cache,
    crop_model_cache,
    KVCacheDiskOffload,
)
from .compact_cache import CompactKVCache, DEFAULT_BLOCK_SIZE
from .quantizer import KVQuantMSE


# ---------------------------------------------------------------------------
# Lazy model cache  (avoid reloading on repeated calls in a notebook / REPL)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, tuple] = {}

# Weight-placement modes that internally set device_map (via bitsandbytes or
# accelerate) and therefore OWN device placement — callers must NOT also call
# mdl.to(device) afterwards.  bitsandbytes-quantized models actively reject .to().
_MANAGED_PLACEMENT_MODES = frozenset({"4bit", "8bit", "offload"})


def _mode_manages_placement(weights: str | None) -> bool:
    """True if the weights mode places the model itself (skip mdl.to(device))."""
    return weights in _MANAGED_PLACEMENT_MODES


def _default_max_gpu_mem() -> str:
    """~90% of currently-free VRAM as a GiB string, e.g. '7GiB'.  Fallback '4GiB'."""
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        gib = max(1, int((free_bytes * 0.9) / (1024 ** 3)))
        return f"{gib}GiB"
    return "4GiB"


def _build_load_kwargs(
    weights: str | None,
    device_map: str | None,
    max_gpu_mem: str | None,
    max_cpu_mem: str | None,
    weights_disk_dir: str | None,
) -> dict:
    """
    Build the from_pretrained kwargs for the requested weight-placement mode.

    Modes:
      None / "full" — current behaviour: fp16/auto weights, optional device_map.
      "4bit"/"8bit" — bitsandbytes on-GPU quantization (fast, low VRAM).
      "offload"     — accelerate tiered placement GPU -> CPU RAM -> SSD (fits
                      anything, slow).
    """
    if weights in (None, "full"):
        load_kwargs: dict = {"torch_dtype": "auto"}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        return load_kwargs

    if weights in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401  (ensures the backend is importable)
        except ImportError as e:
            raise ImportError(
                f"weights={weights!r} needs bitsandbytes. Install it with:\n"
                '    pip install "kvquant-plus-plus[quant]"\n'
                "or:  pip install bitsandbytes"
            ) from e
        if weights == "4bit":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        return {
            "quantization_config": qcfg,
            "device_map": device_map or "auto",
        }

    if weights == "offload":
        try:
            import accelerate  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "weights='offload' needs accelerate. Install it with:\n"
                '    pip install "kvquant-plus-plus[quant]"\n'
                "or:  pip install accelerate"
            ) from e
        gpu_mem = max_gpu_mem or _default_max_gpu_mem()
        cpu_mem = max_cpu_mem or "12GiB"
        max_memory: dict = {"cpu": cpu_mem}
        if torch.cuda.is_available():
            max_memory[0] = gpu_mem
        disk_dir = weights_disk_dir or tempfile.mkdtemp(prefix="kv_weights_offload_")
        os.makedirs(disk_dir, exist_ok=True)
        return {
            "torch_dtype": "auto",
            "device_map": "auto",
            "max_memory": max_memory,
            "offload_folder": disk_dir,
            "offload_state_dict": True,
            "low_cpu_mem_usage": True,
        }

    raise ValueError(
        f"Unknown weights mode {weights!r}. "
        "Choose from: None/'full', '4bit', '8bit', 'offload'."
    )


def _load_model(
    model_name: str,
    device_map: str | None,
    weights: str | None = None,
    max_gpu_mem: str | None = None,
    max_cpu_mem: str | None = None,
    weights_disk_dir: str | None = None,
):
    """
    Load and cache a model+tokenizer. Reuses cached instance on repeat calls.

    ``weights`` selects how model weights are placed so large models fit on
    low-VRAM GPUs (see _build_load_kwargs for the modes).  The mode is part of
    the cache key so switching modes in one session reloads correctly.
    """
    cache_key = f"{model_name}::{device_map}::{weights}::{max_gpu_mem}::{max_cpu_mem}"
    if cache_key not in _MODEL_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        load_kwargs = _build_load_kwargs(
            weights, device_map, max_gpu_mem, max_cpu_mem, weights_disk_dir
        )
        mdl = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        mdl.eval()
        _MODEL_CACHE[cache_key] = (mdl, tok)
    return _MODEL_CACHE[cache_key]


def _model_dims(model):
    cfg = model.config
    n_heads  = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    n_kv     = getattr(cfg, "num_key_value_heads", n_heads)
    hidden   = getattr(cfg, "hidden_size",         getattr(cfg, "n_embd", None))
    head_dim = getattr(cfg, "head_dim",            hidden // n_heads)
    return n_heads, n_kv, head_dim


def _model_device(model) -> torch.device:
    """Return the device of the first parameter (works for single-GPU and CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _format_prompt(tok, prompt: str, raw: bool, system: Optional[str] = None) -> torch.Tensor:
    has_template = hasattr(tok, "apply_chat_template") and bool(tok.chat_template)
    if has_template and not raw:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if system and not raw:
        text = f"{system}\n\nQ: {prompt.rstrip('?').strip()}?\nA:"
    elif not raw:
        text = f"Q: {prompt.rstrip('?').strip()}?\nA:"
    else:
        text = prompt
    return tok(text, return_tensors="pt")["input_ids"]


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample next token from logits with temperature and nucleus (top-p) sampling."""
    if temperature <= 0 or temperature == 1.0 and top_p >= 1.0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / max(temperature, 1e-8)
    if top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Remove tokens after the cumulative probability exceeds top_p
        remove = cumulative - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        token = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, token)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def _apply_repetition_penalty(logits: torch.Tensor, seen: list[int], penalty: float):
    if penalty == 1.0 or not seen:
        return logits
    ids = torch.tensor(list(set(seen)), dtype=torch.long, device=logits.device)
    sc = logits[:, ids]
    logits[:, ids] = torch.where(sc > 0, sc / penalty, sc * penalty)
    return logits


def _get_suppress_ids(tok) -> list[int]:
    """
    Token IDs to mask to -inf at every generation step.

    Suppresses:
    - Leading newlines  (cosmetic — first token is almost always a newline otherwise)
    - <think> token    (prevents Qwen2.5/Qwen3/any updated instruct model from entering
                        thinking mode and consuming the entire token budget inside a
                        <think>...</think> block, leaving only a few tail tokens visible
                        after _clean_text strips the block)

    We only suppress <think> when it is a *single* special token in the vocabulary
    (i.e. tok.encode("<think>") returns exactly one ID).  That is true for Qwen3 and
    newer Qwen2.5 checkpoints.  If "<think>" encodes to multiple tokens (not a
    special token) we leave it alone — it is just ordinary text in that model.
    """
    ids: list[int] = []
    nl = tok.encode("\n", add_special_tokens=False)
    if nl:
        ids.append(nl[0])
    think = tok.encode("<think>", add_special_tokens=False)
    if len(think) == 1:
        ids.append(think[0])
    return ids


def _suppress(logits: torch.Tensor, suppress_ids: list[int]) -> torch.Tensor:
    if suppress_ids:
        logits[:, suppress_ids] = float("-inf")
    return logits


def _clean_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$",         "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def _chunked_prefill(mdl, ids, chunk_size: int):
    """
    Prefill the model in chunks of `chunk_size` tokens to avoid OOM on long
    documents.  Each chunk is fed with the previous chunk's KV cache as
    past_key_values, so attention over the full context is preserved.

    Returns the same object as a single full-context forward pass
    (past_key_values covering all T_p tokens).
    """
    T = ids.shape[1]
    past = None
    for start in range(0, T, chunk_size):
        chunk = ids[:, start : start + chunk_size]
        with torch.no_grad():
            out = mdl(chunk, past_key_values=past, use_cache=True)
        past = out.past_key_values
    return past


def _build_quantized_cache(
    mdl, tok, input_ids, bits, correction_rank,
    prefill_chunk_size: int = 512, quantize: bool = True,
):
    """
    Run prefill on input_ids, calibrate per-layer quantizers on the resulting
    KV vectors, compress the cache with the paper's method, and return
    (past, avg_bits, T_p, ids).

    Long prompts (contracts, documents) are prefilled in chunks of
    `prefill_chunk_size` tokens so the O(T²) attention matrix never exceeds
    VRAM — the full context is still captured in the KV cache.

    The cache used for generation is the outlier-aware Lloyd-Max codec (the
    paper's §3.1 MSE quantizer, split across the outlier/regular channel groups
    of its §4.3 aside), using KVQuantMSE for BOTH K and V.  MSE reconstructs
    faithfully per coordinate, which is what attention needs — the IP quantizer
    used for PPL scoring is an inner-product *estimator* (§3.2, Theorem 2:
    unbiased in ``<y, x>``, not per-coordinate) and would garble generation, so
    we force ``k_quantizer_cls=KVQuantMSE`` here.  The paper does not say which
    of its two quantizers to use for K versus V; this choice is ours, and
    ``test_mse_key_reconstructs_far_better_than_ip`` is what justifies it.

    Each layer gets its own quantizer calibrated on that layer's own prefill KV.
    That too is our decision — see the note in ``eval_ppl`` — since the paper is
    data-oblivious by design and specifies no calibration at all.  avg_bits is
    the real calibrated average.

    When ``quantize=False`` (the disk-offload path), the raw float prefill is
    returned uncompressed: KVCacheDiskOffload is then the SOLE compressor, so we
    avoid quantizing twice (lossy-on-lossy).  avg_bits is still reported so the
    caller can show the offload bit budget.
    """
    n_heads, n_kv_heads, head_dim = _model_dims(mdl)
    gqa      = n_heads // n_kv_heads
    n_outlier = max(4, head_dim // 4)

    device = _model_device(mdl)
    ids = input_ids.to(device)
    T_p = ids.shape[1]

    # Prefill — chunked to keep each attention matrix at chunk_size² instead of T²
    native_cache = _chunked_prefill(mdl, ids, prefill_chunk_size)
    kvs = kvs_from_cache(native_cache)

    # One KVCacheQuantizer PER LAYER, each calibrated on its own layer's KV
    # (pooling across layers mis-identifies the per-layer outlier channels).
    # MSE-for-K makes reconstruction faithful for generation.  The
    # outlier_bits=bits+1 / regular_bits=max(bits-1,1) schedule is our
    # generalisation of the single worked example in §4.3 — the paper gives one
    # configuration and no formula.  (Its stated example, 32@3 + 96@2 = "2.5"
    # bits, actually averages 2.25; see the note in outlier.py.)  Left implicit
    # rather than passed: KVCacheQuantizer derives exactly this, and passing a
    # capped copy is how the three schedules drifted apart in the first place.
    kvc = []
    for k, v in kvs:
        k3 = k.reshape(-1, k.shape[-2], head_dim)
        v3 = v.reshape(-1, v.shape[-2], head_dim)
        q = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=n_outlier,
            gqa_factor=gqa,
            k_quantizer_cls=KVQuantMSE,
        )
        q.calibrate(k3, v3)
        kvc.append(q)
    avg_bits = kvc[0].avg_bits if kvc else float(bits)

    if not quantize:
        # Offload path: leave the prefill in raw float; the offload codec is the
        # sole (single-pass) compressor.  See docstring.
        return native_cache, avg_bits, T_p, ids

    # Compress the prefill once with the paper codec.  The low-rank residual
    # correction is ours — the paper has no low-rank or SVD step anywhere.  (Its
    # only residual is the 1-bit QJL on r = x - Q_mse(x) in §3.2, which is a
    # different construction entirely.)  It only helps below 4-bit: at 4-bit the
    # residual is small enough that a rank-r SVD fits numerical noise, so gate
    # it off.
    eff_rank = correction_rank if (correction_rank > 0 and bits < 4) else 0
    past = quantize_model_cache(native_cache, kvc, correction_rank=eff_rank)
    return past, avg_bits, T_p, ids


def _build_compact_cache(
    mdl, input_ids, bits, prefill_chunk_size: int = 512,
    block_size: int = DEFAULT_BLOCK_SIZE, codec: str = "paper-outlier",
):
    """
    Prefill into a cache that stores compressed codes, and return it ready to
    generate from.

    This is the path that actually saves VRAM.  ``_build_quantized_cache``
    compresses the prefill and immediately decompresses it back to float, so it
    measures quantization's *quality* cost while banking none of its memory
    benefit — and it only ever touches the prefill, because the cache the model
    mutates during generation is a plain float ``DynamicCache``.  Here the codes
    are the storage of record for prefill and every generated token alike.

    Sequencing matters and is not interchangeable:

    1. Prefill ``ids[:, :-1]`` into an ordinary float cache.  One token short so
       the caller can feed the final token itself and get first-token logits at
       the right position — the same invariant ``crop_model_cache`` enforces for
       the float path, obtained here without a crop, since truncating inside an
       already-compressed block would mean re-encoding it and breaking
       compress-exactly-once.
    2. Calibrate each layer on *that layer's own* prefill KV.  Outlier channels
       are layer-specific; pooling across layers mis-identifies them.  (Our
       finding — the paper is data-oblivious and prescribes no calibration.)
    3. Ingest the float KV block by block, then drop the float cache.

    Calibration must precede ingest: a layer that compresses before it is
    calibrated falls back to fitting on its first block alone, which is a
    smaller and less representative sample than the full prefill.

    Returns ``(cache, avg_bits, T_p, ids)`` where ``avg_bits`` is *measured*
    from what is stored — sidecar norms included — not the nominal bit-width.
    """
    device = _model_device(mdl)
    ids = input_ids.to(device)
    T_p = ids.shape[1]

    n_heads, n_kv_heads, _ = _model_dims(mdl)

    # Prefill one token short (see docstring) in a float cache we then discard.
    float_cache = _chunked_prefill(mdl, ids[:, : T_p - 1], prefill_chunk_size)
    kvs = kvs_from_cache(float_cache)
    if not kvs:
        raise RuntimeError(
            "prefill produced no KV tensors; the compact cache cannot be built. "
            "Pass compact_cache=False for this model."
        )

    cache = CompactKVCache(
        n_layers=len(kvs),
        bits=bits,
        block_size=block_size,
        codec=codec,
        gqa_factor=n_heads // n_kv_heads,
    )
    cache.calibrate_from(kvs)
    cache.ingest(kvs)

    del kvs, float_cache
    _gpu_gc()

    return cache, cache.avg_bits_per_dim, T_p, ids


def _build_offload(max_vram_tokens, warm_size, disk_dir, offload_codec="paper-outlier",
                   bits=3, device=None, gqa_factor=1):
    """
    Build a KVCacheDiskOffload for tiered VRAM->RAM->SSD storage of the KV cache.

    Codecs:
      - "paper-outlier" (default): outlier-aware Lloyd-Max (paper §3.1 quantizer,
        §4.3 channel split), calibrated once on the prefill.  Best fidelity on
        real KV tensors
        (cosine ~0.999) because a few high-magnitude channels get extra bits.
      - "paper": plain rotate + Lloyd-Max, indices bit-packed to `bits` (2-4).
        Smallest SSD footprint but degrades on outlier-heavy KV.

    `device` is the model's device — staged (dequantized) tensors are placed there
    so offload works on CPU / MPS / any GPU, not just CUDA.

    `gqa_factor` (n_heads / n_kv_heads) bumps the codec's effective bit-width to
    offset GQA error amplification — without it a grouped-query model (Qwen2.5-7B,
    g=7) under-quantizes and generates gibberish.

    Returns a manager that has NOT yet stored anything (caller decides when).
    """
    return KVCacheDiskOffload(
        max_vram_tokens=max_vram_tokens,
        warm_size=warm_size,
        disk_dir=disk_dir,
        codec=offload_codec,
        bits=bits,
        device=device,
        gqa_factor=gqa_factor,
    )


def _gpu_gc():
    """Collect Python garbage and empty the CUDA cache (bounds VRAM between steps)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_cache_mode(compact_cache: bool, offload_to_disk: bool) -> None:
    """
    Reject the one combination that would compress twice.

    Both paths are lossy codecs over the same tensors, and the disk tier would
    be handed an already-dequantized reconstruction to re-encode.  Lloyd-Max
    error compounds under re-encoding — measured in PAPER.md §3.3, where a
    4-bit value demoted to 3 bits and promoted back lands at MSE 0.0492, worse
    than the 3-bit state it came from.  Choosing silently for the caller would
    hide that; raising names the tradeoff instead.
    """
    if compact_cache and offload_to_disk:
        raise ValueError(
            "compact_cache=True and offload_to_disk=True cannot be combined: "
            "both compress the same KV, and running one over the other's "
            "output compounds quantization error. Choose one — "
            "compact_cache=True keeps compressed codes in VRAM (fast, "
            "~4-5x smaller), offload_to_disk=True spills them to SSD (slower, "
            "for contexts that exceed RAM). Pass compact_cache=False to use "
            "the offload path."
        )


def _compact_decode(
    mdl, tok, input_ids, bits, max_new_tokens, temperature, top_p,
    repetition_penalty, prefill_chunk_size, block_size,
):
    """
    Run the whole compact-cache generation loop, yielding one token id tensor
    at a time and finally the cache itself.

    A generator so ``generate()`` and ``stream()`` share one implementation
    rather than two copies that can drift apart — the float path's duplication
    is exactly how its offload and in-memory branches came to differ in their
    ``_gpu_gc`` throttling.  ``stream()`` decodes each yielded token as it
    arrives; ``generate()`` drains the generator and joins.

    The model stays attached to the compact attention implementation for the
    entire loop.  ``attached()`` is the single mechanism that restores it — on
    normal exit, on an exception, and on ``GeneratorExit`` when a caller
    abandons a partly-consumed stream.  Deliberately not belt-and-braces with a
    second ``finally: cache.detach()``: a redundant restore makes the primary
    one untestable, since breaking either alone leaves the model correctly
    restored by the other.
    """
    cache, _, _T_p, ids = _build_compact_cache(
        mdl, input_ids, bits, prefill_chunk_size, block_size
    )
    suppress_ids = _get_suppress_ids(tok)

    with cache.attached(mdl):
        # The cache already holds positions 0..T_p-2 (prefilled one short), so
        # feeding the final prompt token puts it at the right position.
        with torch.no_grad():
            out = mdl(ids[:, -1:], past_key_values=cache, use_cache=True)
        logits = _suppress(out.logits[:, -1, :], suppress_ids)

        next_tok = _sample_next(logits, temperature, top_p)
        seen_ids = ids[0].tolist()
        seen_ids.append(next_tok.item())
        yield next_tok

        for step_i in range(max_new_tokens - 1):
            if next_tok.item() == tok.eos_token_id:
                break
            with torch.no_grad():
                step = mdl(next_tok, past_key_values=cache, use_cache=True)
            logits = _suppress(step.logits[:, -1, :], suppress_ids)
            logits = _apply_repetition_penalty(
                logits, seen_ids, repetition_penalty
            )
            next_tok = _sample_next(logits, temperature, top_p)
            seen_ids.append(next_tok.item())
            yield next_tok
            if step_i % 64 == 63:
                _gpu_gc()

    # Seal the decode tail before anyone measures.  Generated tokens land in
    # the same pending window as prefill, so a run that stops mid-block leaves
    # up to block_size-1 positions in float — which makes the achieved ratio
    # sawtooth with max_new_tokens rather than describe the codec: measured
    # 3.56x at 128 generated tokens but 1.17x at 127, on the same prompt.
    # Generation is over here, so there is nothing to lose by compressing the
    # remainder, and each token is still compressed exactly once.
    cache.flush()

    # Yielded last so the caller can report measured storage.  Everything
    # before this is a token; this is the cache.
    yield cache


def _stream_compact(
    mdl, tok, input_ids, bits, max_new_tokens, temperature, top_p,
    repetition_penalty, prefill_chunk_size, block_size,
):
    """
    Yield text fragments from the compact decode loop.

    Re-decodes the whole token list each step and yields only the tail, rather
    than decoding tokens individually: byte-level BPE tokenizers split
    multi-byte characters across tokens, so a per-token decode emits U+FFFD
    replacement characters mid-emoji.  Fragments still containing U+FFFD are
    held back until the next token completes the sequence.
    """
    generated: list[torch.Tensor] = []
    prev_len = 0
    for item in _compact_decode(
        mdl, tok, input_ids, bits, max_new_tokens, temperature, top_p,
        repetition_penalty, prefill_chunk_size, block_size,
    ):
        if isinstance(item, CompactKVCache):
            continue  # not `break`: let the inner generator run its own finally
        generated.append(item)
        text = tok.decode(
            torch.cat(generated, dim=1)[0], skip_special_tokens=True
        )
        fragment = text[prev_len:].replace("�", "")
        if fragment:
            yield fragment
        prev_len = len(text)


def _generate_compact(
    mdl, tok, model, prompt, input_ids, bits, max_new_tokens, temperature,
    top_p, repetition_penalty, prefill_chunk_size, block_size,
) -> "GenerateResult":
    """Drain the compact decode loop into a GenerateResult with measured bytes."""
    generated: list[torch.Tensor] = []
    cache = None
    for item in _compact_decode(
        mdl, tok, input_ids, bits, max_new_tokens, temperature, top_p,
        repetition_penalty, prefill_chunk_size, block_size,
    ):
        if isinstance(item, CompactKVCache):
            cache = item
        else:
            generated.append(item)

    text = _clean_text(
        tok.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    )
    breakdown = cache.byte_breakdown()
    return GenerateResult(
        text=text,
        bits=bits,
        avg_bits_per_dim=breakdown.bits_per_coord,
        compression_ratio=breakdown.compression_ratio,
        model=model,
        prompt=prompt,
        cache_bytes=breakdown.total,
        sidecar_bytes=breakdown.sidecar_bytes,
        measured=True,
    )


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class GenerateResult:
    """Return value from generate()."""
    text: str
    """Generated text (prompt stripped, think-blocks cleaned)."""
    bits: int
    """Nominal bit-width requested."""
    avg_bits_per_dim: float
    """
    Bits per KV coordinate.

    On the compact path this is *measured* from the bytes actually held at the
    end of generation, side information included — the per-vector float32 norms
    the codec stores alongside the packed indices are real memory and are
    counted here.  It is therefore higher than the nominal bit-width, and that
    gap is the point: the nominal figure describes an aspiration, this one
    describes an allocation.
    """
    compression_ratio: float
    """
    float16-equivalent bytes divided by bytes actually stored.

    Compact path: measured.  Float path: the nominal ``16 / avg_bits``, which
    describes only the compressed prefill — the generated tokens stay
    full-precision there, so it overstates the saving.
    """
    model: str
    prompt: str
    cache_bytes: int = 0
    """Bytes held by the KV cache at the end of generation (0 if unmeasured)."""
    sidecar_bytes: int = 0
    """Of ``cache_bytes``, how many are side information rather than payload."""
    measured: bool = False
    """
    True when the two ratio fields come from counting bytes rather than from
    the nominal bit-width.  False on the float and offload paths.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    bits: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.3,
    correction_rank: int = 0,
    raw: bool = False,
    system: Optional[str] = None,
    prefill_chunk_size: int = 512,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    weights: Optional[str] = None,
    max_gpu_mem: Optional[str] = None,
    max_cpu_mem: Optional[str] = None,
    weights_disk_dir: Optional[str] = None,
    offload_to_disk: bool = False,
    max_vram_tokens: int = 512,
    warm_size: int = 16,
    disk_dir: Optional[str] = None,
    offload_codec: str = "paper-outlier",
    compact_cache: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> GenerateResult:
    """
    Generate text from a prompt using a quantized KV cache.

    Works with any HuggingFace causal LM — instruct, base, hybrid (Qwen, Llama,
    Phi, Mistral, Falcon, Gemma, Mamba …).  The model is downloaded on first call
    and cached in memory for subsequent calls.

    Args:
        prompt:             Input text — a question, instruction, or long document.
                            The prompt itself is used as calibration data: the KV
                            vectors produced from this exact context are the most
                            accurate calibration source, so no separate corpus is
                            needed.
        model:              HuggingFace model ID.
        bits:               KV cache bit-width (2 = most compressed, 4 = best quality).
                            3 is the recommended default: ~5x compression.
        max_new_tokens:     Maximum tokens to generate.
        temperature:        Sampling temperature.  0 = greedy (deterministic).
                            0.7 is a good starting point for creative tasks.
        top_p:              Nucleus sampling probability.  1.0 = disabled.
        repetition_penalty: Values > 1 reduce repetition loops.
        correction_rank:    Low-rank SVD correction of the quantization residual
                            (0 = disabled, 4 = recommended for 2/3-bit).
        raw:                Tokenize the prompt as plain text, skipping the model's
                            chat template.  Useful for sentence-completion prompts.
        system:             Optional system prompt.  Prepended before the user prompt
                            in the chat template (e.g. "You are a helpful assistant.").
                            Its KV vectors are quantized together with the user prompt
                            as part of the same prefill pass — no extra compute needed.
                            Ignored when raw=True.
        prefill_chunk_size: Tokens processed per prefill step (default 512).
                            Prefill attention is O(T²) — a long document on a
                            consumer GPU (8 GB) will OOM with a single full-context
                            forward pass.  Chunked prefill keeps each step at
                            chunk_size² while the full context is still captured
                            in the KV cache.  Lower values (256) use less VRAM;
                            higher values (1024+) are faster on GPUs with more memory.
        device:             "cuda", "cpu", or None (auto-detect).
        device_map:         HuggingFace device_map for multi-GPU or CPU offload,
                            e.g. "auto", "balanced", "sequential".  When set,
                            `device` is ignored.
        weights:            How to place model WEIGHTS so large models fit on
                            low-VRAM GPUs.  This is the fix for "OOM loading a 7B
                            model on an 8 GB card" — it controls the weights, not
                            the KV cache.  Options:
                              None / "full" — default, fp16/auto weights (today's
                                              behaviour, unchanged).
                              "4bit"        — 4-bit NF4 weights (bitsandbytes).
                                              A 7B fits in ~5 GB VRAM and stays
                                              fast.  Recommended for 8 GB GPUs.
                              "8bit"        — 8-bit weights (bitsandbytes).
                                              A 7B needs ~8 GB VRAM.
                              "offload"     — accelerate tiered placement
                                              GPU -> CPU RAM -> SSD.  Fits any
                                              model size but is slow (SSD-bound).
                            "4bit"/"8bit" need bitsandbytes; "offload" needs
                            accelerate.  Install both with
                            `pip install "kvquant-plus-plus[quant]"`.
        max_gpu_mem:        VRAM cap for weights="offload" (e.g. "6GiB").
                            Default: ~90% of free VRAM.
        max_cpu_mem:        CPU-RAM cap for weights="offload" (e.g. "12GiB").
        weights_disk_dir:   SSD folder for offloaded weight shards
                            (weights="offload" only).  Default: an auto-cleaned
                            temp directory.
        offload_to_disk:    Spill the KV *cache* across VRAM->CPU RAM->SSD so long
                            contexts do not OOM (separate from weights= offload).
        max_vram_tokens:    Token positions kept dequantized (hot) in VRAM.
        warm_size:          Compressed layer entries kept in CPU RAM before disk.
        disk_dir:           SSD folder for spilled cache (auto-cleaned temp if None).
        offload_codec:      Compression for spilled cache.
                            "paper-outlier" (default): outlier-aware Lloyd-Max
                            (paper §3.1 + §4.3), calibrated on the prefill — best fidelity on
                            real KV.  "paper": plain Lloyd-Max, bit-packed (smallest,
                            degrades on outlier-heavy KV).
        compact_cache:      Keep the KV cache as compressed codes in memory
                            (default True).  This is what makes the compression
                            reduce VRAM rather than only measure quality: codes
                            are the storage of record, decompressed one block at
                            a time inside attention and dropped immediately, for
                            both the prefill and every generated token.
                            compact_cache=False restores the previous behaviour,
                            where the prefill is compressed and immediately
                            expanded back to float and generated tokens are never
                            compressed at all — useful for A/B comparison, but it
                            saves no memory.  Mutually exclusive with
                            offload_to_disk.
        block_size:         Tokens per compressed block (default 128).  Larger
                            blocks amortise the per-block sidecar over more
                            tokens (slightly better ratio) at the cost of a
                            larger transient during attention; smaller blocks do
                            the reverse.

    Returns:
        GenerateResult with .text, .bits, .avg_bits_per_dim, .compression_ratio.
        On the compact path .measured is True and .avg_bits_per_dim /
        .compression_ratio are counted from the bytes actually held, side
        information included, rather than derived from the nominal bit-width.

    Example::

        from kvquant import generate

        # Simple Q&A
        out = generate("What is the capital of France?")
        print(out.text)

        # Long document — context is its own calibration data
        out = generate(
            document + "\\n\\nSummarise the above in three bullet points:",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            bits=3,
        )
        print(out.text)

        # Creative writing with sampling
        out = generate("Once upon a time", bits=4, temperature=0.8, top_p=0.95)
        print(out.text)

        # Large model across multiple GPUs
        out = generate("Explain quantum computing",
                       model="meta-llama/Llama-3.1-8B-Instruct",
                       bits=3, device_map="auto")
        print(out.text)
        print(f"{out.compression_ratio:.1f}x vs float16")

        # 7B model on an 8 GB GPU without OOM (4-bit weights, fast)
        out = generate("What is machine learning?",
                       model="Qwen/Qwen2.5-7B-Instruct", weights="4bit")
        print(out.text)
    """
    # Resolve device
    if device_map is not None:
        _device = None     # HF handles placement
        _map    = device_map
    else:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _map    = None

    mdl, tok = _load_model(
        model, _map, weights, max_gpu_mem, max_cpu_mem, weights_disk_dir
    )
    # Skip .to() when the weights mode owns placement (bitsandbytes / offload set
    # device_map internally, and bnb-quantized models reject .to()).
    if _device and _map is None and not _mode_manages_placement(weights):
        mdl = mdl.to(_device)

    _check_cache_mode(compact_cache, offload_to_disk)

    input_ids = _format_prompt(tok, prompt, raw, system=system)

    if compact_cache:
        return _generate_compact(
            mdl, tok, model, prompt, input_ids, bits, max_new_tokens,
            temperature, top_p, repetition_penalty, prefill_chunk_size,
            block_size,
        )

    past, avg_bits, T_p, ids = _build_quantized_cache(
        mdl, tok, input_ids, bits, correction_rank, prefill_chunk_size,
        quantize=not offload_to_disk,
    )

    suppress_ids = _get_suppress_ids(tok)

    # Crop to T_p-1 so the last prefill token is processed at the correct position.
    # This MUST happen before any offload store — otherwise the last token is fed
    # twice (once already in the cache, once via ids[:, -1:]) and every position
    # misaligns, producing garbage output.
    past_crop = crop_model_cache(past, T_p - 1)

    offload = None
    if offload_to_disk:
        # Tiered VRAM->RAM->SSD storage of the KV cache.  Store the CROPPED cache,
        # then stage it back for the first-token forward — identical positions to
        # the in-memory path, just spilled to disk between steps.
        _nh, _nkv, _ = _model_dims(mdl)
        offload = _build_offload(max_vram_tokens, warm_size, disk_dir, offload_codec, bits,
                                 device=_model_device(mdl), gqa_factor=_nh // _nkv)
        del past
        offload.store(past_crop)
        del past_crop
        _gpu_gc()
        past_first = offload.stage_for_forward()
        with torch.no_grad():
            first_out = mdl(ids[:, -1:], past_key_values=past_first, use_cache=True)
        first_logits = first_out.logits[:, -1, :]
        offload.replace(first_out.past_key_values)
        del past_first
        _gpu_gc()
    else:
        # In-memory path: keep the updated cache (use_cache=True) for the loop.
        # This loop needs the returned cache to append to, so use_cache=True is
        # required here regardless of how any version treats use_cache=False.
        # (Verified on transformers 4.57.6: past_key_values IS still attended to
        # with use_cache=False — eval_ppl.py relies on that, and
        # test_use_cache_false_still_attends_to_past pins it so an upgrade that
        # changes the behaviour fails loudly instead of silently scoring garbage.)
        with torch.no_grad():
            first_out = mdl(ids[:, -1:], past_key_values=past_crop, use_cache=True)
        first_logits = first_out.logits[:, -1, :]
        past = first_out.past_key_values   # T_p KVs — quantized prefill + exact last token

    first_logits = _suppress(first_logits, suppress_ids)

    generated = [_sample_next(first_logits, temperature, top_p)]
    seen_ids  = ids[0].tolist()

    for step_i in range(max_new_tokens - 1):
        if offload is not None:
            past_step = offload.stage_for_forward()
            with torch.no_grad():
                step = mdl(generated[-1], past_key_values=past_step, use_cache=True)
            offload.replace(step.past_key_values)
            del past_step
            # empty_cache()/gc.collect() are expensive (CUDA sync + heap walk);
            # calling them every token dominates step time.  Throttle to every
            # 32 tokens — the append-only offload already frees floats via `del`.
            if step_i % 32 == 31:
                _gpu_gc()
        else:
            with torch.no_grad():
                step = mdl(generated[-1], past_key_values=past, use_cache=True)
            past = step.past_key_values
        logits = step.logits[:, -1, :]
        logits = _suppress(logits, suppress_ids)
        logits = _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
        next_tok = _sample_next(logits, temperature, top_p)
        tok_id = next_tok.item()          # one D2H sync, reused below
        seen_ids.append(tok_id)
        generated.append(next_tok)
        if tok_id == tok.eos_token_id:
            break

    if offload is not None:
        offload.close()

    gen_ids = torch.cat(generated, dim=1)
    text = _clean_text(tok.decode(gen_ids[0], skip_special_tokens=True))

    return GenerateResult(
        text=text,
        bits=bits,
        avg_bits_per_dim=avg_bits,
        # Nominal, and deliberately left so: on this path the compressed prefill
        # is expanded back to float and generated tokens are never compressed,
        # so there is no compact byte count to report.  `measured=False` marks
        # the difference rather than dressing this figure up as a measurement.
        compression_ratio=16.0 / avg_bits,
        model=model,
        prompt=prompt,
        measured=False,
    )


def stream(
    prompt: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    bits: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.3,
    correction_rank: int = 0,
    raw: bool = False,
    system: Optional[str] = None,
    prefill_chunk_size: int = 512,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    weights: Optional[str] = None,
    max_gpu_mem: Optional[str] = None,
    max_cpu_mem: Optional[str] = None,
    weights_disk_dir: Optional[str] = None,
    offload_to_disk: bool = False,
    max_vram_tokens: int = 512,
    warm_size: int = 16,
    disk_dir: Optional[str] = None,
    offload_codec: str = "paper-outlier",
    compact_cache: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Generator[str, None, None]:
    """
    Stream generated text token-by-token from a quantized KV cache.

    Same parameters as generate() — including ``weights`` ("4bit"/"8bit"/
    "offload") to fit large models on low-VRAM GPUs without OOM.  Yields decoded
    text fragments as each token is produced — use this when you want to display
    output incrementally rather than waiting for the full response.

    Example::

        from kvquant import stream

        for token in stream("Explain quantum entanglement simply"):
            print(token, end="", flush=True)
        print()

        # With a specific model and sampling
        for token in stream("Write a short poem about rain",
                            model="Qwen/Qwen2.5-1.5B-Instruct",
                            bits=3, temperature=0.8, top_p=0.95):
            print(token, end="", flush=True)
    """
    if device_map is not None:
        _device = None
        _map    = device_map
    else:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _map    = None

    mdl, tok = _load_model(
        model, _map, weights, max_gpu_mem, max_cpu_mem, weights_disk_dir
    )
    if _device and _map is None and not _mode_manages_placement(weights):
        mdl = mdl.to(_device)

    _check_cache_mode(compact_cache, offload_to_disk)

    input_ids = _format_prompt(tok, prompt, raw, system=system)

    if compact_cache:
        yield from _stream_compact(
            mdl, tok, input_ids, bits, max_new_tokens, temperature, top_p,
            repetition_penalty, prefill_chunk_size, block_size,
        )
        return

    past, _, T_p, ids = _build_quantized_cache(
        mdl, tok, input_ids, bits, correction_rank, prefill_chunk_size,
        quantize=not offload_to_disk,
    )

    suppress_ids = _get_suppress_ids(tok)

    # Crop BEFORE storing (see generate() for why this ordering is essential).
    past_crop = crop_model_cache(past, T_p - 1)

    offload = None
    if offload_to_disk:
        _nh, _nkv, _ = _model_dims(mdl)
        offload = _build_offload(max_vram_tokens, warm_size, disk_dir, offload_codec, bits,
                                 device=_model_device(mdl), gqa_factor=_nh // _nkv)
        del past
        offload.store(past_crop)
        del past_crop
        _gpu_gc()
        past_first = offload.stage_for_forward()
        with torch.no_grad():
            first_out = mdl(ids[:, -1:], past_key_values=past_first, use_cache=True)
        first_logits = first_out.logits[:, -1, :]
        offload.replace(first_out.past_key_values)
        del past_first
        _gpu_gc()
    else:
        with torch.no_grad():
            first_out = mdl(ids[:, -1:], past_key_values=past_crop, use_cache=True)
        first_logits = first_out.logits[:, -1, :]
        past = first_out.past_key_values   # T_p KVs — quantized prefill + exact last token

    first_logits = _suppress(first_logits, suppress_ids)

    generated  = [_sample_next(first_logits, temperature, top_p)]
    seen_ids   = ids[0].tolist()
    prev_len   = 0     # track previously decoded text length to yield only the delta

    for _ in range(max_new_tokens - 1):
        # Decode all tokens so far; yield only the new fragment.
        # Skip any fragment that contains \ufffd — these are incomplete
        # multi-byte sequences from byte-level BPE tokenizers that will
        # be completed by the next token.
        current_text = tok.decode(
            torch.cat(generated, dim=1)[0], skip_special_tokens=True
        )
        fragment = current_text[prev_len:].replace("\ufffd", "")
        if fragment:
            yield fragment
        prev_len = len(current_text)

        if offload is not None:
            past_step = offload.stage_for_forward()
            with torch.no_grad():
                step = mdl(generated[-1], past_key_values=past_step, use_cache=True)
            offload.replace(step.past_key_values)
            del past_step
            _gpu_gc()
        else:
            with torch.no_grad():
                step = mdl(generated[-1], past_key_values=past, use_cache=True)
            past = step.past_key_values
        logits = step.logits[:, -1, :]
        logits = _suppress(logits, suppress_ids)
        logits   = _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
        next_tok = _sample_next(logits, temperature, top_p)
        seen_ids.append(next_tok.item())
        generated.append(next_tok)
        if next_tok.item() == tok.eos_token_id:
            break

    if offload is not None:
        offload.close()

    # Yield any remaining text after the last token, skipping incomplete sequences.
    # Use the same raw decode as the loop (no _clean_text) so whitespace is consistent.
    final_text = tok.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    final_fragment = final_text[prev_len:].replace("\ufffd", "")
    if final_fragment:
        yield final_fragment
