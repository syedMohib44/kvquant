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

    The cache used for generation is the paper's Section 5 outlier-aware
    Lloyd-Max codec (KVQuantMSE for BOTH K and V).  MSE reconstructs faithfully
    per coordinate, which is what attention needs — the IP quantizer used for
    PPL scoring is an inner-product *estimator* and would garble generation, so
    we force ``k_quantizer_cls=KVQuantMSE`` here.  Each layer gets its own
    quantizer calibrated on that layer's own prefill KV (per-layer calibration,
    paper §Per-layer calibration).  avg_bits is the real calibrated average.

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
    # (paper §Per-layer calibration — pooling across layers mis-identifies the
    # per-layer outlier channels).  MSE-for-K makes reconstruction faithful for
    # generation; the paper §5.1 outlier config is outlier_bits=min(bits+1,4),
    # regular_bits=max(bits-1,1).
    kvc = []
    for k, v in kvs:
        k3 = k.reshape(-1, k.shape[-2], head_dim)
        v3 = v.reshape(-1, v.shape[-2], head_dim)
        q = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=n_outlier,
            outlier_bits=min(bits + 1, 4),
            regular_bits=max(bits - 1, 1),
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

    # Compress the prefill once with the paper codec.  Low-rank residual
    # correction (paper §3.4/§5.3) only helps below 4-bit — at 4-bit the residual
    # is small enough that a rank-r SVD fits numerical noise, so gate it off.
    eff_rank = correction_rank if (correction_rank > 0 and bits < 4) else 0
    past = quantize_model_cache(native_cache, kvc, correction_rank=eff_rank)
    return past, avg_bits, T_p, ids


def _build_offload(max_vram_tokens, warm_size, disk_dir, offload_codec="paper-outlier",
                   bits=3, device=None):
    """
    Build a KVCacheDiskOffload for tiered VRAM->RAM->SSD storage of the KV cache.

    Codecs:
      - "paper-outlier" (default): the paper's Section 5 outlier-aware Lloyd-Max,
        calibrated once on the prefill.  Best fidelity on real KV tensors
        (cosine ~0.999) because a few high-magnitude channels get extra bits.
      - "paper": plain rotate + Lloyd-Max, indices bit-packed to `bits` (2-4).
        Smallest SSD footprint but degrades on outlier-heavy KV.

    `device` is the model's device — staged (dequantized) tensors are placed there
    so offload works on CPU / MPS / any GPU, not just CUDA.

    Returns a manager that has NOT yet stored anything (caller decides when).
    """
    return KVCacheDiskOffload(
        max_vram_tokens=max_vram_tokens,
        warm_size=warm_size,
        disk_dir=disk_dir,
        codec=offload_codec,
        bits=bits,
        device=device,
    )


def _gpu_gc():
    """Collect Python garbage and empty the CUDA cache (bounds VRAM between steps)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    """Effective bits/dim after GQA compensation."""
    compression_ratio: float
    """Storage reduction vs float16  (16 / avg_bits_per_dim)."""
    model: str
    prompt: str


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
                            "paper-outlier" (default): paper Section 5 outlier-aware
                            Lloyd-Max, calibrated on the prefill — best fidelity on
                            real KV.  "paper": plain Lloyd-Max, bit-packed (smallest,
                            degrades on outlier-heavy KV).

    Returns:
        GenerateResult with .text, .bits, .avg_bits_per_dim, .compression_ratio.

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

    input_ids = _format_prompt(tok, prompt, raw, system=system)
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
        offload = _build_offload(max_vram_tokens, warm_size, disk_dir, offload_codec, bits,
                                 device=_model_device(mdl))
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
        # Older transformers honoured past_key_values with use_cache=False;
        # transformers 5.x ignores it, so we always use use_cache=True here.
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
        compression_ratio=16.0 / avg_bits,
        model=model,
        prompt=prompt,
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

    input_ids = _format_prompt(tok, prompt, raw, system=system)
    past, _, T_p, ids = _build_quantized_cache(
        mdl, tok, input_ids, bits, correction_rank, prefill_chunk_size,
        quantize=not offload_to_disk,
    )

    suppress_ids = _get_suppress_ids(tok)

    # Crop BEFORE storing (see generate() for why this ordering is essential).
    past_crop = crop_model_cache(past, T_p - 1)

    offload = None
    if offload_to_disk:
        offload = _build_offload(max_vram_tokens, warm_size, disk_dir, offload_codec, bits,
                                 device=_model_device(mdl))
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
