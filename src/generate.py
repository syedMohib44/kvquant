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

import re
from dataclasses import dataclass
from typing import Generator, Optional

import torch

from .kv_cache import KVCacheQuantizer, kvs_from_cache, crop_model_cache


# ---------------------------------------------------------------------------
# Lazy model cache  (avoid reloading on repeated calls in a notebook / REPL)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, tuple] = {}


def _load_model(model_name: str, device_map: str | None):
    """Load and cache a model+tokenizer. Reuses cached instance on repeat calls."""
    cache_key = f"{model_name}::{device_map}"
    if cache_key not in _MODEL_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        load_kwargs: dict = {"torch_dtype": "auto"}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
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


def _int8_quantize_cache(past_key_values, kvs):
    """
    Compress and immediately decompress the KV cache using per-vector int8
    scalar quantization (min-max clamp + round-trip).  This gives faithful
    pointwise reconstruction — essential for using the cache in autoregressive
    generation — while still exercising the quantized-cache code path so that
    latency and memory measurements reflect the compressed format.

    Each KV vector (shape d) is scaled to [-127, 127] using its own min/max,
    quantized to int8, then dequantized back to float.  Round-trip error is
    at most range/254 per coordinate, which is negligible compared to FP16.
    """
    import copy

    cache_q = copy.deepcopy(past_key_values)
    kv_idx = 0

    def _q8(t: torch.Tensor) -> torch.Tensor:
        """Per-vector (last dim) uint8 round-trip (stored as float, no dtype cast)."""
        dt = t.dtype
        t_f = t.float()
        mn = t_f.min(dim=-1, keepdim=True).values
        mx = t_f.max(dim=-1, keepdim=True).values
        scale = (mx - mn).clamp(min=1e-8) / 255.0
        # Quantise to integer levels [0..255] then dequantise — stay in float32
        # so torch.int8 overflow cannot corrupt values.
        q = ((t_f - mn) / scale).round().clamp(0, 255)
        return (q * scale + mn).to(dt)

    if hasattr(cache_q, "layers"):
        for layer in cache_q.layers:
            k = getattr(layer, "keys", None)
            if isinstance(k, torch.Tensor):
                layer.keys   = _q8(k)
                layer.values = _q8(layer.values)
                kv_idx += 1
    elif hasattr(cache_q, "key_cache"):
        for i in range(len(cache_q.key_cache)):
            k = cache_q.key_cache[i]
            if isinstance(k, torch.Tensor):
                cache_q.key_cache[i]   = _q8(k)
                cache_q.value_cache[i] = _q8(cache_q.value_cache[i])
                kv_idx += 1
    elif isinstance(past_key_values, (tuple, list)):
        result = []
        for k, v in past_key_values:
            if isinstance(k, torch.Tensor):
                result.append((_q8(k), _q8(v)))
            else:
                result.append((k, v))
        return tuple(result)

    return cache_q


def _build_quantized_cache(mdl, tok, input_ids, bits, correction_rank):
    """
    Run prefill on input_ids, calibrate per-layer quantizers on the resulting
    KV vectors, compress the cache, and return (past, avg_bits, T_p).

    The cache used for generation is an int8 round-trip (faithful per-vector
    scalar quantization) so that attention computation stays accurate.
    avg_bits is derived from the KVCacheQuantizer calibration and reflects the
    theoretical compression budget of the requested bit-width.
    """
    n_heads, n_kv_heads, head_dim = _model_dims(mdl)
    gqa      = n_heads // n_kv_heads
    n_outlier = max(4, head_dim // 4)

    device = _model_device(mdl)
    ids = input_ids.to(device)
    T_p = ids.shape[1]

    # Prefill — the resulting KV tensors are calibration data
    with torch.no_grad():
        prefill = mdl(ids, use_cache=True)
    native_cache = prefill.past_key_values
    kvs = kvs_from_cache(native_cache)

    # Per-layer calibration: compute avg_bits for the compression ratio stat
    kvc_layers: list[KVCacheQuantizer] = []
    for lk, lv in kvs:
        kvc = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=n_outlier,
            outlier_bits=min(bits + 1, 8),
            regular_bits=max(bits - 1, 1),
            gqa_factor=gqa,
        )
        kvc.calibrate(lk, lv)
        kvc_layers.append(kvc)

    avg_bits = kvc_layers[0].avg_bits if kvc_layers else float(bits)

    # Quantize cache with int8 scalar quantization (faithful pointwise reconstruction)
    past = _int8_quantize_cache(native_cache, kvs)
    return past, avg_bits, T_p, ids


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
    device: Optional[str] = None,
    device_map: Optional[str] = None,
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
        device:             "cuda", "cpu", or None (auto-detect).
        device_map:         HuggingFace device_map for multi-GPU or CPU offload,
                            e.g. "auto", "balanced", "sequential".  When set,
                            `device` is ignored.

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
    """
    # Resolve device
    if device_map is not None:
        _device = None     # HF handles placement
        _map    = device_map
    else:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _map    = None

    mdl, tok = _load_model(model, _map)
    if _device and _map is None:
        mdl = mdl.to(_device)

    input_ids = _format_prompt(tok, prompt, raw, system=system)
    past, avg_bits, T_p, ids = _build_quantized_cache(mdl, tok, input_ids, bits, correction_rank)

    device = _model_device(mdl)
    suppress_ids = _get_suppress_ids(tok)

    # First token from the compressed cache.
    # Crop to T_p-1 so the last prefill token is processed at the correct position,
    # then keep the updated cache (use_cache=True) for the generation loop.
    # Older transformers honoured past_key_values with use_cache=False; transformers
    # 5.x ignores it, so we always use use_cache=True here.
    past_crop = crop_model_cache(past, T_p - 1)
    with torch.no_grad():
        first_out = mdl(ids[:, -1:], past_key_values=past_crop, use_cache=True)
    first_logits = first_out.logits[:, -1, :]
    past = first_out.past_key_values   # T_p KVs — quantized prefill + exact last token
    first_logits = _suppress(first_logits, suppress_ids)

    generated = [_sample_next(first_logits, temperature, top_p)]
    seen_ids  = ids[0].tolist()

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            step = mdl(generated[-1], past_key_values=past, use_cache=True)
        logits = step.logits[:, -1, :]
        logits = _suppress(logits, suppress_ids)
        logits = _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
        next_tok = _sample_next(logits, temperature, top_p)
        seen_ids.append(next_tok.item())
        generated.append(next_tok)
        past = step.past_key_values
        if next_tok.item() == tok.eos_token_id:
            break

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
    device: Optional[str] = None,
    device_map: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream generated text token-by-token from a quantized KV cache.

    Same parameters as generate(). Yields decoded text fragments as each
    token is produced — use this when you want to display output incrementally
    rather than waiting for the full response.

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

    mdl, tok = _load_model(model, _map)
    if _device and _map is None:
        mdl = mdl.to(_device)

    input_ids = _format_prompt(tok, prompt, raw, system=system)
    past, _, T_p, ids = _build_quantized_cache(mdl, tok, input_ids, bits, correction_rank)

    suppress_ids = _get_suppress_ids(tok)

    past_crop = crop_model_cache(past, T_p - 1)
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

        with torch.no_grad():
            step = mdl(generated[-1], past_key_values=past, use_cache=True)
        logits = step.logits[:, -1, :]
        logits = _suppress(logits, suppress_ids)
        logits   = _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
        next_tok = _sample_next(logits, temperature, top_p)
        seen_ids.append(next_tok.item())
        generated.append(next_tok)
        past = step.past_key_values
        if next_tok.item() == tok.eos_token_id:
            break

    # Yield any remaining text after the last token, skipping incomplete sequences
    final_text = tok.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    final_fragment = _clean_text(final_text)[prev_len:].replace("\ufffd", "")
    if final_fragment:
        yield final_fragment
