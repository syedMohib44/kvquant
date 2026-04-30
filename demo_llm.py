"""
Real-world KVQuant demo: compress KV caches from a language model.

What this does
--------------
1. Loads a model and runs a forward pass on real text.
2. Captures the actual K and V tensors from every attention layer.
3. Calibrates OutlierKVQuant on those tensors.
4. Compresses and decompresses the KV cache with KVQuant.
5. Reports:
   - Per-layer KV reconstruction MSE
   - Attention score error  (Q @ K^T  vs  Q @ K_hat^T)
   - Output logit difference (full KV vs quantized KV)

Run (default benchmark):
    python -m kvquant.demo_llm

Interactive generation (any model):
    python -m kvquant.demo_llm --prompt "Hello, how are you?"
    python -m kvquant.demo_llm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Once upon a time"
    python -m kvquant.demo_llm --model Qwen/Qwen3.5-0.8B --prompt "What is AI?"

Auto-detects hybrid models (Qwen3.5, Mamba, etc.) and uses native-cache quantization
so that both pure-transformer and hybrid architectures work transparently.
"""

import argparse
import copy
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import KVCacheQuantizer
from src.correction import _randomized_svd

# ---------------------------------------------------------------------------
DEFAULT_MODEL = "distilgpt2"
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hellow my name is some one",
    "Artificial intelligence is transforming the world in unexpected ways.",
    "The history of the Roman Empire spans several centuries of conquest.",
    "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole.",
    "To be or not to be, that is the question every programmer asks at 3am.",
]
BITS_LIST = [2, 3, 4]
# ---------------------------------------------------------------------------


def load_model(name):
    print(f"Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    model.eval()
    return model, tok


def get_model_dims(model):
    """Return (n_layers, n_heads, n_kv_heads, head_dim) for any architecture."""
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    return n_layers, n_heads, n_kv_heads, head_dim


# ---------------------------------------------------------------------------
# Cache helpers (architecture-agnostic)
# ---------------------------------------------------------------------------


def _is_hybrid_model(model):
    """
    True if the model mixes standard transformer attention with linear/state-space
    attention layers (e.g. Qwen3.5, Jamba, Zamba).  Detects by:
      1. Class name of any submodule containing a known hybrid keyword.
      2. Presence of 'linear_attn' as a named child (Qwen3.5 pattern).
      3. Any submodule exposing has_previous_state (Mamba cache marker).
    """
    for _, module in model.named_modules():
        cls = type(module).__name__
        if any(
            x in cls
            for x in (
                "LinearAttention",
                "MambaLayer",
                "Mamba2Layer",
                "Rwkv",
                "RetNet",
                "GatedLinearAttention",
            )
        ):
            return True
        # Qwen3.5 names its linear-attention child 'linear_attn'
        if "linear_attn" in {n for n, _ in module.named_children()}:
            return True
        if hasattr(module, "has_previous_state"):
            return True
    return False


def _kvs_from_cache(native_cache):
    """
    Extract list of (k, v) tensors from whatever cache type the model returned.
    Handles three formats:
      • new DynamicCache  (transformers >= 4.50): .layers[i].keys / .values
      • old DynamicCache  (transformers 4.38-4.49): .key_cache[i] / .value_cache[i]
      • tuple-of-tuples   (transformers < 4.38): ((k0,v0), (k1,v1), …)
    Skips None / missing entries (linear-attn / sliding-window layers).
    """
    # New-style DynamicCache: layers list with DynamicLayer objects
    if hasattr(native_cache, "layers"):
        result = []
        for layer in native_cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if isinstance(k, torch.Tensor):
                result.append((k, v))
        return result

    # Old-style DynamicCache / HybridCache
    if hasattr(native_cache, "key_cache"):
        return [
            (native_cache.key_cache[i], native_cache.value_cache[i])
            for i in range(len(native_cache.key_cache))
            if native_cache.key_cache[i] is not None
        ]

    # Legacy tuple-of-tuples
    if isinstance(native_cache, (tuple, list)):
        return [(k, v) for k, v in native_cache if isinstance(k, torch.Tensor)]

    return []


def _apply_lowrank_correction(
    x: torch.Tensor, x_hat: torch.Tensor, rank: int
) -> torch.Tensor:
    """Add rank-r SVD correction to x_hat using residual (x - x_hat).

    The quantized cache x_hat has error R = x - x_hat. That error is not
    random - it tends to be low-rank (a few directions account for most of
    the energy). We approximate R with a rank-r SVD and add it back, recovering
    most of the lost precision at a fraction of the storage cost.
    """
    # Save original shape (B, H, T, d) to restore at the end.
    # The model expects this exact shape - we can't return anything different.
    orig_shape = x_hat.shape

    # SVD requires exactly 3 dims: (batch, rows, cols).
    # The KV cache is 4D: (B, H, T, d) - B=batch, H=heads, T=tokens, d=head_dim.
    # Flatten B and H into a single batch dimension N = B*H so SVD sees (N, T, d).
    # reshape() is a free view - no data is copied.
    N = x_hat.reshape(-1, orig_shape[-2], orig_shape[-1]).shape[0]

    # Compute the residual error: how far is the quantized cache from the real cache.
    # Use float32 to avoid precision loss during subtraction (x_hat may be float16).
    R = (x.float() - x_hat.float()).reshape(N, orig_shape[-2], orig_shape[-1])
    T = R.shape[1]  # number of tokens in the cache

    # Clamp rank so it never exceeds what SVD can produce: min(T, d) - 1.
    # Without this, SVD would crash on very short sequences (e.g. T=2, rank=4).
    r = min(rank, T - 1, orig_shape[-1] - 1)
    if r < 1:
        # Sequence too short to apply any correction - return as-is.
        return x_hat

    # Choose SVD method based on sequence length:
    #   T >= 64 -> randomized SVD: faster (2.5x at T=256+), slight approximation
    #   T <  64 -> exact SVD then manually truncate to top-r components
    if T >= 64:
        U, S, Vh = _randomized_svd(R, rank=r)
    else:
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        # SVD returns all singular components; keep only the top-r.
        # These hold the most energy (singular values are sorted largest->smallest).
        U, S, Vh = U[..., :r], S[..., :r], Vh[..., :r, :]

    # Reconstruct the rank-r approximation of the residual error.
    # Equivalent to U @ diag(S) @ Vh but avoids allocating a diagonal matrix.
    # S.unsqueeze(-2) broadcasts S from (N, r) to (N, 1, r) so it scales each
    # row of U: (N, T, r) * (N, 1, r) = (N, T, r), then @ Vh (N, r, d) = (N, T, d).
    correction = (U * S.unsqueeze(-2)) @ Vh  # (N, T, d)

    # Add correction to quantized cache, reshape back to (B, H, T, d),
    # and cast back to the original dtype (e.g. float16).
    return (x_hat.float() + correction.reshape(orig_shape)).to(x_hat.dtype)


def _crop_cache(native_cache, seq_len: int):
    """
    Return a deep copy of native_cache with KV tensors truncated to seq_len
    positions along the sequence dimension.  Non-KV state (Mamba, linear-attn)
    is preserved unchanged.  Used to obtain the T_p-1 cache needed to compute
    an accurate quantized logit for the first generated token.
    """
    cache = copy.deepcopy(native_cache)
    if hasattr(cache, "layers"):  # new-style DynamicCache
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            if isinstance(k, torch.Tensor) and k.shape[-2] > seq_len:
                layer.keys = k[..., :seq_len, :]
                layer.values = layer.values[..., :seq_len, :]
    elif hasattr(cache, "key_cache"):  # old-style DynamicCache / HybridCache
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i]
            if isinstance(k, torch.Tensor) and k.shape[-2] > seq_len:
                cache.key_cache[i] = k[..., :seq_len, :]
                cache.value_cache[i] = cache.value_cache[i][..., :seq_len, :]
    return cache


def _quantize_cache(native_cache, kvc, correction_rank: int = 0):
    """
    Return a COPY of native_cache with K, V quantized in transformer-attention
    layers.  Linear/state-space state is left untouched so hybrid models (Qwen3.5,
    Mamba, …) continue to work correctly.
    """
    cache_q = copy.deepcopy(native_cache)

    def _quant_pair(k, v):
        """Quantize a (k, v) pair, optionally apply low-rank correction, cast back."""
        dtype = k.dtype
        k_hat, v_hat = kvc.decompress_kv(*kvc.compress_kv(k.float(), v.float()))
        if correction_rank > 0:
            k_hat = _apply_lowrank_correction(k, k_hat, correction_rank)
            v_hat = _apply_lowrank_correction(v, v_hat, correction_rank)
        return k_hat.to(dtype), v_hat.to(dtype)

    # New-style DynamicCache
    if hasattr(cache_q, "layers"):
        for layer in cache_q.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if isinstance(k, torch.Tensor):
                layer.keys, layer.values = _quant_pair(k, v)
        return cache_q

    # Old-style DynamicCache / HybridCache
    if hasattr(cache_q, "key_cache"):
        for i in range(len(cache_q.key_cache)):
            k, v = cache_q.key_cache[i], cache_q.value_cache[i]
            if isinstance(k, torch.Tensor):
                cache_q.key_cache[i], cache_q.value_cache[i] = _quant_pair(k, v)
        return cache_q

    # Legacy tuple-of-tuples (immutable - rebuild)
    if isinstance(native_cache, (tuple, list)):
        result = []
        for k, v in native_cache:
            if isinstance(k, torch.Tensor):
                result.append(_quant_pair(k, v))
            else:
                result.append((k, v))
        return tuple(result)

    return cache_q  # unknown type - return deep copy as-is


# ---------------------------------------------------------------------------
# Legacy helper used by the default (non-prompt) benchmark sections
# ---------------------------------------------------------------------------


def _make_cache_dynamic(kvs, kvc):
    """Build a quantized cache from a list of (k, v) pairs (used by default benchmark)."""
    pairs = [kvc.decompress_kv(*kvc.compress_kv(k, v)) for k, v in kvs]
    try:
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k_hat, v_hat) in enumerate(pairs):
            cache.update(k_hat, v_hat, layer_idx=i)
        return cache
    except (ImportError, AttributeError, TypeError):
        return tuple(pairs)


def capture_kv(model, input_ids):
    """
    Run one forward pass, return (kvs, native_cache) where kvs is a list of
    (k, v) tensors and native_cache is the raw cache object from the model.
    """
    with torch.no_grad():
        out = model(input_ids, output_attentions=False, use_cache=True)
    pkv = out.past_key_values
    kvs = _kvs_from_cache(pkv)
    return kvs, pkv


def attn_score_error(q, k_true, k_hat):
    scale = q.shape[-1] ** -0.5
    s_true = (q @ k_true.transpose(-2, -1)) * scale
    s_hat = (q @ k_hat.transpose(-2, -1)) * scale
    return (s_true - s_hat).abs().mean().item()


def sep(title=""):
    w = 64
    if title:
        print(f"\n---- {title} {'-' * (w - len(title) - 6)}")
    else:
        print("-" * w)


def _apply_repetition_penalty(logits, generated_ids, penalty):
    """
    Penalize tokens that appear in generated_ids.
    Scores > 0 are divided by penalty; scores < 0 are multiplied by penalty.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    seen = torch.tensor(list(set(generated_ids)), dtype=torch.long)
    scores = logits[:, seen]
    logits[:, seen] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def _clean(text):
    """Strip <think>…</think> reasoning blocks (closed or unclosed) and collapse whitespace."""
    # Closed think block -> drop entirely
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed think block (hit token limit mid-think) -> drop tag and all content after
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="KVQuant LLM demo - benchmark or interactive generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL}). "
        "Works with pure-transformer and hybrid (Qwen3.5, Mamba, …) models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for interactive generation (skips default benchmark).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Tokens to generate in --prompt mode (default: 40).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.5,
        help="Repetition penalty applied during quantized greedy decode (default: 1.5). "
        "Values > 1 discourage repeating already-generated tokens. "
        "Increase to 2.0+ if you see repetition loops at low bit-widths.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip the chat template and tokenize the prompt as plain text. "
        "Use for sentence-completion prompts; leave off for Q&A prompts.",
    )
    parser.add_argument(
        "--correction-rank",
        type=int,
        default=0,
        help="Apply rank-r low-rank error correction to the quantized KV cache "
        "(0 = disabled, 4 = recommended). Reduces quantization error at the "
        "cost of storing r*(T+d) extra floats per layer.",
    )
    parser.add_argument(
        "--product-quant",
        action="store_true",
        help="Also run Product Quantization. Shown as an extra PQ line after the "
        "scalar bit-width results.",
    )
    parser.add_argument(
        "--pq-bits",
        type=int,
        default=8,
        help="Bits per PQ subspace (default: 8 -> K=256 centroids). "
        "Higher = better quality, larger codebook. "
        "e.g. 4->K=16 (1b/dim), 8->K=256 (2b/dim), 12->K=4096 (3b/dim).",
    )
    parser.add_argument(
        "--pq-subspaces",
        type=int,
        default=16,
        help="Number of PQ subspaces M (default: 16). "
        "Must divide head_dim. Fewer subspaces = larger sub_dim = richer codebook.",
    )
    args = parser.parse_args()

    model, tok = load_model(args.model)
    n_layers, n_heads, n_kv_heads, head_dim = get_model_dims(model)
    n_outlier = max(4, head_dim // 4)
    hybrid = _is_hybrid_model(model)

    print(
        f"  arch   : {'hybrid (transformer + linear attn)' if hybrid else 'pure transformer'}"
    )
    print(
        f"  layers : {n_layers}  heads : {n_heads}  kv_heads : {n_kv_heads}  head_dim : {head_dim}"
    )

    # -----------------------------------------------------------------------
    # --prompt  Interactive generation
    # -----------------------------------------------------------------------
    if args.prompt:
        sep("Interactive generation")
        print(f"  Model  : {args.model}")
        print(f"  Prompt : {args.prompt!r}")
        # Determine prompt mode:
        #   chat template  - model has a chat template (instruct/chat models)
        #   qa-format      - base model (no chat template), auto-wrap as Q:/A:
        #   raw            - user passed --raw, or sentence-completion style
        raw_prompt = args.prompt
        has_template = hasattr(tok, "apply_chat_template") and bool(tok.chat_template)
        use_template = has_template and not args.raw
        use_qa_fmt = not has_template and not args.raw

        if use_template:
            mode_label = "chat template"
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                formatted = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            enc = tok(formatted, return_tensors="pt", add_special_tokens=False)
        elif use_qa_fmt:
            # Base model: format as "Q: ...\nA:" so the model completes the answer
            mode_label = "Q/A format (base model)"
            formatted = f"Q: {raw_prompt.rstrip('?').strip()}?\nA:"
            enc = tok(formatted, return_tensors="pt")
        else:
            mode_label = "raw"
            enc = tok(raw_prompt, return_tensors="pt")

        print(f"  Mode   : {mode_label}\n")
        input_ids = enc["input_ids"]
        T_p = input_ids.shape[1]

        # Single prefill pass - gets native cache (DynamicCache or HybridCache)
        # and logits at the last prompt position.
        with torch.no_grad():
            prefill_out = model(input_ids, use_cache=True)
        native_cache_orig = prefill_out.past_key_values

        # Calibration pool - use TEXTS corpus for better outlier-channel estimation.
        # The prompt alone is too short (< 20 tokens) to reliably identify outliers.
        cal_enc = tok(
            TEXTS,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            add_special_tokens=True,
        )
        with torch.no_grad():
            cal_out = model(cal_enc["input_ids"], use_cache=True)
        cal_kvs = _kvs_from_cache(cal_out.past_key_values)
        cal_T = cal_enc["input_ids"].shape[1]
        all_k_p = torch.cat(
            [kv[0].reshape(-1, cal_T, head_dim) for kv in cal_kvs], dim=0
        )
        all_v_p = torch.cat(
            [kv[1].reshape(-1, cal_T, head_dim) for kv in cal_kvs], dim=0
        )

        # Newline suppression (varies by tokenizer)
        newline_ids = tok.encode("\n", add_special_tokens=False)
        bad_words = [[t] for t in newline_ids] if newline_ids else None
        newline_id = newline_ids[0] if newline_ids else -1

        attn_mask = torch.ones_like(input_ids)

        # Suppress HuggingFace warnings that fire on every greedy generate() call
        # (attention_mask inference warning, temperature/top_p/top_k validity warning).
        import logging as _logging

        _hf = _logging.getLogger("transformers")
        _prev_hf = _hf.level
        _hf.setLevel(_logging.ERROR)

        # Unquantized baseline
        with torch.no_grad():
            true_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                pad_token_id=tok.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                bad_words_ids=bad_words,
                repetition_penalty=args.repetition_penalty,
            )

        _hf.setLevel(_prev_hf)
        gen_only = true_ids[0, T_p:]
        print(f"  Unquant: {_clean(tok.decode(gen_only, skip_special_tokens=True))}\n")

        # Quantized generation - one pass per bit-width
        for bits in BITS_LIST:
            kvc = KVCacheQuantizer(
                head_dim=head_dim,
                num_bits=bits,
                use_outlier=True,
                n_outlier=n_outlier,
                outlier_bits=min(bits + 1, 4),
                regular_bits=max(bits - 1, 1),
            )
            kvc.calibrate(all_k_p, all_v_p)

            # _quantize_cache deep-copies native_cache and quantizes K,V in-place.
            # For hybrid models the Mamba/linear-attn state is preserved in the copy,
            # so the subsequent forward passes work without error.
            #
            # Correction rank is applied selectively: at 4-bit the quantization
            # residual is already small (0.011 MSE bound), so a rank-4 SVD picks
            # up numerical noise rather than signal and can hurt quality. At 2-bit
            # and 3-bit the residual is large and structured, so correction helps.
            effective_rank = args.correction_rank if bits < 4 else 0
            past = _quantize_cache(
                native_cache_orig, kvc, correction_rank=effective_rank
            )

            # Greedy decode: get the first-token logit from the QUANTIZED cache
            # by cropping to T_p-1 positions and running the last prompt token
            # through. Using the unquantized prefill logit here would make all
            # bit-widths produce the same first token - hiding the real degradation.
            past_crop = _crop_cache(past, T_p - 1)
            with torch.no_grad():
                q1_out = model(
                    input_ids[:, -1:], past_key_values=past_crop, use_cache=False
                )
            first_logits_m = q1_out.logits[:, -1, :].clone()
            if newline_id >= 0:
                first_logits_m[:, newline_id] = float("-inf")
            generated = [first_logits_m.argmax(-1, keepdim=True)]
            seen_ids = input_ids[0].tolist()  # seed with prompt tokens

            for _ in range(args.max_new_tokens - 1):
                with torch.no_grad():
                    step = model(generated[-1], past_key_values=past, use_cache=True)
                logits_s = step.logits[:, -1, :].clone()
                if newline_id >= 0:
                    logits_s[:, newline_id] = float("-inf")
                logits_s = _apply_repetition_penalty(
                    logits_s, seen_ids, args.repetition_penalty
                )
                next_tok = logits_s.argmax(-1, keepdim=True)
                seen_ids.append(next_tok.item())
                generated.append(next_tok)
                past = step.past_key_values
                if next_tok.item() == tok.eos_token_id:
                    break

            gen_ids = torch.cat(generated, dim=1)
            print(
                f"  {bits}-bit  : {_clean(tok.decode(gen_ids[0], skip_special_tokens=True))}"
            )

        # Product Quantization run (optional, --product-quant flag)
        if args.product_quant:
            from .src.product_quantizer import ProductKVCache

            # PQ config: M=16 subspaces x b=8 bits  →  128 bits/vector = 2 bits/dim.
            # M (subspaces), b (bits each)
            #
            # Space consequences:
            #   + Per-vector storage is HALVED vs 4-bit scalar (128 vs 256 bits).
            #   + Fixed codebook overhead: M × K × sub_dim = 16 × 256 × 4 = 16,384
            #     floats (64 KB) per K/V quantizer. Negligible at >500 tokens.
            #
            # Time consequences:
            #   - Calibration: k-means++ × M subspaces (one-time at prefill, seconds).
            #   - Encode (per new token): O(N × M × K × sub_dim) with K=256 centroids
            #     vs O(N × d × 16) for 4-bit scalar 16× slower per append.
            #     Current impl uses M sequential cdist calls (Python loop); a batched
            #     version would close most of this gap.
            #     Decode: O(N × d) table lookups same as scalar.
            #     Attention: same cost if KV is reconstructed before Q·Kᵀ.
            #     Asymmetric distance (precompute q to centroid per subspace, then
            #     sum M lookups) avoids reconstruction entirely and is faster.
            #
            # Other combinations worth trying
            # num_subspaces=8, bits_per_subspace=8
            # num_subspaces=16, bits_per_subspace=4
            # num_subspaces=8, bits_per_subspace=4

            pq_kvc = ProductKVCache(head_dim, num_subspaces=args.pq_subspaces, bits_per_subspace=args.pq_bits)
            pq_kvc.calibrate(all_k_p, all_v_p)

            past = _quantize_cache(native_cache_orig, pq_kvc, correction_rank=args.correction_rank)
            past_crop = _crop_cache(past, T_p - 1)
            with torch.no_grad():
                q1_out = model(
                    input_ids[:, -1:], past_key_values=past_crop, use_cache=False
                )
            first_logits_pq = q1_out.logits[:, -1, :].clone()
            if newline_id >= 0:
                first_logits_pq[:, newline_id] = float("-inf")
            generated = [first_logits_pq.argmax(-1, keepdim=True)]
            seen_ids = input_ids[0].tolist()

            for _ in range(args.max_new_tokens - 1):
                with torch.no_grad():
                    step = model(generated[-1], past_key_values=past, use_cache=True)
                logits_s = step.logits[:, -1, :].clone()
                if newline_id >= 0:
                    logits_s[:, newline_id] = float("-inf")
                logits_s = _apply_repetition_penalty(
                    logits_s, seen_ids, args.repetition_penalty
                )
                next_tok = logits_s.argmax(-1, keepdim=True)
                seen_ids.append(next_tok.item())
                generated.append(next_tok)
                past = step.past_key_values
                if next_tok.item() == tok.eos_token_id:
                    break

            gen_ids = torch.cat(generated, dim=1)
            ratio = pq_kvc.k_quant.compression_ratio()
            eff = pq_kvc.effective_bits_per_dim
            print(
                f"  PQ({eff:.0f}b) : {_clean(tok.decode(gen_ids[0], skip_special_tokens=True))}"
                f"\n           [M={pq_kvc.k_quant.M} subspaces x {pq_kvc.k_quant.b} bits, {ratio:.1f}x smaller than 4-bit scalar]"
            )

        sep()
        return

    # -----------------------------------------------------------------------
    # Default benchmark (pure-transformer models work best here)
    # -----------------------------------------------------------------------
    encoded = tok(
        TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=64
    )
    input_ids = encoded["input_ids"]
    B, T = input_ids.shape

    sep("Capturing real KV caches")
    kvs, _ = capture_kv(model, input_ids)
    print(f"  {len(kvs)} layers captured, K shape: {list(kvs[0][0].shape)}")

    all_k = torch.cat([kv[0].reshape(-1, T, head_dim) for kv in kvs], dim=0)
    all_v = torch.cat([kv[1].reshape(-1, T, head_dim) for kv in kvs], dim=0)
    print(f"  Calibration pool: {all_k.shape[0]} KV vectors of dim {head_dim}")

    torch.manual_seed(0)
    q_proxy = torch.randn_like(kvs[0][0])

    sep("Per-layer KV compression results")
    print(
        f"{'bits':>5}  {'avg_bits':>9}  {'K MSE':>10}  {'V MSE':>10}  {'Attn err':>10}"
    )
    sep()

    for bits in BITS_LIST:
        kvc = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=n_outlier,
            outlier_bits=min(bits + 1, 4),
            regular_bits=max(bits - 1, 1),
        )
        kvc.calibrate(all_k, all_v)

        k_mse_l, v_mse_l, attn_l = [], [], []
        for k, v in kvs:
            k_c, v_c = kvc.compress_kv(k, v)
            k_hat, v_hat = kvc.decompress_kv(k_c, v_c)
            k_mse_l.append(((k - k_hat) ** 2).mean().item())
            v_mse_l.append(((v - v_hat) ** 2).mean().item())
            attn_l.append(attn_score_error(q_proxy, k, k_hat))

        print(
            f"{bits:>5}  {kvc.avg_bits:>9.2f}"
            f"  {sum(k_mse_l)/len(k_mse_l):>10.5f}"
            f"  {sum(v_mse_l)/len(v_mse_l):>10.5f}"
            f"  {sum(attn_l)/len(attn_l):>10.5f}"
        )

    sep("Output logit difference (last token)")
    with torch.no_grad():
        out_true = model(input_ids, use_cache=False)
    logits_true = out_true.logits[:, -1, :]

    print(f"{'bits':>5}  {'avg_bits':>9}  {'logit MAE':>12}  {'top-1 match':>12}")
    sep()

    for bits in BITS_LIST:
        kvc = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=n_outlier,
            outlier_bits=min(bits + 1, 4),
            regular_bits=max(bits - 1, 1),
        )
        kvc.calibrate(all_k, all_v)
        quant_kvs = _make_cache_dynamic(kvs, kvc)

        dummy = torch.full((B, 1), tok.eos_token_id)
        with torch.no_grad():
            out_q = model(dummy, past_key_values=quant_kvs, use_cache=False)
        logits_q = out_q.logits[:, -1, :]

        mae = (logits_true - logits_q).abs().mean().item()
        match = (logits_true.argmax(-1) == logits_q.argmax(-1)).float().mean().item()
        print(f"{bits:>5}  {kvc.avg_bits:>9.2f}  {mae:>12.5f}  {match:>11.0%}")

    sep(f"Top-5 next-token predictions  (bits={BITS_LIST[-1]})")
    kvc = KVCacheQuantizer(
        head_dim=head_dim,
        num_bits=BITS_LIST[-1],
        use_outlier=True,
        n_outlier=n_outlier,
        outlier_bits=4,
        regular_bits=3,
    )
    kvc.calibrate(all_k, all_v)
    quant_kvs = _make_cache_dynamic(kvs, kvc)

    dummy = torch.full((B, 1), tok.eos_token_id)
    with torch.no_grad():
        out_q = model(dummy, past_key_values=quant_kvs, use_cache=False)
    logits_q = out_q.logits[:, -1, :]

    for i, text in enumerate(TEXTS):
        true_top5 = logits_true[i].topk(5).indices.tolist()
        quant_top5 = logits_q[i].topk(5).indices.tolist()
        t_words = [tok.decode([t]).strip() for t in true_top5]
        q_words = [tok.decode([t]).strip() for t in quant_top5]
        overlap = len(set(true_top5) & set(quant_top5))
        print(f"\n  Text : {text[:55]}...")
        print(f"  True : {t_words}")
        print(f"  Quant: {q_words}  (top-5 overlap: {overlap}/5)")

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
