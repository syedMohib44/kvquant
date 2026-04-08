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
    n_layers  = getattr(cfg, "num_hidden_layers",  getattr(cfg, "n_layer", None))
    n_heads   = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head",  None))
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    hidden    = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    head_dim  = getattr(cfg, "head_dim", hidden // n_heads)
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
        if any(x in cls for x in ("LinearAttention", "MambaLayer", "Mamba2Layer",
                                   "Rwkv", "RetNet", "GatedLinearAttention")):
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
      • new DynamicCache  (transformers >= ~4.50): .layers[i].keys / .values
      • old DynamicCache  (transformers ~4.38-4.49): .key_cache[i] / .value_cache[i]
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


def _quantize_cache(native_cache, kvc):
    """
    Return a COPY of native_cache with K, V quantized in transformer-attention
    layers.  Linear/state-space state is left untouched so hybrid models (Qwen3.5,
    Mamba, …) continue to work correctly.
    """
    cache_q = copy.deepcopy(native_cache)

    def _quant_pair(k, v):
        """Quantize a (k, v) pair and cast back to the original dtype."""
        dtype = k.dtype
        k_hat, v_hat = kvc.decompress_kv(*kvc.compress_kv(k.float(), v.float()))
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

    # Legacy tuple-of-tuples (immutable — rebuild)
    if isinstance(native_cache, (tuple, list)):
        result = []
        for k, v in native_cache:
            if isinstance(k, torch.Tensor):
                result.append(_quant_pair(k, v))
            else:
                result.append((k, v))
        return tuple(result)

    return cache_q  # unknown type — return deep copy as-is


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
    s_hat  = (q @ k_hat.transpose(-2, -1)) * scale
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
    # Closed think block → drop entirely
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed think block (hit token limit mid-think) → drop tag and all content after
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="KVQuant LLM demo — benchmark or interactive generation"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL}). "
             "Works with pure-transformer and hybrid (Qwen3.5, Mamba, …) models.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom prompt for interactive generation (skips default benchmark).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=40,
        help="Tokens to generate in --prompt mode (default: 40).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.3,
        help="Repetition penalty applied during quantized greedy decode (default: 1.3). "
             "Values > 1 discourage repeating already-generated tokens.",
    )
    args = parser.parse_args()

    model, tok = load_model(args.model)
    n_layers, n_heads, n_kv_heads, head_dim = get_model_dims(model)
    n_outlier = max(4, head_dim // 4)
    hybrid = _is_hybrid_model(model)

    print(f"  arch   : {'hybrid (transformer + linear attn)' if hybrid else 'pure transformer'}")
    print(f"  layers : {n_layers}  heads : {n_heads}  kv_heads : {n_kv_heads}  head_dim : {head_dim}")

    # -----------------------------------------------------------------------
    # --prompt  Interactive generation
    # -----------------------------------------------------------------------
    if args.prompt:
        sep("Interactive generation")
        print(f"  Model  : {args.model}")
        print(f"  Prompt : {args.prompt!r}")
        print(f"  Mode   : {'chat template' if (hasattr(tok, 'apply_chat_template') and tok.chat_template) else 'raw'}\n")

        # For chat/thinking models, apply the chat template so the model gets
        # the proper BOS + system prompt structure.  Pass enable_thinking=False
        # for Qwen3 models (suppresses the <think> block without /no_think hack).
        raw_prompt = args.prompt
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                formatted = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            enc = tok(formatted, return_tensors="pt", add_special_tokens=False)
        else:
            enc = tok(raw_prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        T_p = input_ids.shape[1]

        # Single prefill pass — gets native cache (DynamicCache or HybridCache)
        # and logits at the last prompt position.
        with torch.no_grad():
            prefill_out = model(input_ids, use_cache=True)
        native_cache_orig = prefill_out.past_key_values
        first_logits = prefill_out.logits[:, -1, :]   # (1, vocab)

        # Calibration pool from captured KV tensors
        kvs_p = _kvs_from_cache(native_cache_orig)
        all_k_p = torch.cat([kv[0].reshape(-1, T_p, head_dim) for kv in kvs_p], dim=0)
        all_v_p = torch.cat([kv[1].reshape(-1, T_p, head_dim) for kv in kvs_p], dim=0)

        # Newline suppression (varies by tokenizer)
        newline_ids = tok.encode("\n", add_special_tokens=False)
        bad_words   = [[t] for t in newline_ids] if newline_ids else None
        newline_id  = newline_ids[0] if newline_ids else -1

        # Unquantized baseline
        with torch.no_grad():
            true_ids = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, bad_words_ids=bad_words,
                repetition_penalty=args.repetition_penalty,
            )
        gen_only = true_ids[0, T_p:]
        print(f"  Unquant: {_clean(tok.decode(gen_only, skip_special_tokens=True))}\n")

        # Quantized generation — one pass per bit-width
        for bits in BITS_LIST:
            kvc = KVCacheQuantizer(
                head_dim=head_dim, num_bits=bits,
                use_outlier=True, n_outlier=n_outlier,
                outlier_bits=min(bits + 1, 4),
                regular_bits=max(bits - 1, 1),
            )
            kvc.calibrate(all_k_p, all_v_p)

            # _quantize_cache deep-copies native_cache and quantizes K,V in-place.
            # For hybrid models the Mamba/linear-attn state is preserved in the copy,
            # so the subsequent forward passes work without error.
            past = _quantize_cache(native_cache_orig, kvc)

            # Greedy decode: first token from prefill logits, rest from model.
            first_logits_m = first_logits.clone()
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
                logits_s = _apply_repetition_penalty(logits_s, seen_ids, args.repetition_penalty)
                next_tok = logits_s.argmax(-1, keepdim=True)
                seen_ids.append(next_tok.item())
                generated.append(next_tok)
                past = step.past_key_values
                if next_tok.item() == tok.eos_token_id:
                    break

            gen_ids = torch.cat(generated, dim=1)
            print(f"  {bits}-bit  : {_clean(tok.decode(gen_ids[0], skip_special_tokens=True))}")

        sep()
        return

    # -----------------------------------------------------------------------
    # Default benchmark (pure-transformer models work best here)
    # -----------------------------------------------------------------------
    encoded = tok(TEXTS, return_tensors="pt", padding=True,
                  truncation=True, max_length=64)
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
    print(f"{'bits':>5}  {'avg_bits':>9}  {'K MSE':>10}  {'V MSE':>10}  {'Attn err':>10}")
    sep()

    for bits in BITS_LIST:
        kvc = KVCacheQuantizer(
            head_dim=head_dim, num_bits=bits,
            use_outlier=True, n_outlier=n_outlier,
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

        print(f"{bits:>5}  {kvc.avg_bits:>9.2f}"
              f"  {sum(k_mse_l)/len(k_mse_l):>10.5f}"
              f"  {sum(v_mse_l)/len(v_mse_l):>10.5f}"
              f"  {sum(attn_l)/len(attn_l):>10.5f}")

    sep("Output logit difference (last token)")
    with torch.no_grad():
        out_true = model(input_ids, use_cache=False)
    logits_true = out_true.logits[:, -1, :]

    print(f"{'bits':>5}  {'avg_bits':>9}  {'logit MAE':>12}  {'top-1 match':>12}")
    sep()

    for bits in BITS_LIST:
        kvc = KVCacheQuantizer(
            head_dim=head_dim, num_bits=bits,
            use_outlier=True, n_outlier=n_outlier,
            outlier_bits=min(bits + 1, 4),
            regular_bits=max(bits - 1, 1),
        )
        kvc.calibrate(all_k, all_v)
        quant_kvs = _make_cache_dynamic(kvs, kvc)

        dummy = torch.full((B, 1), tok.eos_token_id)
        with torch.no_grad():
            out_q = model(dummy, past_key_values=quant_kvs, use_cache=False)
        logits_q = out_q.logits[:, -1, :]

        mae   = (logits_true - logits_q).abs().mean().item()
        match = (logits_true.argmax(-1) == logits_q.argmax(-1)).float().mean().item()
        print(f"{bits:>5}  {kvc.avg_bits:>9.2f}  {mae:>12.5f}  {match:>11.0%}")

    sep(f"Top-5 next-token predictions  (bits={BITS_LIST[-1]})")
    kvc = KVCacheQuantizer(
        head_dim=head_dim, num_bits=BITS_LIST[-1],
        use_outlier=True, n_outlier=n_outlier,
        outlier_bits=4, regular_bits=3,
    )
    kvc.calibrate(all_k, all_v)
    quant_kvs = _make_cache_dynamic(kvs, kvc)

    dummy = torch.full((B, 1), tok.eos_token_id)
    with torch.no_grad():
        out_q = model(dummy, past_key_values=quant_kvs, use_cache=False)
    logits_q = out_q.logits[:, -1, :]

    for i, text in enumerate(TEXTS):
        true_top5  = logits_true[i].topk(5).indices.tolist()
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
