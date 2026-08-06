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
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import KVCacheQuantizer, kvs_from_cache, quantize_model_cache, crop_model_cache

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


def load_model(name, weights=None, max_gpu_mem=None, max_cpu_mem=None, weights_disk_dir=None):
    """
    Load a model + tokenizer, optionally with weight-placement offload so large
    models fit on low-VRAM GPUs (see --weights).

    Reuses _build_load_kwargs from kvquant.generate so the offload logic lives in
    one place (4bit/8bit via bitsandbytes, offload via accelerate).
    """
    from kvquant.generate import _build_load_kwargs

    mode = None if weights in (None, "full") else weights
    print(f"Loading {name}  (weights={mode or 'full'})...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = _build_load_kwargs(
        mode, None, max_gpu_mem, max_cpu_mem, weights_disk_dir
    )
    model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
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
    kvs = kvs_from_cache(pkv)
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
    parser.add_argument(
        "--weights",
        choices=["full", "4bit", "8bit", "offload"],
        default="full",
        help="How to place model WEIGHTS so large models fit on low-VRAM GPUs "
        "(the fix for OOM loading e.g. a 7B model on an 8 GB card). "
        "full = fp16/auto (default). "
        "4bit = 4-bit NF4 weights (bitsandbytes); a 7B fits in ~5 GB VRAM, fast. "
        "8bit = 8-bit weights (bitsandbytes); a 7B needs ~8 GB VRAM. "
        "offload = accelerate tiered GPU->RAM->SSD placement; fits any size, slow. "
        'Needs the [quant] extra: pip install "kvquant-plus-plus[quant]".',
    )
    parser.add_argument(
        "--max-gpu-mem",
        type=str,
        default=None,
        help="VRAM cap for --weights offload (e.g. '6GiB'). Default: ~90%% of free VRAM.",
    )
    parser.add_argument(
        "--max-cpu-mem",
        type=str,
        default=None,
        help="CPU-RAM cap for --weights offload (e.g. '12GiB').",
    )
    parser.add_argument(
        "--weights-disk-dir",
        type=str,
        default=None,
        help="SSD folder for offloaded weight shards (--weights offload only). "
        "Default: an auto-cleaned temp directory.",
    )
    args = parser.parse_args()

    model, tok = load_model(
        args.model,
        weights=args.weights,
        max_gpu_mem=args.max_gpu_mem,
        max_cpu_mem=args.max_cpu_mem,
        weights_disk_dir=args.weights_disk_dir,
    )
    n_layers, n_heads, n_kv_heads, head_dim = get_model_dims(model)
    n_outlier = max(4, head_dim // 4)
    gqa_factor = n_heads // n_kv_heads  # 1 for MHA; >1 for GQA (Qwen, Llama-3, …)
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

        # Prefill: run the model on the full prompt once.
        # The resulting KV tensors are both the calibration data and the cache to
        # compress — they capture this context's exact outlier-channel distribution.
        with torch.no_grad():
            prefill_out = model(input_ids, use_cache=True)
        native_cache_orig = prefill_out.past_key_values
        cal_kvs = kvs_from_cache(native_cache_orig)

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
            # Per-layer quantizers: calibrate each layer independently so that
            # outlier channels are detected from that layer's own statistics.
            # Layers can have very different KV distributions (early vs late layers),
            # so a single shared quantizer misidentifies outliers for most layers.
            kvc_layers = []
            for lk, lv in cal_kvs:
                kvc_l = KVCacheQuantizer(
                    head_dim=head_dim,
                    num_bits=bits,
                    use_outlier=True,
                    n_outlier=n_outlier,
                    outlier_bits=min(bits + 1, 8),
                    regular_bits=max(bits - 1, 1),
                    gqa_factor=gqa_factor,
                )
                kvc_l.calibrate(lk, lv)
                kvc_layers.append(kvc_l)
            kvc = kvc_layers[0]  # for avg_bits label only

            # quantize_model_cache deep-copies native_cache and quantizes K,V in-place.
            # For hybrid models the Mamba/linear-attn state is preserved in the copy,
            # so the subsequent forward passes work without error.
            #
            # Correction rank is applied selectively: at 4-bit the quantization
            # residual is already small (0.011 MSE bound), so a rank-4 SVD picks
            # up numerical noise rather than signal and can hurt quality. At 2-bit
            # and 3-bit the residual is large and structured, so correction helps.
            effective_rank = args.correction_rank if (bits < 4 or gqa_factor > 1) else 0
            past = quantize_model_cache(
                native_cache_orig, kvc_layers, correction_rank=effective_rank
            )

            # Greedy decode: get the first-token logit from the QUANTIZED cache
            # by cropping to T_p-1 positions and running the last prompt token
            # through. Using the unquantized prefill logit here would make all
            # bit-widths produce the same first token - hiding the real degradation.
            past_crop = crop_model_cache(past, T_p - 1)
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
            eff_bits = kvc.avg_bits
            label = (
                f"{bits}-bit"
                if gqa_factor == 1
                else f"{bits}-bit→{eff_bits:.0f}b (g={gqa_factor})"
            )
            print(
                f"  {label}  : {_clean(tok.decode(gen_ids[0], skip_special_tokens=True))}"
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

            pq_kvc = ProductKVCache(
                head_dim,
                num_subspaces=args.pq_subspaces,
                bits_per_subspace=args.pq_bits,
            )
            # Calibrate PQ codebooks on the actual prefill KV vectors (cal_kvs),
            # stacking all layers into one pool for a richer training set.
            all_k_cal = torch.cat([kv[0].reshape(-1, head_dim) for kv in cal_kvs])
            all_v_cal = torch.cat([kv[1].reshape(-1, head_dim) for kv in cal_kvs])
            pq_kvc.calibrate(all_k_cal, all_v_cal)

            past = quantize_model_cache(
                native_cache_orig, pq_kvc, correction_rank=args.correction_rank
            )
            past_crop = crop_model_cache(past, T_p - 1)
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
            outlier_bits=min(bits + 1, 8),
            regular_bits=max(bits - 1, 1),
            gqa_factor=gqa_factor,
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
            outlier_bits=min(bits + 1, 8),
            regular_bits=max(bits - 1, 1),
            gqa_factor=gqa_factor,
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
        gqa_factor=gqa_factor,
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
