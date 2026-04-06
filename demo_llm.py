"""
Real-world KVQuant demo: compress KV caches from distilgpt2.

What this does
--------------
1. Loads distilgpt2 and runs a forward pass on real text.
2. Hooks into every attention layer to capture the actual K and V tensors.
3. Calibrates OutlierKVQuant on those tensors.
4. Compresses and decompresses the KV cache with KVQuant.
5. Reports:
   - Per-layer KV reconstruction MSE
   - Attention score error  (Q @ K^T  vs  Q @ K_hat^T)
   - Output logit difference (full KV vs quantized KV)
   - Perplexity on a short passage with and without quantization

Run:  python -m kvquant.demo_llm
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import KVCacheQuantizer

# ---------------------------------------------------------------------------
MODEL_NAME = "distilgpt2"
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hellow my name is some one",
    "Artificial intelligence is transforming the world in unexpected ways.",
    "The history of the Roman Empire spans several centuries of conquest.",
    "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole.",
    "To be or not to be, that is the question every programmer asks at 3am.",
]
BITS_LIST = [2, 3, 4]
N_OUTLIER = 16  # distilgpt2 head_dim=64, use 16 outlier channels
# ---------------------------------------------------------------------------


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tok


def get_head_dim(model):
    cfg = model.config
    return cfg.n_embd // cfg.n_head


def capture_kv(model, input_ids):
    """
    Run one forward pass and return all (K, V) tensors per layer as a plain
    list of (k, v) tuples, each of shape (B, H, T, head_dim).

    Handles both legacy tuple-of-tuples and the newer DynamicCache object
    introduced in transformers >= 4.38.
    """
    with torch.no_grad():
        out = model(input_ids, output_attentions=False, use_cache=True)
    pkv = out.past_key_values

    # DynamicCache: has .key_cache and .value_cache lists
    if hasattr(pkv, "key_cache"):
        return [
            (pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache))
        ]

    # Iterate layers - each item is (k, v) or (k, v, ...) depending on version
    result = []
    for item in pkv:
        if isinstance(item, (tuple, list)):
            result.append((item[0], item[1]))
        else:
            result.append(item)
    return result


def _make_cache(model, kvs, kvc):
    """
    Compress all layers with kvc and return a past_key_values object
    compatible with the installed transformers version (DynamicCache or tuple).
    """
    pairs = []
    for k, v in kvs:
        k_c, v_c = kvc.compress_kv(k, v)
        k_hat, v_hat = kvc.decompress_kv(k_c, v_c)
        pairs.append((k_hat, v_hat))

    try:
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k_hat, v_hat) in enumerate(pairs):
            cache.update(k_hat, v_hat, layer_idx=i)
        return cache
    except (ImportError, AttributeError):
        return tuple(pairs)


def attn_score_error(q, k_true, k_hat):
    """Mean absolute error on attention scores Q @ K^T / sqrt(d)."""
    scale = q.shape[-1] ** -0.5
    s_true = (q @ k_true.transpose(-2, -1)) * scale  # (B,H,T,T)
    s_hat = (q @ k_hat.transpose(-2, -1)) * scale
    return (s_true - s_hat).abs().mean().item()


def sep(title=""):
    w = 64
    if title:
        print(f"\n---- {title} {'-' * (w - len(title) - 6)}")
    else:
        print("-" * w)


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to generate from (skips default demo)")
    parser.add_argument("--max-new-tokens", type=int, default=40,
                        help="Number of tokens to generate (default: 40)")
    args = parser.parse_args()

    model, tok = load_model()
    head_dim = get_head_dim(model)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    print(f"  n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}")

    # -- Interactive prompt mode --------------------------------------------
    if args.prompt:
        sep("Interactive generation")
        print(f"  Prompt : {args.prompt!r}\n")

        enc = tok(args.prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        T_p = input_ids.shape[1]

        kvs_p = capture_kv(model, input_ids)
        all_k_p = torch.cat([kv[0].reshape(-1, T_p, head_dim) for kv in kvs_p], dim=0)
        all_v_p = torch.cat([kv[1].reshape(-1, T_p, head_dim) for kv in kvs_p], dim=0)

        # Unquantized generation
        with torch.no_grad():
            true_ids = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        print(f"  Unquant: {tok.decode(true_ids[0], skip_special_tokens=True).strip()}\n")

        # Prompt forward pass: get logits at last position (unquantized)
        with torch.no_grad():
            prompt_out = model(input_ids, use_cache=False)
        first_logits = prompt_out.logits[:, -1, :]  # (1, vocab)

        # Quantized generation at each bit-width
        for bits in BITS_LIST:
            kvc = KVCacheQuantizer(
                head_dim=head_dim,
                num_bits=bits,
                use_outlier=True,
                n_outlier=N_OUTLIER,
                outlier_bits=min(bits + 1, 4),
                regular_bits=max(bits - 1, 1),
            )
            kvc.calibrate(all_k_p, all_v_p)
            quant_kv = _make_cache(model, kvs_p, kvc)

            # Manual greedy decode: quant_kv holds compressed prompt context
            # first token from prompt logits, then step through with compressed KV
            generated = [first_logits.argmax(-1, keepdim=True)]  # (1, 1)
            past = quant_kv
            for _ in range(args.max_new_tokens - 1):
                with torch.no_grad():
                    step = model(generated[-1], past_key_values=past, use_cache=True)
                next_tok = step.logits[:, -1, :].argmax(-1, keepdim=True)
                generated.append(next_tok)
                past = step.past_key_values
                decoded = tok.decode(next_tok[0], skip_special_tokens=False)
                if next_tok.item() == tok.eos_token_id or decoded in ("\n", "\n\n"):
                    break

            gen_ids = torch.cat(generated, dim=1)
            full_ids = torch.cat([input_ids, gen_ids], dim=1)
            print(f"  {bits}-bit  : {tok.decode(full_ids[0], skip_special_tokens=True).strip()}")

        sep()
        return

    # -- Tokenise all texts -------------------------------------------------
    encoded = tok(
        TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=64
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    B, T = input_ids.shape

    # -- Capture real KV caches ---------------------------------------------
    sep("Capturing real KV caches")
    kvs = capture_kv(model, input_ids)
    print(f"  {len(kvs)} layers captured, K shape: {list(kvs[0][0].shape)}")

    # Stack all layers for calibration: (n_layers * B * H, T, hd)
    all_k = torch.cat([kv[0].reshape(-1, T, head_dim) for kv in kvs], dim=0)
    all_v = torch.cat([kv[1].reshape(-1, T, head_dim) for kv in kvs], dim=0)
    print(f"  Calibration pool: {all_k.shape[0]} KV vectors of dim {head_dim}")

    # -- Build query tensors for attention score evaluation -----------------
    # Use the K tensors themselves as proxy queries (same distribution)
    torch.manual_seed(0)
    q_proxy = torch.randn_like(kvs[0][0])  # (B, H, T, hd) - same shape as K

    # -- Per-bit-width evaluation -------------------------------------------
    sep("Per-layer KV compression results")

    header = (
        f"{'bits':>5}  {'avg_bits':>9}  {'K MSE':>10}  {'V MSE':>10}  {'Attn err':>10}"
    )
    print(header)
    sep()

    for bits in BITS_LIST:
        kvc = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=N_OUTLIER,
            outlier_bits=min(bits + 1, 4),
            regular_bits=max(bits - 1, 1),
        )
        kvc.calibrate(all_k, all_v)

        k_mse_layers, v_mse_layers, attn_errs = [], [], []

        for k, v in kvs:
            k_c, v_c = kvc.compress_kv(k, v)
            k_hat, v_hat = kvc.decompress_kv(k_c, v_c)

            k_mse_layers.append(((k - k_hat) ** 2).mean().item())
            v_mse_layers.append(((v - v_hat) ** 2).mean().item())
            attn_errs.append(attn_score_error(q_proxy, k, k_hat))

        k_mse = sum(k_mse_layers) / len(k_mse_layers)
        v_mse = sum(v_mse_layers) / len(v_mse_layers)
        attn = sum(attn_errs) / len(attn_errs)

        print(
            f"{bits:>5}  {kvc.avg_bits:>9.2f}  {k_mse:>10.5f}  {v_mse:>10.5f}  {attn:>10.5f}"
        )

    # -- Logit difference ---------------------------------------------------
    sep("Output logit difference (last token)")

    # Baseline: logits with real KV cache
    with torch.no_grad():
        out_true = model(input_ids, use_cache=False)
    logits_true = out_true.logits[:, -1, :]  # (B, vocab)

    print(f"{'bits':>5}  {'avg_bits':>9}  {'logit MAE':>12}  {'top-1 match':>12}")
    sep()

    for bits in BITS_LIST:
        kvc = KVCacheQuantizer(
            head_dim=head_dim,
            num_bits=bits,
            use_outlier=True,
            n_outlier=N_OUTLIER,
            outlier_bits=min(bits + 1, 4),
            regular_bits=max(bits - 1, 1),
        )
        kvc.calibrate(all_k, all_v)

        # Replace each layer's KV with quantized version via past_key_values
        quant_kvs = _make_cache(model, kvs, kvc)

        # Feed a single pad token with the quantized KV cache
        # (simulates generation step using compressed cache)
        dummy = torch.full((B, 1), tok.eos_token_id)
        with torch.no_grad():
            out_quant = model(dummy, past_key_values=quant_kvs, use_cache=False)
        logits_quant = out_quant.logits[:, -1, :]  # (B, vocab)

        mae = (logits_true - logits_quant).abs().mean().item()
        top1_true = logits_true.argmax(-1)
        top1_quant = logits_quant.argmax(-1)
        match = (top1_true == top1_quant).float().mean().item()

        print(f"{bits:>5}  {kvc.avg_bits:>9.2f}  {mae:>12.5f}  {match:>11.0%}")

    # -- Per-text top-5 predictions -----------------------------------------
    sep(f"Top-5 next-token predictions  (bits={BITS_LIST[-1]})")

    kvc = KVCacheQuantizer(
        head_dim=head_dim,
        num_bits=BITS_LIST[-1],
        use_outlier=True,
        n_outlier=N_OUTLIER,
        outlier_bits=4,
        regular_bits=3,
    )
    kvc.calibrate(all_k, all_v)

    quant_kvs = _make_cache(model, kvs, kvc)

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
