"""
Real-world test of the 4 novel KVQuant extensions using distilgpt2 KV caches.

Run:  python -m kvquant.demo_extensions
"""

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import (
    KVQuantMSE,
    KVQuantIP,
    AttentionWeightedQuantizer,
    weighted_distortion,
    DeltaKVCache,
    AdaptiveKVCache,
    LowRankCorrection,
)

# ---------------------------------------------------------------------------
MODEL_NAME = "distilgpt2"
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "The history of the Roman Empire spans several centuries.",
    "In a hole in the ground there lived a hobbit.",
    "To be or not to be, that is the question.",
]
# ---------------------------------------------------------------------------


def sep(title=""):
    w = 66
    if title:
        print(f"\n---- {title} {'-' * (w - len(title) - 6)}")
    else:
        print("-" * w)


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    )
    model.eval()
    return model, tok


def capture_kv_and_attn(model, input_ids):
    """Return KV caches and attention weights per layer."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=True)

    pkv = out.past_key_values
    if hasattr(pkv, "key_cache"):
        kvs = [
            (pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache))
        ]
    else:
        kvs = [(item[0], item[1]) for item in pkv]

    # attentions: tuple of (B, H, T, T) per layer
    attns = out.attentions  # tuple of (B, H, T, T)
    return kvs, attns


def main():
    model, tok = load_model()
    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    n_heads = cfg.n_head
    n_layers = cfg.n_layer
    print(f"  n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}")

    encoded = tok(
        TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=48
    )
    input_ids = encoded["input_ids"]
    B, T = input_ids.shape

    kvs, attns = capture_kv_and_attn(model, input_ids)
    # kvs[l] = (k, v) each (B, H, T, d)
    # attns[l] = (B, H, T, T)

    # Flatten all layers into one pool for calibration
    all_k = torch.cat([kv[0].reshape(-1, T, head_dim) for kv in kvs], 0)
    all_v = torch.cat([kv[1].reshape(-1, T, head_dim) for kv in kvs], 0)
    print(f"  KV pool: {all_k.shape}  (layers×B×H, T, d)\n")

    # Baseline: uniform 3-bit KVQuantMSE
    baseline = KVQuantMSE(head_dim, num_bits=3)
    k0, v0 = kvs[0]  # use layer 0 as representative
    k0_hat = baseline(k0)
    base_mse = ((k0 - k0_hat) ** 2).mean().item()

    # -- 1. Attention-Weighted Quantization ---------------------------------
    sep("1. Attention-Weighted Quantization")
    print("  Compares uniform 3-bit vs AWQ (hi=4bit top-50%, lo=2bit rest)")
    print(
        f"\n  {'layer':>6}  {'uniform WD':>12}  {'AWQ WD':>10}  {'improvement':>12}  {'avg_bits':>9}"
    )
    sep()

    for l in range(n_layers):
        k, v = kvs[l]  # (B, H, T, d)
        attn = attns[l]  # (B, H, T, T)
        # Use the last query row = most recent token query
        q_vec = k[:, :, -1, :]  # (B, H, d) as proxy query

        awq = AttentionWeightedQuantizer(
            head_dim, hi_bits=4, lo_bits=2, top_fraction=0.5, seed=l
        )
        k_awq = awq(k, q_vec)

        k_uni = baseline(k)

        wd_uni = weighted_distortion(q_vec, k, k_uni).item()
        wd_awq = weighted_distortion(q_vec, k, k_awq).item()
        improvement = (wd_uni - wd_awq) / wd_uni * 100

        print(
            f"  {l:>6}  {wd_uni:>12.5f}  {wd_awq:>10.5f}  {improvement:>11.1f}%  {awq.avg_bits:>9.1f}"
        )

    # -- 2. Delta KV Cache ---------------------------------------------------
    sep("2. Delta KV Cache  (correlated token stream)")
    print("  Compresses k_t - k_{t-1} instead of k_t directly\n")
    print(
        f"  {'layer':>6}  {'standard MSE':>13}  {'delta MSE':>11}  {'speedup':>8}  {'delta norm avg':>15}"
    )
    sep()

    ip3 = KVQuantIP(head_dim, num_bits=3)

    for l in range(n_layers):
        k, _ = kvs[l]  # (B, H, T, d)
        # Treat each (B*H) row as one head stream
        k_bh = k.reshape(B * n_heads, T, head_dim)

        # Standard: compress each token independently
        k_flat = k_bh.reshape(-1, head_dim)
        std_mse = ((k_flat - ip3(k_flat)) ** 2).mean().item()

        # Delta: compress token-to-token differences
        cache = DeltaKVCache(head_dim, num_bits=3)
        for t in range(T):
            cache.push(k_bh[:, t, :], k_bh[:, t, :])

        k_rec, _ = cache.get()  # (T, B*H, d)
        k_rec = k_rec.permute(1, 0, 2)  # (B*H, T, d)
        delta_mse = ((k_bh - k_rec) ** 2).mean().item()

        ratio = std_mse / (delta_mse + 1e-9)
        dnorms = cache.delta_norms()
        mean_delta = dnorms.mean().item() if len(dnorms) else 0.0

        print(
            f"  {l:>6}  {std_mse:>13.5f}  {delta_mse:>11.5f}  {ratio:>7.1f}x  {mean_delta:>13.4f}"
        )

    # -- 3. Adaptive bit allocation ------------------------------------------
    sep("3. Adaptive Bit Allocation  (real attention scores)")
    print("  Tokens with high cumulative attention get more bits\n")

    # Use layer 0, all 5 sentences
    k, v = kvs[0]  # (B, H, T, d)
    attn_l = attns[0]  # (B, H, T, T)

    acache = AdaptiveKVCache(
        head_dim=head_dim,
        hi_bits=4,
        mid_bits=3,
        lo_bits=2,
        evict_bits=1,
        hi_threshold=0.08,
        lo_threshold=0.015,
        evict_threshold=0.002,
    )

    k_bh = k.reshape(B * n_heads, T, head_dim)
    for t in range(T):
        acache.push(k_bh[:, t, :], k_bh[:, t, :])

    # Feed real attention weights from each query position
    for qi in range(T):
        # attn_l[:, :, qi, :T] = attention from query qi to all keys
        w = attn_l[:, :, qi, :T].reshape(B * n_heads, T)
        acache.attend(w)

    alloc = acache.bit_allocation()
    avg_b = acache.avg_bits()
    k_rec_a, _ = acache.get()
    k_true = k_bh.permute(1, 0, 2)  # (T, B*H, d) to match get() output

    ada_mse = ((k_true - k_rec_a) ** 2).mean().item()
    uni_mse = ((k_bh - baseline(k_bh)) ** 2).mean().item()

    print(f"  Bit allocation after {T} attention steps:")
    for b_tier in sorted(alloc, reverse=True):
        bar = "#" * alloc[b_tier]
        print(f"    {b_tier}-bit: {alloc[b_tier]:>3} tokens  {bar}")
    print(
        f"\n  Avg bits: {avg_b:.2f}  |  Adaptive MSE: {ada_mse:.5f}"
        f"  |  Uniform 3-bit MSE: {uni_mse:.5f}"
    )

    # -- 4. Low-rank error correction ----------------------------------------
    sep("4. Low-Rank Error Correction")
    print("  Rank-r SVD of residual R = K - K_hat added on top of KVQuant\n")
    print(
        f"  {'bits':>5}  {'base MSE':>10}  {'rank-2':>8}  {'rank-4':>8}"
        f"  {'rank-8':>8}  {'storage(r=4)':>13}"
    )
    sep()

    for b in (2, 3, 4):
        base_q = KVQuantMSE(head_dim, b)
        k_mat = all_k.reshape(-1, head_dim)  # (N, d)

        mse_base = ((k_mat - base_q(k_mat)) ** 2).mean().item()

        mses = []
        for r in (2, 4, 8):
            corr = LowRankCorrection(base_q, rank=r)
            k_cor = corr(k_mat)
            mses.append(((k_mat - k_cor) ** 2).mean().item())

        ratio = LowRankCorrection(base_q, rank=4).storage_ratio(k_mat.shape[0])
        print(
            f"  {b:>5}  {mse_base:>10.5f}  {mses[0]:>8.5f}  {mses[1]:>8.5f}"
            f"  {mses[2]:>8.5f}  {ratio:>12.3f}x"
        )

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
