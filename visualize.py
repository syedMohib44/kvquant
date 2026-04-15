"""
KVQuant Visualizations.

Generates plots explaining how each component works using real distilgpt2 data.

Run:  python -m kvquant.visualize
Saves: kvquant/plots/*.png
"""

import math
import os
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import (
    KVQuantMSE,
    KVQuantIP,
    HadamardRotation,
    RandomRotation,
    PRECOMPUTED_CENTROIDS,
    build_codebook,
    AttentionWeightedQuantizer,
    weighted_distortion,
    DeltaKVCache,
    AdaptiveKVCache,
    LowRankCorrection,
    entropy_bits,
    analyse,
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "blue": "#2166ac",
    "green": "#1a9641",
    "orange": "#d73027",
    "red": "#d73027",
    "purple": "#762a83",
    "teal": "#4dac26",
    "yellow": "#f4a582",
    "gray": "#888888",
}


def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
def load_data():
    print("Loading distilgpt2...")
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", attn_implementation="eager"
    )
    model.eval()

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The history of the Roman Empire spans several centuries.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=48)
    with torch.no_grad():
        out = model(enc["input_ids"], output_attentions=True, use_cache=True)

    pkv = out.past_key_values
    if hasattr(pkv, "key_cache"):
        kvs = [
            (pkv.key_cache[i], pkv.value_cache[i]) for i in range(len(pkv.key_cache))
        ]
    else:
        kvs = [(item[0], item[1]) for item in pkv]

    cfg = model.config
    return kvs, out.attentions, cfg.n_embd // cfg.n_head, cfg.n_head, cfg.n_layer


# ---------------------------------------------------------------------------
# 1. MSE Distortion Bounds
# ---------------------------------------------------------------------------
def plot_mse_bounds():
    print("Plotting MSE bounds...")
    bits = [1, 2, 3, 4]
    lower = [1 / 4**b for b in bits]
    upper = [math.sqrt(3) * math.pi / 2 / 4**b for b in bits]

    torch.manual_seed(0)
    d = 128
    x = torch.randn(2000, d)
    x = x / x.norm(dim=-1, keepdim=True)

    actual_qr = []
    actual_had = []
    for b in bits:
        q_qr = KVQuantMSE(d, b, use_hadamard=False)
        q_had = KVQuantMSE(d, b, use_hadamard=True)
        actual_qr.append(((x - q_qr(x)) ** 2).mean(-1).mean().item())
        actual_had.append(((x - q_had(x)) ** 2).mean(-1).mean().item())

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(bits))
    w = 0.2

    ax.bar(
        x_pos - w * 1.5,
        lower,
        w,
        label="Informal lower bound (1/4^b)",
        color=COLORS["gray"],
        alpha=0.8,
    )
    ax.bar(
        x_pos - w * 0.5,
        upper,
        w,
        label="Paper upper bound",
        color=COLORS["blue"],
        alpha=0.8,
    )
    ax.bar(
        x_pos + w * 0.5,
        actual_qr,
        w,
        label="KVQuant QR (ours)",
        color=COLORS["green"],
        alpha=0.8,
    )
    ax.bar(
        x_pos + w * 1.5,
        actual_had,
        w,
        label="KVQuant Hadamard (ours)",
        color=COLORS["orange"],
        alpha=0.8,
    )

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.set_ylabel("Per-element MSE  ||x - x_hat||^2 / d")
    ax.set_title("MSE Distortion vs Bit-Width (unit sphere vectors, d=128)")
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    savefig("1_mse_bounds.png")


# ---------------------------------------------------------------------------
# 2. Codebook Centroids
# ---------------------------------------------------------------------------
def plot_codebook():
    print("Plotting codebook...")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    d = 128

    # Sample the TRUE sphere marginal - same method used to fit the codebook
    torch.manual_seed(42)
    g = torch.randn(50000, d)
    u = g / g.norm(dim=-1, keepdim=True)
    x_sphere = u[:, 0].numpy()  # one coordinate of a uniform unit vector

    # Gaussian approximation for comparison
    x_gauss = (torch.randn(50000) / math.sqrt(d)).numpy()

    for i, b in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        centroids = build_codebook(b, d).numpy()

        ax.hist(
            x_sphere,
            bins=80,
            density=True,
            color=COLORS["blue"],
            alpha=0.5,
            label="True sphere marginal",
        )
        ax.hist(
            x_gauss,
            bins=80,
            density=True,
            color=COLORS["gray"],
            alpha=0.3,
            label="N(0,1/d) approx",
        )
        for c in centroids:
            ax.axvline(c, color=COLORS["orange"], linewidth=2, alpha=0.9)
        ax.scatter(
            centroids,
            np.zeros_like(centroids) - 0.5,
            color=COLORS["orange"],
            s=60,
            zorder=5,
            label="Centroids (fitted to sphere)",
        )

        ax.set_title(f"b={b}  ({2**b} centroids)")
        ax.set_xlabel("Coordinate value")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(axis="y")

    fig.suptitle(
        "Lloyd-Max Codebook: Centroids fitted to True Sphere Marginal (not Gaussian)",
        y=1.02,
    )
    plt.tight_layout()
    savefig("2_codebook_centroids.png")


# ---------------------------------------------------------------------------
# 3. Rotation Effect
# ---------------------------------------------------------------------------
def plot_rotation(kvs, head_dim):
    print("Plotting rotation effect...")
    k, _ = kvs[2]  # layer 2 has most variance
    k_flat = k.reshape(-1, head_dim).float()
    k_unit = k_flat / k_flat.norm(dim=-1, keepdim=True)

    rot_qr = RandomRotation(head_dim, seed=0)
    rot_had = HadamardRotation(head_dim, seed=0)

    y_qr = rot_qr(k_unit)
    y_had = rot_had(k_unit)

    # Pick one coordinate
    coord_before = k_unit[:, 0].detach().numpy()
    coord_qr = y_qr[:, 0].detach().numpy()
    coord_had = y_had[:, 0].detach().numpy()
    expected_std = 1.0 / math.sqrt(head_dim)

    x_range = np.linspace(-0.4, 0.4, 200)
    gauss_pdf = (1 / (expected_std * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * (x_range / expected_std) ** 2
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bins = 50

    axes[0].hist(coord_before, bins=bins, density=True, color=COLORS["red"], alpha=0.8)
    axes[0].set_title("Before Rotation\n(raw KV coord)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")

    axes[1].hist(
        coord_qr,
        bins=bins,
        density=True,
        color=COLORS["green"],
        alpha=0.8,
        label="QR rotated",
    )
    axes[1].plot(
        x_range, gauss_pdf, color=COLORS["yellow"], linewidth=2, label="N(0,1/d)"
    )
    axes[1].set_title("After QR Rotation\n(O(d^2))")
    axes[1].set_xlabel("Value")
    axes[1].legend(fontsize=8)

    axes[2].hist(
        coord_had,
        bins=bins,
        density=True,
        color=COLORS["purple"],
        alpha=0.8,
        label="Hadamard rotated",
    )
    axes[2].plot(
        x_range, gauss_pdf, color=COLORS["yellow"], linewidth=2, label="N(0,1/d)"
    )
    axes[2].set_title("After Hadamard Rotation\n(O(d log d))")
    axes[2].set_xlabel("Value")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis="y")

    fig.suptitle("Rotation Transforms KV Coordinates into N(0,1/d)", y=1.02)
    plt.tight_layout()
    savefig("3_rotation_effect.png")


# ---------------------------------------------------------------------------
# 4. Attention-Weighted Quantization
# ---------------------------------------------------------------------------
def plot_awq(kvs, attns, head_dim, n_heads):
    print("Plotting attention-weighted quantization...")
    layer = 2
    k, _ = kvs[layer]  # (B, H, T, d)
    B, H, T, d = k.shape
    q_vec = k[:, :, -1, :]  # last token as query

    attn_w = attns[layer][:, :, -1, :]  # (B, H, T) - attn from last query
    attn_mean = attn_w.mean(dim=(0, 1)).detach().numpy()  # (T,)

    awq = AttentionWeightedQuantizer(d, hi_bits=4, lo_bits=2, top_fraction=0.5)
    qz = awq.quantize(k, q_vec)
    top_mask = qz.top_mask.reshape(B * H, T).float().mean(0).numpy()  # (T,)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Attention weights
    axes[0].bar(
        range(T),
        attn_mean,
        color=[
            COLORS["orange"] if top_mask[t] > 0.5 else COLORS["blue"] for t in range(T)
        ],
    )
    axes[0].set_title(f"Layer {layer}: Attention Weights from Last Query Token")
    axes[0].set_ylabel("Attention weight")
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(color=COLORS["orange"], label="4-bit (hi-attention)"),
            Patch(color=COLORS["blue"], label="2-bit (lo-attention)"),
        ],
        fontsize=8,
    )

    # Bit assignment
    bits_assigned = np.where(top_mask > 0.5, 4, 2)
    axes[1].bar(
        range(T),
        bits_assigned,
        color=[COLORS["orange"] if b == 4 else COLORS["blue"] for b in bits_assigned],
    )
    axes[1].set_yticks([2, 4])
    axes[1].set_title("Bit Assignment per Token Position")
    axes[1].set_ylabel("Bits")
    axes[1].set_ylim(0, 5)

    # Per-token attention-weighted distortion comparison
    base = KVQuantMSE(d, 3)
    k_uni = base(k)
    k_awq = awq(k, q_vec)
    k_flat = k.reshape(B * H, T, d)

    # Attention weights from last query (B*H, T)
    scores = (q_vec.reshape(B * H, 1, d) @ k_flat.transpose(-2, -1)).squeeze(1)
    attn_tok = F.softmax(scores / math.sqrt(d), dim=-1)  # (B*H, T)

    err_uni = ((k_flat - k_uni.reshape(B * H, T, d)) ** 2).mean(-1)  # (B*H, T)
    err_awq = ((k_flat - k_awq.reshape(B * H, T, d)) ** 2).mean(-1)  # (B*H, T)

    # Weight each token's error by its attention probability
    wdist_uni = (attn_tok * err_uni).mean(0).detach().numpy()  # (T,)
    wdist_awq = (attn_tok * err_awq).mean(0).detach().numpy()  # (T,)

    axes[2].plot(
        range(T),
        wdist_uni,
        color=COLORS["blue"],
        linewidth=2,
        label="Uniform 3-bit (attn-weighted error)",
        marker="o",
        markersize=4,
    )
    axes[2].plot(
        range(T),
        wdist_awq,
        color=COLORS["orange"],
        linewidth=2,
        label="AWQ hi=4,lo=2 (attn-weighted error)",
        marker="s",
        markersize=4,
    )
    # Only shade where AWQ is strictly better (clip negatives so shading is never inverted)
    improvement_lo = np.minimum(wdist_awq, wdist_uni)
    improvement_hi = np.maximum(wdist_awq, wdist_uni)
    awq_better = wdist_awq <= wdist_uni
    axes[2].fill_between(
        range(T),
        improvement_lo,
        improvement_hi,
        where=awq_better,
        alpha=0.25,
        color=COLORS["green"],
        label="AWQ improvement",
    )
    axes[2].fill_between(
        range(T),
        improvement_lo,
        improvement_hi,
        where=~awq_better,
        alpha=0.2,
        color=COLORS["orange"],
        label="Uniform better (token de-prioritised)",
    )
    axes[2].set_title("Per-Token Attention-Weighted Error: Uniform vs AWQ")
    axes[2].set_ylabel("Attn-weighted MSE")
    axes[2].set_xlabel("Token position")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis="y")

    plt.tight_layout()
    savefig("4_attention_weighted.png")


# ---------------------------------------------------------------------------
# 5. Delta Compression
# ---------------------------------------------------------------------------
def plot_delta(kvs, head_dim, n_heads):
    print("Plotting delta compression...")
    layer = 1
    k, _ = kvs[layer]  # (B, H, T, d)
    B, H, T, d = k.shape
    k_bh = k.reshape(B * H, T, d)

    # Token norms
    token_norms = k_bh.norm(dim=-1).mean(0).detach().numpy()  # (T,)

    # Delta norms
    delta_norms = (
        (k_bh[:, 1:, :] - k_bh[:, :-1, :]).norm(dim=-1).mean(0).detach().numpy()
    )

    # MSE comparison per token
    ip3 = KVQuantIP(d, 3)
    cache = DeltaKVCache(d, 3)
    for t in range(T):
        cache.push(k_bh[:, t, :], k_bh[:, t, :])
    k_rec, _ = cache.get()
    k_rec = k_rec.permute(1, 0, 2)

    mse_std = ((k_bh - ip3(k_bh)) ** 2).mean(-1).mean(0).detach().numpy()
    mse_delta = ((k_bh - k_rec) ** 2).mean(-1).mean(0).detach().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Token vs delta norms
    axes[0].plot(
        range(T),
        token_norms,
        color=COLORS["blue"],
        linewidth=2,
        label="Token norm ||k_t||",
        marker="o",
        markersize=4,
    )
    axes[0].plot(
        range(1, T),
        delta_norms,
        color=COLORS["orange"],
        linewidth=2,
        label="Delta norm ||k_t - k_{t-1}||",
        marker="s",
        markersize=4,
    )
    axes[0].set_title(f"Layer {layer}: Token Norms vs Delta Norms")
    axes[0].set_ylabel("L2 Norm")
    axes[0].legend(fontsize=8)
    axes[0].set_yscale("log")

    # Compression ratio illustration
    ratio_text = "Compressing delta = fewer bits for same quality"
    axes[1].fill_between(
        range(T),
        token_norms,
        alpha=0.4,
        color=COLORS["blue"],
        label="Full vector budget",
    )
    axes[1].fill_between(
        range(1, T),
        delta_norms,
        alpha=0.5,
        color=COLORS["green"],
        label="Delta budget (much smaller)",
    )
    axes[1].set_title(
        "Budget Needed for Same Quality\n(smaller magnitude = less distortion at same bits)"
    )
    axes[1].set_ylabel("Magnitude (proxy for bits needed)")
    axes[1].legend(fontsize=8)

    # MSE per token
    axes[2].plot(
        range(T),
        mse_std,
        color=COLORS["blue"],
        linewidth=2,
        label="Standard KVQuant (3-bit)",
        marker="o",
        markersize=4,
    )
    axes[2].plot(
        range(T),
        mse_delta,
        color=COLORS["green"],
        linewidth=2,
        label="Delta KVQuant (3-bit)",
        marker="s",
        markersize=4,
    )
    axes[2].fill_between(
        range(T),
        mse_delta,
        mse_std,
        alpha=0.15,
        color=COLORS["teal"],
        label="Improvement",
    )
    axes[2].set_title("Per-Token Reconstruction MSE")
    axes[2].set_ylabel("MSE")
    axes[2].set_xlabel("Token position")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis="y")

    plt.tight_layout()
    savefig("5_delta_compression.png")


# ---------------------------------------------------------------------------
# 6. Adaptive Bit Allocation
# ---------------------------------------------------------------------------
def plot_adaptive(kvs, attns, head_dim, n_heads):
    print("Plotting adaptive bit allocation...")
    layer = 0
    k, _ = kvs[layer]
    B, H, T, d = k.shape

    # Use a longer synthetic sequence so tier differentiation visibly activates.
    # Build T_long tokens: first few get high repeated attention (become hi-bit),
    # middle tokens get moderate attention (mid-bit), late tokens get almost none (lo/evict).
    T_long = 48
    torch.manual_seed(7)
    k_syn = torch.randn(1, T_long, d)  # (1 batch-head, T, d)

    # Synthetic attention: token 0 always gets 40%, tokens 1-7 share 40%, rest share 20%
    def make_attn(t_query, T_ctx):
        w = torch.zeros(1, T_ctx)
        w[0, 0] = 0.40
        mid = min(8, T_ctx)
        w[0, 1:mid] = 0.40 / max(mid - 1, 1)
        if T_ctx > mid:
            w[0, mid:] = 0.20 / (T_ctx - mid)
        # Add mild random noise so scores drift
        w = w + torch.rand_like(w) * 0.02
        w = w / w.sum()
        return w

    acache = AdaptiveKVCache(
        head_dim=d,
        hi_bits=4,
        mid_bits=3,
        lo_bits=2,
        evict_bits=1,
        hi_threshold=0.08,
        lo_threshold=0.015,
        evict_threshold=0.002,
    )
    for t in range(T_long):
        acache.push(k_syn[:, t, :], k_syn[:, t, :])

    # Record bit allocations after each attention step
    bit_history = []
    score_history = []
    for qi in range(T_long):
        w = make_attn(qi, T_long)
        acache.attend(w)
        bits_now = [e.bits for e in acache._k_entries]
        scores_now = [e.score for e in acache._k_entries]
        bit_history.append(bits_now)
        score_history.append(scores_now)

    T = T_long

    bit_arr = np.array(bit_history)  # (T_steps, T_tokens)
    score_arr = np.array(score_history)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Heatmap of bit assignments over time
    im = axes[0].imshow(
        bit_arr.T, aspect="auto", cmap="RdYlGn", vmin=1, vmax=4, origin="lower"
    )
    plt.colorbar(im, ax=axes[0], label="Bits")
    axes[0].set_title("Bit Allocation per Token Over Generation Steps")
    axes[0].set_ylabel("Token position")
    axes[0].set_xlabel("Generation step")

    # Importance scores over time
    im2 = axes[1].imshow(score_arr.T, aspect="auto", cmap="Blues", origin="lower")
    plt.colorbar(im2, ax=axes[1], label="Importance score")
    axes[1].set_title("Token Importance Scores (EMA of attention weights)")
    axes[1].set_ylabel("Token position")
    axes[1].set_xlabel("Generation step")

    # Average bits over time
    avg_bits = bit_arr.mean(axis=1)
    axes[2].plot(
        range(T),
        avg_bits,
        color=COLORS["green"],
        linewidth=2,
        marker="o",
        markersize=4,
        label="Adaptive avg bits",
    )
    axes[2].axhline(
        3,
        color=COLORS["blue"],
        linewidth=1.5,
        linestyle="--",
        label="Uniform 3-bit baseline",
    )
    axes[2].set_title("Average Bits per Generation Step")
    axes[2].set_ylabel("Avg bits/token")
    axes[2].set_xlabel("Generation step")
    axes[2].set_ylim(0, 5)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y")

    plt.tight_layout()
    savefig("6_adaptive_allocation.png")


# ---------------------------------------------------------------------------
# 7. Low-Rank Error Correction
# ---------------------------------------------------------------------------
def plot_correction(kvs, head_dim):
    print("Plotting low-rank correction...")
    k_all = torch.cat(
        [kv[0].reshape(-1, kvs[0][0].shape[-2], head_dim) for kv in kvs], 0
    )
    k_flat = k_all.reshape(-1, head_dim)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Singular value spectrum of residual
    base2 = KVQuantMSE(head_dim, 2)
    R = (k_flat - base2(k_flat)).reshape(1, -1, head_dim)
    _, S, _ = torch.linalg.svd(R, full_matrices=False)
    S = S.squeeze(0).detach().numpy()
    energy = S**2 / (S**2).sum()
    cum_energy = np.cumsum(energy[:32])

    axes[0].bar(
        range(1, 17),
        energy[:16] * 100,
        color=COLORS["blue"],
        alpha=0.8,
        label="Per-rank energy",
    )
    axes[0].plot(
        range(1, 17),
        cum_energy[:16] * 100,
        color=COLORS["orange"],
        linewidth=2,
        marker="o",
        markersize=5,
        label="Cumulative energy",
    )
    axes[0].axvline(
        4,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.5,
        label="rank=4 (our default)",
    )
    axes[0].set_title("Residual Energy by SVD Rank\n(2-bit base)")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Energy (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y")

    # MSE vs rank for each bit-width
    ranks = [0, 1, 2, 4, 8, 16]
    for b, col in zip([2, 3, 4], [COLORS["red"], COLORS["blue"], COLORS["green"]]):
        base_q = KVQuantMSE(head_dim, b)
        mses = []
        for r in ranks:
            if r == 0:
                m = ((k_flat - base_q(k_flat)) ** 2).mean().item()
            else:
                corr = LowRankCorrection(base_q, rank=r)
                m = ((k_flat - corr(k_flat)) ** 2).mean().item()
            mses.append(m)
        axes[1].plot(
            ranks,
            mses,
            color=col,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{b}-bit",
        )

    axes[1].set_title("MSE vs Correction Rank")
    axes[1].set_xlabel("Rank r")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y")

    # Storage cost vs quality (rank=4 across bit-widths)
    T_vals = [12, 64, 128, 256, 512]
    ratios = [(T_ * 4 + head_dim * 4) / (T_ * head_dim) for T_ in T_vals]
    axes[2].plot(
        T_vals,
        [r * 100 for r in ratios],
        color=COLORS["purple"],
        linewidth=2,
        marker="o",
        markersize=6,
    )
    axes[2].fill_between(
        T_vals, [r * 100 for r in ratios], alpha=0.2, color=COLORS["purple"]
    )
    axes[2].set_title("Correction Storage Overhead (rank=4)\nvs Sequence Length")
    axes[2].set_xlabel("Sequence length T")
    axes[2].set_ylabel("Extra storage (% of KV)")
    axes[2].grid(axis="y")
    axes[2].set_xscale("log")

    plt.tight_layout()
    savefig("7_low_rank_correction.png")


# ---------------------------------------------------------------------------
# 8. Entropy Coding Savings
# ---------------------------------------------------------------------------
def plot_entropy():
    print("Plotting entropy coding savings...")
    dims = [32, 64, 128, 256]
    bits_list = [1, 2, 3, 4]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Savings vs bit-width for d=128
    d = 128
    savings = [analyse(b, d).saving_pct for b in bits_list]
    entropy = [entropy_bits(b, d) for b in bits_list]

    axes[0].bar(
        [f"{b}-bit" for b in bits_list],
        savings,
        color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]],
        alpha=0.85,
    )
    for i, (b, s) in enumerate(zip(bits_list, savings)):
        axes[0].text(i, s + 0.05, f"{s:.1f}%", ha="center", fontsize=10, color="black")
    axes[0].set_title(
        "Huffman Coding Savings vs Bit-Width\n(d=128, distilgpt2 distribution)"
    )
    axes[0].set_ylabel("Bit savings (%)")
    axes[0].set_ylim(0, 8)
    axes[0].grid(axis="y")

    # Raw vs entropy vs huffman
    for b, col in zip(
        bits_list, [COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"]]
    ):
        s = analyse(b, d)
        axes[1].plot(
            b, s.raw_bits, marker="o", color=col, markersize=10, linestyle="none"
        )
        axes[1].plot(
            b, s.entropy, marker="^", color=col, markersize=10, linestyle="none"
        )
        axes[1].plot(
            b, s.huffman_avg, marker="s", color=col, markersize=10, linestyle="none"
        )

    from matplotlib.lines import Line2D

    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markersize=8,
                linestyle="none",
                label="Raw bits",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="black",
                markersize=8,
                linestyle="none",
                label="Shannon entropy",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="black",
                markersize=8,
                linestyle="none",
                label="Huffman avg",
            ),
        ],
        fontsize=8,
    )
    axes[1].set_title("Raw vs Entropy vs Huffman Bits per Symbol")
    axes[1].set_xlabel("Bits")
    axes[1].set_ylabel("Bits per coordinate")
    axes[1].grid(axis="y")
    axes[1].set_xticks(bits_list)

    plt.tight_layout()
    savefig("8_entropy_coding.png")


# ---------------------------------------------------------------------------
# 9. Full Pipeline Comparison
# ---------------------------------------------------------------------------
def plot_pipeline(kvs, attns, head_dim, n_heads):
    print("Plotting full pipeline comparison...")
    layer = 2
    k, _ = kvs[layer]
    B, H, T, d = k.shape
    q_vec = k[:, :, -1, :]
    k_flat = k.reshape(-1, head_dim)

    configs = {
        "2-bit plain": KVQuantMSE(d, 2),
        "3-bit plain": KVQuantMSE(d, 3),
        "4-bit plain": KVQuantMSE(d, 4),
        "3-bit + AWQ*": None,
        "3-bit + LR(r=4)": LowRankCorrection(KVQuantMSE(d, 3), rank=4),
    }

    mses = []
    labels = []

    for name, q in configs.items():
        if name == "3-bit + AWQ*":
            awq = AttentionWeightedQuantizer(d, 4, 2, 0.5)
            k_h = awq(k, q_vec)
        else:
            k_h = q(k_flat).reshape(k.shape)
        mse = weighted_distortion(q_vec, k, k_h).item()
        mses.append(mse)
        labels.append(name)

    colors = [
        COLORS["red"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["purple"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, mses, color=colors, alpha=0.85)
    for bar, mse in zip(bars, mses):
        ax.text(
            mse + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{mse:.4f}",
            va="center",
            fontsize=9,
            color="black",
        )

    ax.set_xlabel("Attention-Weighted MSE (lower = better; * = AWQ's native metric)")
    ax.set_title(f"Layer {layer} KV Reconstruction MSE: Method Comparison")
    ax.grid(axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    savefig("9_pipeline_comparison.png")


# ---------------------------------------------------------------------------
# 10. Distortion Ratio Check  (2.7× claim)
# ---------------------------------------------------------------------------
def plot_ratio_check():
    print("Plotting distortion ratio check...")
    bits_list = [1, 2, 3, 4]
    d = 128
    torch.manual_seed(0)
    x = torch.randn(5000, d)
    x = x / x.norm(dim=-1, keepdim=True)

    ratios_qr, ratios_had = [], []
    for b in bits_list:
        lower = 1.0 / (4**b)
        q_qr = KVQuantMSE(d, b, use_hadamard=False)
        q_had = KVQuantMSE(d, b, use_hadamard=True)
        mse_qr = ((x - q_qr(x)) ** 2).mean(-1).mean().item()
        mse_had = ((x - q_had(x)) ** 2).mean(-1).mean().item()
        ratios_qr.append(mse_qr / lower)
        ratios_had.append(mse_had / lower)

    bound = math.sqrt(3) * math.pi / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(bits_list))
    w = 0.3
    ax.bar(
        x_pos - w / 2,
        ratios_qr,
        w,
        label="KVQuant QR",
        color=COLORS["green"],
        alpha=0.85,
    )
    ax.bar(
        x_pos + w / 2,
        ratios_had,
        w,
        label="KVQuant Hadamard",
        color=COLORS["orange"],
        alpha=0.85,
    )
    ax.axhline(
        bound,
        color=COLORS["red"],
        linestyle="--",
        linewidth=2,
        label=f"Paper upper bound (sqrt3·pi/2 ≈ {bound:.2f})",
    )
    ax.axhline(
        1.0,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.5,
        label="Lower bound (1.0)",
    )

    for i, (r_qr, r_had) in enumerate(zip(ratios_qr, ratios_had)):
        ax.text(
            i - w / 2, r_qr * 1.15, f"{r_qr:.3f}", ha="center", va="bottom", fontsize=8
        )
        ax.text(
            i + w / 2,
            r_had * 1.15,
            f"{r_had:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.001, bound * 5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b}-bit" for b in bits_list])
    ax.set_ylabel("Distortion ratio   actual_MSE / (1/4^b)   [log scale]")
    ax.set_title(
        "2.7× Bound Verification: Distortion Ratio vs Bit-Width\n(d=128 unit sphere - ratio must stay ≤ sqrt3·pi/2 ≈ 2.72)"
    )
    ax.legend()
    ax.grid(axis="y", which="both")
    plt.tight_layout()
    savefig("10_ratio_check.png")


# ---------------------------------------------------------------------------
# 11. Perplexity Validation  (3.5-bit neutral / 2.5-bit marginal)
# ---------------------------------------------------------------------------
def plot_perplexity_validation():
    print("Plotting perplexity validation...")
    from kvquant import KVCacheQuantizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", attn_implementation="eager"
    )
    model.eval()

    texts = [
        "The quick brown fox jumps over the lazy dog and ran away.",
        "Artificial intelligence is transforming the world in remarkable ways.",
        "The history of the Roman Empire spans several centuries of conquest.",
        "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole.",
        "To be or not to be, that is the question every philosopher asks.",
    ]

    # (label, use_outlier, num_bits, outlier_bits, regular_bits)
    configs = [
        ("2-bit", False, 2, None, None),
        ("2.5-bit", True, 3, 3, 2),
        ("3-bit", False, 3, None, None),
        ("3.5-bit", True, 3, 4, 3),
        ("4-bit", False, 4, None, None),
    ]

    head_dim = model.config.n_embd // model.config.n_head  # 64

    def _run_prefix(prefix_ids):
        """Run prefix forward; return (pkv, kvs_list) where kvs_list[i] = (K, V) per layer."""
        with torch.no_grad():
            pkv = model(prefix_ids, use_cache=True).past_key_values
        kvs = [
            (pkv.layers[i].keys, pkv.layers[i].values) for i in range(len(pkv.layers))
        ]
        return pkv, kvs

    def _inject_quantized(prefix_ids, pairs):
        """Fresh prefix cache with K,V replaced by quantized pairs."""
        with torch.no_grad():
            pkv_q = model(prefix_ids, use_cache=True).past_key_values
        for i, (k_hat, v_hat) in enumerate(pairs):
            pkv_q.layers[i].keys = k_hat
            pkv_q.layers[i].values = v_hat
        return pkv_q

    # -- Process each text individually (no padding) -----------------------
    baseline_nlls = []
    config_nlls = {name: [] for name, *_ in configs}

    for text in texts:
        ids = tok(text, return_tensors="pt")["input_ids"]  # (1, T)
        T = ids.shape[1]
        T_split = T // 2
        if T_split < 2:
            continue

        prefix_ids = ids[:, :T_split]
        suffix_ids = ids[:, T_split:]

        # -- Baseline: fresh prefix run, unquantized -----------------------
        pkv_bl, kvs_ref = _run_prefix(prefix_ids)
        with torch.no_grad():
            bl = model(suffix_ids, past_key_values=pkv_bl, labels=suffix_ids)
        baseline_nlls.append(bl.loss.item())

        for name, use_outlier, num_bits, ob, rb in configs:
            kvc = KVCacheQuantizer(
                head_dim=head_dim,
                num_bits=num_bits,
                use_outlier=use_outlier,
                n_outlier=16,
                outlier_bits=ob,
                regular_bits=rb,
            )
            # calibrate always (required even for non-outlier configs)
            k_cal = torch.cat([kv[0] for kv in kvs_ref], dim=1)
            v_cal = torch.cat([kv[1] for kv in kvs_ref], dim=1)
            kvc.calibrate(k_cal, v_cal)

            pairs = []
            for k, v in kvs_ref:
                k_c, v_c = kvc.compress_kv(k, v)
                pairs.append(kvc.decompress_kv(k_c, v_c))
            pkv_q = _inject_quantized(prefix_ids, pairs)

            with torch.no_grad():
                qo = model(suffix_ids, past_key_values=pkv_q, labels=suffix_ids)
            config_nlls[name].append(qo.loss.item())

    baseline_ppl = math.exp(sum(baseline_nlls) / len(baseline_nlls))
    labels = [name for name, *_ in configs]
    ppls = [math.exp(sum(config_nlls[n]) / len(config_nlls[n])) for n in labels]

    # -- Plot --------------------------------------------------------------
    bar_colors = [
        COLORS["red"],
        COLORS["orange"],
        COLORS["blue"],
        COLORS["teal"],
        COLORS["green"],
    ]
    x_pos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute PPL
    axes[0].bar(x_pos, ppls, color=bar_colors, alpha=0.85)
    axes[0].axhline(
        baseline_ppl,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Baseline (no quant) = {baseline_ppl:.1f}",
    )
    for i, p in enumerate(ppls):
        axes[0].text(i, p + 0.3, f"{p:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Perplexity (lower = better)")
    axes[0].set_title("Suffix Perplexity: Quantized KV Cache vs Baseline")
    axes[0].legend()
    axes[0].grid(axis="y")

    # Right: ratio PPL / baseline
    rel = [p / baseline_ppl for p in ppls]
    axes[1].bar(x_pos, rel, color=bar_colors, alpha=0.85)
    axes[1].axhline(
        1.0, color="black", linestyle="--", linewidth=2, label="Baseline (1.00×)"
    )
    axes[1].axhline(
        1.05, color=COLORS["gray"], linestyle=":", linewidth=1.5, label="5% degradation"
    )
    for i, v in enumerate(rel):
        axes[1].text(i, v + 0.005, f"{v:.2f}×", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("PPL ratio  (quantized / baseline)")
    axes[1].set_title(
        "Relative PPL Degradation\n(paper: 3.5-bit ≈ neutral, 2.5-bit ≈ marginal)"
    )
    axes[1].legend()
    axes[1].grid(axis="y")

    plt.tight_layout()
    savefig("11_perplexity_validation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    kvs, attns, head_dim, n_heads, n_layers = load_data()
    print(f"  head_dim={head_dim}, n_heads={n_heads}, n_layers={n_layers}\n")

    plot_mse_bounds()
    plot_codebook()
    plot_rotation(kvs, head_dim)
    plot_awq(kvs, attns, head_dim, n_heads)
    plot_delta(kvs, head_dim, n_heads)
    plot_adaptive(kvs, attns, head_dim, n_heads)
    plot_correction(kvs, head_dim)
    plot_entropy()
    plot_pipeline(kvs, attns, head_dim, n_heads)
    plot_ratio_check()
    plot_perplexity_validation()

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
