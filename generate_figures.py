"""
Generate all figures for PAPER.md.
Run: python generate_figures.py
Outputs PNG files to figures/
"""

import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 — Codebook distribution: Gaussian vs sphere marginal
# ---------------------------------------------------------------------------


def sphere_marginal_pdf(t, d):
    """f(t) ∝ (1 - t²)^((d-3)/2) normalised over [-1, 1]."""
    from scipy.special import gamma

    vals = np.clip(1 - t**2, 0, None) ** ((d - 3) / 2)
    # normalise
    vals /= vals.sum() * (t[1] - t[0])
    return vals


def gaussian_pdf(t, d):
    sigma = 1 / math.sqrt(d)
    return np.exp(-0.5 * (t / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


t = np.linspace(-1, 1, 1000)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

for ax, d, title in zip(axes, [8, 64], ["d = 8  (low dim)", "d = 64  (high dim)"]):
    g = gaussian_pdf(t, d)
    g /= g.max()
    s = sphere_marginal_pdf(t, d)
    s /= s.max()

    ax.plot(t, g, label="Gaussian approx", color="#e07b54", linewidth=2, linestyle="--")
    ax.plot(t, s, label="True sphere marginal", color="#4c72b0", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("coordinate value $t$")
    ax.set_ylabel("density (normalised)")
    ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, None)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 1. True post-rotation coordinate distribution vs Gaussian approximation.\n"
    "At low $d$ the gap is large; Lloyd-Max centroids should be fitted to the true distribution.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("figures/fig1_distribution.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig1_distribution.png")


# ---------------------------------------------------------------------------
# Figure 2 — Attention-weighted quantization
# ---------------------------------------------------------------------------

tokens = [f"t{i}" for i in range(1, 9)]
attn = np.array([0.38, 0.22, 0.14, 0.09, 0.06, 0.05, 0.03, 0.03])
bits = np.array([4, 4, 4, 4, 2, 2, 2, 2])  # top-50% = 4-bit

colors = ["#4c72b0" if b == 4 else "#c0c0c0" for b in bits]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Left: attention weights
ax = axes[0]
bars = ax.bar(tokens, attn, color=colors, edgecolor="white", linewidth=0.8)
ax.axvline(x=3.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
ax.text(1.5, 0.33, "4-bit\n(top 50%)", ha="center", fontsize=9, color="#4c72b0")
ax.text(5.5, 0.33, "2-bit\n(bottom 50%)", ha="center", fontsize=9, color="#888888")
ax.set_xlabel("Token")
ax.set_ylabel("Attention weight $a_i$")
ax.set_title("Attention weights → bit assignment")
ax.set_ylim(0, 0.44)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: distortion comparison
ax = axes[1]
layers = [f"L{i}" for i in range(6)]
uniform_wd = [0.07099, 0.07603, 0.18383, 0.07854, 0.04318, 0.03048]
awq_wd = [0.03132, 0.03991, 0.05497, 0.03685, 0.01979, 0.01277]
x = np.arange(len(layers))
w = 0.35
ax.bar(
    x - w / 2, uniform_wd, w, label="Uniform 3-bit", color="#c0c0c0", edgecolor="white"
)
ax.bar(x + w / 2, awq_wd, w, label="AWQ 3-bit", color="#4c72b0", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.set_xlabel("Layer")
ax.set_ylabel("Attention-weighted distortion")
ax.set_title("Per-layer distortion: uniform vs AWQ")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 2. Attention-weighted quantization assigns more bits to high-attention tokens.\n"
    "Right: 47–70% distortion reduction per layer on distilgpt2 at the same 3-bit average.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("figures/fig2_awq.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig2_awq.png")


# ---------------------------------------------------------------------------
# Figure 3 — Low-rank correction concept
# ---------------------------------------------------------------------------


def draw_matrix(ax, x, y, w, h, color, label, sublabel=""):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2 + (0.05 if sublabel else 0),
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.12,
            sublabel,
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
        )


fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis("off")

# R = U @ Sigma @ Vh
draw_matrix(ax, 0.1, 0.3, 1.4, 1.4, "#f0e6d3", "R", "T × d\n(error)")
ax.text(1.7, 1.0, "≈", ha="center", va="center", fontsize=18)
draw_matrix(ax, 2.0, 0.3, 0.6, 1.4, "#cde6f5", "U", "T × r")
draw_matrix(ax, 2.8, 0.6, 0.8, 0.8, "#d5f0cd", "Σ", "r × r\ndiag")
draw_matrix(ax, 3.8, 0.85, 1.4, 0.3, "#f5cdd5", "Vᵀ", "r × d")

ax.text(5.4, 1.0, "→", ha="center", va="center", fontsize=16)

# Storage comparison
ax.text(5.9, 1.55, "Full residual", fontsize=9, color="#c0392b")
ax.text(5.9, 1.25, "T × d  =  360 × 64  =  23,040 floats", fontsize=9, color="#c0392b")
ax.plot([5.9, 9.6], [1.18, 1.18], color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(5.9, 0.90, "Rank-4 correction", fontsize=9, color="#27ae60")
ax.text(
    5.9, 0.60, "r(T+d) = 4×(360+64) = 1,696 floats  (7.4%)", fontsize=9, color="#27ae60"
)

fig.suptitle(
    "Figure 3. Low-rank correction stores U·Σ·Vᵀ instead of the full residual,\n"
    "capturing ~96% of error energy at 7.4% of the storage cost.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("figures/fig3_lowrank.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig3_lowrank.png")


# ---------------------------------------------------------------------------
# Figure 4 — PPL vs bit-width
# ---------------------------------------------------------------------------

models = ["distilgpt2", "gpt2-medium", "TinyLlama-1.1B"]
fp32_ppl = [33.51, 13.38, 4.78]

# ΔPPL without correction
delta_no = {
    "2-bit": [276.64, 173.61, 274.06],
    "3-bit": [21.44, 6.02, 0.87],
    "4-bit": [1.71, 1.10, 0.25],
}
# ΔPPL with rank-4 (TinyLlama omitted → NaN)
delta_r4 = {
    "2-bit+r4": [10.89, 5.95, None],
    "3-bit+r4": [4.27, 1.55, None],
    "4-bit+r4": [0.67, 0.47, None],
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: all models, all bit-widths (log scale for 2-bit)
ax = axes[0]
bw_labels = ["2-bit", "3-bit", "4-bit"]
colors_m = ["#4c72b0", "#e07b54", "#55a868"]
x = np.arange(len(bw_labels))
w = 0.25

for i, (model, color) in enumerate(zip(models, colors_m)):
    vals = [delta_no[bw][i] for bw in bw_labels]
    ax.bar(x + (i - 1) * w, vals, w, label=model, color=color, edgecolor="white")

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(bw_labels)
ax.set_ylabel("ΔPPL (log scale)")
ax.set_title("PPL degradation by bit-width")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: rank-4 correction effect on distilgpt2 and gpt2-medium
ax = axes[1]
bw_pairs = [("2-bit", "2-bit+r4"), ("3-bit", "3-bit+r4"), ("4-bit", "4-bit+r4")]
x = np.arange(len(bw_pairs))
w = 0.3

for i, (model, color) in enumerate(zip(models[:2], colors_m[:2])):
    base_vals = [delta_no[p[0]][i] for p in bw_pairs]
    r4_vals = [delta_r4[p[1]][i] for p in bw_pairs]
    ax.bar(
        x + (i - 0.5) * w,
        base_vals,
        w,
        color=color,
        alpha=0.35,
        edgecolor=color,
        linewidth=1.2,
        label=f"{model} (no corr)",
    )
    ax.bar(
        x + (i - 0.5) * w,
        r4_vals,
        w,
        color=color,
        edgecolor="white",
        label=f"{model} +rank-4",
    )

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(["2-bit", "3-bit", "4-bit"])
ax.set_ylabel("ΔPPL (log scale)")
ax.set_title("Effect of rank-4 correction")
ax.legend(fontsize=7.5, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 4. Left: PPL degradation across models and bit-widths (log scale).\n"
    "Right: rank-4 correction recovers 96–97% of 2-bit degradation.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("figures/fig4_ppl.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig4_ppl.png")


# ---------------------------------------------------------------------------
# Figure 5 — First-token fix (_crop_cache)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


def draw_box(ax, xy, w, h, text, color="#cde6f5", fontsize=9):
    x, y = xy
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor=color,
        edgecolor="#444444",
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=3,
        wrap=True,
        multialignment="center",
    )


def arrow(ax, x0, y0, x1, y1, color="black"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
    )


for ax, title, good in zip(axes, ["BEFORE (bug)", "AFTER (fix)"], [False, True]):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
        color="#c0392b" if not good else "#27ae60",
    )

    draw_box(ax, (1, 5.5), 2, 0.7, "Prompt tokens", "#f5e6cd")
    arrow(ax, 2, 5.5, 2, 5.0)
    draw_box(ax, (0.3, 4.1), 3.4, 0.8, "Prefill → FP32 KV cache", "#cde6f5")
    arrow(ax, 2, 4.1, 2, 3.5)
    draw_box(ax, (0.3, 2.6), 3.4, 0.8, "Quantize → quant KV cache", "#d5f0cd")

    if not good:
        # Bug: first logit comes from fp32 prefill output
        draw_box(
            ax,
            (0.3, 1.1),
            3.4,
            0.8,
            "first_logits = fp32 prefill output\n(SAME for all bit-widths!)",
            "#f5cdd5",
            fontsize=8,
        )
        arrow(ax, 2, 2.6, 2, 1.9)
        ax.text(
            2,
            0.6,
            "Token 1 identical\nfor 2-bit / 3-bit / 4-bit",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#c0392b",
            fontweight="bold",
        )
    else:
        # Fix: crop + re-run
        draw_box(
            ax,
            (0.3, 1.8),
            3.4,
            0.7,
            "crop cache to T_p−1\nre-run last prompt token",
            "#fffacc",
            fontsize=8,
        )
        draw_box(
            ax,
            (0.3, 0.8),
            3.4,
            0.7,
            "first_logits = quantized output\n(different per bit-width [OK])",
            "#d5f0cd",
            fontsize=8,
        )
        arrow(ax, 2, 2.6, 2, 2.5)
        arrow(ax, 2, 1.8, 2, 1.5)

fig.suptitle(
    "Figure 5. First-token fix: crop the quantized cache to T_p−1, re-run the last prompt token.\n"
    "Before: all bit-widths share the same unquantized logit. After: each uses its own quantized logit.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig("figures/fig5_firsttoken.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig5_firsttoken.png")

print("\nAll figures saved to figures/")
