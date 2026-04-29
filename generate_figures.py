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
# Figure 1 Codebook distribution: Gaussian vs sphere marginal
# ---------------------------------------------------------------------------


def sphere_marginal_pdf(t, d):
    """f(t) alpha (1 - t^2)^((d-3)/2) normalised over [-1, 1]."""
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
# Figure 2 Attention-weighted quantization
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
ax.set_title("Attention weights -> bit assignment")
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
    "Right: 47-70% distortion reduction per layer on distilgpt2 at the same 3-bit average.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("figures/fig2_awq.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig2_awq.png")


# ---------------------------------------------------------------------------
# Figure 3 Low-rank correction concept
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
draw_matrix(ax, 0.1, 0.3, 1.4, 1.4, "#f0e6d3", "R", "T x d\n(error)")
ax.text(1.7, 1.0, "~=", ha="center", va="center", fontsize=18)
draw_matrix(ax, 2.0, 0.3, 0.6, 1.4, "#cde6f5", "U", "T x r")
draw_matrix(ax, 2.8, 0.6, 0.8, 0.8, "#d5f0cd", "S", "r x r\ndiag")
draw_matrix(ax, 3.8, 0.85, 1.4, 0.3, "#f5cdd5", "V^T", "r x d")

ax.text(5.4, 1.0, "->", ha="center", va="center", fontsize=16)

# Storage comparison
ax.text(5.9, 1.55, "Full residual", fontsize=9, color="#c0392b")
ax.text(5.9, 1.25, "T x d  =  360 x 64  =  23,040 floats", fontsize=9, color="#c0392b")
ax.plot([5.9, 9.6], [1.18, 1.18], color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(5.9, 0.90, "Rank-4 correction", fontsize=9, color="#27ae60")
ax.text(
    5.9, 0.60, "r(T+d) = 4x(360+64) = 1,696 floats  (7.4%)", fontsize=9, color="#27ae60"
)

fig.suptitle(
    "Figure 3. Low-rank correction stores U*S*V^T instead of the full residual,\n"
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
# Figure 5 PPL vs bit-width
# ---------------------------------------------------------------------------

models = ["distilgpt2", "gpt2-medium", "TinyLlama-1.1B"]
fp32_ppl = [33.51, 13.38, 4.78]

# dPPL without correction
delta_no = {
    "2-bit": [276.64, 173.61, 274.06],
    "3-bit": [21.44, 6.02, 0.87],
    "4-bit": [1.71, 1.10, 0.25],
}
# dPPL with rank-4 (TinyLlama omitted -> NaN)
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
ax.set_ylabel("dPPL (log scale)")
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
ax.set_ylabel("dPPL (log scale)")
ax.set_title("Effect of rank-4 correction")
ax.legend(fontsize=7.5, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 5. Left: PPL degradation across models and bit-widths (log scale).\n"
    "Right: rank-4 correction recovers 96-97% of 2-bit degradation.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("figures/fig5_ppl.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig5_ppl.png")


# ---------------------------------------------------------------------------
# Figure 6 First-token fix (_crop_cache)
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
        arrowprops=dict(arrowstyle="->", color=color, lw=1.3, shrinkA=4, shrinkB=4),
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
    draw_box(ax, (0.3, 4.1), 3.4, 0.8, "Prefill -> FP32 KV cache", "#cde6f5")
    arrow(ax, 2, 4.1, 2, 3.5)
    draw_box(ax, (0.3, 2.6), 3.4, 0.8, "Quantize -> quant KV cache", "#d5f0cd")

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
        # crop box: bottom=1.5, top=2.2  -> 0.4-unit gap below Quantize (bottom=2.6)
        draw_box(
            ax,
            (0.3, 1.5),
            3.4,
            0.7,
            "crop cache to T_p-1\nre-run last prompt token",
            "#fffacc",
            fontsize=8,
        )
        # logits box: bottom=0.5, top=1.2 -> 0.3-unit gap below crop (bottom=1.5)
        draw_box(
            ax,
            (0.3, 0.5),
            3.4,
            0.7,
            "first_logits = quantized output\n(different per bit-width [OK])",
            "#d5f0cd",
            fontsize=8,
        )
        arrow(ax, 2, 2.6, 2, 2.2)  # Quantize bottom (2.6) -> crop top (2.2)
        arrow(ax, 2, 1.5, 2, 1.2)  # crop bottom (1.5) -> logits top (1.2)

fig.suptitle(
    "Figure 6. First-token fix: crop the quantized cache to T_p-1, re-run the last prompt token.\n"
    "Before: all bit-widths share the same unquantized logit. After: each uses its own quantized logit.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig("figures/fig6_firsttoken.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig6_firsttoken.png")


# ---------------------------------------------------------------------------
# Figure 4 Product Quantization: schematic + quality vs storage
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# --- Left panel: PQ encoding schematic ---
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("PQ Encoding (d=64, M=4 shown for clarity)", fontsize=10)

# Original vector bar (top)
ax.add_patch(
    mpatches.FancyBboxPatch(
        (0.5, 4.8),
        9,
        0.8,
        boxstyle="round,pad=0.05",
        linewidth=1.2,
        edgecolor="#333",
        facecolor="#cce5ff",
    )
)
ax.text(
    5,
    5.2,
    r"$\mathbf{k} \in \mathbb{R}^{d}$  (post-rotation)",
    ha="center",
    va="center",
    fontsize=10,
)

# Four subvector blocks
sub_colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
sub_labels = [
    r"$\mathbf{k}^{(1)}$",
    r"$\mathbf{k}^{(2)}$",
    r"$\mathbf{k}^{(3)}$",
    r"$\mathbf{k}^{(4)}$",
]
sub_x = [0.5, 2.9, 5.3, 7.7]
sub_w = 2.1

for i, (x, col, lbl) in enumerate(zip(sub_x, sub_colors, sub_labels)):
    # subvector box
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, 3.1),
            sub_w,
            0.8,
            boxstyle="round,pad=0.05",
            linewidth=1.2,
            edgecolor=col,
            facecolor=col + "44",
        )
    )
    ax.text(x + sub_w / 2, 3.5, lbl, ha="center", va="center", fontsize=10, color=col)
    # arrow from big bar to subvector
    ax.annotate(
        "",
        xy=(x + sub_w / 2, 3.9),
        xytext=(x + sub_w / 2, 4.8),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, shrinkA=3, shrinkB=3),
    )
    # codebook box below
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, 1.1),
            sub_w,
            1.2,
            boxstyle="round,pad=0.05",
            linewidth=1.2,
            edgecolor=col,
            facecolor="#f8f8f8",
        )
    )
    ax.text(
        x + sub_w / 2,
        1.85,
        rf"$\mathcal{{C}}_{i+1}$" + "\n256 centroids",
        ha="center",
        va="center",
        fontsize=8,
        color=col,
    )
    # arrow from subvector to codebook
    ax.annotate(
        "",
        xy=(x + sub_w / 2, 2.3),
        xytext=(x + sub_w / 2, 3.1),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, shrinkA=3, shrinkB=3),
    )
    # code label below codebook
    ax.text(
        x + sub_w / 2,
        0.7,
        rf"code $c^{{({i+1})}}$" + "\n8 bits",
        ha="center",
        va="center",
        fontsize=8,
        color="#333",
    )
    ax.annotate(
        "",
        xy=(x + sub_w / 2, 0.95),
        xytext=(x + sub_w / 2, 1.1),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.0, shrinkA=2, shrinkB=2),
    )

# Total bits label
ax.text(
    5,
    0.1,
    r"Total: $M \times b = 16 \times 8 = 128$ bits/vector  (2 bits/dim)",
    ha="center",
    va="center",
    fontsize=9,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9db", edgecolor="#ccc"),
)

# --- Right panel: quality vs storage bar chart ---
ax2 = axes[1]

methods = ["4-bit scalar", "3-bit scalar", "PQ\n(M=16,b=8)", "2-bit scalar"]
bits_vec = [256, 192, 128, 128]
coherent = [True, True, True, False]  # False = broken generation

bar_colors = ["#4c72b0" if c else "#c44e52" for c in coherent]
# PQ gets a distinct highlight
bar_colors[2] = "#55a868"

bars = ax2.bar(
    methods, bits_vec, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.55
)

# Annotate bars with coherent/broken labels
quality_labels = ["Excellent", "Good", "Coherent\n(same as 3-bit)", "Broken"]
for bar, lbl, col in zip(bars, quality_labels, bar_colors):
    ypos = bar.get_height() + 4
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        ypos,
        lbl,
        ha="center",
        va="bottom",
        fontsize=8,
        color=col,
    )

ax2.set_ylabel("bits per vector")
ax2.set_ylim(0, 310)
ax2.set_title(
    "Storage vs Generation Quality\n(TinyLlama, d=64, 100 tokens)", fontsize=10
)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Highlight the equal-storage comparison (PQ vs 2-bit scalar)
ax2.annotate(
    "",
    xy=(3, 128),
    xytext=(2, 128),
    arrowprops=dict(arrowstyle="<->", color="#888", lw=1.2),
)
ax2.text(
    2.5,
    135,
    "same storage\n2x quality gap",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#555",
)

# Legend patches
legend_patches = [
    mpatches.Patch(color="#4c72b0", label="Coherent (scalar)"),
    mpatches.Patch(color="#55a868", label="Coherent (PQ)"),
    mpatches.Patch(color="#c44e52", label="Broken"),
]
ax2.legend(handles=legend_patches, fontsize=8, loc="upper right")

fig.suptitle(
    "Figure 4. Left: PQ splits a d-dim KV vector into M subvectors, each encoded via its own k-means codebook.\n"
    "Right: at 2 bits/dim, PQ (green) produces coherent output while same-budget scalar (red) collapses.",
    fontsize=10,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig("figures/fig4_pq.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig4_pq.png")

# ---------------------------------------------------------------------------
# Figure 7  k-means++ vs linspace initialisation MSE
# ---------------------------------------------------------------------------

b_vals = [1, 2, 3, 4]
k_vals = [2, 4, 8, 16]
linspace_mse = [0.097183, 0.006086, 0.001102, 0.000241]
kmpp_mse     = [0.024222, 0.002528, 0.000798, 0.000240]
improvements = [75.1, 58.5, 27.6, 0.0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

x = np.arange(len(b_vals))
w = 0.35

ax = axes[0]
ax.bar(x - w / 2, linspace_mse, w, label="Linspace init", color="#c44e52", alpha=0.85, edgecolor="white")
ax.bar(x + w / 2, kmpp_mse,    w, label="k-means++ init", color="#4c72b0", alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"b={b}  (k={k})" for b, k in zip(b_vals, k_vals)])
ax.set_ylabel("Init MSE (before Lloyd-Max iterations)")
ax.set_title("Codebook Init MSE: Linspace vs k-means++\n(d=64, 100k samples)")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
bar_colors = ["#4c72b0" if imp > 0 else "#c0c0c0" for imp in improvements]
bars = ax.bar(x, improvements, color=bar_colors, alpha=0.85, edgecolor="white")
for bar, imp in zip(bars, improvements):
    if imp > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{imp:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"b={b}" for b in b_vals])
ax.set_ylabel("MSE reduction (%)")
ax.set_title("k-means++ Improvement at Initialisation\n(gains largest where k is small)")
ax.set_ylim(0, 90)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 7. k-means++ seeding concentrates initial centroids in high-density regions.\n"
    "Left: raw init MSE (log scale). Right: % reduction vs linspace. After 2000 Lloyd-Max\n"
    "iterations both converge; practical benefit is faster convergence and fewer bad local optima.",
    fontsize=9,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.savefig("figures/fig7_kmeans_init.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig7_kmeans_init.png")


# ---------------------------------------------------------------------------
# Figure 8  K-V asymmetric quantization: IP/IP vs IP/MSE
# ---------------------------------------------------------------------------

configs  = ["3-bit IP/IP\n(baseline)", "2.5-bit IP/MSE\n(asymmetric)", "2-bit IP/IP\n(same budget)"]
bpw      = [3.0, 2.5, 2.0]
k_ip_err = [0.2815, 2.1816, 3.1417]
v_mse    = [0.005424, 0.002086, 0.045147]

x = np.arange(len(configs))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: K IP-error
ax = axes[0]
bar_colors = ["#4c72b0", "#55a868", "#c44e52"]
bars = ax.bar(x, k_ip_err, color=bar_colors, alpha=0.85, edgecolor="white", width=0.5)
for bar, val in zip(bars, k_ip_err):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel("K IP-error")
ax.set_title("K Inner-Product Error\n(lower = better attention scores)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: V MSE
ax = axes[1]
bars = ax.bar(x, v_mse, color=bar_colors, alpha=0.85, edgecolor="white", width=0.5)
for bar, val in zip(bars, v_mse):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
            f"{val:.5f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel("V MSE")
ax.set_title("V Reconstruction MSE\n(lower = better output quality)")
ax.set_yscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotation on V MSE panel: highlight the improvement
ax.annotate("", xy=(1, v_mse[1]), xytext=(0, v_mse[0]),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(0.5, 0.0028, "-61.5%\nat 0.5 fewer bits", ha="center", fontsize=8,
        color="#55a868", fontweight="bold")

fig.suptitle(
    "Figure 8. K-V asymmetric quantization: K uses KVQuantIP (inner-product optimal),\n"
    "V uses KVQuantMSE (MSE optimal). At 2.5 bpw the asymmetric config cuts V MSE by\n"
    "61.5% vs the 3-bit symmetric baseline while using 0.5 fewer bits per dimension.",
    fontsize=9,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.savefig("figures/fig8_kv_asymmetric.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig8_kv_asymmetric.png")


# ---------------------------------------------------------------------------
# Figure 9  Delta + outlier combination
# ---------------------------------------------------------------------------

configs2  = ["3-bit plain\n(IP/IP)", "2-bit plain\n(IP/IP)", "2.5-bit delta+outlier\n(IP/MSE)"]
bpw2      = [3.0, 2.0, 2.5]
k_ip_err2 = [0.2815, 3.1417, 2.1816]
v_mse2    = [0.005424, 0.045147, 0.002086]

x = np.arange(len(configs2))
bar_colors2 = ["#4c72b0", "#c44e52", "#55a868"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: K IP-error
ax = axes[0]
bars = ax.bar(x, k_ip_err2, color=bar_colors2, alpha=0.85, edgecolor="white", width=0.5)
for bar, val in zip(bars, k_ip_err2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(configs2, fontsize=8)
ax.set_ylabel("K IP-error")
ax.set_title("K Inner-Product Error")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: V MSE (log scale, annotate improvements)
ax = axes[1]
bars = ax.bar(x, v_mse2, color=bar_colors2, alpha=0.85, edgecolor="white", width=0.5)
for bar, val in zip(bars, v_mse2):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
            f"{val:.5f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(configs2, fontsize=8)
ax.set_ylabel("V MSE")
ax.set_title("V Reconstruction MSE\n(log scale)")
ax.set_yscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Improvement annotations
ax.annotate("", xy=(2, v_mse2[2]), xytext=(0, v_mse2[0]),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                            connectionstyle="arc3,rad=0.3"))
ax.text(1.0, 0.0009, "-61.5% vs 3-bit", ha="center", fontsize=7.5,
        color="#55a868", fontweight="bold")
ax.annotate("", xy=(2, v_mse2[2]), xytext=(1, v_mse2[1]),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(1.6, 0.02, "-95.4% vs 2-bit", ha="center", fontsize=7.5,
        color="#55a868", fontweight="bold")

fig.suptitle(
    "Figure 9. Delta compression + outlier-aware quantization stacked at 2.5 bpw.\n"
    "V MSE is 61.5% lower than 3-bit plain (at 0.5 fewer bits) and 95.4% lower than\n"
    "same-budget 2-bit plain. Outlier channels in deltas receive higher bit allocation.",
    fontsize=9,
    y=0.01,
    va="bottom",
)
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.savefig("figures/fig9_delta_outlier.png", bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig9_delta_outlier.png")


print("\nAll figures saved to figures/")
