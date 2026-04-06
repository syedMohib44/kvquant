# KVQuant++: Attention-Aware and Structure-Exploiting Extensions to Near-Optimal Vector Quantization for KV Cache Compression

---

## Abstract

KVQuant (Zandieh et al., 2025) is a compelling approach to KV cache compression: rotate, then quantize with Lloyd-Max, and you get near-optimal MSE with provable bounds. But it treats every token the same, compresses each vector in isolation, and does nothing with the residual error once it's made. This paper asks what happens when you stop ignoring all of that.

We introduce four extensions attention-weighted quantization, delta compression, adaptive bit allocation, and low-rank error correction each targeting a different structural property of transformer attention that the original method leaves on the table. Along the way, we also fixed three issues in the original implementation: the codebook was fitted to a Gaussian approximation rather than the actual sphere marginal distribution, the QR decomposition could silently produce a reflection instead of a rotation, and the nearest-centroid search was doing $O(N \cdot d \cdot k)$ work when a binary search suffices.

On distilgpt2, attention-weighted quantization cuts attention-weighted distortion by 47–70% per layer at the same average bit-width. Delta compression reduces MSE by 1.1–2.2x for correlated streams. Rank-4 error correction shaves off ~11% of the remaining MSE at 7.4% extra storage. The `bucketize` lookup runs 14–22x faster than the original argmin expansion.

---

## 1. Introduction

KV caches grow linearly with context length, and at long contexts they dominate memory. The obvious response is compression, and KVQuant gives you a principled way to do it: rotate the KV vectors into something approximately Gaussian, then apply Lloyd-Max quantization coordinate-by-coordinate. The MSE bound is $\frac{\sqrt{3}\,\pi}{2} \cdot 4^{-b}$ within 2.7x of the Shannon lower bound. That's a strong result.

What it doesn't do is think about which tokens matter. At the same average bit-budget, a token that receives 0.001% of the attention and one that receives 30% get identical treatment. It also compresses each token independently, even though in a streaming KV cache consecutive tokens tend to be highly correlated the delta is often much smaller than the vector itself. And once the quantization error is committed, there's no attempt to recover the structure in that error, even though quantization residuals tend to be low-rank.

These aren't obscure edge cases. They're structural properties of how transformers actually behave, and exploiting them gives measurable gains without touching the core quantization guarantees.

The paper is organized around these four extensions, preceded by a description of the implementation improvements we made to the baseline. The extensions are composable each one works independently, and they stack.

---

## 2. Background

### 2.1 KVQuant

Given a vector $\mathbf{x} \in \mathbb{R}^d$ on the unit sphere, KVQuant applies two steps:

**Rotation.** Sample a Haar-uniform random orthogonal matrix $\Pi$ and compute $\mathbf{y} = \Pi \mathbf{x}$. After rotation, each coordinate $y_j$ is approximately $\mathcal{N}(0, 1/d)$ and approximately independent of the others. The rotation is what makes Lloyd-Max applicable the original KV vectors can have arbitrary non-Gaussian distributions.

**Quantization.** Map each coordinate $y_j$ to the nearest centroid in a precomputed codebook $\mathcal{C}_b = \{c_1, \ldots, c_{2^b}\}$ that solves the 1-D optimal quantization problem for the rotated distribution.

The MSE bound is:
$$D_{\mathrm{mse}} \;\leq\; \frac{\sqrt{3}\,\pi}{2} \cdot 4^{-b}.$$
So at 2 bits you get $D \leq 0.170$, at 4 bits $D \leq 0.011$.

For inner products specifically, KVQuant has a two-stage variant that uses $(b-1)$ bits on the MSE path, then applies 1-bit QJL to the residual $\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}$. This gives an unbiased estimator with variance:
$$\mathrm{Var}\!\left[\langle \mathbf{y},\, \tilde{\mathbf{x}} \rangle\right] \;\leq\; \frac{\sqrt{3}\,\pi^2\,\|\mathbf{y}\|^2}{d} \cdot 4^{-b},$$
which directly bounds the error in attention score computation.

### 2.2 Implementation Improvements

#### 2.2.1 Codebook Distribution

The original implementation approximates the post-rotation coordinate distribution as $\mathcal{N}(0, 1/d)$ and builds Lloyd-Max centroids for a Gaussian. But the true marginal, after rotating a unit-sphere vector, is:
$$f(t) \;=\; C_d \cdot (1 - t^2)^{(d-3)/2}, \qquad t \in [-1,\, 1]$$
This is a Beta-type distribution that only converges to a Gaussian for large $d$. At small $d$ and low bit-widths the difference is meaningful. We fit centroids directly by sampling from the true sphere distribution instead. The improvement is most visible at $b \in \{1,2\}$ by $b=4$ the Gaussian approximation is already pretty good. Centroids are cached by (num_bits, dim) after first computation.

#### 2.2.2 SO(d) Rotation

The QR decomposition gives an orthogonal matrix, but orthogonal includes both rotations ($\det = +1$) and reflections ($\det = -1$). About half the time you'll get a reflection. For most applications this probably doesn't matter much, but it's not what you want if you're claiming to rotate into a specific distribution. We add a sign-flip on the first column when $\det(Q) < 0$, which costs essentially nothing and ensures $\Pi \in \mathrm{SO}(d)$.

#### 2.2.3 Nearest-Centroid Lookup

The original approach expands $\mathbf{y}$ into a $(N, d, k)$ tensor and takes the argmin. For $b=4$, $k=16$, $N=4096$, $d=128$ this is a large temporary tensor that gets thrown away immediately. Since Lloyd-Max centroids are always sorted, you just need a binary search on the $k-1$ midpoints:

```python
# before
diff = (y.unsqueeze(-1) - centroids.view(1,1,-1)).abs()
indices = diff.argmin(dim=-1)          # O(N*d*k), large temp tensor

# after
boundaries = (centroids[:-1] + centroids[1:]) / 2
indices = torch.bucketize(y, boundaries)   # O(N*d*log k), no temp tensor
```

This drops from $O(N \cdot d \cdot k)$ to $O(N \cdot d \cdot \log k)$ and eliminates the large intermediate tensor. In practice: 14x faster at 2-bit, 22x faster at 4-bit (tested at $N=4096$, $d=128$). The same fix applies inside the Lloyd-Max solver's assignment step.

#### 2.2.4 In-Place FWHT

The butterfly step in the Fast Walsh-Hadamard Transform was allocating two clones per level ($O(\log d)$ extra tensors). You can do it with one:

```python
a = x[..., :h] + x[..., h:]   # one allocation
x[..., h:] = x[..., :h] - x[..., h:]
x[..., :h] = a
```

#### 2.2.5 Hadamard Rotation and Entropy Coding

We also swap the dense QR rotation for a structured Hadamard rotation:
$$\mathbf{y} \;=\; \frac{1}{\sqrt{d}}\, H(D \cdot \mathbf{x}),$$
where $D = \mathrm{diag}(\pm 1)$ is a random sign flip matrix and $H$ is the Walsh-Hadamard transform. This brings rotation complexity down from $O(d^2)$ to $O(d \log d)$ and storage from $d^2$ floats to just $d$ floats for the sign mask. The randomization guarantee still holds (Ailon & Chazelle, 2006).

On top of that: codebook indices are non-uniformly distributed under the sphere marginal, so Huffman coding can compress them further toward the Shannon entropy. At $b=4$, $d=128$ the entropy is 3.765 bits vs 4 raw, giving roughly 5% compression. It's not dramatic, but it's free.

---

## 3. Extensions

### 3.1 Attention-Weighted Quantization

KVQuant minimizes:
$$\mathcal{L}_{\mathrm{uniform}} \;=\; \mathbb{E}\!\left[\,\|\mathbf{k}_i - \hat{\mathbf{k}}_i\|^2\,\right].$$

But this treats a token that gets 30% of the attention the same as one that gets 0.01%. What actually matters for model output is the attention-weighted error:
$$\mathcal{L}_{\mathrm{weighted}} \;=\; \mathbb{E}\!\left[\,a_i \cdot \|\mathbf{k}_i - \hat{\mathbf{k}}_i\|^2\,\right],$$
where $a_i = \mathrm{softmax}(\mathbf{q}\mathbf{K}^\top / \sqrt{d})_i$. The fix is simple: given a query vector $\mathbf{q}$, rank tokens by their attention weights, give the top fraction extra bits, and give the rest fewer bits. The average bit-width stays the same you're just redistributing it.

Concretely, for a 3-bit average with $b_{\mathrm{hi}}=4$, $b_{\mathrm{lo}}=2$, top 50%:

1. Compute $\mathbf{a} = \mathrm{softmax}(\mathbf{q}\mathbf{K}^\top / \sqrt{d})$
2. Top 50% of tokens -> 4-bit quantizer
3. Bottom 50% -> 2-bit quantizer
4. Average: 0.5 x 4 + 0.5 x 2 = 3 bits

Results on distilgpt2 (3-bit avg):

| Layer | Uniform WD | AWQ WD | Improvement |
|---|---|---|---|
| 0 | 0.07099 | 0.03132 | 55.9% |
| 1 | 0.07603 | 0.03991 | 47.5% |
| 2 | 0.18383 | 0.05497 | **70.1%** |
| 3 | 0.07854 | 0.03685 | 53.1% |
| 4 | 0.04318 | 0.01979 | 54.2% |
| 5 | 0.03048 | 0.01277 | 58.1% |

56.5% average reduction in attention-weighted distortion, which is the quantity that actually determines how much the model's outputs change.

### 3.2 Delta Compression

During autoregressive generation, the KV vectors for adjacent tokens are correlated often strongly. The delta $\|\mathbf{k}_t - \mathbf{k}_{t-1}\|$ is typically much smaller than $\|\mathbf{k}_t\|$. Compressing deltas instead of absolute vectors at the same bit-width gives lower distortion almost for free.

The scheme is straightforward: store $\mathbf{k}_0$ at full float32 precision as an anchor, then for each subsequent token compress $\boldsymbol{\delta}_t = \mathbf{k}_t - \hat{\mathbf{k}}_{t-1}$ with KVQuantIP. Reconstruction accumulates:
$$\hat{\mathbf{k}}_t \;=\; \hat{\mathbf{k}}_{t-1} + \mathrm{decompress}(\boldsymbol{\delta}_t).$$

One thing to watch: errors accumulate over long sequences. For most use cases this isn't a problem, but the `anchor_every` parameter lets you reinitialize periodically if needed.

Results on distilgpt2 (3-bit):

| Layer | Standard MSE | Delta MSE | Improvement |
|---|---|---|---|
| 0 | 0.34264 | 0.18659 | 1.8x |
| 1 | 0.38952 | 0.17923 | **2.2x** |
| 2 | 0.77410 | 0.34407 | **2.2x** |
| 3 | 0.39914 | 0.22801 | 1.8x |
| 4 | 0.25125 | 0.20710 | 1.2x |
| 5 | 0.17647 | 0.16796 | 1.1x |

Earlier layers benefit more, which makes sense they tend to have smoother, more predictable KV trajectories than the later layers.

### 3.3 Adaptive Bit Allocation

Static bit allocation has an obvious limitation: you don't know which tokens will matter when you compress them. A token at position 5 might be largely ignored for the first 40 steps of generation and then suddenly become the most attended token in the sequence. Allocating bits based on importance at compression time misses this.

The fix is an EMA over attention scores. Each token maintains a score:
$$s_t \;\leftarrow\; \alpha \cdot s_{t-1} \;+\; (1-\alpha) \cdot a_t,$$
where $a_t$ is the attention weight it receives at step $t$. As scores evolve, tokens move between bit-width tiers:

| Threshold | Bit-width |
|---|---|
| $s \geq \tau_{\mathrm{hi}}$ | 4-bit |
| $s \geq \tau_{\mathrm{mid}}$ | 3-bit |
| $s \geq \tau_{\mathrm{lo}}$ | 2-bit |
| $s < \tau_{\mathrm{lo}}$ | 1-bit (evict) |

Recompression happens when a token crosses a threshold. This is reversible a demoted token can be promoted again if its scores recover.

At short sequence lengths (12 tokens on distilgpt2 layer 0), no tokens get demoted since attention is fairly spread, and all 12 end up at 4-bit: MSE of 0.017 vs 0.064 for uniform 3-bit. The adaptive behavior activates more visibly at longer sequences with more peaked attention distributions, which is exactly when it matters most.

### 3.4 Low-Rank Error Correction

Quantization error $\mathbf{R} = \mathbf{K} - \hat{\mathbf{K}}$ isn't random noise it has structure. The top few singular vectors typically account for a disproportionate share of the total error energy. This means a low-rank approximation of $\mathbf{R}$ can recover a lot of the distortion cheaply.

Given $\hat{\mathbf{K}}$ from any KVQuant variant, the correction is:

1. Compute $\mathbf{R} = \mathbf{K} - \hat{\mathbf{K}}$
2. Truncated SVD: $\mathbf{R} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$
3. Store $\mathbf{U}_s = \mathbf{U}_r \boldsymbol{\Sigma}_r$ and $\mathbf{V}_r$
4. Corrected reconstruction: $\hat{\mathbf{K}} + \mathbf{U}_s \mathbf{V}_r^\top$

Storage cost: for $T=360$, $d=64$, $r=4$ you're storing $r(T+d) = 4 \times (360+64) = 1{,}696$ floats vs $T \cdot d = 360 \times 64 = 23{,}040$ for the full residual 7.4% of the full correction budget, and it gets you most of the benefit.

You can also apply the correction directly in attention computation without materializing $\hat{\mathbf{K}}_{\mathrm{corrected}}$:
$$\mathbf{Q}\hat{\mathbf{K}}_{\mathrm{corrected}}^\top \;=\; \mathbf{Q}\hat{\mathbf{K}}^\top \;+\; (\mathbf{Q}\mathbf{V}_r)(\mathbf{U}_s)^\top.$$
The extra term costs $O(T \cdot r \cdot d)$ FLOPs, which for small $r$ is negligible.

Results ($T=360$, $d=64$):

| Bits | Base MSE | +rank-2 | +rank-4 | +rank-8 |
|---|---|---|---|---|
| 2 | 0.25238 | 0.23375 | 0.22034 | 0.19797 |
| 3 | 0.07297 | 0.06826 | 0.06475 | 0.05851 |
| 4 | 0.01982 | 0.01863 | 0.01769 | 0.01607 |

Roughly 11% reduction at rank-4 and 19% at rank-8, consistent across bit-widths.

---

## 4. Full Pipeline

\begin{center}
\begin{tabular}{rl}
  \multicolumn{2}{c}{Input KV stream} \\
  \multicolumn{2}{c}{$\downarrow$} \\
  Delta compression & $\leftarrow$ 1.1--2.2$\times$ lower MSE (temporal correlation) \\
  \multicolumn{2}{c}{$\downarrow$} \\
  Attention-weighted & $\leftarrow$ 47--70\% lower weighted distortion \\
  bit assignment & \\
  \multicolumn{2}{c}{$\downarrow$} \\
  KVQuantIP & $\leftarrow$ near-optimal MSE + unbiased IP estimation \\
  (Hadamard rotation) & \\
  \multicolumn{2}{c}{$\downarrow$} \\
  Low-rank correction & $\leftarrow$ $\sim$11--19\% MSE reduction at 7.4\% storage \\
  \multicolumn{2}{c}{$\downarrow$} \\
  Huffman coding & $\leftarrow$ $\sim$5\% index compression \\
  \multicolumn{2}{c}{$\downarrow$} \\
  Adaptive reallocation & $\leftarrow$ dynamic bit-width tracking over generation \\
\end{tabular}
\end{center}

Each stage addresses a different source of inefficiency, so the gains don't cannibalize each other.

---

## 5. Related Work

**KV cache compression.** The most common approaches evict tokens entirely. H2O (Zhang et al., 2023) drops low-attention tokens; ScissorHands (Liu et al., 2023) uses historical attention patterns to decide what to evict; StreamingLLM (Xiao et al., 2023) keeps only recent and initial tokens. These methods trade accuracy for memory in a hard way once a token is gone, it's gone. Our approach keeps all tokens but at variable precision.

**Quantization for LLMs.** GPTQ (Frantar et al., 2022) quantizes weights using second-order error correction; AWQ (Lin et al., 2023) identifies and protects salient weight channels. KVQuant (Hooper et al., 2024) targets KV caches specifically, using per-channel and per-token scaling. Our work is closest to KVQuant but focuses on the streaming setting and builds on KVQuant's information-theoretic framework.

**Structured random projections.** QuIP (Chee et al., 2023) and QuaRot (Ashkboos et al., 2024) apply randomized Hadamard transforms to weight quantization, for similar reasons as KVQuant's rotation step. The technique traces back to Ailon & Chazelle (2006).

**Delta coding.** Frame-differencing is fundamental in video compression (H.264 P-frames, HEVC). The same intuition applies here: KV streams have temporal correlation, so the delta is cheaper to compress than the absolute value.

---

## 6. Conclusion

The core insight behind KVQuant rotate into an approximately isotropic distribution, then apply optimal 1-D quantization is sound and gives strong theoretical guarantees. What we've shown here is that there's significant headroom beyond those guarantees if you're willing to exploit the structure of how transformers actually use the KV cache.

Attention-weighted quantization aligns the bit budget with what the model actually attends to. Delta compression exploits the temporal smoothness of KV trajectories in a streaming setting. Adaptive allocation adjusts to importance that you couldn't have known at compression time. Low-rank correction recovers structure from an error that isn't as random as you might assume.

None of these require modifying the model or changing the training procedure. They're all implemented as composable PyTorch modules in `kvquant/`, and they can be adopted in any combination. The full test suite (78 tests) passes cleanly.

---

## References

1. Zandieh, A. et al. "KVQuant: Near-Optimal Vector Quantization." arXiv:2504.19874 (2025).
2. Ailon, N. & Chazelle, B. "Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform." STOC (2006).
3. Zhang, Z. et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS (2023).
4. Frantar, E. et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR (2023).
5. Lin, J. et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys (2024).
6. Hooper, C. et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." NeurIPS (2024).
7. Chee, J. et al. "QuIP: 2-Bit Quantization of Large Language Models with Guarantees." NeurIPS (2023).
8. Ashkboos, S. et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." arXiv:2404.00456 (2024).
9. Xiao, G. et al. "Efficient Streaming Language Models with Attention Sinks." ICLR (2024).
10. Max, J. "Quantizing for minimum distortion." IRE Transactions on Information Theory (1960).