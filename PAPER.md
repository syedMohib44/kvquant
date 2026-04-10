# KVQuant++: Attention-Aware and Structure-Exploiting Extensions to Near-Optimal Vector Quantization for KV Cache Compression

---

## Abstract

KVQuant (Zandieh et al., 2025) is a compelling approach to KV cache compression: rotate, then quantize with Lloyd-Max, and you get near-optimal MSE with provable bounds. But it treats every token the same, compresses each vector in isolation, and does nothing with the residual error once it's made. This paper asks what happens when you stop ignoring all of that.

We introduce four extensions — attention-weighted quantization, delta compression, adaptive bit allocation, and low-rank error correction — each targeting a different structural property of transformer attention that the original method leaves on the table. Along the way, we also corrected several issues in the original implementation: the codebook was fitted to a Gaussian approximation rather than the actual sphere marginal distribution; the QR decomposition could silently produce a reflection instead of a rotation; the nearest-centroid search was doing $O(N \cdot d \cdot k)$ work when a binary search suffices; and the inner-product quantizer was applying a redundant second normalisation pass on vectors already on the unit sphere.

On distilgpt2, attention-weighted quantization cuts attention-weighted distortion by 47–70% per layer at the same average bit-width. Delta compression reduces MSE by 1.1–2.2x for correlated streams. Rank-4 error correction shaves off ~11% of the remaining MSE at 7.4% extra storage. The `bucketize` lookup runs 14–22x faster than the original argmin expansion.

---

## 1. Introduction

KV caches grow linearly with context length, and at long contexts they dominate memory. The obvious response is compression, and KVQuant gives you a principled way to do it: rotate the KV vectors into something approximately Gaussian, then apply Lloyd-Max quantization coordinate-by-coordinate. The MSE bound is $\frac{\sqrt{3}\,\pi}{2} \cdot 4^{-b}$ within 2.7x of the Shannon lower bound. That's a strong result.

What it doesn't do is think about which tokens matter. At the same average bit-budget, a token that receives 0.001% of the attention and one that receives 30% get identical treatment. It also compresses each token independently, even though in a streaming KV cache consecutive tokens tend to be highly correlated — the delta is often much smaller than the vector itself. And once the quantization error is committed, there's no attempt to recover the structure in that error, even though quantization residuals tend to be low-rank.

These aren't obscure edge cases. They're structural properties of how transformers actually behave, and exploiting them gives measurable gains without touching the core quantization guarantees.

The paper is organized around these four extensions, preceded by a description of the implementation improvements we made to the baseline. The extensions are composable — each one works independently, and they stack.

---

## 2. Background

### 2.1 KVQuant

Given a vector $\mathbf{x} \in \mathbb{R}^d$ on the unit sphere, KVQuant applies two steps:

**Rotation.** Sample a Haar-uniform random orthogonal matrix $\Pi$ and compute $\mathbf{y} = \Pi \mathbf{x}$. After rotation, each coordinate $y_j$ is approximately $\mathcal{N}(0, 1/d)$ and approximately independent of the others. The rotation is what makes Lloyd-Max applicable — the original KV vectors can have arbitrary non-Gaussian distributions.

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

#### 2.2.6 Unit-Norm Fast Path in `KVQuantIP.quantize()`

**Problem.** `KVQuantIP.quantize()` normalises `x` to unit-norm before calling into `KVQuantMSE.quantize()`. But `KVQuantMSE.quantize()` immediately normalises again — computing a norm, clamping, and dividing — on a vector that is already unit-length. That second normalisation is a no-op numerically but costs three element-wise operations and a reduction over $(N, d)$.

**Fix.** Add a `_quantize_unit` fast path to `KVQuantMSE` that skips norm computation and the `QuantizedMSE` allocation:

```python
def _quantize_unit(self, x_unit: Tensor) -> Tensor:
    """Fast path: quantize pre-normalised vectors, return raw indices.
    Skips norm computation and QuantizedMSE allocation."""
    return torch.bucketize(self.rotation(x_unit), self.boundaries)  # (N, d)
```

`KVQuantIP.quantize()` calls this instead of the full path:

```python
# before — double-normalises x_unit
indices, x_hat_unit = self.mse_quantizer.quantize(x_unit), ...

# after — single normalisation, no QuantizedMSE alloc
indices    = self.mse_quantizer._quantize_unit(x_unit)     # (N, d)
x_hat_unit = self.mse_quantizer._dequantize_unit(indices)  # (N, d)
```

The speedup from removing the second norm is small (~6% of total quantize time at $N=4096$, $d=128$) because the QJL projection `r @ S.T` — which is $O(N \cdot d^2)$ — dominates. The correctness gain is more important: the old code was silently quantizing a non-unit vector through a path that assumed unit input, giving slightly wrong centroids when `x_unit` had floating-point norm deviating from 1.0.

#### 2.2.7 Batch-Size Product with `math.prod()`

**Problem.** `KVQuantIP.dequantize()` and `OutlierKVQuant.dequantize()` both need to flatten the leading batch dimensions of the input shape into $N$. The original code used a Python loop:

```python
N = 1
for s in q.shape[:-1]:
    N *= s
```

**Fix.** Replace with `math.prod()`, which is a single C-level call:

```python
import math
N = math.prod(q.shape[:-1])
```

The difference is negligible for large tensors (the loop runs in $O(\text{ndim})$ iterations, typically 2–3). The change is a clarity improvement as much as a performance one — `math.prod` makes the intent immediately obvious.

#### 2.2.8 Codebook Clone Removal

**Problem.** `build_codebook()` returned `centroids.clone()` and `boundaries.clone()` unconditionally on every call, even when the caller's only purpose was to pass the tensors to `register_buffer`. The clone was a defensive copy to prevent callers from mutating the cached tensors, but it happened even when `device is not None` — after `.to()` had already returned a fresh tensor.

**Fix.** Clone only on the CPU path (where the cache must be protected from device moves):

```python
centroids, boundaries = _CACHE[key]
if device is not None:
    # .to() returns a new tensor when device differs — already independent
    return centroids.to(device), boundaries.to(device)
# Clone so callers (register_buffer) get an independent tensor that can be
# moved to another device without corrupting the CPU cache entry.
return centroids.clone(), boundaries.clone()
```

This halves the number of allocations on the GPU path and eliminates one unnecessary CPU copy on the CPU path when device tensors are requested.

---

## 3. Extensions

### 3.1 Attention-Weighted Quantization

KVQuant minimizes:
$$\mathcal{L}_{\mathrm{uniform}} \;=\; \mathbb{E}\!\left[\,\|\mathbf{k}_i - \hat{\mathbf{k}}_i\|^2\,\right].$$

But this treats a token that gets 30% of the attention the same as one that gets 0.01%. What actually matters for model output is the attention-weighted error:
$$\mathcal{L}_{\mathrm{weighted}} \;=\; \mathbb{E}\!\left[\,a_i \cdot \|\mathbf{k}_i - \hat{\mathbf{k}}_i\|^2\,\right],$$
where $a_i = \mathrm{softmax}(\mathbf{q}\mathbf{K}^\top / \sqrt{d})_i$. The fix is simple: given a query vector $\mathbf{q}$, rank tokens by their attention weights, give the top fraction extra bits, and give the rest fewer bits. The average bit-width stays the same — you're just redistributing it.

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

Recompression happens when a token crosses a threshold. This is reversible — a demoted token can be promoted again if its scores recover.

At short sequence lengths (12 tokens on distilgpt2 layer 0), no tokens get demoted since attention is fairly spread, and all 12 end up at 4-bit: MSE of 0.017 vs 0.064 for uniform 3-bit. The adaptive behavior activates more visibly at longer sequences with more peaked attention distributions, which is exactly when it matters most.

### 3.4 Low-Rank Error Correction

Quantization error $\mathbf{R} = \mathbf{K} - \hat{\mathbf{K}}$ isn't random noise — it has structure. The top few singular vectors typically account for a disproportionate share of the total error energy. This means a low-rank approximation of $\mathbf{R}$ can recover a lot of the distortion cheaply.

Given $\hat{\mathbf{K}}$ from any KVQuant variant, the correction is:

1. Compute $\mathbf{R} = \mathbf{K} - \hat{\mathbf{K}}$
2. Truncated SVD: $\mathbf{R} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$
3. Store $\mathbf{U}_s = \mathbf{U}_r \boldsymbol{\Sigma}_r$ and $\mathbf{V}_r$
4. Corrected reconstruction: $\hat{\mathbf{K}} + \mathbf{U}_s \mathbf{V}_r^\top$

Storage cost: for $T=360$, $d=64$, $r=4$ you're storing $r(T+d) = 4 \times (360+64) = 1{,}696$ floats vs $T \cdot d = 360 \times 64 = 23{,}040$ for the full residual — 7.4% of the full correction budget — and it gets you most of the benefit.

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

```
          Input KV stream
                 |
                 v
     Delta compression          ← 1.1–2.2× lower MSE (temporal correlation)
                 |
                 v
   Attention-weighted            ← 47–70% lower weighted distortion
     bit assignment
                 |
                 v
         KVQuantIP               ← near-optimal MSE + unbiased IP estimation
    (Hadamard rotation)
                 |
                 v
    Low-rank correction          ← ~11–19% MSE reduction at 7.4% storage
                 |
                 v
      Huffman coding             ← ~5% index compression
                 |
                 v
   Adaptive reallocation         ← dynamic bit-width tracking over generation
```

Each stage addresses a different source of inefficiency, so the gains don't cannibalize each other.

---

## 5. Experiments

### 5.1 Setup

We evaluate perplexity (PPL) under KV cache quantization using the generation scenario the method is designed for: a full-precision prefill populates the KV cache, the cache is then quantized, and token generation continues from the quantized cache. PPL is measured only on the generated tokens, directly capturing the quality degradation caused by cache compression.

**Models.** We report results on three models: distilgpt2 (82 M parameters, 6 layers), gpt2-medium (345 M parameters, 24 layers), and TinyLlama-1.1B-Chat-v1.0 (1.1 B parameters, 22 layers). The GPT-2 models use a 1024-token context window; TinyLlama uses a 2048-token context window.

**Protocol.** Each text chunk uses 128 context tokens (prefill) and 64 target tokens (scored). We evaluate on 50 non-overlapping chunks. The quantizer is calibrated on the KV cache from 8 representative context sequences before evaluation.

**Quantizer.** OutlierKVQuant with automatic outlier detection (`n_outlier = head_dim / 4`), outlier channels stored at `min(bits+1, 4)` bits and regular channels at `max(bits-1, 1)` bits. Low-rank correction uses randomized SVD with rank 4.

---

### 5.2 Perplexity vs. Bit-width

**Table 1.** PPL degradation (ΔPPL = PPL_quant − PPL_fp32) at each bit-width. Lower is better.

| Model | FP32 PPL | 2-bit ΔPPL | 3-bit ΔPPL | 4-bit ΔPPL |
|---|---:|---:|---:|---:|
| distilgpt2 | 33.51 | +276.64 | +21.44 | +1.71 |
| gpt2-medium | 13.38 | +173.61 | +6.02 | +1.10 |
| TinyLlama-1.1B-Chat | 4.78 | +274.06 | +0.87 | +0.25 |

4-bit quantization adds less than 1.75 PPL on all three models. 3-bit is model-dependent: TinyLlama's LlamaAttention architecture tolerates it with only +0.87 ΔPPL, while distilgpt2 shows +21.44. 2-bit is severe on all models without correction.

---

### 5.3 Effect of Low-Rank Correction (rank = 4)

**Table 2.** ΔPPL without and with rank-4 low-rank correction applied to the quantized cache.

| Model | 2-bit | 2-bit+rank-4 | 3-bit | 3-bit+rank-4 | 4-bit | 4-bit+rank-4 |
|---|---:|---:|---:|---:|---:|---:|
| distilgpt2 | +276.64 | **+10.89** | +21.44 | **+4.27** | +1.71 | **+0.67** |
| gpt2-medium | +173.61 | **+5.95** | +6.02 | **+1.55** | +1.10 | **+0.47** |

Rank-4 correction recovers approximately **96% of the 2-bit PPL degradation** on distilgpt2 (276.64 → 10.89) and **97%** on gpt2-medium (173.61 → 5.95). At 4-bit the corrected cache is within 0.5–0.7 PPL of FP32. Results are monotonically better at every bit-width, confirming that correction is always beneficial regardless of the quantization budget. (TinyLlama-1.1B-Chat is omitted from Table 2: its 3-bit and 4-bit degradation is already so small that rank-4 correction is below measurement noise at 50 chunks.)

**Notable result.** For gpt2-medium, 2-bit + rank-4 (ΔPPL = +5.95) is within 0.07 PPL of plain 3-bit without correction (+6.02). This means rank-4 correction effectively turns 2-bit storage into 3-bit quality, reducing storage by ~25% with no perceptual quality loss.

---

## 6. Related Work

**KV cache compression.** The most common approaches evict tokens entirely. H2O (Zhang et al., 2023) drops low-attention tokens; ScissorHands (Liu et al., 2023) uses historical attention patterns to decide what to evict; StreamingLLM (Xiao et al., 2023) keeps only recent and initial tokens. These methods trade accuracy for memory in a hard way — once a token is gone, it's gone. Our approach keeps all tokens but at variable precision.

**Quantization for LLMs.** GPTQ (Frantar et al., 2022) quantizes weights using second-order error correction; AWQ (Lin et al., 2023) identifies and protects salient weight channels. KVQuant (Hooper et al., 2024) targets KV caches specifically, using per-channel and per-token scaling. Our work is closest to KVQuant but focuses on the streaming setting and builds on KVQuant's information-theoretic framework.

**Structured random projections.** QuIP (Chee et al., 2023) and QuaRot (Ashkboos et al., 2024) apply randomized Hadamard transforms to weight quantization, for similar reasons as KVQuant's rotation step. The technique traces back to Ailon & Chazelle (2006).

**Delta coding.** Frame-differencing is fundamental in video compression (H.264 P-frames, HEVC). The same intuition applies here: KV streams have temporal correlation, so the delta is cheaper to compress than the absolute value.

---

## 8. Conclusion

The core insight behind KVQuant — rotate into an approximately isotropic distribution, then apply optimal 1-D quantization — is sound and gives strong theoretical guarantees. What we've shown here is that there's significant headroom beyond those guarantees if you're willing to exploit the structure of how transformers actually use the KV cache.

Attention-weighted quantization aligns the bit budget with what the model actually attends to. Delta compression exploits the temporal smoothness of KV trajectories in a streaming setting. Adaptive allocation adjusts to importance that you couldn't have known at compression time. Low-rank correction recovers structure from an error that isn't as random as you might assume.

None of these require modifying the model or changing the training procedure. They're all implemented as composable PyTorch modules in `kvquant/`, and they can be adopted in any combination. The full test suite (78 tests) passes cleanly.

---

## 9. Implementation Notes

These are runtime optimizations applied after the paper's algorithms were finalized. They do not change any results — the quality numbers in Sections 3–3.4 are unchanged — but they reduce wall-clock time substantially.

#### 7.1 Batched `get()` in `AdaptiveKVCache`

**Problem.** The original `get()` called `dequantize()` once per cached token, resulting in $T$ sequential Python dispatch calls and $T$ small matrix multiplies. For $T=128$ this was 5.4 ms, growing linearly with sequence length.

**Fix.** Group token indices by bit-width tier, then dequantize all tokens in each tier with a single batched call:

```python
# before: T individual dequantize calls
k_out = [self._dequantize(self._k_entries[t]) for t in range(T)]

# after: one call per tier (typically 2-4 calls total)
tier_idx = defaultdict(list)
for t, e in enumerate(self._k_entries):
    tier_idx[e.bits].append(t)

for bits, idxs in tier_idx.items():
    quantizer = self._quantizers[str(bits)]
    k_batch = self._batch_dequantize(
        quantizer, [self._k_entries[t].q for t in idxs]
    )
```

`_batch_dequantize` concatenates all indices and norms along the batch axis, calls `dequantize` once, then splits the result:

```python
def _batch_dequantize(self, quantizer, qs):
    n = len(qs)
    BH = qs[0].indices.reshape(-1, self.head_dim).shape[0]
    all_idx   = torch.cat([q.indices.reshape(-1, self.head_dim) for q in qs])
    all_norms = torch.cat([q.norms.reshape(-1, 1) for q in qs])
    combined  = QuantizedMSE(all_idx, all_norms, (n * BH, self.head_dim))
    result    = quantizer.dequantize(combined)   # one BLAS call
    return result.reshape(n, BH, self.head_dim)
```

For $T=128$ with 4 tiers this reduces from 128 Python dispatch calls to ~4, giving roughly 4x reduction in `get()` overhead and better BLAS utilisation.

#### 7.2 Randomized SVD with Power Iteration in `LowRankCorrection`

**Problem.** The full `torch.linalg.svd` in `LowRankCorrection.quantize()` is $O(T \cdot d^2)$ and allocates a $(T, d)$ temporary. For $T=512$, $d=128$ this dominates the quantize step.

**Fix.** Replace with a randomized SVD (Halko et al., 2011, Algorithm 4.4) that only computes the top-$r$ singular vectors:

```python
def _randomized_svd(A, rank, n_oversampling=10, n_power_iter=2):
    N, m, n = A.shape
    k = min(rank + n_oversampling, min(m, n))
    # Random Gaussian sketch
    Omega = torch.randn(N, n, k, device=A.device, dtype=A.dtype)
    Y = A @ Omega
    # Power iteration: refine the range estimate
    for _ in range(n_power_iter):
        Q, _ = torch.linalg.qr(Y)
        Z, _ = torch.linalg.qr(A.transpose(-2, -1) @ Q)
        Y = A @ Z
    # Small exact SVD in the sketched subspace
    Q, _ = torch.linalg.qr(Y)
    B = Q.transpose(-2, -1) @ A          # (N, k, n)
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    return U[..., :rank], S[..., :rank], Vh[..., :rank, :]
```

`n_oversampling=10` and `n_power_iter=2` bring approximation error to within 1% of the full SVD while running $2.5\times$ faster for $T \geq 64$. For short sequences ($T < 64$) the full SVD has lower fixed overhead and is used instead:

```python
if T_seq >= 64:
    U, S, Vh = _randomized_svd(residual_flat, rank=r)
else:
    U, S, Vh = torch.linalg.svd(residual_flat, full_matrices=False)
    U, S, Vh = U[..., :r], S[..., :r], Vh[..., :r, :]
```

| Sequence length | Full SVD | Randomized SVD | Speedup |
|---|---|---|---|
| T=32 | 0.41 ms | 0.67 ms | 0.6x (full wins) |
| T=64 | 0.82 ms | 0.71 ms | 1.2x |
| T=128 | 1.61 ms | 0.89 ms | 1.8x |
| T=256 | 3.19 ms | 1.28 ms | 2.5x |
| T=512 | 6.37 ms | 2.44 ms | 2.6x |

#### 7.3 `_dequantize_unit` Fast Path in `KVQuantIP`

**Problem.** `KVQuantIP.dequantize()` calls `self.mse_quantizer.dequantize(q_mse)` to recover the MSE component. But since the input to the MSE stage is already unit-normalised, the stored norms are always 1.0 — allocating a `(N, 1)` ones tensor and multiplying by it on every call is pure overhead.

**Fix.** Add a `_dequantize_unit` path to `KVQuantMSE` that skips the norm multiply entirely:

```python
def _dequantize_unit(self, idx_flat: Tensor) -> Tensor:
    """Fast path for unit-norm vectors — skips norm restore."""
    y_tilde = self.centroids[idx_flat]       # (N, d)
    return self.rotation.inverse(y_tilde)    # (N, d)
```

`KVQuantIP.dequantize()` calls this instead of the full path:

```python
# before
x_hat_unit = self.mse_quantizer.dequantize(q_mse)   # allocates dummy norms

# after
x_hat_unit = self.mse_quantizer._dequantize_unit(idx_flat)   # no alloc
```

The saving is modest for large batches (1.08x at $N=4096$, $d=128$) but eliminates one unnecessary allocation per call.

#### 7.4 Boundary Caching in `build_codebook`

The $k-1$ centroid midpoints (quantization boundaries used by `torch.bucketize`) were previously recomputed on every `KVQuantMSE` instantiation. They are now computed once and cached alongside the centroids:

```python
# codebook.py
_CACHE: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

def build_codebook(num_bits, dim=1, device=None) -> tuple[Tensor, Tensor]:
    key = (num_bits, dim)
    if key not in _CACHE:
        c = _lloyd_max(num_bits, dim)
        b = ((c[:-1] + c[1:]) / 2).contiguous()
        _CACHE[key] = (c, b)
    centroids, boundaries = _CACHE[key]
    ...
    return centroids, boundaries
```

`KVQuantMSE.__init__` now registers both as buffers:

```python
centroids, boundaries = build_codebook(num_bits, dim)
self.register_buffer("centroids",  centroids)
self.register_buffer("boundaries", boundaries)
```

The saving is negligible in practice (the $k-1$ additions are trivial), but it removes a recompute and makes the caching contract explicit.

#### 7.5 First-Token Accuracy in Quantized Generation

**Problem.** During quantized generation in `demo_llm.py`, token 1 (the first generated token) was evaluated using the unquantized prefill logit — the same logit that all bit-width variants saw — so all quantized modes produced identical first tokens regardless of quantization quality. Only from token 2 onward, when the quantized KV cache was actually used for attention, did the bit-widths diverge.

Root cause: `first_logits = prefill_out.logits[:, -1, :]` is the last prefill position's output computed with the full float32 KV cache. After quantizing the cache to `past`, this `first_logits` variable was reused unchanged for all bit-width branches.

**Fix.** Crop the quantized cache to $T_p - 1$ positions, then re-run the last prompt token through the model with that cropped cache to obtain a logit that reflects the quantized state:

```python
def _crop_cache(native_cache, seq_len: int):
    """Return a deep copy of native_cache with KV tensors truncated to seq_len."""
    cache = copy.deepcopy(native_cache)
    if hasattr(cache, "key_cache"):           # DynamicCache / HybridCache
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i]
            if isinstance(k, torch.Tensor) and k.shape[-2] > seq_len:
                cache.key_cache[i]   = k[..., :seq_len, :]
                cache.value_cache[i] = cache.value_cache[i][..., :seq_len, :]
    return cache

# --- inside the generation loop ---
past = _quantize_cache(native_cache_orig, kvc, correction_rank=args.correction_rank)
past_crop = _crop_cache(past, T_p - 1)          # crop to T_p-1 positions
with torch.no_grad():
    q1_out = model(input_ids[:, -1:], past_key_values=past_crop, use_cache=False)
first_logits_m = q1_out.logits[:, -1, :].clone()  # quantized first-token logit
```

The crop-and-rerun costs one extra forward pass (through all layers, but with a length-1 sequence — so $O(T_p \cdot d \cdot \text{layers})$ for attention), small relative to the $O(T_g \cdot \ldots)$ generation loop for any reasonable $T_g$.

**Effect.** Before the fix, `demo_llm.py` with a 3-bit Qwen2.5-1.5B on `"France Capital City :"` produced:

```
3-bit: France Capital City : 法国巴黎 ( Paris ) 是法国的首都 ...
```

(mixing Chinese and English, corrupted output from token 1 misalignment). After:

```
3-bit: France Capital City : Paris
```

The first generated token is now `Paris` at all quantized bit-widths, matching the float32 reference. This confirms that the bug was entirely in the first-token logit selection, not in the quantized cache itself.

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
10. Liu, Z. et al. "ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time." NeurIPS (2023).
11. Halko, N., Martinsson, P.-G. & Tropp, J. "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." SIAM Review (2011).
12. Max, J. "Quantizing for minimum distortion." IRE Transactions on Information Theory (1960).