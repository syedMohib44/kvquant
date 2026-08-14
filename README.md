**[Read the Paper (PDF)](https://osf.io/9wskz/files/xzb7d)** | **[DOI: 10.17605/OSF.IO/9WSKZ](https://doi.org/10.17605/OSF.IO/9WSKZ)** | **[View on GitHub](https://github.com/syedMohib44/kvquant)**

# KVQUANT

**Attention-aware KV cache quantization for LLM inference.**

---

## The problem

Every token an LLM generates requires storing its Key and Value vectors across all transformer layers the KV cache. It grows linearly with sequence length. For a 70B model at FP16, that is roughly 140 MB per 1,000 tokens. A 32K context window consumes ~4.5 GB of VRAM just for the cache, on top of model weights. On consumer hardware this means truncated context, slow CPU offloading, or out-of-memory errors.

Naïve quantization (uniform grid) fails because KV vectors have heavy-tailed outliers a few large-magnitude coordinates dominate the distribution. Rounding those aggressively collapses attention scores.

KVQUANT compresses the KV cache from 16-bit floats down to 2–4 bits while preserving attention accuracy. It isolates outlier dimensions at higher precision, fits per-layer non-uniform codebooks to the actual KV distribution, and minimises distortion directly on the `<Q,K>` dot products rather than pointwise values.

**Measured end-to-end**, on Qwen2.5-0.5B-Instruct with a ~300-token prompt and 64 generated tokens, `generate()` holds **2.6–3.6× fewer cache bytes** than fp16 and cuts **peak VRAM from 1225 MiB to 265 MiB (4.6×)** at equal throughput. Those numbers are counted from bytes actually resident, sidecars included — see [What is measured, and how](#what-is-measured-and-how). The nominal `16/bits` figure (4–8×) is roughly double the truth, because it ignores the per-vector norms the codec must store.

---

KVQUANT extends [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) with five novel extensions and several implementation improvements. It achieves near-optimal KV cache compression by combining information-theoretically grounded vector quantization with transformer-specific structure exploitation.

---

## Relationship to the TurboQuant paper

TurboQuant is a 25-page theory paper: three theorems, two algorithms, and four
experiment subsections. It ends at §4.4 — there is no §5, §6, §7, and no
appendix. What we take from it, and what is ours, is worth stating plainly.

**From the paper.** The MSE-optimal quantizer — random rotation, then a
per-coordinate Lloyd-Max codebook fitted to the sphere marginal of Lemma 1
(§3.1, Algorithm 1, Theorem 1). The inner-product-optimal two-stage quantizer,
`(b-1)`-bit MSE plus 1-bit QJL on the residual (§3.2, Algorithm 2, Theorem 2).
The distortion bounds we test against.

**Ours, not the paper's.** These are engineering decisions the paper does not
make, and they need to stand on their own measurements:

| Decision | What the paper says |
|---|---|
| `KVQuantMSE` for **both** K and V | Nothing. The paper never distinguishes keys from values, and never says which of its two quantizers to use for either. Our choice is pinned by `test_mse_key_reconstructs_far_better_than_ip`. |
| Outlier channels chosen by **empirical per-channel variance** | The outlier split is a three-sentence aside in §4.3 with **no selection criterion** — no threshold, no statistic, no mention of whether the choice is static or calibrated. |
| **Per-layer calibration** on real prefill KVs | Nothing. TurboQuant is explicitly *data-oblivious* ("apply instantly without needing data-specific tuning or calibrations", §1.2), so calibrating at all is a departure — one the outlier split forces on us. |
| Bit schedule `outlier=min(b+1,4)`, `regular=max(b-1,1)` | The paper gives one configuration and no formula to generalise it. |
| **GQA bump** `+ceil(log4(g))` | Nothing. Grouped-query attention is never mentioned outside a bibliography entry. |
| **Hadamard** rotation for power-of-2 dims | Nothing. The paper says only "a random rotation matrix"; fast/structured transforms are never discussed. Sound (Ailon & Chazelle 2006), but ours. |
| **det(Π) = +1** (SO(d), not O(d)) | Nothing. The determinant is never discussed, and the paper's argument holds for reflections too. We pin it so the QR and Hadamard backends stay interchangeable. |
| **Low-rank correction**, delta, adaptive, PQ, Huffman | Not in the paper. PQ appears only as a baseline to beat (§4.4); the ~5% entropy-coding gain is noted in §3.1 and explicitly declined. |

**Two errata in the source.** §4.3 states its worked example as
"(32×3+96×2)/128 = 2.5" — that sum is **2.25**. And the paper's bits-per-channel
figures are payload-only: §1.3 concedes that non-unit vectors need their L2
norms stored "in floating-point precision", but no bit-width figure anywhere
accounts for them. At head_dim 64 that omission is substantial, and it is the
main reason our measured ratios land near half the nominal ones. We count the
norms as `sidecar_bytes` and report them separately.

---

## Results at a glance

| Metric | Value |
|---|---|
| Attention-weighted distortion reduction | **47--70%** per layer (same bit-width) |
| Delta compression MSE improvement | **1.1--2.2x** on correlated streams |
| Low-rank correction MSE reduction | **~11%** at rank-4, ~19% at rank-8 |
| Codebook lookup speedup vs argmin | **14--22x** (bucketize) |
| Measured cache-byte reduction, end to end | **2.6--3.6x** (sidecars included) |
| Measured peak-VRAM reduction, end to end | **4.6x** (1225 -> 265 MiB) |
| Test suite | full suite green; counts change every commit, so run `pytest tests/` |

---

## How it works

### 1. MSE Distortion vs Bit-Width

KVQuant achieves per-element MSE well within the theoretical upper bound `(sqrt3 pi/2) . 4^{-b}` across all bit-widths. Both QR and Hadamard rotations stay below the bound.

![MSE Bounds](https://github.com/syedMohib44/kvquant/blob/main/plots/1_mse_bounds.png?raw=true)

---

### 2. Codebook Centroids (True Sphere Distribution)

Unlike the original KVQuant which uses a Gaussian approximation, we fit Lloyd-Max centroids directly to the **true unit-sphere marginal distribution** `f(t) alpha (1 - t^2)^{(d-3)/2}`. This gives tighter quantization, especially at low bit-widths and small `d`.

![Codebook Centroids](https://github.com/syedMohib44/kvquant/blob/main/plots/2_codebook_centroids.png?raw=true)

---

### 3. Rotation Effect

Random rotation (QR or Hadamard) transforms raw KV coordinates - which have arbitrary non-Gaussian distributions - into approximately N(0, 1/d), enabling optimal per-coordinate Lloyd-Max quantization.

![Rotation Effect](https://github.com/syedMohib44/kvquant/blob/main/plots/3_rotation_effect.png?raw=true)

---

### 4. Attention-Weighted Quantization (Extension 1)

Rather than allocating bits uniformly, AWQ assigns more bits to tokens that receive high attention from queries. At the same average bit-width, this reduces **attention-weighted distortion by 56.5% on average** across all distilgpt2 layers.

![Attention-Weighted Quantization](https://github.com/syedMohib44/kvquant/blob/main/plots/4_attention_weighted.png?raw=true)

---

### 5. Delta Compression (Extension 2)

Consecutive KV vectors are highly correlated. Compressing token-to-token deltas `delta_t = k_t - k(cap)_{t-1}` instead of absolute vectors gives **1.1--2.2x lower MSE**, especially in early layers with smoother trajectories.

![Delta Compression](https://github.com/syedMohib44/kvquant/blob/main/plots/5_delta_compression.png?raw=true)

---

### 6. Adaptive Bit Allocation (Extension 3)

Token importance evolves during generation. An EMA-based importance tracker dynamically reassigns bit-widths as attention scores accumulate, giving more bits to tokens that prove important over time.

![Adaptive Allocation](https://github.com/syedMohib44/kvquant/blob/main/plots/6_adaptive_allocation.png?raw=true)

---

### 7. Low-Rank Error Correction (Extension 4)

Quantization error `R = K - k(cap)` is structured - its top singular vectors capture most of the error energy. A rank-r SVD correction reduces MSE by **~11% at rank-4** using only 7.4% extra storage (at `T=360, d=64`; the overhead is `r(T+d)/(T*d)`, so it shrinks as context grows — 3.5% at `T=1024, d=128`).

![Low-Rank Correction](https://github.com/syedMohib44/kvquant/blob/main/plots/7_low_rank_correction.png?raw=true)

---

### 8. Entropy Coding

Codebook indices are non-uniformly distributed after rotation. Huffman coding reduces storage toward the Shannon entropy (3.771 bits at 4-bit, d=128) - the Huffman average is 3.815 bits, giving ~4.6% savings.

![Entropy Coding](https://github.com/syedMohib44/kvquant/blob/main/plots/8_entropy_coding.png?raw=true)

---

### 9. Full Pipeline Comparison

End-to-end comparison of baseline KVQuant vs the combined KVQUANT pipeline across all layers and bit-widths.

![Pipeline Comparison](https://github.com/syedMohib44/kvquant/blob/main/plots/9_pipeline_comparison.png?raw=true)

---

### 10. Compression Ratio

Storage cost vs reconstruction quality across configurations.

![Compression Ratio](https://github.com/syedMohib44/kvquant/blob/main/plots/10_ratio_check.png?raw=true)

---

### 11. Perplexity Validation

End-to-end perplexity on distilgpt2 confirms that quantization quality improvements preserve language modelling performance.

![Perplexity Validation](https://github.com/syedMohib44/kvquant/blob/main/plots/11_perplexity_validation.png?raw=true)

---

---

# Using the package

> Install via pip and use from any Python environment. No repo clone needed.

---

## Installation

```bash
pip install kvquant-plus-plus           # GPU kernels (Triton JIT) included
pip install "kvquant-plus-plus[cuda]"   # same + explicit Triton version pin
```

**Requirements:** Python >= 3.10, PyTorch >= 2.1, Transformers >= 4.40

GPU kernels (Triton JIT softmax, flash attention, matmul) are included in the base install no `[cuda]` extra needed. All paths fall back to pure PyTorch automatically on CPU, AMD, or Apple MPS.

---

## generate() full response

Pass a prompt string, get back text. The model is downloaded on first call and cached in memory for subsequent calls.

```python
from kvquant import generate

out = generate("What is machine learning?")
print(out.text)
print(f"{out.compression_ratio:.1f}x smaller than float16")

# With a system prompt
out = generate(
    "What is machine learning?",
    system="You are a helpful assistant. Be concise.",
)
print(out.text)
```

`GenerateResult` fields: `.text`, `.bits`, `.avg_bits_per_dim`, `.compression_ratio`, `.model`, `.prompt`

---

## stream() print tokens as they arrive

```python
from kvquant import stream

for token in stream("Explain transformers in simple terms"):
    print(token, end="", flush=True)
print()
```

---

## System prompt role, persona, or instructions

Pass any combination of system and user prompt. Both are quantized together in a single prefill pass no extra compute cost.

```python
from kvquant import generate, stream

# System prompt sets the model's role; user prompt is the question
out = generate(
    "What is quantum entanglement?",
    system="You are a physics professor. Answer at undergraduate level.",
    bits=3,
)
print(out.text)

# Long document analysis system sets the task, user supplies the document.
# prefill_chunk_size keeps each attention step at chunk² instead of T²,
# so even a 10 000-token contract fits on an 8 GB GPU.
contract = open("contract.txt").read()
out = generate(
    contract + "\n\nList all payment terms.",
    system="You are a legal analyst. Extract facts only, be concise.",
    bits=3,
    max_new_tokens=300,
    prefill_chunk_size=512,   # lower (256) = less VRAM, higher (1024) = faster
)
print(out.text)

# Streaming with a system prompt
for token in stream(
    "Explain gradient descent.",
    system="You are a machine learning tutor. Use simple analogies.",
    bits=3,
):
    print(token, end="", flush=True)
print()
```

---

## Bit-width options (2 / 3 / 4)

```python
from kvquant import generate

# 2-bit: most compressed (~8x smaller than float16), some quality loss
out = generate("Summarise the French Revolution", bits=2)
print(f"{out.compression_ratio:.1f}x  {out.text}")

# 3-bit: recommended default (~5x smaller, minimal quality loss)
out = generate("Summarise the French Revolution", bits=3)
print(f"{out.compression_ratio:.1f}x  {out.text}")

# 4-bit: best quality (~4x smaller, near-lossless)
out = generate("Summarise the French Revolution", bits=4)
print(f"{out.compression_ratio:.1f}x  {out.text}")
```

---

## Sampling creative / varied output

```python
from kvquant import generate

out = generate("Once upon a time", bits=3, temperature=0.0)   # greedy
out = generate("Once upon a time", bits=3, temperature=0.8)   # sampling
out = generate("Once upon a time", bits=3, temperature=0.8, top_p=0.95)  # nucleus

print(out.text)
```

---

## Different models

Works with any HuggingFace causal LM instruct, base, and hybrid architectures (Qwen, Llama, Phi, Mistral, Falcon, Gemma …).

```python
from kvquant import generate

out = generate("What is quantum entanglement?",
               model="Qwen/Qwen2.5-1.5B-Instruct", bits=3)       # default

out = generate("What is the capital of France?",
               model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", bits=3)  # fast/small

out = generate("Explain the Turing test",
               model="Qwen/Qwen2.5-7B-Instruct", bits=3)          # larger

out = generate("The Eiffel Tower is located in",
               model="distilgpt2", bits=3, raw=True)               # base model

print(out.text)
```

---

## Multi-GPU large models

```python
from kvquant import generate

out = generate(
    "Explain quantum computing in detail",
    model="meta-llama/Llama-3.1-8B-Instruct",
    bits=3,
    device_map="auto",   # spreads model across all available GPUs
)
print(out.text)
```

---

## Avoiding out-of-memory (OOM) on low-VRAM GPUs

There are **two different** OOMs, and they need different fixes. Diagnose first:

| Symptom | Cause | Fix |
|---|---|---|
| Crashes **while loading the model**, before any output | Model **weights** too big (a 7B is ~15 GB, an 8 GB GPU cannot hold it) | `weights="4bit"` or `weights="offload"` (below) |
| Loads fine, then crashes **during a long prompt** | Prefill attention is O(T²) | `prefill_chunk_size=256` (see [Long documents](#long-documents-contracts-papers-codebases)) |
| Runs, but VRAM creeps up over a long generation | KV **cache** grows per token | lower `bits` (2 or 3) more compression, or `offload_to_disk=True` to spill the cache to SSD (below) |

All the memory levers on `generate()` / `stream()`:

| Lever | Shrinks / moves | Example |
|---|---|---|
| `weights="4bit"` | model weights → 4-bit on GPU | fits a 7B in ~5 GB |
| `weights="8bit"` | model weights → 8-bit on GPU | fits a 7B in ~8 GB |
| `weights="offload"` | model weights → CPU RAM → **SSD** | fits **any** size, slow |
| `max_gpu_mem` / `max_cpu_mem` | caps for the offload tiers | `"6GiB"` / `"12GiB"` |
| `weights_disk_dir` | where SSD weight shards go | `"D:/kv_weights"` |
| `device_map="auto"` | spreads weights across GPUs / CPU | multi-GPU |
| `bits` (2/3/4) | KV **cache** precision | `bits=2` = most compression |
| `offload_to_disk=True` | KV **cache** → CPU RAM → **SSD** (paper Lloyd-Max codec) | fits any context, slow |
| `prefill_chunk_size` | peak prefill VRAM | `256` = safest |

---

### Offload model weights to SSD (`weights=`)

A 7B model is ~15 GB in float16 it cannot fit in an 8 GB GPU (RTX 3070/3060) **at
load time**, before generation even starts. The `weights=` option controls how the
model **weights** are placed so large models fit on small GPUs. (This is separate
from KV-cache compression `bits` shrinks the cache, `weights` shrinks/offloads
the model itself.)

| `weights=` | What it does | 7B on 8 GB? | Speed | Needs |
|---|---|---|---|---|
| `None` / `"full"` (default) | float16/auto weights | no | fast | |
| `"4bit"` | 4-bit NF4 weights on GPU | **yes, ~5 GB** | **fast** | bitsandbytes |
| `"8bit"` | 8-bit weights on GPU | tight, ~8 GB | fast | bitsandbytes |
| `"offload"` | weights spilled GPU → CPU RAM → **SSD** | **any size** | slow | accelerate |

Install the extra once:

```bash
pip install "kvquant-plus-plus[quant]"     # bitsandbytes + accelerate
```

**Recommended for an 8 GB GPU 4-bit weights (fast, fits):**

```python
from kvquant import generate

out = generate(
    "What is machine learning?",
    model="Qwen/Qwen2.5-7B-Instruct",
    weights="4bit",           # a 7B now fits in ~5 GB VRAM
)
print(out.text)
```

**Offload weights to SSD when even 4-bit will not fit (any size, slower):**

> ⚠️ **`weights="offload"` is the last resort, not the default fix.** It streams the
> model's weights from SSD **every token**, so it is *extremely* slow (often <0.2
> tok/s — a 40-token reply can take minutes to hours). On an **8 GB GPU a 7B fits
> entirely in 4-bit (~5 GB) and runs fast** use `weights="4bit"` above. Reach for
> `"offload"` only for models too big even for 4-bit (14B/32B+ on a small GPU).

```python
from kvquant import generate

out = generate(
    "Explain quantum computing in detail",
    model="Qwen/Qwen2.5-7B-Instruct",
    weights="offload",              # GPU → CPU RAM → SSD  (slow — see warning above)
    max_gpu_mem="6GiB",             # cap VRAM (default: ~90% of free VRAM)
    max_cpu_mem="12GiB",            # cap CPU RAM
    weights_disk_dir="D:/kv_weights",  # SSD folder for weight shards
)
print(out.text)
```

### How to test it

Quick CPU-safe check (no model download needed) confirms the option is wired up:

```bash
python -c "import inspect, kvquant; print('weights' in inspect.signature(kvquant.generate).parameters)"
# -> True
```

Run the real thing from the CLI (watch VRAM in a second terminal with `nvidia-smi`):

```bash
python run_offload.py --model Qwen/Qwen3-30B-A3B --hf-home D:/huggingface-models --weights offload --max-gpu-mem 6GiB --weights-disk-dir D:/kv_weights --disk-dir D:/kv_cache --prompt "What is machine learning?"

# 4-bit  should complete without OOM; nvidia-smi shows ~5 GB VRAM
python -m kvquant.demo_llm --model Qwen/Qwen2.5-7B-Instruct --weights 4bit \
    --prompt "What is machine learning?" --max-new-tokens 40

# SSD offload  fits any size; weights_disk_dir fills with shards during load
python -m kvquant.demo_llm --model Qwen/Qwen2.5-7B-Instruct --weights offload \
    --max-gpu-mem 6GiB --weights-disk-dir D:/kv_weights \
    --prompt "What is machine learning?" --max-new-tokens 40

# 4-bit weights + KV-cache offload on an 8 GB GPU (paper-outlier codec, bits=4).
# The GQA bump is applied automatically for Qwen2.5-7B (g=7 → effective 5.25 bits).
python run_offload.py --model Qwen/Qwen2.5-7B-Instruct --weights 4bit \
    --offload-to-disk --offload-codec paper-outlier --bits 4 --device cuda \
    --prompt "Explain machine learning"
```

> **Tip.** KV-cache offload round-trips to SSD every token (slow — ~0.3 tok/s). On a
> 4-bit 7B the cache usually fits in VRAM, so drop `--offload-to-disk` for a large
> speed-up and only enable it when the *context* is long enough that the cache itself
> won't fit. `--weights 8bit` needs ~8 GB and won't fit an 8 GB card alongside
> activations — use `--weights 4bit` there.

Or from Python:

```python
from kvquant import generate

# Baseline (may OOM on 8 GB) vs 4-bit (fits) prove the difference:
out = generate("What is the capital of France?",
               model="Qwen/Qwen2.5-7B-Instruct", weights="4bit", max_new_tokens=20)
print(out.text)
print(f"{out.compression_ratio:.1f}x KV compression")
```

If bitsandbytes/accelerate is missing, the call raises a clear error telling you to
`pip install "kvquant-plus-plus[quant]"`.

`weights=` is accepted by both `generate()` and `stream()`, and by the
`--weights {full,4bit,8bit,offload}` CLI flag in `demo_llm`.

### Combine everything (worst case: big model + long prompt + 8 GB GPU)

The levers stack. For a large model on a small GPU with a long document, use 4-bit
weights (so the model fits) **and** chunked prefill **and** aggressive KV bits:

```python
from kvquant import generate

doc = open("contract.txt").read()
out = generate(
    doc + "\n\nSummarise the key obligations.",
    model="Qwen/Qwen2.5-7B-Instruct",
    weights="4bit",            # model fits in ~5 GB VRAM
    bits=2,                    # most KV-cache compression
    prefill_chunk_size=256,    # keep prefill attention small
    max_new_tokens=200,
)
print(out.text)
```

If 4-bit still will not fit (e.g. a 14B/32B model), swap `weights="4bit"` for
`weights="offload"` + `max_gpu_mem="6GiB"` fits any size, just slower.

### No GPU at all (CPU / RAM only)

Everything runs on CPU with no changes it is slow but never OOMs on VRAM:

```python
out = generate("What is machine learning?",
               model="Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
print(out.text)
```

---

### Offload the KV cache to SSD (`offload_to_disk=`)

`weights=` moves the **model**; `offload_to_disk=` moves the **KV cache**. Use this
when the model itself fits but the *cache* is what overflows a very long context
whose per-token K/V no longer fits in VRAM. The cache is spilled across three tiers
**VRAM → CPU RAM → SSD**.

Decoded floats are kept between calls on purpose — that is what makes the
incremental decode O(T) rather than O(T²) — and `max_vram_tokens` bounds how many
token positions stay resident, evicting least-recently-staged layers past the
budget. Eviction is lossless: it drops decoded floats only, never codes, so a
re-staged layer decodes bit-identical. Layers the current forward pass is about to
read are exempt, so a budget below one pass's worth bounds nothing; check
`memory_summary()["vram_tokens"]` for what is actually resident.

**This path is for when the cache does not fit in RAM at all.** It does not
reduce peak VRAM within a forward pass — the model still receives full float
tensors. For real VRAM savings, use the default in-memory compact cache
(`compact_cache=True`), which never materialises them: measured 265 MiB vs
1225 MiB on a ~300-token prompt.

Two codecs are available via `offload_codec` — both are the paper's method:

- **`"paper-outlier"` (default)** outlier-aware Lloyd-Max — the paper's §3.1
  MSE quantizer applied separately to the outlier and regular channel groups it
  sketches in §4.3 — calibrated once on the prefill KVs. Real attention K/V tensors have a
  few high-magnitude "outlier" channels; this codec gives them extra bits and
  quantizes the rest at `bits`, reaching near-lossless fidelity (cosine ~0.999) at
  3-4 bits/dim. **This is the recommended paper-faithful codec** plain Lloyd-Max
  wastes precision on the outliers and degrades real-model output.
- **`"paper"`** plain rotate → **Lloyd-Max**, indices **bit-packed** to `bits`
  (2-4). Smallest SSD footprint (~`bits`/coordinate), but on outlier-heavy KV it
  loses coherence — use only when footprint matters more than quality. Uses the
  MSE path for both K and V (not the KVQuant-**IP** estimator, which cannot
  reconstruct a usable cache).

**Append-only (why generation stays coherent).** Each token's K/V is compressed
**exactly once** and then frozen the offloader appends the newly generated token
each step and never re-touches earlier ones. This mirrors the paper (which quantizes
each vector once) and is essential for the lossy Lloyd-Max codec: re-quantizing an
already-quantized token every step makes it drift and, over a long generation,
degrades into gibberish. Compressing each token once and freezing it keeps the paper
codec faithful over long generations.

**Bit-width guidance for the paper codecs:** use **`bits=4`** (closest to lossless)
or **`bits=3`** (paper's sweet spot). **`bits=2` is very lossy** 4 Lloyd-Max levels
per coordinate is at the edge of usable and can visibly hurt quality; raise `bits`
if output degrades — there is no non-paper fallback.

**GQA models are handled automatically (paper §GQA compensation).** Grouped-query
models — Qwen2.5-7B, Llama-3, most modern 7B+ checkpoints — share each KV head
across `g = n_heads / n_kv_heads` query heads, which amplifies quantization error by
~`g`×. To hold quality constant the codec adds `ceil(log4(g))` bits per coordinate to
**both** the outlier and regular channels (Qwen2.5-7B has `g=7` → +2 bits, so
`bits=4` is stored at an *effective* 5.25 bits/dim). This matches the in-memory path
and is applied for you — you still pass `bits=4`; the bump happens internally and is
reflected in the reported `avg_bits_per_dim`. Without it, a GQA model under-quantizes
and generates gibberish, so **don't try to hand-tune around it** — just pick 3 or 4.

**Runs on any device.** The staged (dequantized) cache follows the model's device, so
offload works on CPU, MPS, or any GPU — not just CUDA. It is placed automatically from
the model; nothing extra to configure.

```python
from kvquant import generate

doc = open("long_document.txt").read()
out = generate(
    doc + "\n\nSummarise the key points.",
    model="Qwen/Qwen2.5-7B-Instruct",
    offload_to_disk=True,          # spill KV cache VRAM → RAM → SSD
    offload_codec="paper-outlier", # outlier-aware (default); "paper" also valid
    bits=4,                        # pack width (2-4); GQA bump raises effective bits
    device="cuda",                 # model + staged cache device (auto if omitted)
    max_vram_tokens=512,           # token positions kept dequantized in VRAM
    warm_size=16,                  # layer entries kept in CPU RAM before disk spill
    disk_dir="D:/kv_cache",        # SSD folder for spilled cache (auto-cleaned)
    prefill_chunk_size=256,        # also keep prefill attention small
)
print(out.text)
```

| Lever | Effect | Default |
|---|---|---|
| `offload_to_disk=True` | enable the VRAM → RAM → SSD KV-cache spill | `False` |
| `offload_codec` | `"paper-outlier"` (outlier-aware, best fidelity) or `"paper"` (Lloyd-Max, bit-packed, smallest) | `"paper-outlier"` |
| `bits` | pack width for the paper codec (2-4); also the in-memory KV precision. GQA models get an automatic effective-bit bump | `3` |
| `device` | compute + staged-cache device (`"cuda"`, `"cpu"`, `"mps"`); follows the model when omitted | auto |
| `max_vram_tokens` | token positions kept dequantized between forward passes; excess layers are evicted least-recently-staged first (lossless — codes are untouched). `0` = unbounded | `512` |
| `warm_size` | compressed layer entries kept in CPU RAM before spilling to SSD | `16` |
| `disk_dir` | SSD folder for spilled cache files (temp dir if unset, auto-cleaned) | `None` |

Both `offload_to_disk=` and `weights=` can be combined the KV-cache path is
independent of where the weights live:

```python
out = generate(
    doc + "\n\nSummarise.",
    model="Qwen/Qwen2.5-7B-Instruct",
    weights="4bit",            # model weights fit in ~5 GB VRAM
    offload_to_disk=True,      # and the growing KV cache spills to SSD
    disk_dir="D:/kv_cache",
    prefill_chunk_size=256,
)
```

`offload_to_disk=` and its knobs are accepted by both `generate()` and `stream()`.

---

## Long documents contracts, papers, codebases

Prefill attention is O(T²). A 6 000-token contract on an 8 GB GPU OOMs in a single forward pass. `prefill_chunk_size` processes context in chunks each step stays at `chunk² × heads` while the full context accumulates in the KV cache. No information is lost.

```python
from kvquant import generate, stream

contract = open("contract.txt").read()
out = generate(
    contract + "\n\nList all payment terms.",
    system="You are a legal analyst. Extract facts only.",
    bits=3,
    max_new_tokens=300,
    prefill_chunk_size=512,
)
print(out.text)

# Tune for your GPU:
#   prefill_chunk_size=256   →  ~0.5 GB prefill VRAM  (safe, slower)
#   prefill_chunk_size=512   →  ~1.5 GB prefill VRAM  (default)
#   prefill_chunk_size=1024  →  ~4.0 GB prefill VRAM  (faster)

for token in stream(
    contract + "\n\nWhat are the termination conditions?",
    system="You are a legal analyst.",
    bits=3,
    prefill_chunk_size=256,
):
    print(token, end="", flush=True)
print()
```

---

## Low-rank error correction (+11% quality)

```python
from kvquant import generate

out = generate(
    "Explain the history of the Roman Empire",
    bits=3,
    correction_rank=4,    # 0 = off (default), 4 = recommended at 2–3 bit
    max_new_tokens=300,
)
print(out.text)
```

---

## Repetition control

```python
from kvquant import generate

# Default repetition_penalty=1.3 handles most cases
# Raise to 1.5–2.0 if you see looping at 2-bit
out = generate("Tell me about AI", bits=2, repetition_penalty=1.5)
print(out.text)
```

---

## Low-level API

For integrating directly into a training loop, custom transformer, or research pipeline. All classes accept KV tensors as-is `(T, head_dim)`, `(B, H, T, head_dim)`, or anything in between.

### KVCacheQuantizer

```python
from kvquant import KVCacheQuantizer

quant = KVCacheQuantizer(head_dim=head_dim, num_bits=3)
quant.calibrate(keys, values)           # calibrate on actual prefill KVs

k_c, v_c     = quant.compress_kv(keys, values)
k_hat, v_hat = quant.decompress_kv(k_c, v_c)

print(f"avg bits/dim: {quant.avg_bits:.2f}")
```

### AttentionWeightedQuantizer

```python
from kvquant import AttentionWeightedQuantizer

awq = AttentionWeightedQuantizer(dim=head_dim, hi_bits=4, lo_bits=2, top_fraction=0.5)
compressed = awq.quantize(keys, query)
k_hat      = awq.dequantize(compressed)
print(f"avg bits: {awq.avg_bits:.2f}")
```

### DeltaKVCache

```python
from kvquant import DeltaKVCache

cache = DeltaKVCache(head_dim=head_dim, num_bits=3)
for k_new, v_new in token_stream:
    cache.push(k_new, v_new)

K_hat, V_hat = cache.get()
```

### AdaptiveKVCache

```python
from kvquant import AdaptiveKVCache

cache = AdaptiveKVCache(head_dim=head_dim)
for k_new, v_new in token_stream:
    cache.push(k_new, v_new)

cache.attend(attn_weights)    # attn_weights: (..., T)
K_hat, V_hat = cache.get()
print(cache.bit_allocation()) # e.g. {4: 8, 3: 50, 2: 60, 1: 10}
```

### LowRankCorrection

```python
from kvquant import LowRankCorrection, KVQuantMSE

lrc = LowRankCorrection(quantizer=KVQuantMSE(dim=head_dim, num_bits=2), rank=4)
k_corrected = lrc.forward(keys)
energy = lrc.residual_rank_analysis(keys, max_rank=8)
```

### ProductQuantizer

```python
from kvquant import ProductQuantizer

pq = ProductQuantizer(dim=head_dim, num_subspaces=16, bits_per_subspace=8)
pq.calibrate(keys.reshape(-1, head_dim))

compressed = pq.quantize(keys.reshape(-1, head_dim))
k_hat      = pq.dequantize(compressed)
print(f"compression ratio: {pq.compression_ratio():.1f}x vs float32")
```

---

## GPU Acceleration

GPU kernels (Triton JIT) are included in the base install and compile on first use no build step needed. Falls back to PyTorch automatically on CPU, AMD, or Apple MPS.

For full details on supported GPUs, backends, and performance numbers see the [cuda-triton-multiarch README](https://github.com/syedMohib44/cuda-triton-multiarch/blob/main/README.md).

---

## Implementation improvements over reference

| | Reference | KVQUANT |
|---|---|---|
| Codebook distribution | Gaussian approximation | **True sphere marginal** |
| Rotation | May produce reflections (det = -1) | **SO(d) enforced** (det = +1) |
| Nearest-centroid lookup | `argmin` - O(N.d.k) | **`bucketize`** - O(N.d.log k), 14--22x faster |
| FWHT | 2 clones per butterfly level | **In-place**, 1 allocation per level |
| FWHT (CUDA) | Multiple kernel launches | **`torch.compile`** fused, 2--3x faster |
| PQ encode (CUDA) | M sequential `cdist` calls | **Triton kernel**, all subspaces in 1 launch, 10--15x faster |
| Attention softmax (CUDA) | Eager `F.softmax` | **Triton fused** exp+sum+div, 3.5--8x faster |
| QJL projection (CUDA) | Separate matmul + compare | **`torch.compile`** fused, 1.5--2x faster |
| Extensions | None | **5 novel extensions** |
| Entropy coding | Not present | **Huffman coding** on indices |
| High-level API | None | **`KVCacheQuantizer`** for (B,H,T,d) tensors |

---

---

# For contributors

> Clone the repo and set up a local dev environment.

---

## Setup

```bash
conda create -n kvquant python=3.11 -y
conda activate kvquant

# Install PyTorch follow https://pytorch.org/get-started/locally/ or CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install package in editable mode
pip install -e .          # base + GPU kernels
pip install -e ".[dev]"   # + pytest
```

---

## Running demos

```bash
# Basic quantization (no model download needed)
python -m kvquant.demo

# All 5 extensions on real data
python -m kvquant.demo_extensions

# Regenerate all plots
python -m kvquant.visualize

# Perplexity benchmark
python -m kvquant.eval_ppl
python -m kvquant.eval_ppl --model gpt2-medium --correction-rank 4

# Interactive generation at different bit-widths
python -m kvquant.demo_llm --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Explain ML" --max-new-tokens 80
```

---

## Running tests

```bash
python -m pytest tests/ -q
```

Tests marked `cuda` skip automatically without a device. The suite is green on
`main`; a hard-coded count in a README goes stale on the next commit, so run it
rather than trusting a number here.

---

## What is measured, and how

Every compression figure in this README is counted from bytes actually
resident, not derived from the nominal bit-width. The two rulers:

**Cache bytes** (`tests/cache_bytes.py::cache_nbytes`) walks the live cache
object and classifies every tensor it holds:

- `code_bytes` — the bit-packed Lloyd-Max indices, the payload.
- `sidecar_bytes` — everything else the codec needs to reconstruct: the
  **per-vector float32 L2 norms** (two per quantized vector, one for each of the
  outlier and regular channel groups), channel permutations, and codebooks.
  This is 16–22% of the total at head_dim 64, and it is what the paper's
  bits-per-channel figures leave out.
- `float_bytes` — the **pending window**. Tokens accumulate in float until they
  fill a block, so up to `block_size - 1` positions are uncompressed at any
  moment. `generate()` flushes at the end, but mid-stream this is real memory.

`GenerateResult.compression_ratio` is `fp16_bytes / (code + sidecar + float)`,
and `GenerateResult.measured` is `True` when it was counted rather than assumed.
`test_reported_ratio_equals_measured` pins the two together.

**Peak VRAM** (`tests/vram.py::measure_peak_vram`) uses
`torch.cuda.max_memory_allocated`, minus a baseline captured after
`gc.collect()` + `empty_cache()` + `reset_peak_memory_stats()`. Three choices
worth naming: the baseline is **subtracted** (model weights would swamp the
delta); it reads `memory_allocated`, **not** `memory_reserved` (the reserved
pool is sticky and would mask the win); and float and compact runs happen
**back-to-back in one process** with a cache flush between, so they see the same
allocator state. A self-test allocates a known 64 MiB and asserts the harness
reports it within 10% — a broken ruler otherwise "proves" whatever you like.

**Warm the codec before measuring.** The first compact-cache forward on a device
costs roughly 9 MB more than every later one: Lloyd-Max codebooks are solved and
cached per `(dim, num_bits)` in a process-global dict, rotation buffers migrate to
the device, and cuBLAS picks its workspace. Attribute that to the cache and the
compact path can read as *worse* than float — measuring `bits ∈ {2,3,4}` in one
process without warming reported 1225 MiB for whichever ran first and 265 MiB for
the rest, which looks exactly like a bit-width regression and is not one. The
session-scoped `warm_cuda_codec` fixture in `tests/conftest.py` pays this once.

**Quality** is judged on logits and perplexity, never on generated text.
Sampling makes text a bad instrument: with `repetition_penalty=1.3`, a ~4e-4
cosine difference in reconstruction flips one greedy argmax and the whole
continuation diverges, which says nothing about codec quality. On a random-weight
test fixture it is worse than useless — deleting the causal mask entirely leaves
the output byte-identical while moving max logit error from 0.068 to 2.07.

Perplexity on held-out text (Qwen2.5-0.5B-Instruct, 256-token context):
fp32 baseline **22.90**; 4-bit **30.35**, 3-bit **38.93**, 2-bit **46.35**.
Monotone in bits, as reconstruction error is at every level of the codec.

---

## Building CUDA flash attention extensions (optional)

Build instructions, requirements, and supported SM targets are in the [cuda-triton-multiarch README](https://github.com/syedMohib44/cuda-triton-multiarch/blob/main/README.md).

---

## Architecture

```
kvquant/
|-- quantizer.py          # KVQuantMSE, KVQuantIP
|-- rotation.py           # RandomRotation (QR), HadamardRotation (WHT)
|-- codebook.py           # Lloyd-Max on true sphere distribution
|-- entropy.py            # Huffman coding on indices
|-- outlier.py            # Per-channel bit allocation for outliers
|-- kv_cache.py           # KVCacheQuantizer - high-level (B,H,T,d) API
|-- attn_weighted.py      # Extension 1: attention-weighted quantization
|-- delta.py              # Extension 2: delta / temporal compression
|-- adaptive.py           # Extension 3: EMA-based adaptive bit allocation
|-- correction.py         # Extension 4: low-rank error correction
|-- product_quantizer.py  # Extension 5: product quantization
+-- csrc/                 # GPU acceleration
    |-- pq_encode.py      # Triton PQ encode kernel (10-15x vs cdist loop)
    |-- softmax.py        # Triton row-wise softmax (3.5-8x vs F.softmax)
    +-- attention.py      # Flash attention bridge: WMMA/CuTe/Triton/SDPA
```

## Citation

```bibtex
@misc{kvquantpp2025,
  title   = {KVQuant: Attention Aware, Structure Exploiting Extensions to KV Cache Compression via Near Optimal Vector Quantization},
  author  = {Uddin, Syed Muheeb},
  year    = {2025},
  doi     = {10.17605/OSF.IO/9WSKZ},
  url     = {https://doi.org/10.17605/OSF.IO/9WSKZ},
  note    = {Extensions of TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (Zandieh et al., arXiv:2504.19874)}
}
```
