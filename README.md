**[Read the Paper (PDF)](https://osf.io/9wskz/files/xzb7d)** | **[DOI: 10.17605/OSF.IO/9WSKZ](https://doi.org/10.17605/OSF.IO/9WSKZ)** | **[View on GitHub](https://github.com/syedMohib44/kvquant)**

# KVQUANT

**Attention-aware KV cache quantization for LLM inference.**

KVQUANT extends [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) with five novel extensions and several implementation improvements. It achieves near-optimal KV cache compression by combining information-theoretically grounded vector quantization with transformer-specific structure exploitation.

---

## Results at a glance

| Metric | Value |
|---|---|
| Attention-weighted distortion reduction | **47--70%** per layer (same bit-width) |
| Delta compression MSE improvement | **1.1--2.2x** on correlated streams |
| Low-rank correction MSE reduction | **~11%** at rank-4, ~19% at rank-8 |
| Codebook lookup speedup vs argmin | **14--22x** (bucketize) |
| Test suite | **88/88 passing** |

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

Quantization error `R = K - k(cap)` is structured - its top singular vectors capture most of the error energy. A rank-r SVD correction reduces MSE by **~11% at rank-4** using only 6.3% extra storage.

![Low-Rank Correction](https://github.com/syedMohib44/kvquant/blob/main/plots/7_low_rank_correction.png?raw=true)

---

### 8. Entropy Coding

Codebook indices are non-uniformly distributed after rotation. Huffman coding reduces storage toward the Shannon entropy - at 4-bit, d=128: ~5% savings.

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

## Installation

### 1. Create and activate a conda environment

```bash
conda create -n kvquant python=3.11 -y
conda activate kvquant
```

### 2. Install PyTorch

Follow the instructions at https://pytorch.org/get-started/locally/ for your platform, or for a CPU-only install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the package (editable)

Run this from the `kvquant/` directory (the folder that contains `pyproject.toml`):

```bash
pip install -e .           # includes GPU kernels (Triton JIT) + PyTorch fallbacks
pip install -e ".[dev]"    # + pytest
pip install -e ".[cuda]"   # + explicit Triton version pin (optional)
```

This installs all dependencies (`torch`, `transformers`, `numpy`, `matplotlib`, `cuda-triton-kernels`) and makes the `kvquant` package importable from anywhere as long as the environment is active.

**Requirements:** Python >= 3.10, PyTorch >= 2.1, Transformers >= 4.40

GPU kernels (Triton JIT softmax, flash attention, matmul) are included in the base install — no `[cuda]` extra needed. All paths fall back to pure PyTorch automatically on CPU, AMD, or Apple MPS. For maximum flash attention performance, build the optional WMMA/CUTLASS CUDA extensions from source (see [GPU Acceleration](#gpu-acceleration) below).

### 4. Configure VS Code (optional)

1. Open VS Code in the `kvquant` folder.
2. Press `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose the `kvquant` conda environment (path will look like `~\anaconda3\envs\kvquant\python.exe`).
3. Press `Ctrl+Shift+P` -> **Developer: Reload Window**.

If Pylance still shows `Import "kvquant" could not be resolved`, add this to your `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": [".."]
}
```

(The `..` points one level above the `kvquant` folder so that `import kvquant` resolves correctly.)

---

## Running demos

Run from the repo root with the `kvquant` conda env active:

```bash
# Basic quantization examples (no model download needed)
python -m kvquant.demo

# All 5 novel extensions on real data
python -m kvquant.demo_extensions

# Regenerate all plots
python -m kvquant.visualize

# Perplexity benchmark (distilgpt2 vs 2/3/4-bit KV cache)
python -m kvquant.eval_ppl
python -m kvquant.eval_ppl --model gpt2-medium --correction-rank 4
```

### demo_llm — interactive generation at different bit-widths

Runs the same prompt through unquantized, 2-bit, 3-bit, and 4-bit KV caches side by side so you can see the quality difference directly.

```bash
# Default benchmark (distilgpt2, no download)
python -m kvquant.demo_llm

# Base model — auto Q/A prompt format
python -m kvquant.demo_llm \
  --model distilgpt2 \
  --prompt "What is the capital of France?" \
  --max-new-tokens 30

# Instruct model — uses the model's built-in chat template
python -m kvquant.demo_llm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain machine learning in simple terms" \
  --max-new-tokens 80

# TinyLlama — fast, recommended for quick tests
python -m kvquant.demo_llm \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is nihilism?" \
  --max-new-tokens 100

# Hybrid model (Qwen3.5) — auto-detected, uses native cache
python -m kvquant.demo_llm \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "What is AI?" \
  --max-new-tokens 60

# Sentence completion — skip chat template with --raw
python -m kvquant.demo_llm \
  --model distilgpt2 \
  --prompt "The Eiffel Tower is located in" \
  --raw --max-new-tokens 20

# Low-rank correction — +11% quality at 3-bit
python -m kvquant.demo_llm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Describe the water cycle" \
  --correction-rank 4

# Product Quantization — 2 bits/dim via learned codebooks (73x faster encode)
python -m kvquant.demo_llm \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is nihilism?" \
  --max-new-tokens 100 \
  --product-quant

# PQ + low-rank correction — strongest combination at 2 bits/dim
# PQ captures inter-dimension correlations; correction cleans up the residual
python -m kvquant.demo_llm \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain the Turing test" \
  --product-quant --pq-bits 8 --pq-subspaces 16 --correction-rank 4

# Large model (requires enough VRAM or use device_map in Python API)
python -m kvquant.demo_llm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt "Explain quantum entanglement" \
  --max-new-tokens 80 --correction-rank 4

# Mac / Apple MPS (suppress tokenizer parallelism warning)
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m kvquant.demo_llm \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "What is the capital of France?" \
  --max-new-tokens 20
```

---

## Quick start

### Install from PyPI

```bash
pip install kvquant-plus-plus
```

---

### generate() — full response

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

### stream() — print tokens as they arrive

```python
from kvquant import stream

for token in stream("Explain transformers in simple terms"):
    print(token, end="", flush=True)
print()
```

---

### System prompt — role, persona, or instructions

Pass any combination of system and user prompt. Both are quantized together in a single prefill pass — no extra compute cost.

```python
from kvquant import generate, stream

# System prompt only sets the model's role; user prompt is the question
out = generate(
    "What is quantum entanglement?",
    system="You are a physics professor. Answer at undergraduate level.",
    bits=3,
)
print(out.text)

# User prompt only (system omitted — same as before)
out = generate("What is quantum entanglement?", bits=3)
print(out.text)

# Long document analysis — system sets the task, user supplies the document
contract = open("contract.txt").read()
out = generate(
    contract + "\n\nList all payment terms.",
    system="You are a legal analyst. Extract facts only, be concise.",
    bits=3,
    max_new_tokens=300,
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

### Bit-width options (2 / 3 / 4)

```python
from kvquant import generate

# 2-bit: most compressed (~8x smaller than float16), some quality loss
out = generate("Summarise the French Revolution", bits=2)
print(f"{out.compression_ratio:.1f}x  — {out.text}")

# 3-bit: recommended default (~5x smaller, minimal quality loss)
out = generate("Summarise the French Revolution", bits=3)
print(f"{out.compression_ratio:.1f}x  — {out.text}")

# 4-bit: best quality (~4x smaller, near-lossless)
out = generate("Summarise the French Revolution", bits=4)
print(f"{out.compression_ratio:.1f}x  — {out.text}")
```

---

### Sampling — creative / varied output

```python
from kvquant import generate

# Greedy (deterministic, default)
out = generate("Once upon a time", bits=3, temperature=0.0)

# Sampling with temperature
out = generate("Once upon a time", bits=3, temperature=0.8)

# Nucleus (top-p) sampling
out = generate("Once upon a time", bits=3, temperature=0.8, top_p=0.95)

print(out.text)
```

---

### Different models

Works with any HuggingFace causal LM — instruct, base, and hybrid architectures (Qwen, Llama, Phi, Mistral, Falcon, Gemma …).

```python
from kvquant import generate

# Small instruct model (default, downloads ~1 GB)
out = generate("What is quantum entanglement?",
               model="Qwen/Qwen2.5-1.5B-Instruct", bits=3)

# TinyLlama — fast, great for testing
out = generate("What is the capital of France?",
               model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", bits=3)

# Larger instruct model
out = generate("Explain the Turing test",
               model="Qwen/Qwen2.5-7B-Instruct", bits=3)

# Base (non-instruct) model — use raw=True for sentence completion
out = generate("The Eiffel Tower is located in",
               model="distilgpt2", bits=3, raw=True)

print(out.text)
```

---

### Multi-GPU — large models

```python
from kvquant import generate

# device_map="auto" spreads the model across all available GPUs
out = generate(
    "Explain quantum computing in detail",
    model="meta-llama/Llama-3.1-8B-Instruct",
    bits=3,
    device_map="auto",
)
print(out.text)
print(f"{out.compression_ratio:.1f}x vs float16")
```

---

### Low-rank error correction (+11% quality)

At 2-bit and 3-bit, a rank-4 SVD correction of the quantization residual recovers ~11% of the MSE at only 6–7% extra storage. Recommended for low bit-widths.

```python
from kvquant import generate

out = generate(
    "Explain the history of the Roman Empire",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    bits=3,
    correction_rank=4,    # 0 = disabled (default), 4 = recommended
    max_new_tokens=300,
)
print(out.text)
```

---

### Repetition control

```python
from kvquant import generate

# Default repetition_penalty=1.3 already handles most cases
# Increase to 1.5-2.0 if you see looping at 2-bit
out = generate("Tell me about AI", bits=2, repetition_penalty=1.5)
print(out.text)
```

---

> **What are keys and values?**  When a language model processes text it stores intermediate
> representations called the KV cache — one key and value vector per token per layer.
> These tensors grow with context length and consume most of GPU memory during inference.
> kvquant compresses them from 16 bits to 2–4 bits with minimal quality loss.

---

## Low-level API

For integrating directly into a training loop, custom transformer, or research pipeline. All classes accept KV tensors as-is — `(T, head_dim)`, `(B, H, T, head_dim)`, or anything in between.

---

### KVCacheQuantizer — compress / decompress a KV cache

```python
from kvquant import KVCacheQuantizer

# keys, values: KV tensors from your transformer, shape (..., head_dim)
head_dim = keys.shape[-1]

quant = KVCacheQuantizer(head_dim=head_dim, num_bits=3)

# One-shot calibration — use the actual prefill KVs from your context
quant.calibrate(keys, values)

# Compress (K uses inner-product-optimal, V uses MSE-optimal quantization)
k_c, v_c     = quant.compress_kv(keys, values)
k_hat, v_hat = quant.decompress_kv(k_c, v_c)

print(f"avg bits/dim: {quant.avg_bits:.2f}")
```

---

### AttentionWeightedQuantizer — more bits for tokens the model attends to

Sink tokens (positions 0–3) are always pinned to `hi_bits` regardless of their attention score.

```python
from kvquant import AttentionWeightedQuantizer

awq = AttentionWeightedQuantizer(dim=head_dim, hi_bits=4, lo_bits=2, top_fraction=0.5)

# keys: (..., T, head_dim)   query: (..., head_dim)
compressed = awq.quantize(keys, query)
k_hat      = awq.dequantize(compressed)

print(f"avg bits: {awq.avg_bits:.2f}")   # 3.0 at top_fraction=0.5
```

---

### DeltaKVCache — streaming / autoregressive generation

Compresses token-to-token deltas instead of absolute vectors. Works because consecutive KV vectors are highly correlated (1.1–2.2x lower MSE than absolute compression at the same bit-width).

```python
from kvquant import DeltaKVCache

cache = DeltaKVCache(head_dim=head_dim, num_bits=3)

# During generation — call once per new token
for k_new, v_new in token_stream:     # k_new, v_new: (..., head_dim)
    cache.push(k_new, v_new)

K_hat, V_hat = cache.get()            # full reconstructed cache, O(1)
print(f"cache length: {cache.length}")
```

---

### AdaptiveKVCache — importance-based bit allocation

Tracks a running EMA importance score per token position and dynamically reassigns bits. High-importance tokens get more bits; low-importance tokens are compressed further. Sink tokens (positions 0–3) are always kept at `hi_bits`.

```python
from kvquant import AdaptiveKVCache

cache = AdaptiveKVCache(head_dim=head_dim)   # tiers: 4 / 3 / 2 / 1 bits

# Prefill
for k_new, v_new in token_stream:
    cache.push(k_new, v_new)

# After each forward pass, supply the softmax attention weights
cache.attend(attn_weights)    # attn_weights: (..., T) from your softmax

K_hat, V_hat = cache.get()
print(cache.bit_allocation()) # e.g. {4: 8, 3: 50, 2: 60, 1: 10}
print(f"avg bits: {cache.avg_bits():.2f}")
```

---

### LowRankCorrection — SVD residual correction

Captures the structured part of quantization error with a rank-r SVD. Reduces MSE by ~11% at rank-4, ~19% at rank-8, using only 6–7% extra storage.

```python
from kvquant import LowRankCorrection, KVQuantMSE

lrc = LowRankCorrection(quantizer=KVQuantMSE(dim=head_dim, num_bits=2), rank=4)

# Quantize + correct in one call
k_corrected = lrc.forward(keys)

# Analyse how much energy each rank captures
energy = lrc.residual_rank_analysis(keys, max_rank=8)
print(energy)   # cumulative fraction at each rank
```

---

### ProductQuantizer — subspace codebook quantization

Splits each vector into M subspaces and encodes each independently using a learned codebook. Encode is 73x faster than a cdist loop via the included Triton kernel.

```python
from kvquant import ProductQuantizer

pq = ProductQuantizer(dim=head_dim, num_subspaces=16, bits_per_subspace=8)

# Calibrate codebooks on a pool of vectors
pq.calibrate(keys.reshape(-1, head_dim))

compressed = pq.quantize(keys.reshape(-1, head_dim))
k_hat      = pq.dequantize(compressed)

print(f"compression ratio: {pq.compression_ratio():.1f}x vs float32")
print(f"effective bits/dim: {pq.effective_bits:.2f}")
```

---

## Running tests

```bash
# From the repo root
python -m pytest test_kvquant.py -v   # 88 tests, all pass
# or simply:
pytest
```

---

## GPU Acceleration

GPU kernels are included in the base install — no extra flag needed:

```bash
pip install kvquant-plus-plus
```

For an explicit Triton version pin:

```bash
pip install "kvquant-plus-plus[cuda]"
```

| Hot path | Mechanism | Speedup |
|---|---|---|
| PQ encode (M=16 subspace lookups) | Triton kernel — all M subspaces in one GPU launch | **10--15x** |
| FWHT (d=128, 7 butterfly iterations) | `torch.compile` — fuses 7 kernel launches into 1 | **2--3x** |
| Attention softmax (AWQ path) | Triton row-wise fused exp+sum+div | **3.5--8x** |
| QJL projection `r @ S.T > 0` | `torch.compile` — fuses matmul + compare | **1.5--2x** |
| Flash attention (post-decompression) | WMMA / CuTe CUDA → Triton → PyTorch SDPA cascade | **up to native FA-2 speed** |

Kernels are adapted from [cuda-triton-multiarch](https://github.com/syedMohib44/cuda-triton-multiarch) and compile JIT on first call — no `nvcc`, no build step.

### Flash attention backend

`pip install "kvquant-plus-plus[cuda]"` also installs `cuda-triton-kernels`, which provides `attention_bhsd` — a multi-head flash attention function for `(B, H, T, d)` tensors (KVQuant's native layout):

```python
from kvquant.csrc.attention import attention_bhsd, attention_backend

# q, k, v: (batch, heads, seqlen, head_dim)
out = attention_bhsd(q, k, v, is_causal=False)

print(attention_backend())
# → "flash_attn_cuda (WMMA)"           fastest — requires CUDA build step (see below)
# → "flash_attn_cutlass (CuTe)"        fast — requires CUDA build step (see below)
# → "flash_attention_triton (Triton)"  JIT — available with [cuda] install, no build
# → "scaled_dot_product_attention (PyTorch)"  always available fallback
```

The Triton backend is active out of the box after `pip install "kvquant-plus-plus[cuda]"`. The WMMA and CuTe CUDA backends give additional speedups but require building C++ extensions from source (see below).

### Optional: build CUDA flash attention extensions

For maximum flash attention performance, build the CUDA extensions from [cuda-triton-multiarch](https://github.com/syedMohib44/cuda-triton-multiarch). These require `nvcc` (CUDA Toolkit 12.x/13.x) and, on Windows, Visual Studio Build Tools 2019+.

```bash
git clone https://github.com/syedMohib44/cuda-triton-multiarch.git
cd cuda-triton-multiarch

# Linux / WSL2
make build-fac            # WMMA FlashAttention (SM75/80/86/89/120)
make build-fac-cutlass    # CuTe/CUTLASS FlashAttention (SM80+)

# Windows (Native)
powershell -ExecutionPolicy Bypass -File Makefile.windows.ps1 build-fac
powershell -ExecutionPolicy Bypass -File Makefile.windows.ps1 build-fac-cutlass
```

After building, `attention_bhsd` automatically picks up the compiled extensions — no code changes needed. Check the active backend with `attention_backend()`.

**GPU support:** any NVIDIA GPU SM >= 7.5 (RTX 20xx / 30xx / 40xx / 50xx, A100, H100). Falls back silently to PyTorch on CPU, AMD, or Apple MPS.

| GPU family | SM | Supported |
|---|---|---|
| RTX 20xx / T4 | SM 75 | Yes |
| RTX 30xx / A10 | SM 86 | Yes |
| RTX 40xx / L40S | SM 89 | Yes |
| RTX 50xx (Blackwell) | SM 120 | Yes |
| A100 | SM 80 | Yes |
| H100 | SM 90 | Yes |
| CPU / Apple MPS | -- | PyTorch fallback |

**Windows support:** `pip install "kvquant-plus-plus[cuda]"` automatically installs `triton-windows` on Windows and `triton` on Linux/macOS — no manual steps. GPU kernel performance is identical across platforms (same PTX/CUBIN). Kernels are adapted from [cuda-triton-multiarch](https://github.com/syedMohib44/cuda-triton-multiarch) which also supports Windows natively.

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

## Architecture

```
kvquant/
|-- quantizer.py       # KVQuantMSE, KVQuantIP
|-- rotation.py        # RandomRotation (QR), HadamardRotation (WHT)
|-- codebook.py        # Lloyd-Max on true sphere distribution
|-- entropy.py         # Huffman coding on indices
|-- outlier.py         # Per-channel bit allocation for outliers
|-- kv_cache.py        # KVCacheQuantizer - high-level (B,H,T,d) API
|-- attn_weighted.py   # Extension 1: attention-weighted quantization
|-- delta.py           # Extension 2: delta / temporal compression
|-- adaptive.py        # Extension 3: EMA-based adaptive bit allocation
|-- correction.py      # Extension 4: low-rank error correction
|-- product_quantizer.py  # Extension 5: product quantization
+-- csrc/              # GPU acceleration (optional, requires [cuda] extra)
    |-- pq_encode.py   # Triton PQ encode kernel (10-15x vs cdist loop)
    |-- softmax.py     # Triton row-wise softmax (3.5-8x vs F.softmax)
    +-- attention.py   # Flash attention bridge — WMMA/CuTe/Triton/SDPA cascade
```

## Example Output

```bash
python -m kvquant.demo_llm --prompt "Hi how are you?"
```
---- Interactive generation ------------------------------------
  Prompt : 'Hi how are you?'

  Unquant: Hi how are you? I'm a big fan of the game and I'm really excited to see how it will be released. I'm really excited to see how it will be released. I'm really excited to see how

  2-bit  : Hi how are you? I's a guy who is a guy who is a guy who is a guy who is a guy who is a guy who is a guy who is a guy who is a guy who is

  3-bit  : Hi how are you? I'm a little bit shy, but I'm a little bit shy, but I'm a little bit shy, but I'm a little bit shy, but
  
  4-bit  : Hi how are you? I'm a big fan of the game and I'm really excited to see what you do. I'm really excited to see what you do. I'm a big fan of the game and I'm
----------------------------------------------------------------

---

## Publishing to PyPI

### Build the package

Run from the **parent** directory of `kvquant/` (i.e., one level up):

```bash
cd ..
python -m build kvquant
```

This produces two files in `kvquant/dist/`:
- `kvquant_plus_plus-x.x.x-py3-none-any.whl`
- `kvquant_plus_plus-x.x.x.tar.gz`

### Upload to PyPI

```bash
twine upload kvquant/dist/kvquant_plus_plus-x.x.x*
```

Credentials are read from `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Or via environment variables:

```bash
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
twine upload kvquant/dist/*
```

### Version bumps

Update `version` in `pyproject.toml` before each release. PyPI does not allow overwriting an existing version.

```toml
[project]
version = "0.1.2"
```

Then rebuild and upload:

```bash
python -m build kvquant
twine upload kvquant/dist/kvquant_plus_plus-0.1.2*
```

### Install from PyPI

```bash
pip install kvquant-plus-plus              # GPU kernels included (Triton JIT)
pip install "kvquant-plus-plus[cuda]"      # same + explicit Triton version pin
```

PyPI page: https://pypi.org/project/kvquant-plus-plus/

---

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