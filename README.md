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

![MSE Bounds](https://github.com/syedMohib44/kvquant/blob/main/plots/1_mse_bounds?raw=true)

---

### 2. Codebook Centroids (True Sphere Distribution)

Unlike the original KVQuant which uses a Gaussian approximation, we fit Lloyd-Max centroids directly to the **true unit-sphere marginal distribution** `f(t) alpha (1 - t^2)^{(d-3)/2}`. This gives tighter quantization, especially at low bit-widths and small `d`.

![Codebook Centroids](https://github.com/syedMohib44/kvquant/blob/main/plots/2_codebook_centroids?raw=true)

---

### 3. Rotation Effect

Random rotation (QR or Hadamard) transforms raw KV coordinates - which have arbitrary non-Gaussian distributions - into approximately N(0, 1/d), enabling optimal per-coordinate Lloyd-Max quantization.

![Rotation Effect](https://github.com/syedMohib44/kvquant/blob/main/plots/3_rotation_effect?raw=true)

---

### 4. Attention-Weighted Quantization (Extension 1)

Rather than allocating bits uniformly, AWQ assigns more bits to tokens that receive high attention from queries. At the same average bit-width, this reduces **attention-weighted distortion by 56.5% on average** across all distilgpt2 layers.

![Attention-Weighted Quantization](https://github.com/syedMohib44/kvquant/blob/main/plots/4_attention_weighted?raw=true)

---

### 5. Delta Compression (Extension 2)

Consecutive KV vectors are highly correlated. Compressing token-to-token deltas `delta_t = k_t - k(cap)_{t-1}` instead of absolute vectors gives **1.1--2.2x lower MSE**, especially in early layers with smoother trajectories.

![Delta Compression](https://github.com/syedMohib44/kvquant/blob/main/plots/5_delta_compression?raw=true)

---

### 6. Adaptive Bit Allocation (Extension 3)

Token importance evolves during generation. An EMA-based importance tracker dynamically reassigns bit-widths as attention scores accumulate, giving more bits to tokens that prove important over time.

![Adaptive Allocation](https://github.com/syedMohib44/kvquant/blob/main/plots/6_adaptive_allocation?raw=true)

---

### 7. Low-Rank Error Correction (Extension 4)

Quantization error `R = K - k(cap)` is structured - its top singular vectors capture most of the error energy. A rank-r SVD correction reduces MSE by **~11% at rank-4** using only 6.3% extra storage.

![Low-Rank Correction](https://github.com/syedMohib44/kvquant/blob/main/plots/7_low_rank_correction?raw=true)

---

### 8. Entropy Coding

Codebook indices are non-uniformly distributed after rotation. Huffman coding reduces storage toward the Shannon entropy - at 4-bit, d=128: ~5% savings.

![Entropy Coding](https://github.com/syedMohib44/kvquant/blob/main/plots/8_entropy_coding?raw=true)

---

### 9. Full Pipeline Comparison

End-to-end comparison of baseline KVQuant vs the combined KVQUANT pipeline across all layers and bit-widths.

![Pipeline Comparison](https://github.com/syedMohib44/kvquant/blob/main/plots/9_pipeline_comparison?raw=true)

---

### 10. Compression Ratio

Storage cost vs reconstruction quality across configurations.

![Compression Ratio](https://github.com/syedMohib44/kvquant/blob/main/plots/10_ratio_check?raw=true)

---

### 11. Perplexity Validation

End-to-end perplexity on distilgpt2 confirms that quantization quality improvements preserve language modelling performance.

![Perplexity Validation](https://github.com/syedMohib44/kvquant/blob/main/plots/11_perplexity_validation?raw=true)

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
pip install -e .
OR
pip install -e ".[dev]"
```

This installs all dependencies (`torch`, `transformers`, `numpy`, `matplotlib`) and `pytest`, and makes the `kvquant` package importable from anywhere as long as the `kvquant` conda environment is active.

**Requirements:** Python >= 3.10, PyTorch >= 2.1, Transformers >= 4.40

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

Run demos as **modules** from inside the `kvquant` directory with the `kvquant` env active:

```bash
python -m kvquant.demo              # basic quantization examples
python -m kvquant.demo_llm          # real distilgpt2 KV cache 

#### Prompt testing ####
python -m kvquant.demo_llm --prompt "Hi how are you?" --max-new-tokens 60 # Pure transformer (default)
python -m kvquant.demo_llm --model Qwen/Qwen3.5-0.8B --prompt "What is AI?" # Hybrid model (Qwen3.5) - auto-detected, uses native cache
python -m kvquant.demo_llm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Once upon a time" # Recommended alternative models
python -m kvquant.demo_llm --model microsoft/phi-2 --prompt "Explain transformers" # Recommended alternative models
# Base model - auto Q/A format
python -m kvquant.demo_llm --model distilgpt2 --prompt "What is the capital of France?" --max-new-tokens 20
# Instruction model - chat template
python -m kvquant.demo_llm --model Qwen/Qwen2.5-1.5B-Instruct --prompt "What is the capital of France?" --max-new-tokens 20
# Hybrid thinking model - chat template + enable_thinking=False
python -m kvquant.demo_llm --model Qwen/Qwen3.5-0.8B --prompt "What is the capital of France?" --max-new-tokens 20
# Rank 4 will get us 11% MSE reduction at only 7.4% extra storage
python -m kvquant.demo_llm --model Qwen/Qwen2.5-1.5B-Instruct --prompt "What is the capital of France?" --max-new-tokens 20 --correction-rank 4
# Test for Mac M1
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m kvquant.demo_llm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "What is the capital of France?" --max-new-tokens 20
# Product Quant (PQ) good with accuracy this will use PQ instead of BS 
python -m kvquant.demo_llm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "What is Nihilism?" --max-new-tokens 100 --product-quant

# The combination of PQ + low rank correction at the same 2 bits/dim storage is the strongest option PQ captures inter-dimension correlations, correction cleans up the residual error that PQ leaves behind. Use:
python -m kvquant.demo_llm --model TinyLlama/... --prompt "..." \
  --product-quant --pq-bits 8 --pq-subspaces 16 --correction-rank 4

#########################

python -m kvquant.demo_extensions   # all 5 extensions on real data
python -m kvquant.visualize         # regenerate all plots -> https://github.com/syedMohib44/kvquant/blob/main/plots/
```

---

## Quick start

```python
from kvquant import KVCacheQuantizer

# Standard 2-bit KV cache quantization
quant = KVCacheQuantizer(head_dim=64, num_bits=2)
quant.calibrate(keys)
k_compressed = quant.compress(keys)
k_reconstructed = quant.decompress(k_compressed)

# Attention-weighted quantization
from kvquant import AttentionWeightedQuantizer
awq = AttentionWeightedQuantizer(head_dim=64, hi_bits=4, lo_bits=2, top_fraction=0.5)
q_weighted = awq.quantize(keys, attn_weights=attn_scores)

# Delta compression for streaming
from kvquant import DeltaKVCache
delta = DeltaKVCache(head_dim=64, num_bits=3)
for token_key in stream:
    compressed = delta.push(token_key)

# Low-rank error correction
from kvquant import LowRankCorrection
lrc = LowRankCorrection(head_dim=64, num_bits=2, rank=4)
k_corrected = lrc.forward(keys)
```

---

## Running tests

```bash
python -m pytest test_kvquant.py -v   # 88 tests, all pass
# or simply:
pytest
```

---

## Implementation improvements over reference

| | Reference | KVQUANT |
|---|---|---|
| Codebook distribution | Gaussian approximation | **True sphere marginal** |
| Rotation | May produce reflections (det = -1) | **SO(d) enforced** (det = +1) |
| Nearest-centroid lookup | `argmin` - O(N.d.k) | **`bucketize`** - O(N.d.log k), 14--22x faster |
| FWHT | 2 clones per butterfly level | **In-place**, 1 allocation per level |
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
+-- correction.py      # Extension 4: low-rank error correction
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