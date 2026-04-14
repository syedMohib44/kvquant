# KVQuant++ - Complete Code-Level Study

---

## Centroids

Centroids are a small fixed set of representative values (e.g. 4 for 2-bit, 16 for 4-bit). Every real value in the vector gets replaced by its nearest centroid - that's quantization. The centroid index (2 bits) is stored instead of the full float (32 bits).

The optimal centroid positions depend on the distribution of input values.


If data is uniform:         centroids are equally spaced
If data is bell-shaped:     centroids cluster in the middle (more data there)
If data is sphere marginal: centroids are slightly different from Gaussian bell

Wrong distribution -> wrong centroid positions -> more error for the same 

# In code: how centroids are used

# centroids is a 1D tensor, sorted ascending
centroids = tensor([-0.87, -0.29, +0.29, +0.87])   # shape (4,) for 2-bit

# ---- QUANTIZE: find nearest centroid index ----
# Instead of: diff = (y - centroids).abs(); index = diff.argmin()
# We use binary search on the midpoints:
boundaries = (centroids[:-1] + centroids[1:]) / 2   # [-0.58, 0.0, +0.58]
index = torch.bucketize(y, boundaries)
# y = 0.31 -> falls in bucket 2 -> index = 2

# ---- DEQUANTIZE: look up centroid value ----
y_reconstructed = centroids[index]    # centroids[2] = +0.29


## Notation Reference

| Notation | Reads as | Concrete meaning |
|---|---|---|
| `k ∈ ℝᵈ` | k is in R-d | k is a d-dimensional vector of real numbers |
| `∈` | "element of" / "belongs to" | the thing on the left is a member of the set on the right |
| `ℝ` | "real numbers" | all ordinary decimal numbers (not complex, not integers only) |
| `ℝᵈ` | R-d | the set of all vectors with exactly d real-number entries |
| `ℝᵀˣᵈ` | R T-by-d | the set of all T×d matrices (T rows, d columns) |
| `K ∈ ℝᵀˣᵈ` | K is in R T-by-d | K is a matrix: T tokens, each a d-dimensional vector |
| `Π ∈ ℝᵈˣᵈ` | Pi is in R d-by-d | Π is a square d×d matrix (the rotation matrix) |
| `y = Πk` | y equals Pi times k | matrix-vector multiply - rotate k into y |
| `‖k‖` | norm of k | length of the vector: sqrt(k₁² + k₂² + ... + kᵈ²) |
| `k / ‖k‖` | k over norm-k | unit vector - same direction, length exactly 1 |
| `Sᵈ⁻¹` | S d-minus-1 | the unit sphere in d dimensions (all vectors with length = 1) |
| `⟨q, k⟩` | inner product of q and k | dot product: q₁k₁ + q₂k₂ + ... + qᵈkᵈ |
| `f(t) ∝ g(t)` | f proportional to g | same shape, different scale constant |
| `f(t) = C_d·(1-t²)^((d-3)/2)` | sphere marginal | probability density of one coordinate of a random unit vector in ℝᵈ |
| `4⁻ᵇ` | 4 to the minus b | 1/4ᵇ - gets smaller as bits b increase (better quality) |
| `O(N·d·log k)` | big-O of N d log k | how runtime scales - the bucketize lookup complexity |
| `det(Q)` | determinant of Q | scalar: +1 for rotations, -1 for reflections |
| `SO(d)` | special orthogonal group | all proper rotations in d dimensions (det = +1, no reflections) |
| `N(0, 1/d)` | Gaussian mean 0, variance 1/d | the distribution each rotated coordinate approximately follows |
| `b` | bits | bits per coordinate used for quantization (1–4 in this codebase) |
| `k` (codebook) | number of centroids | k = 2ᵇ - e.g. 4-bit -> 16 centroids |
| `T` | sequence length | number of tokens in the KV cache |
| `d` | head dimension | size of each key/value vector (e.g. 64, 128, 256) |
| `r` | rank | number of singular vectors used in low-rank correction |
| `α` (alpha) | EMA decay factor | controls how fast importance scores update (e.g. 0.9) |
| `H(p)` | Shannon entropy | -∑ pᵢ log₂(pᵢ) - the theoretical minimum bits needed |
| `Φ` | normal CDF | cumulative distribution function of a Gaussian |
| `B, H` | batch, heads | batch size and number of attention heads in transformer shapes |

### Geometry in one picture

```
ℝ¹: the number line           [x]
ℝ²: the x-y plane            [x, y]
ℝ³: x-y-z space              [x, y, z]
ℝᵈ: d-dimensional space      [k₁, k₂, ..., kᵈ]

S⁰: two points {-1, +1}
S¹: unit circle in ℝ²        all [x,y] where x²+y²=1
S²: unit sphere in ℝ³        all [x,y,z] where x²+y²+z²=1
Sᵈ⁻¹: unit sphere in ℝᵈ     all k where ‖k‖=1

k_unit = k / ‖k‖   ->   projects any vector onto Sᵈ⁻¹
```

In code: `x_unit = x / x.norm()` - projecting onto Sᵈ⁻¹.

---

## Package Structure (`__init__.py`)

The package exposes everything from one flat namespace:

```python
from .codebook  import build_codebook, PRECOMPUTED_CENTROIDS
from .rotation  import RandomRotation, HadamardRotation
from .quantizer import KVQuantMSE, KVQuantIP, QuantizedMSE, QuantizedIP, CompressedMSE
from .outlier   import OutlierKVQuant
from .kv_cache  import KVCacheQuantizer
from .entropy   import HuffmanCodec, codebook_probs, entropy_bits, analyse
from .attn_weighted import AttentionWeightedQuantizer, weighted_distortion
from .delta     import DeltaKVCache
from .adaptive  import AdaptiveKVCache
from .correction import LowRankCorrection
```

Every public class is an `nn.Module` with `.quantize()` / `.dequantize()` and returns a typed `NamedTuple`.
This is the contract that lets them compose - one module's output is another's input.

---

## 1. The Foundation: What a vector looks like going in

Every key vector `k ∈ ℝᵈ` in the KV cache is a point on (or near) the unit sphere `Sᵈ⁻¹`.
If `k` isn't unit-norm, it gets normalized first and the norm is saved separately.

```
k  ->  norm = ||k||  ->  k_unit = k / norm   (on unit sphere Sᵈ⁻¹)
                                  ↓
                              [quantize]
                                  ↓
k_hat = dequantize(...) * norm    (restore original scale)
```

This norm-save-and-restore pattern appears in **every quantizer** in the codebase - MSE, IP, Outlier.
Without it, quantizing a large-magnitude vector with small-magnitude codebook centroids would produce huge errors.

---

## 2. [`codebook.py`] - Building the optimal scalar quantizer

### Why a custom codebook?

After rotation, each coordinate `yⱼ` follows the **true sphere marginal**:

```
f(t) = C_d · (1 - t²)^((d-3)/2)    for t ∈ [-1, 1]
```

This is a **Beta distribution**, not Gaussian. The original KVQuant used hardcoded Gaussian centroids
(`_KNOWN_GAUSSIAN_CENTROIDS` dict in the file - kept for reference). At `b=1,2` and small `d` the
difference is visible; by `b=4` it barely matters.

### Step 1 - Sample the true distribution

```python
def _sample_sphere_coord(dim: int, num_samples: int, seed: int = 42) -> Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    g = torch.randn(num_samples, dim, generator=gen)   # (N, d) Gaussian
    u = g / g.norm(dim=-1, keepdim=True)               # (N, d) unit vectors
    return u[:, 0]   # first coordinate - any coordinate has the same marginal
```

Why does this work? A Gaussian vector normalized to the sphere is **Haar-uniform** on Sᵈ⁻¹.
By symmetry, every coordinate has the same marginal distribution `f(t) = C_d·(1-t²)^((d-3)/2)`.
Picking column 0 is arbitrary - any column gives the same distribution.

### Step 2 - Run Lloyd-Max iteration

Lloyd-Max is the standard algorithm for 1-D optimal scalar quantization:
- **Assignment step:** assign each sample to its nearest centroid
- **Update step:** move each centroid to the mean of its assigned samples
- Repeat until convergence

```python
def _lloyd_max(num_bits, dim, num_steps=2000, num_samples=500_000) -> Tensor:
    k = 2 ** num_bits
    samples = _sample_sphere_coord(dim, num_samples).contiguous()

    # Initialize centroids uniformly over the data range
    c_max = float(samples.abs().quantile(0.999))
    centroids = torch.linspace(-c_max, c_max, k)   # (k,) evenly spaced

    for _ in range(num_steps):
        # --- Assignment (binary search, O(n log k)) ---
        boundaries = ((centroids[:-1] + centroids[1:]) / 2).contiguous()  # (k-1,) midpoints
        assignments = torch.bucketize(samples, boundaries)                 # (N,) each sample -> bucket index

        # --- Update (mean per bucket) ---
        new_centroids = torch.zeros(k)
        counts        = torch.zeros(k)
        new_centroids.scatter_add_(0, assignments, samples)  # sum samples per bucket
        counts.scatter_add_(0, assignments, torch.ones(num_samples))  # count per bucket

        mask = counts > 0
        new_centroids[mask]  /= counts[mask]         # mean
        new_centroids[~mask]  = centroids[~mask]     # keep old if bucket empty

        if (new_centroids - centroids).abs().max() < 1e-7:
            break                                    # converged
        centroids = new_centroids

    return centroids.sort().values   # always return sorted ascending
```

Why `torch.bucketize`? Because Lloyd-Max centroids are always **sorted**, binary search suffices for
assignment. The naive approach (`argmin` over an expanded tensor) is O(N·k); `bucketize` is O(N·log k).

### Step 3 - Cache by `(num_bits, dim)`

```python
_CACHE: dict[tuple[int, int], Tensor] = {}

def build_codebook(num_bits, dim, device=None) -> Tensor:
    key = (num_bits, dim)
    if key not in _CACHE:
        _CACHE[key] = _lloyd_max(num_bits, dim)   # compute once, ~2s first time
    return _CACHE[key].clone().to(device)
```

Result: a `(2**num_bits,)` tensor of sorted centroids, scaled to the true sphere marginal.
No extra `/ sqrt(d)` rescaling needed at quantize time - the centroids are already at the right scale.

---

## 3. [`rotation.py`] - Making coordinates Gaussian

### Why rotate at all?

Raw KV vectors have arbitrary distributions - peaked, skewed, correlated across dimensions.
Lloyd-Max is only optimal when the input distribution matches the distribution used to build the codebook.
After a Haar-uniform random rotation, each coordinate becomes approximately `N(0, 1/d)` and nearly
independent of others. This is a consequence of the Johnson-Lindenstrauss lemma.

### `RandomRotation` - dense QR

```python
def _make_qr_rotation(dim: int, seed: int) -> Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen)   # random Gaussian matrix
    Q, R = torch.linalg.qr(G)                 # G = Q @ R, Q orthogonal, R upper-triangular

    # Make the QR decomposition unique: force diagonal of R to be positive
    signs = torch.sign(torch.diag(R))          # (d,) each ±1
    Q = Q * signs.unsqueeze(0)                 # scale columns of Q

    # Enforce SO(d): det must be +1, not -1
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]                    # flip one column -> flips the determinant sign

    return Q   # (d, d) proper rotation matrix
```

**Why the double sign fix?**
1. `signs = sign(diag(R))` then `Q = Q * signs`: This is the standard "unique QR" fix - it makes
   the decomposition deterministic (without it, `torch.linalg.qr` may return different `Q` matrices for
   the same input on different hardware).
2. `if det(Q) < 0: Q[:, 0] = -Q[:, 0]`: QR gives orthogonal matrices (|det| = 1), but roughly half
   the time you get det = -1 (a **reflection**). Reflections are geometrically different from rotations -
   they flip handedness. Flipping one column changes sign of the determinant at zero cost.

**Forward / Inverse:**
```python
def forward(self, x):  return x @ self.Pi.T   # rotate:   y = x Πᵀ
def inverse(self, y):  return y @ self.Pi      # unrotate: x = y Π  (Π⁻¹ = Πᵀ for orthogonal)
```

**Cost:** O(d²) per vector, stores `d²` floats (e.g. d=128 -> 16,384 floats = 64KB).

### `HadamardRotation` - structured, fast

```python
class HadamardRotation(nn.Module):
    def __init__(self, dim, seed=0):
        gen = torch.Generator(); gen.manual_seed(seed)
        signs = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).float()
        # signs ∈ {-1, +1}^d, stored as (d,) buffer
        self.register_buffer("signs", signs)

    def forward(self, x):
        return _fwht(x * self.signs) / math.sqrt(self.dim)
        # 1. Multiply each coordinate by ±1  (random sign flip)
        # 2. Apply Walsh-Hadamard transform
        # 3. Normalize by 1/sqrt(d)

    def inverse(self, y):
        return self.signs * (_fwht(y) / math.sqrt(self.dim))
        # H is its own inverse up to a 1/d factor
        # H⁻¹ = H/d, so: x = D · H(y · sqrt(d)) / (sqrt(d) · d)
        #              = D · H(y) / (d/sqrt(d)) = D · H(y) / sqrt(d)
```

**The FWHT butterfly:**

```python
def _fwht(x: Tensor) -> Tensor:
    x = x.clone()
    d = x.shape[-1]
    h = 1
    while h < d:                               # log₂(d) iterations
        x = x.reshape(*x.shape[:-1], d//(2*h), 2*h)
        #     split last dim into pairs of size h
        a = x[..., :h] + x[..., h:]           # sum   of each pair
        x[..., h:] = x[..., :h] - x[..., h:] # diff  of each pair (in-place)
        x[..., :h] = a                         # store sum          (in-place)
        x = x.reshape(*x.shape[:-2], d)
        h *= 2
    return x
```

At each level `h`, the array is split into blocks of `2h`. Within each block:
- Left half  ← left + right (sum)
- Right half ← left - right (difference)

This is exactly the 1D Haar wavelet / Walsh-Hadamard butterfly. `log₂(d)` levels, one `a` allocation per level.

**Cost comparison:**

| | Complexity | Storage |
|---|---|---|
| RandomRotation | O(d²) per vector | d² floats |
| HadamardRotation | O(d log d) per vector | d floats (sign mask) |

For d=128: QR uses 16,384 floats; Hadamard uses 128 floats + O(128 log 128) work instead of O(128²).

---

## 4. [`quantizer.py`] - The two core quantizers

### Return types (NamedTuples)

```python
class QuantizedMSE(NamedTuple):
    indices: Tensor   # (... , d)  LongTensor - codebook index per coordinate
    norms:   Tensor   # (..., 1)   FloatTensor - original vector L2 norms
    shape:   tuple    # original input shape before flattening

class QuantizedIP(NamedTuple):
    indices:   Tensor   # (..., d) LongTensor - (b-1)-bit MSE indices
    qjl_bits:  Tensor   # (..., d) BoolTensor  - QJL sign bits (1 bit each)
    r_norm:    Tensor   # (..., 1) FloatTensor - residual L2 norm
    vec_norms: Tensor   # (..., 1) FloatTensor - original vector norms
    shape:     tuple
```

Using NamedTuples means the compressed representation is:
- Self-documenting (field names, not integer offsets)
- Serializable (just a tuple under the hood)
- Typed - Python/mypy can catch mismatches

### `KVQuantMSE` - minimize reconstruction error

**Goal:** minimize `E[‖k - k̂‖²]`

```python
class KVQuantMSE(nn.Module):
    def __init__(self, dim, num_bits=2, seed=0, use_hadamard=False):
        # Build and store the rotation (either QR or Hadamard)
        self.rotation = HadamardRotation(dim, seed) if use_hadamard else RandomRotation(dim, seed)

        # Build Lloyd-Max codebook for the true sphere marginal, shape (2**num_bits,)
        centroids = build_codebook(num_bits, dim)
        self.register_buffer("centroids", centroids)
        # register_buffer -> saved in state_dict, moved to GPU with .cuda(), not a parameter
```

**Quantize - step by step:**

```python
def quantize(self, x: Tensor) -> QuantizedMSE:
    shape = x.shape
    x_flat = x.reshape(-1, self.dim)            # flatten to (N, d)

    # 1. Save norm, project to unit sphere
    norms  = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    x_unit = x_flat / norms                     # (N, d), each row has ‖·‖=1

    # 2. Rotate coordinates to ~N(0, 1/d)
    y = self.rotation(x_unit)                   # (N, d)

    # 3. Nearest centroid via binary search - O(N·d·log k), no temp tensor
    boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2  # (k-1,) midpoints
    indices = torch.bucketize(y, boundaries)    # (N, d) LongTensor in [0, k-1]

    return QuantizedMSE(
        indices=indices.reshape(*shape[:-1], self.dim),
        norms=norms.reshape(*shape[:-1], 1),
        shape=shape,
    )
```

`torch.bucketize(y, boundaries)` for each value in `y` finds its position in the sorted
`boundaries` array using binary search. Since centroids are sorted, this is identical to finding
the nearest centroid but `log k` times faster.

**Dequantize - step by step:**

```python
def dequantize(self, q: QuantizedMSE) -> Tensor:
    idx_flat   = q.indices.reshape(-1, self.dim)    # (N, d) LongTensor
    norms_flat = q.norms.reshape(-1, 1)             # (N, 1)

    # 1. Index into centroids - fancy indexing
    y_tilde = self.centroids[idx_flat]              # (N, d) float - centroid values
    # self.centroids has shape (k,). Indexing with (N,d) LongTensor gives (N,d) float.

    # 2. Unrotate to get back to original coordinate system
    x_unit_hat = self.rotation.inverse(y_tilde)     # (N, d)

    # 3. Restore original scale
    x_hat = x_unit_hat * norms_flat                 # (N, d)
    return x_hat.reshape(q.shape)
```

**The old vs new lookup (why bucketize):**

```python
# OLD - O(N·d·k), allocates (N, d, k) intermediate tensor:
diff    = (y.unsqueeze(-1) - centroids.view(1, 1, -1)).abs()  # (N, d, k) - huge
indices = diff.argmin(dim=-1)                                  # (N, d)

# NEW - O(N·d·log k), no intermediate tensor:
boundaries = (centroids[:-1] + centroids[1:]) / 2             # (k-1,)
indices    = torch.bucketize(y, boundaries)                    # (N, d)
```

At N=4096, d=128, k=16: old allocates 4096×128×16 = 8M floats (~32MB temp). New allocates nothing.
Speedup measured: 14× at 2-bit, 22× at 4-bit.

---

### `KVQuantIP` - minimize inner product error (unbiased)

**The problem with MSE quantization for attention:** Attention computes `softmax(QKᵀ/√d)`.
MSE-optimal quantization `k̂` minimizes `‖k - k̂‖²`, but this introduces **bias** in the inner product:
`E[⟨q, k̂⟩] ≠ ⟨q, k⟩`. The softmax then assigns wrong probabilities to tokens.

**The fix: two-stage quantization**

```python
class KVQuantIP(nn.Module):
    def __init__(self, dim, num_bits=2, seed=0, qjl_seed=1, use_hadamard=False):
        self.mse_bits = max(0, num_bits - 1)   # (b-1) bits for MSE stage

        if self.mse_bits > 0:
            self.mse_quantizer = KVQuantMSE(dim, self.mse_bits, seed, use_hadamard)
        else:
            self.mse_quantizer = None   # at b=1, skip MSE stage entirely

        # QJL random matrix S ~ N(0,1)^{d×d}, fixed for life of this quantizer
        gen = torch.Generator(); gen.manual_seed(qjl_seed)
        S = torch.randn(dim, dim, generator=gen)
        self.register_buffer("S", S)   # (d, d)
```

**Quantize:**

```python
def quantize(self, x: Tensor) -> QuantizedIP:
    # Save norm, normalize
    vec_norms = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_unit    = x_flat / vec_norms

    # --- Stage 1: MSE quantize with (b-1) bits ---
    if self.mse_quantizer is not None:
        q_mse       = self.mse_quantizer.quantize(x_unit)     # quantize unit vector
        x_hat_unit  = self.mse_quantizer.dequantize(q_mse)    # reconstruct
        indices     = q_mse.indices
    else:
        x_hat_unit  = torch.zeros_like(x_unit)                # b=1: no MSE stage
        indices     = torch.zeros(N, self.dim, dtype=torch.long)

    # --- Stage 2: QJL on the residual ---
    r_unit   = x_unit - x_hat_unit          # residual on unit sphere (N, d)
    r_norm   = r_unit.norm(dim=-1, keepdim=True)   # (N, 1) residual magnitude

    qjl_proj = r_unit @ self.S.T            # (N, d) - project residual through S
    qjl_bits = qjl_proj > 0                 # (N, d) bool - 1 bit per dimension: just the sign

    return QuantizedIP(indices, qjl_bits, r_norm, vec_norms, shape)
```

**Why take only the sign?** The QJL (Quantized Johnson-Lindenstrauss) step uses the fact that:
for a Gaussian random matrix `S` and any vector `r`:

```
E[ sign(S @ r) ]  can be used to reconstruct  E[ ⟨q, r⟩ ]
```

Specifically, there exists a constant `c = sqrt(π/2)` such that:

```
E[ c/d · r_norm · (S.T @ sign(S @ r)) ]  =  r
```

So the 1-bit sign is sufficient to get an **unbiased estimate** of the residual.

**Dequantize (the unbiasedness math):**

```python
def dequantize(self, q: QuantizedIP) -> Tensor:
    # Recover MSE part
    if self.mse_quantizer is not None:
        dummy_norms  = torch.ones(N, 1)               # MSE quantizer expects unit norms
        q_mse_inner  = QuantizedMSE(idx_flat, dummy_norms, shape=(N, self.dim))
        x_hat_unit   = self.mse_quantizer.dequantize(q_mse_inner)  # (N, d)
    else:
        x_hat_unit   = torch.zeros(N, self.dim)

    # QJL residual correction
    signs      = 2.0 * bits_flat - 1.0               # {-1, +1} from {0, 1} bool
    # E[<y, (sqrt(π/2)/d) · r_norm · S.T @ sign(S·r)>] = <y, r>
    correction = (math.sqrt(math.pi / 2.0) / self.dim) * r_norm_flat * (signs @ self.S)
    #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^   ^^^^^^^^^^^^^^
    #            unbiasedness constant                      scale         recover direction

    x_tilde_unit = x_hat_unit + correction            # (N, d) - combined estimate
    return (x_tilde_unit * vec_norms_flat).reshape(q.shape)  # restore scale
```

The `sqrt(π/2)/d` constant: for `s ~ N(0,1)` and fixed unit vector `r`, we have
`E[sign(s) · s] = sqrt(2/π)`, so the inverse scaling is `sqrt(π/2)`. The `1/d` accounts
for averaging over all `d` dimensions of `S`.

**Compress with Huffman (bonus method):**

```python
def compress(self, x: Tensor) -> CompressedMSE:
    from .entropy import HuffmanCodec
    q     = self.quantize(x)
    codec = HuffmanCodec(self.num_bits, self.dim)
    bits  = codec.encode(q.indices)     # variable-length bit stream
    return CompressedMSE(bits=bits, norms=q.norms, shape=q.shape,
                         codec=codec, indices_len=q.indices.numel())
```

---

## 5. [`entropy.py`] - Huffman coding on codebook indices

### Why entropy coding works here

After Lloyd-Max quantization, the `k = 2ᵇ` codebook indices are **not uniformly distributed**.
Indices near the middle of the distribution are assigned more often (more samples fall there),
outer indices less often. This non-uniformity means fixed-length codes waste bits.

**Shannon entropy** is the theoretical minimum:
```
H = -∑ pᵢ log₂(pᵢ)
```
At b=4, d=128: H ≈ 3.765 bits vs 4 raw bits -> ~5% free compression.

### Step 1 - Compute symbol probabilities

```python
def codebook_probs(num_bits: int, dim: int) -> Tensor:
    centroids = build_codebook(num_bits, dim)   # (k,) sorted
    k         = len(centroids)
    std       = 1.0 / math.sqrt(dim)            # N(0, 1/d) standard deviation

    # Decision boundaries: midpoints + ±∞ at the ends
    bounds      = torch.full((k + 1,), float("inf"))
    bounds[0]   = float("-inf")
    bounds[-1]  = float("inf")
    bounds[1:-1] = (centroids[:-1] + centroids[1:]) / 2   # (k-1,) interior boundaries

    # P(symbol i) = area under N(0, std) between bounds[i] and bounds[i+1]
    from torch.distributions import Normal
    normal = Normal(0.0, std)
    lo    = normal.cdf(bounds[:-1])   # (k,) CDF at left boundary
    hi    = normal.cdf(bounds[1:])    # (k,) CDF at right boundary
    probs = (hi - lo).clamp(min=1e-12)
    return probs / probs.sum()        # normalize (numerical safety)
```

Each probability is the integral of `N(0, 1/d)` over that centroid's Voronoi cell.
The `Normal.cdf` call computes `Φ(x/std)` - the probability mass to the left of `x`.

### Step 2 - Build Huffman tree

```python
def _build_huffman(probs):
    k    = len(probs)
    heap = [_Node(prob=p, symbol=i) for i, p in enumerate(probs)]
    heapq.heapify(heap)           # min-heap by probability

    while len(heap) > 1:
        lo = heapq.heappop(heap)  # lowest probability node
        hi = heapq.heappop(heap)  # second lowest
        # merge: create parent with combined probability
        heapq.heappush(heap, _Node(prob=lo.prob + hi.prob, left=lo, right=hi))

    # Assign codes: left branch = 0, right branch = 1
    root  = heap[0]
    codes = [[] for _ in range(k)]
    _assign_codes(root, [], codes)
    return codes, [len(c) for c in codes]
```

Huffman property: **most frequent symbols get the shortest codes**.
For a 4-bit codebook (16 symbols), the middle symbols (indices 7, 8) might get 2-3 bit codes,
while rare outer symbols (indices 0, 15) might get 5-6 bit codes. Average comes out below 4.

### Step 3 - Encode/Decode

```python
class HuffmanCodec:
    def encode(self, indices: Tensor) -> list[int]:
        bits = []
        for idx in indices.flatten().tolist():
            bits.extend(self._codes[int(idx)])   # variable-length code per index
        return bits   # flat bit list

    def decode(self, bits: list[int], length: int) -> Tensor:
        symbols = []
        node    = ""
        for bit in bits:
            node += str(bit)                      # accumulate bits
            if node in self._decode_table:        # if it's a valid codeword
                symbols.append(self._decode_table[node])
                node = ""                         # reset for next symbol
        return torch.tensor(symbols, dtype=torch.long)
```

The decode table is a dict: `{"0": 7, "10": 8, "110": 6, ...}` - codeword string -> symbol integer.

```python
def entropy_bits(num_bits, dim) -> float:
    p = codebook_probs(num_bits, dim)
    return -(p * p.log2()).sum().item()    # H = -∑ pᵢ log₂(pᵢ)

def analyse(num_bits, dim) -> EntropyStats:
    codec = HuffmanCodec(num_bits, dim)
    raw   = float(num_bits)
    h     = codec.entropy           # Shannon bound
    avg   = codec.avg_bits          # actual Huffman bits
    pct   = (raw - avg) / raw * 100
    return EntropyStats(raw, h, avg, pct)
    # Example: EntropyStats(raw_bits=4, entropy=3.816, huffman_avg=3.847, saving_pct=3.83)
```

---

## 6. [`outlier.py`] - Handling spiky channels

### The problem

Real transformer KV caches have **outlier channels** - a handful of dimensions with variance
10-100× higher than the others. Quantizing them at the same bit-width as regular channels causes
disproportionate error: the codebook is sized for the typical spread, not these giants.

### Calibration - find outlier channels once

```python
def calibrate(self, x: Tensor) -> None:
    flat = x.reshape(-1, self.dim).float()   # (N, d) - all tokens
    var  = flat.var(dim=0)                   # (d,)  - variance per channel

    # Identify top-n_outlier channels by variance
    _, top_idx    = var.topk(self.n_outlier)         # (n_outlier,)
    outlier_idx   = top_idx.sort().values             # keep sorted for reproducibility

    all_idx       = torch.arange(self.dim)
    mask          = torch.ones(self.dim, dtype=torch.bool)
    mask[outlier_idx] = False
    regular_idx   = all_idx[mask]                    # (d - n_outlier,)

    self.outlier_idx = outlier_idx
    self.regular_idx = regular_idx

    # Build sub-quantizers with correct sub-dimensions
    self._outlier_q = KVQuantIP(dim=self.n_outlier, num_bits=self.outlier_bits, ...)
    self._regular_q = KVQuantIP(dim=self.n_regular, num_bits=self.regular_bits, ...)
    self._calibrated = True
```

Each sub-quantizer gets its own rotation matrix sized to its sub-dimension - not the full `d`.
This is important: the sphere marginal for a 32-dim subspace is different from a 128-dim one.

### Quantize

```python
def quantize(self, x: Tensor) -> OutlierQuantized:
    flat  = x.reshape(-1, self.dim)

    x_out = flat[:, self.outlier_idx]   # (N, n_outlier) - high-variance channels
    x_reg = flat[:, self.regular_idx]   # (N, n_regular) - normal channels

    return OutlierQuantized(
        outlier_q   = self._outlier_q.quantize(x_out),   # quantized at outlier_bits
        regular_q   = self._regular_q.quantize(x_reg),   # quantized at regular_bits
        outlier_idx = self.outlier_idx,                  # store indices for reconstruction
        regular_idx = self.regular_idx,
        shape       = x.shape,
    )
```

### Dequantize

```python
def dequantize(self, q: OutlierQuantized) -> Tensor:
    x_out = self._outlier_q.dequantize(q.outlier_q)   # (N, n_outlier)
    x_reg = self._regular_q.dequantize(q.regular_q)   # (N, n_regular)

    out = torch.empty(N, self.dim)
    out[:, q.outlier_idx] = x_out    # scatter back to original positions
    out[:, q.regular_idx] = x_reg
    return out.reshape(q.shape)
```

**Weighted average bits:** `(n_outlier × outlier_bits + n_regular × regular_bits) / d`
Example: `(32×4 + 96×2) / 128 = 2.5 bits`

---

## 7. [`kv_cache.py`] - The high-level (B, H, T, d) API

### Why a wrapper?

The raw quantizers work on `(N, d)` tensors. Transformers use `(B, H, T, d)` tensors
(batch × heads × sequence × dim). `KVCacheQuantizer` handles the reshape and manages
separate quantizers for K and V.

```python
class KVCacheQuantizer(nn.Module):
    def __init__(self, head_dim, num_bits=3, use_outlier=True,
                 n_outlier=32, outlier_bits=None, regular_bits=None, seed=0):

        if use_outlier:
            ob = outlier_bits or min(num_bits + 1, 4)   # e.g. 3->4-bit outliers
            rb = regular_bits or max(num_bits - 1, 1)   # e.g. 3->2-bit regular
            self.k_quant = OutlierKVQuant(head_dim, n_outlier, ob, rb, seed=seed)
            self.v_quant = OutlierKVQuant(head_dim, n_outlier, ob, rb, seed=seed+100)
            # Different seeds -> independent rotation matrices for K and V
        else:
            self.k_quant = KVQuantIP(head_dim, num_bits, seed=seed,   qjl_seed=seed+1)
            self.v_quant = KVQuantIP(head_dim, num_bits, seed=seed+2, qjl_seed=seed+3)
```

Different seeds for K and V: important because K and V have different distributions
(keys are used in dot products with queries; values are used for weighted sums).
Using the same rotation for both would introduce subtle correlations.

### Calibration flow

```python
def calibrate(self, k: Tensor, v: Tensor) -> None:
    # k, v: (B, H, T, d) representative samples
    k_flat = k.reshape(-1, self.head_dim)   # (B*H*T, d)
    v_flat = v.reshape(-1, self.head_dim)
    self.k_quant.calibrate(k_flat)           # finds top-n_outlier channels by variance
    self.v_quant.calibrate(v_flat)
    self._calibrated = True
```

### Compress/Decompress

```python
def compress(self, x: Tensor, is_value=False) -> CompressedKV:
    quant = self.v_quant if is_value else self.k_quant
    return quant.quantize(x)    # returns QuantizedIP or OutlierQuantized

def decompress(self, q: CompressedKV, is_value=False) -> Tensor:
    quant = self.v_quant if is_value else self.k_quant
    return quant.dequantize(q)  # returns float tensor, original shape

# Convenience:
def compress_kv(self, k, v):    return self.compress(k), self.compress(v, True)
def decompress_kv(self, k_c, v_c): return self.decompress(k_c), self.decompress(v_c, True)
```

---

## 8. [`attn_weighted.py`] - Allocate bits where attention goes

### The insight

Standard KVQuant assigns the same bit-width to all tokens in the cache.
But the model's output depends on `softmax(QKᵀ/√d)` - a token with 0.001% attention weight
has essentially zero impact. Why give it the same 3 bits as a token with 30% weight?

**Objective:** minimize `L_weighted = E[aᵢ · ‖kᵢ - k̂ᵢ‖²]` not `E[‖kᵢ - k̂ᵢ‖²]`

### Quantize

```python
def quantize(self, keys: Tensor, query: Tensor) -> AttentionWeightedQuantized:
    # keys: (B, H, T, d) or (T, d)
    # query: (B, H, d) or (d,)
    keys_flat  = keys.reshape(-1, T, self.dim)    # (N, T, d)
    query_flat = query.reshape(-1, 1, self.dim)   # (N, 1, d)

    # Compute attention weights
    scores  = (query_flat @ keys_flat.transpose(-2, -1)).squeeze(1)  # (N, T)
    scores  = scores / math.sqrt(self.dim)
    weights = F.softmax(scores, dim=-1)                              # (N, T)

    # Split: top fraction -> hi_bits
    k_hi     = max(1, int(T * self.top_fraction))       # number of hi-attention tokens
    _, top_idx = weights.topk(k_hi, dim=-1)             # (N, k_hi) indices
    top_mask   = torch.zeros(N, T, dtype=torch.bool)
    top_mask.scatter_(1, top_idx, True)                  # (N, T) boolean mask

    # Gather and quantize each group separately
    hi_keys = keys_flat[top_mask].reshape(N, k_hi, self.dim)        # (N, k_hi, d)
    lo_keys = keys_flat[~top_mask].reshape(N, T - k_hi, self.dim)   # (N, T-k_hi, d)

    hi_q = self.hi_quantizer.quantize(hi_keys)   # KVQuantMSE at hi_bits
    lo_q = self.lo_quantizer.quantize(lo_keys)   # KVQuantMSE at lo_bits

    return AttentionWeightedQuantized(hi_q, lo_q, top_mask, shape, self.dim)
```

Note: uses `KVQuantMSE` (not IP) because the bit allocation step itself is already
optimizing the attention-weighted objective - IP's unbiasedness is not needed here.

### Dequantize

```python
def dequantize(self, q: AttentionWeightedQuantized) -> Tensor:
    hi_keys = self.hi_quantizer.dequantize(q.hi_q)   # (N, k_hi, d)
    lo_keys = self.lo_quantizer.dequantize(q.lo_q)   # (N, T-k_hi, d)

    out = torch.empty(N, T, self.dim)
    out[q.top_mask]  = hi_keys.reshape(-1, self.dim)   # scatter back to original positions
    out[~q.top_mask] = lo_keys.reshape(-1, self.dim)
    return out.reshape(q.shape)
```

The `top_mask` boolean tensor is the map between "sorted by importance" and "original token order".

### Analysis helper

```python
def weighted_distortion(q, K, K_hat) -> Tensor:
    # q: (..., d) query; K: (..., T, d) true keys; K_hat: (..., T, d) reconstructed
    scores  = (q.unsqueeze(-2) @ K.transpose(-2, -1)).squeeze(-2) / math.sqrt(d)
    weights = F.softmax(scores, dim=-1)       # (..., T)
    per_tok = ((K - K_hat) ** 2).mean(-1)     # (..., T) - per-token MSE
    return (weights * per_tok).sum(-1).mean() # weighted sum, averaged over batch
```

---

## 9. [`delta.py`] - Exploiting temporal correlation

### The principle

In autoregressive generation, tokens are added one at a time. Key vector at step `t` is
similar to `t-1` - the model is attending to the same context. The delta `‖kₜ - kₜ₋₁‖`
is often much smaller than `‖kₜ‖`.

If we compress the delta at the same bit-width, the absolute error is much smaller.
Equivalently: same distortion with fewer bits.

### Data structures

```python
class DeltaKVCache(nn.Module):
    def __init__(self, head_dim, num_bits=3, anchor_every=0, seed=0):
        self.k_quantizer = KVQuantIP(head_dim, num_bits, ...)  # for deltas
        self.v_quantizer = KVQuantIP(head_dim, num_bits, ...)

        self._k_store: list[QuantizedIP | Tensor] = []  # entry per token
        self._v_store: list[QuantizedIP | Tensor] = []
        self._k_prev:  Tensor | None = None   # last reconstructed key (running state)
        self._v_prev:  Tensor | None = None
        self._anchors: list[int] = []         # which positions are full-precision anchors
```

### Push - adding one token

```python
def push(self, k: Tensor, v: Tensor) -> None:
    t = len(self._k_store)
    is_anchor = (t == 0) or (self.anchor_every > 0 and t % self.anchor_every == 0)

    if is_anchor:
        # Store raw float32 - one vector, cheap
        self._k_store.append(k.detach().clone())
        self._v_store.append(v.detach().clone())
        self._anchors.append(t)
        self._k_prev = k.detach().clone()
        self._v_prev = v.detach().clone()
    else:
        # Compute and compress the delta
        dk = k - self._k_prev    # (head_dim,) or (B, H, head_dim) - much smaller vector
        dv = v - self._v_prev

        qk = self.k_quantizer.quantize(dk)   # compress delta
        qv = self.v_quantizer.quantize(dv)
        self._k_store.append(qk)
        self._v_store.append(qv)

        # CRITICAL: update _prev with reconstructed delta, not true delta
        # This ensures get() accumulates the same errors as push() saw
        self._k_prev = self._k_prev + self.k_quantizer.dequantize(qk)
        self._v_prev = self._v_prev + self.v_quantizer.dequantize(qv)
```

**Why use reconstructed delta to update `_prev`?**

If we used the true delta: `_k_prev = k` (true key)
Then at get() time we'd accumulate reconstructed deltas on top of a true anchor ->
the running sum would drift from what we stored.

By using `_k_prev += dequantize(qk)`, both `push()` and `get()` see the same accumulated
quantization error - they stay synchronized. This prevents error divergence.

### Get - reconstruct full cache

```python
def get(self) -> tuple[Tensor, Tensor]:
    k_list, v_list = [], []
    k_running = None

    for t in range(T):
        if t in self._anchors:
            k_running = self._k_store[t]        # float32 anchor - no error
        else:
            dk = self.k_quantizer.dequantize(self._k_store[t])  # decompress delta
            k_running = k_running + dk           # accumulate

        k_list.append(k_running)

    return torch.stack(k_list, dim=0), torch.stack(v_list, dim=0)
    # returns (T, ..., head_dim) - all T positions reconstructed
```

**Error accumulation concern:** With `anchor_every=0` (default), errors from step 1 onwards
accumulate monotonically. The `anchor_every` parameter inserts fresh float32 snapshots periodically
(e.g. every 128 tokens) to reset accumulated error.

---

## 10. [`adaptive.py`] - Dynamic bit-width per token

### The problem with static allocation

Attention-weighted quantization still assigns bits at compression time based on
one query vector. But a token's importance can change: ignored at step 5, but heavily
attended at step 50. You compressed it at 2-bit and now need 4-bit accuracy from it.

### Internal data structure

```python
class _CacheEntry(NamedTuple):
    q:     QuantizedMSE   # the compressed token at current bit-width
    bits:  int            # which tier it's in (1, 2, 3, or 4)
    score: float          # EMA importance score - how much attention it receives
```

One `_CacheEntry` per token per cache (K-cache and V-cache stored separately).

### Four quantizers - one per tier

```python
self._quantizers = nn.ModuleDict({
    str(b): KVQuantMSE(head_dim, b, seed=seed+b)
    for b in (hi_bits, mid_bits, lo_bits, evict_bits)   # e.g. (4, 3, 2, 1)
})
# Access: self._quantizers["4"].quantize(x)  - quantizer for 4-bit tier
```

### Push - start every token at hi_bits

```python
def push(self, k: Tensor, v: Tensor) -> None:
    qk = self._quantizers[str(self.hi_bits)].quantize(k)
    qv = self._quantizers[str(self.hi_bits)].quantize(v)
    self._k_entries.append(_CacheEntry(q=qk, bits=self.hi_bits, score=1.0))
    self._v_entries.append(_CacheEntry(q=qv, bits=self.hi_bits, score=1.0))
    # New tokens start at max quality - importance unknown, assume important
```

### Attend - update scores and recompress changed tiers

```python
def attend(self, attn_weights: Tensor) -> None:
    # attn_weights: (..., T) softmax attention from current step
    T = len(self._k_entries)
    w = attn_weights.reshape(-1, T).mean(0).tolist()   # (T,) average over batch/heads

    new_k, new_v = [], []
    for t in range(T):
        # EMA update: blend old score with new attention weight
        new_score    = self.ema_decay * self._k_entries[t].score + (1 - self.ema_decay) * w[t]
        #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #              old score decays                              new attention weight blends in

        target_bits  = self._score_to_bits(new_score)   # which tier?

        if target_bits != self._k_entries[t].bits:
            # Bit-width changed -> must recompress
            k_hat = self._dequantize(self._k_entries[t])   # decompress at old bit-width
            v_hat = self._dequantize(self._v_entries[t])
            qk    = self._quantize(k_hat, target_bits)     # recompress at new bit-width
            qv    = self._quantize(v_hat, target_bits)
        else:
            qk = self._k_entries[t].q   # no change - keep existing compressed form
            qv = self._v_entries[t].q

        new_k.append(_CacheEntry(q=qk, bits=target_bits, score=new_score))
        new_v.append(_CacheEntry(q=qv, bits=target_bits, score=new_score))

    self._k_entries = new_k
    self._v_entries = new_v
```

### Score -> tier mapping

```python
def _score_to_bits(self, score: float) -> int:
    if score >= self.hi_threshold:      # e.g. ≥ 0.1   -> 4-bit
        return self.hi_bits
    if score >= self.lo_threshold:      # e.g. ≥ 0.01  -> 3-bit
        return self.mid_bits
    if score >= self.evict_threshold:   # e.g. ≥ 0.001 -> 2-bit
        return self.lo_bits
    return self.evict_bits              # e.g. < 0.001  -> 1-bit (effectively evicted)
```

**Hysteresis is implicit:** EMA smoothing means a token's score changes slowly.
A token won't rapidly oscillate between tiers because the old score is weighted by `ema_decay=0.9`.

### Analysis helpers

```python
def bit_allocation(self) -> dict[int, int]:
    # Returns e.g. {4: 12, 3: 20, 2: 45, 1: 83} - how many tokens at each tier
    counts = {}
    for e in self._k_entries:
        counts[e.bits] = counts.get(e.bits, 0) + 1
    return counts

def avg_bits(self) -> float:
    return sum(e.bits for e in self._k_entries) / len(self._k_entries)
```

---

## 11. [`correction.py`] - Low-rank error recovery

### Why quantization error is low-rank

The rotation step spreads energy uniformly across all dimensions. Then coordinate-wise quantization
rounds each dimension independently. The rounding error for each token depends on which Voronoi cell
it falls in - this is correlated across similar tokens. Tokens with similar rotation outputs will
have similar rounding patterns, creating low-rank structure in `R = K - K̂`.

The singular value spectrum of `R` decays fast - the top few singular values capture most of the
error energy.

### Quantize with correction

```python
def quantize(self, x: Tensor) -> CorrectedQuantized:
    shape = x.shape               # (..., T, d)
    x_2d  = x.reshape(-1, self.dim)   # (NT, d) - flatten for base quantizer

    # Base quantization
    base_q  = self.quantizer.quantize(x_2d)    # KVQuantMSE or KVQuantIP
    x_hat   = self.quantizer.dequantize(base_q)   # (NT, d)

    # Compute residual in original shape
    residual = (x_2d - x_hat).reshape(*shape)   # (..., T, d)

    # SVD of residual: apply per-sample (per batch dimension)
    residual_flat = residual.reshape(-1, shape[-2], self.dim)   # (N, T, d)
    U, S, Vh = torch.linalg.svd(residual_flat, full_matrices=False)
    # U:  (N, T, min(T,d))
    # S:  (N, min(T,d))
    # Vh: (N, min(T,d), d)

    # Truncate to rank r
    r  = min(self.rank, S.shape[-1])
    U  = U[..., :r]     # (N, T, r) - left singular vectors
    S  = S[..., :r]     # (N, r)    - singular values (descending)
    Vh = Vh[..., :r, :] # (N, r, d) - right singular vectors (transposed)

    # Absorb singular values into U: U_scaled = U · diag(S)
    # So correction = U_scaled @ Vh = (U·S) @ Vh ≈ R
    U_scaled = U * S.unsqueeze(-2)   # (N, T, r) - broadcast multiply
    V        = Vh.transpose(-2, -1)  # (N, d, r) - un-transpose

    return CorrectedQuantized(
        base_q = base_q,
        U      = U_scaled.reshape(*shape[:-1], r),     # (..., T, r)
        V      = V.reshape(*shape[:-2], self.dim, r),  # (..., d, r)
        shape  = shape,
    )
```

### Dequantize with correction applied

```python
def dequantize(self, q: CorrectedQuantized) -> Tensor:
    x_hat      = self.quantizer.dequantize(q.base_q).reshape(q.shape)  # (..., T, d)
    correction = q.U @ q.V.transpose(-2, -1)   # (..., T, r) @ (..., r, d) = (..., T, d)
    return x_hat + correction
```

### Applying correction directly in attention (without materializing corrected K)

```
Q·Kcorrᵀ = Q·(K̂ + U_s·Vᵀ)ᵀ
          = Q·K̂ᵀ + Q·V·U_sᵀ
```

In code:
```python
attn_base       = Q @ K_hat.T                     # standard attention
attn_correction = (Q @ V) @ U_scaled.T            # O(T·r·d) extra - tiny for small r
attn_corrected  = attn_base + attn_correction
```

### Storage analysis

```python
def storage_ratio(self, T: int) -> float:
    # correction stores: T*r (for U_scaled) + d*r (for V) floats
    # full residual:     T*d floats
    return (T * self.rank + self.dim * self.rank) / (T * self.dim)
    # At T=360, d=64, r=4: (360*4 + 64*4) / (360*64) = 1696/23040 = 0.074  (7.4%)
```

### Residual rank analysis helper

```python
def residual_rank_analysis(self, x: Tensor, max_rank: int = 16) -> Tensor:
    # Returns cumulative energy fraction for each rank 1..max_rank
    # Useful for choosing r for a new model
    R = x - self.quantizer.dequantize(self.quantizer.quantize(x))
    _, S, _ = torch.linalg.svd(R.reshape(1, -1, self.dim), full_matrices=False)
    S       = S.squeeze(0)
    energy  = S ** 2
    return energy[:max_rank].cumsum(0) / energy.sum()
    # e.g. [0.42, 0.61, 0.74, 0.83, ...] -> rank-4 captures 83% of error energy
```

---

## The Full Data Flow End-to-End

```
Input k ∈ ℝ^(B,H,T,d)
         │
         ▼
[delta.py - DeltaKVCache.push()]
  if anchor (t=0):  store k as float32, _prev = k
  else:             dk = k - _prev
                    _prev = _prev + dequantize(quantize(dk))  ← propagate error
         │
         ▼
[outlier.py - OutlierKVQuant.quantize()]
  var = x.var(dim=0)
  outlier channels -> sub-quantizer at outlier_bits
  regular channels -> sub-quantizer at regular_bits
         │
         ▼
[quantizer.py - KVQuantMSE.quantize() or KVQuantIP.quantize()]
  norms = x.norm(); x_unit = x / norms        ← normalize to unit sphere
  y = rotation(x_unit)                         ← rotate: QR O(d²) or Hadamard O(d log d)
  boundaries = (centroids[:-1]+centroids[1:])/2
  indices = torch.bucketize(y, boundaries)     ← O(N·d·log k), no temp tensor

  [IP variant adds:]
  r = x_unit - dequantize(mse_indices)
  qjl_bits = (r @ S.T) > 0                    ← 1 bit per dim for unbiased IP
         │
         ▼
[entropy.py - HuffmanCodec.encode()]
  probs = codebook_probs(num_bits, dim)        ← area under N(0,1/d) per Voronoi cell
  build Huffman tree from probs
  bits = huffman_encode(indices)               ← ~5% compression over raw bits
         │
         ▼
[STORED: indices(int8) + norms(float32) + qjl_bits(bool) + r_norm + huffman bits]
         │
         ▼
[correction.py - LowRankCorrection.quantize()]
  R = K - K_hat
  U, S, Vh = svd(R)[:r]
  store U_scaled=(U·S), V=Vhᵀ                 ← r(T+d) floats vs T·d for full R
         │
         ▼
[adaptive.py - AdaptiveKVCache.attend()]
  score = α·score + (1-α)·attn_weight          ← EMA update
  new_tier = score_to_bits(score)
  if tier changed: recompress at new bit-width ← promote or demote
         │
         ▼
[attn_weighted.py - AttentionWeightedQuantizer.quantize()]
  weights = softmax(q @ K.T / sqrt(d))
  top 50% tokens -> hi_quantizer (4-bit)
  bot 50% tokens -> lo_quantizer (2-bit)        ← average still 3 bits
```

---

## Key Results Summary

| Extension | What it exploits | Mechanism | Result |
|---|---|---|---|
| Attention-weighted | Tokens differ in importance | topk by softmax weight -> different bit-widths | 47-70% reduction in attention-weighted distortion |
| Delta compression | Adjacent KV correlated | compress `kₜ - kₜ₋₁` not `kₜ` | 1.1-2.2× MSE improvement |
| Low-rank correction | Quantization error is low-rank | truncated SVD of residual R | ~11% MSE at rank-4 (7.4% storage), ~19% at rank-8 |
| Adaptive allocation | Token importance changes over time | EMA score -> tier promotion/demotion | handles late-emerging important tokens |
| Hadamard rotation | O(d²) -> O(d log d) | butterfly FWHT, d-float sign mask | same guarantees, much faster and cheaper |
| True sphere marginal | Post-rotation distribution is Beta not Gaussian | sample ℝᵈ Gaussians, normalize, take one coord | tighter centroids at b=1,2 |
| Bucketize lookup | Sorted centroids allow binary search | `torch.bucketize` on midpoints | 14-22× speedup over argmin expansion |
| In-place FWHT | O(log d) allocations -> 1 per level | one `a` buffer reused in butterfly | reduced memory pressure |
| Outlier channels | Few channels dominate variance | calibrate -> split -> quantize separately | avoids destroying high-variance channels at low bits |
| Huffman coding | Index distribution is non-uniform | build tree from Voronoi cell areas | ~5% compression over raw bit representation |

---

## Class Hierarchy and Composition

```
nn.Module
├── RandomRotation           (d² storage, O(d²) per vector)
├── HadamardRotation         (d storage,  O(d log d) per vector)
├── KVQuantMSE            (uses one Rotation + one codebook)
├── KVQuantIP             (uses KVQuantMSE + QJL matrix S)
│     └── .compress()        (wraps HuffmanCodec on top)
├── OutlierKVQuant        (uses two KVQuantIP - one per channel group)
├── KVCacheQuantizer         (uses two OutlierKVQuant/KVQuantIP - K and V)
│
├── AttentionWeightedQuantizer (uses two KVQuantMSE - hi and lo)
├── DeltaKVCache             (uses two KVQuantIP - for K and V deltas)
├── AdaptiveKVCache          (uses four KVQuantMSE - one per tier)
└── LowRankCorrection        (wraps any KVQuantMSE/KVQuantIP)

Return types (all NamedTuples, composable):
  QuantizedMSE    -> indices, norms, shape
  QuantizedIP     -> indices, qjl_bits, r_norm, vec_norms, shape
  CompressedMSE   -> bits, norms, shape, codec, indices_len
  OutlierQuantized -> outlier_q, regular_q, outlier_idx, regular_idx, shape
  AttentionWeightedQuantized -> hi_q, lo_q, top_mask, shape, dim
  CorrectedQuantized -> base_q, U, V, shape
```