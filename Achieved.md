The Core Problem
Transformers store a KV cache — one key vector and one value vector per token per layer. At long contexts (10k-100k tokens) this dominates memory. You want to compress it aggressively (2-4 bits) without destroying model quality.

What KVQuant (the base paper) does
The original algorithm is elegant but blunt:


1. Take key vector k ∈ ℝᵈ (unit sphere)
2. Rotate it:  y = Π·k   (random orthogonal Π)
   → After rotation, each coordinate is ~N(0, 1/d), nearly independent
3. Quantize each coordinate independently with Lloyd-Max optimal scalar quantizer
4. MSE guarantee: D ≤ (√3·π/2) · 4⁻ᵇ   (within 2.7× of Shannon bound)
The inner product variant matters because attention scores are q·kᵀ / √d. You need softmax(QKᵀ) to be accurate, not just MSE. KVQuant-IP uses (b-1) bits on the MSE path + 1 bit QJL on the residual to get an unbiased attention score estimator.

What's Wrong With It (and what you fixed)
Bug Fixes
Bug	Problem	Your Fix
Codebook distribution	Used Gaussian N(0,1/d) to fit Lloyd-Max centroids. True post-rotation marginal is f(t) ∝ (1-t²)^((d-3)/2) — a Beta, not a Gaussian. Matters most at b=1,2.	Sample from true sphere marginal directly
SO(d) rotation	QR gives orthogonal matrix but ~50% chance det=-1 (reflection, not rotation). Reflection flips handedness, wrong distribution claim.	Sign-flip first column when det(Q) < 0
Codebook lookup	argmin over expanded (N, d, k) tensor = O(N·d·k) with huge temp allocation. Lloyd-Max centroids are sorted so binary search suffices.	torch.bucketize on midpoints = O(N·d·log k), 14-22× faster
Performance Improvement
Rotation	Complexity	Storage
Dense QR	O(d²)	d² floats
Hadamard (your impl)	O(d log d)	d floats (sign mask only)
Your 4 Extensions (what makes this novel)
1. Attention-Weighted Quantization
Problem: KVQuant gives 2-bit to both a 30%-attention token and a 0.001%-attention token.
Fix: Compute softmax(q·Kᵀ/√d), rank tokens by attention weight. Top 50% → 4 bits, bottom 50% → 2 bits. Average stays 3 bits.
Result: 47-70% reduction in attention-weighted distortion.

2. Delta Compression
Problem: Adjacent KV vectors in autoregressive generation are highly correlated. Compressing the full absolute vector wastes bits.
Fix: Store k₀ as float32 anchor. For each t > 0, compress δₜ = kₜ - k̂ₜ₋₁. Deltas are much smaller → same bit-width → much lower MSE.
Result: 1.1-2.2× MSE improvement (earlier layers benefit more — smoother trajectories).

3. Adaptive Bit Allocation
Problem: You don't know which tokens will matter when you first compress them. Static assignment is wrong.
Fix: EMA tracker per token: sₜ ← α·sₜ₋₁ + (1-α)·aₜ. Four tiers: 4-bit (hot), 3-bit, 2-bit, 1-bit (evict). Recompress when a token crosses a threshold boundary. Tokens can be promoted or demoted dynamically.

4. Low-Rank Error Correction
Problem: Quantization residual R = K - K̂ is not random noise — it's structured (low-rank).
Fix: Truncated SVD of residual: R ≈ UₛVᵣᵀ. Store r(T+d) floats instead of T·d (7.4% storage at rank-4). Apply in attention directly:


Q·K̂corrᵀ = Q·K̂ᵀ + (Q·Vᵣ)·Uₛᵀ
Result: ~11% MSE reduction at rank-4, ~19% at rank-8.

The Full Pipeline (composable)

KV stream
  -> Delta compression          (exploit temporal correlation)
  -> Attention-weighted bits    (align budget to what model attends to)
  -> KVQuantIP + Hadamard    (near-optimal MSE + unbiased IP)
  -> Low-rank correction        (recover structured residual)
  -> Huffman coding             (~5% index compression, free)
  -> Adaptive reallocation      (dynamic tier tracking over generation)
Each stage attacks a different inefficiency, so gains stack rather than cannibalize.

