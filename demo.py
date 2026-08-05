"""
KVQuant hands-on demo.

Run:  python -m kvquant.demo

Memory tips for RTX 3070 / 16 GB RAM
-------------------------------------
1. Use bits=3 (default) or bits=2 for smaller KV cache.
2. Use prefill_chunk_size=256 in generate() to reduce peak VRAM during prefill.
3. Enable offload_to_disk=True in generate() / stream() to spill quantized cache
   to RAM and disk between forward passes (uses KVCacheDiskOffload).
4. Use device_map="auto" for CPU offload of model weights (CPU handles some
   layers, GPU handles others).
5. For very long contexts (>2K tokens), use --offload --max-vram-tokens 256.

Sections
--------
1. MSE distortion vs bit-width  (Table 1 of the paper)
2. Inner-product bias + variance (Table 2 of the paper)
3. Outlier channel handling
4. KV-cache round-trip with synthetic attention tensors
"""

import math
import torch

from kvquant import (
    KVQuantMSE,
    KVQuantIP,
    OutlierKVQuant,
    KVCacheQuantizer,
)

torch.manual_seed(0)

HEAD_DIM = 128  # typical LLM head dim
N_VECTORS = 2000  # number of test vectors
N_HEADS = 8
SEQ_LEN = 64
BATCH = 2

# -- helpers ----------------------------------------------------------------


def sep(title=""):
    width = 60
    if title:
        print(f"\n---- {title} {'-'*(width - len(title) - 6)}")
    else:
        print("-" * width)


def unit_sphere(n, d):
    x = torch.randn(n, d)
    return x / x.norm(dim=-1, keepdim=True)


# -- 1. MSE distortion ------------------------------------------------------

sep("1. MSE distortion vs bit-width  (unit-sphere vectors)")
print(
    f"{'bits':>5}  {'MSE (total)':>12}  {'lower bound':>12}  {'upper bound':>12}  {'within?':>8}"
)
sep()

x = unit_sphere(N_VECTORS, HEAD_DIM)

for b in (1, 2, 3, 4):
    q = KVQuantMSE(HEAD_DIM, num_bits=b)
    x_hat = q(x)
    mse = ((x - x_hat) ** 2).sum(-1).mean().item()
    lo = 1.0 / 4**b
    hi = math.sqrt(3) * math.pi / 2 / 4**b
    ok = "ok" if lo <= mse <= hi * 1.15 else "FAIL"
    print(f"{b:>5}  {mse:>12.5f}  {lo:>12.5f}  {hi:>12.5f}  {ok:>8}")


# -- 2. Inner-product bias & variance --------------------------------------

sep("2. Inner-product quality  (unit-sphere vectors)")
print(
    f"{'bits':>5}  {'bias':>10}  {'variance':>10}  {'var bound':>10}  {'unbiased?':>10}"
)
sep()

x = unit_sphere(N_VECTORS, HEAD_DIM)
y = unit_sphere(N_VECTORS, HEAD_DIM)

for b in (1, 2, 3, 4):
    q = KVQuantIP(HEAD_DIM, num_bits=b, seed=b * 7, qjl_seed=b * 7 + 1)
    x_tilde = q(x)
    true_ip = (x * y).sum(-1)
    est_ip = (x_tilde * y).sum(-1)
    bias = (true_ip - est_ip).mean().item()
    var = (true_ip - est_ip).var().item()
    var_hi = math.sqrt(3) * math.pi**2 / HEAD_DIM / 4**b
    ok = "ok" if abs(bias) < 0.02 else "FAIL"
    print(f"{b:>5}  {bias:>10.5f}  {var:>10.5f}  {var_hi:>10.5f}  {ok:>10}")


# -- 3. Outlier channel handling --------------------------------------------

sep("3. Outlier channel handling")

# Inject large-magnitude outliers into 32 channels
x_raw = torch.randn(N_VECTORS, HEAD_DIM)
x_raw[:, 10:42] *= 8.0  # synthetic outlier channels

configs = [
    (32, 3, 2, "2.5-bit"),
    (32, 4, 3, "3.5-bit"),
]

print(f"{'config':>10}  {'avg bits':>9}  {'recon MSE':>10}  {'outliers found':>15}")
sep()

for n_out, ob, rb, label in configs:
    oq = OutlierKVQuant(HEAD_DIM, n_outlier=n_out, outlier_bits=ob, regular_bits=rb)
    oq.calibrate(x_raw)
    x_rec = oq(x_raw)
    mse = ((x_raw - x_rec) ** 2).mean().item()

    detected = set(oq.outlier_idx.tolist())
    injected = set(range(10, 42))

    # Set intersection (&) only works on sets, not tensors or lists.
    # It gives you the channels that appear in both
    # i.e correctly detected outliers.
    # If they used lists they had need a nested loop
    # sets do it in O(min(n,m)).
    overlap = len(detected & injected)
    print(f"{label:>10}  {oq.avg_bits:>9.2f}  {mse:>10.5f}  {overlap:>10}/32")


# -- 4. KV-cache round-trip -------------------------------------------------

sep("4. KV-cache round-trip  (B,H,T,d) tensors")

k = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)
v = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)

print(
    f"Input shape: {list(k.shape)}  (B={BATCH}, H={N_HEADS}, T={SEQ_LEN}, d={HEAD_DIM})\n"
)
print(
    f"{'config':>12}  {'avg bits':>9}  {'K MSE':>10}  {'V MSE':>10}  {'K IP bias':>10}"
)
sep()

for bits, use_out, label in [
    (2, False, "2-bit plain"),
    (3, False, "3-bit plain"),
    (3, True, "2.5-bit+out"),
    (4, True, "3.5-bit+out"),
]:
    kvc = KVCacheQuantizer(HEAD_DIM, num_bits=bits, use_outlier=use_out)
    if use_out:
        kvc.calibrate(k, v)

    k_c, v_c = kvc.compress_kv(k, v)
    k_hat, v_hat = kvc.decompress_kv(k_c, v_c)

    k_mse = ((k - k_hat) ** 2).mean().item()
    v_mse = ((v - v_hat) ** 2).mean().item()

    # Inner-product bias: query @ key^T  (simulate attention scores)
    q_rand = torch.randn_like(k)
    true_scores = (q_rand * k).sum(-1)
    est_scores = (q_rand * k_hat).sum(-1)
    ip_bias = (true_scores - est_scores).mean().item()

    print(
        f"{label:>12}  {kvc.avg_bits:>9.2f}  {k_mse:>10.5f}  {v_mse:>10.5f}  {ip_bias:>10.5f}"
    )

sep()
print("Done.")
