"""
Lloyd-Max codebook for KVQuant.

After random rotation, each coordinate of a unit-sphere vector follows:
    f(t) = [Γ(d/2) / (sqrt(pi) Γ((d-1)/2))] * (1 - t^2)^((d-3)/2)   for t (belongs to) [-1, 1]

This is the true marginal - not Gaussian - and Lloyd-Max centroids should be
fitted to this distribution for each dimension d.  For large d the distribution
concentrates near N(0, 1/d), but fitting directly to the true distribution gives
tighter quantization error, especially at low bit-widths and small d.

Centroids are cached by (num_bits, dim) and computed lazily on first use.
"""

import math
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# True distribution sampler
# ---------------------------------------------------------------------------

def _sample_sphere_coord(dim: int, num_samples: int, seed: int = 42) -> Tensor:
    """
    Sample the marginal distribution of one coordinate of a uniformly random
    unit vector in R^d:

        f(t) = C_d * (1 - t^2)^((d-3)/2),   t (belongs to) [-1, 1]

    Sampled exactly by drawing Gaussian vectors and normalising.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    g = torch.randn(num_samples, dim, generator=gen)
    u = g / g.norm(dim=-1, keepdim=True)
    return u[:, 0]  # any single coordinate has the same marginal


# ---------------------------------------------------------------------------
# Lloyd-Max solver against the true distribution
# ---------------------------------------------------------------------------

def _lloyd_max(
    num_bits: int,
    dim: int,
    num_steps: int = 2000,
    num_samples: int = 500_000,
) -> Tensor:
    """
    Solve the 1-D Lloyd-Max problem for the true unit-sphere marginal
    distribution at dimension `dim`.

    Returns centroids of shape (2**num_bits,), sorted ascending, already
    scaled to the true distribution (no further rescaling needed).
    """
    k = 2 ** num_bits
    samples = _sample_sphere_coord(dim, num_samples).contiguous()

    # Initialise centroids uniformly over the empirical support
    c_max = float(samples.abs().quantile(0.999))
    centroids = torch.linspace(-c_max, c_max, k)

    for _ in range(num_steps):
        # Assignment via binary search on sorted centroids - O(n log k)
        boundaries = ((centroids[:-1] + centroids[1:]) / 2).contiguous()
        assignments = torch.bucketize(samples, boundaries)

        # Update: centroid ← mean of assigned samples
        new_centroids = torch.zeros(k)
        counts = torch.zeros(k)
        new_centroids.scatter_add_(0, assignments, samples)
        counts.scatter_add_(0, assignments, torch.ones(num_samples))

        mask = counts > 0
        new_centroids[mask] /= counts[mask]
        new_centroids[~mask] = centroids[~mask]   # keep old if cell is empty

        if (new_centroids - centroids).abs().max() < 1e-7:
            break
        centroids = new_centroids

    return centroids.sort().values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Cache keyed by (num_bits, dim) - centroids are dimension-specific
_CACHE: dict[tuple[int, int], Tensor] = {}


def build_codebook(num_bits: int, dim: int = 1, device: torch.device | None = None) -> Tensor:
    """
    Return Lloyd-Max centroids fitted to the true unit-sphere marginal
    distribution for the given `num_bits` and embedding dimension `dim`.

    The returned centroids are already at the correct scale for comparing
    against rotated unit vectors - no additional rescaling by 1/sqrtd is needed.

    Args:
        num_bits: Bits per coordinate (1–4).
        dim:      Embedding dimension d.
        device:   Target device.

    Returns:
        centroids: Tensor of shape (2**num_bits,).
    """
    assert 1 <= num_bits <= 4, "Only 1–4 bits/coordinate are supported."
    key = (num_bits, dim)
    if key not in _CACHE:
        _CACHE[key] = _lloyd_max(num_bits, dim)
    c = _CACHE[key].clone()
    if device is not None:
        c = c.to(device)
    return c


# ---------------------------------------------------------------------------
# Legacy Gaussian centroids - kept for visualisation / reference only.
# These are N(0,1) centroids from Max (1960); callers must rescale by 1/sqrt(d).
# ---------------------------------------------------------------------------

_B1_CENTROID = math.sqrt(2.0 / math.pi)

_KNOWN_GAUSSIAN_CENTROIDS: dict[int, list[float]] = {
    2: [-1.5104176, -0.4527801,  0.4527801,  1.5104176],
    3: [-2.1520000, -1.3439900, -0.7560500, -0.2451500,
         0.2451500,  0.7560500,  1.3439900,  2.1520000],
    4: [-2.7326000, -2.0691000, -1.6181000, -1.2562000,
        -0.9423000, -0.6568000, -0.3882000, -0.1282000,
         0.1282000,  0.3882000,  0.6568000,  0.9423000,
         1.2562000,  1.6181000,  2.0691000,  2.7326000],
}


def _gaussian_centroids(num_bits: int) -> Tensor:
    if num_bits == 1:
        return torch.tensor([-_B1_CENTROID, _B1_CENTROID])
    return torch.tensor(_KNOWN_GAUSSIAN_CENTROIDS[num_bits])


# Exposed for visualisation - unscaled N(0,1) reference centroids
PRECOMPUTED_CENTROIDS: dict[int, Tensor] = {
    b: _gaussian_centroids(b) for b in (1, 2, 3, 4)
}
