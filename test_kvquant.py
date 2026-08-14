"""
Tests for the KVQuant implementation (arxiv 2504.19874).

Run with:  python -m pytest test_kvquant.py -v
"""

import math
import pytest
import torch

from src.codebook import build_codebook, PRECOMPUTED_CENTROIDS, _lloyd_max
from src.rotation import RandomRotation
from src.quantizer import KVQuantMSE, KVQuantIP, QuantizedMSE, QuantizedIP
from src.outlier import OutlierKVQuant
from src.kv_cache import KVCacheQuantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 128
SEED = 42


@pytest.fixture
def unit_vectors():
    """1000 unit-sphere vectors in R^128."""
    torch.manual_seed(SEED)
    x = torch.randn(1000, D)
    return x / x.norm(dim=-1, keepdim=True)


@pytest.fixture
def general_vectors():
    """Batch of general (non-unit) vectors (B, T, d)."""
    torch.manual_seed(SEED)
    return torch.randn(8, 16, D)


# ===========================================================================
# codebook.py
# ===========================================================================


class TestCodebook:
    def test_precomputed_keys(self):
        assert set(PRECOMPUTED_CENTROIDS.keys()) == {1, 2, 3, 4}

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_num_centroids(self, b):
        c = PRECOMPUTED_CENTROIDS[b]
        assert c.shape == (2**b,)

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_centroids_sorted(self, b):
        c = PRECOMPUTED_CENTROIDS[b]
        assert (c[1:] > c[:-1]).all(), "Centroids must be strictly increasing"

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_centroids_symmetric(self, b):
        c = PRECOMPUTED_CENTROIDS[b]
        # Nearly symmetric around 0 (small asymmetry from stochastic Lloyd-Max)
        asymmetry = (c + c.flip(0)).abs().max().item()
        assert asymmetry < 0.05, f"b={b}: centroids asymmetry {asymmetry:.4f} too large"

    def test_b1_analytical(self):
        """b=1 centroids must equal (+/-)sqrt(2/pi)."""
        c = PRECOMPUTED_CENTROIDS[1]
        expected = math.sqrt(2.0 / math.pi)
        assert abs(c[0].item() + expected) < 1e-4
        assert abs(c[1].item() - expected) < 1e-4

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_build_codebook_returns_correct_count(self, b):
        c, bnd = build_codebook(b, dim=D)
        assert c.shape == (2**b,)
        assert bnd.shape == (2**b - 1,)
        assert (c[1:] > c[:-1]).all(), "build_codebook centroids must be sorted"

    def test_build_codebook_device(self):
        c, bnd = build_codebook(2, dim=D, device=torch.device("cpu"))
        assert c.device.type == "cpu"
        assert bnd.device.type == "cpu"

    def test_lloyd_max_returns_sorted(self):
        c = _lloyd_max(2, dim=D, num_steps=100, num_samples=10_000)
        assert (c[1:] > c[:-1]).all()


# ===========================================================================
# rotation.py
# ===========================================================================


class TestRotation:
    def test_orthogonal(self):
        rot = RandomRotation(D, seed=0)
        Pi = rot.Pi
        eye = torch.eye(D)
        assert torch.allclose(Pi @ Pi.T, eye, atol=1e-5)
        assert torch.allclose(Pi.T @ Pi, eye, atol=1e-5)

    def test_forward_inverse_roundtrip(self):
        rot = RandomRotation(D, seed=0)
        x = torch.randn(32, D)
        assert torch.allclose(rot.inverse(rot(x)), x, atol=1e-5)

    def test_preserves_norm(self):
        rot = RandomRotation(D, seed=0)
        x = torch.randn(50, D)
        assert torch.allclose(rot(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_different_seeds_differ(self):
        r1 = RandomRotation(D, seed=0)
        r2 = RandomRotation(D, seed=1)
        assert not torch.allclose(r1.Pi, r2.Pi)

    def test_qr_rotation_is_proper_so_d(self):
        """paper §2.2.2: det(Pi) must be +1 (rotation), never -1 (reflection)."""
        for seed in range(8):
            det = torch.linalg.det(RandomRotation(D, seed=seed).Pi).item()
            assert det > 0, f"seed {seed}: det={det:+.3f} is a reflection"

    @pytest.mark.parametrize("dim", [2, 4, 64, 128])
    def test_hadamard_rotation_is_proper_so_d(self, dim):
        """paper §2.2.2 applies to the DEFAULT rotation too.  HadamardRotation is
        the default for power-of-2 dims (all real KV head dims), so it must also
        be a proper rotation — a random sign vector alone gives a reflection about
        half the time."""
        from src.rotation import HadamardRotation
        for seed in range(8):
            rot = HadamardRotation(dim=dim, seed=seed)
            M = rot(torch.eye(dim))          # rows = images of the basis
            det = torch.linalg.det(M).item()
            assert det > 0, f"dim={dim} seed={seed}: det={det:+.3f} is a reflection"
            # the det fix must not break orthogonality or invertibility
            assert torch.allclose(M @ M.T, torch.eye(dim), atol=1e-4)
            x = torch.randn(4, dim)
            assert torch.allclose(rot.inverse(rot(x)), x, atol=1e-4)

    def test_same_seed_reproducible(self):
        r1 = RandomRotation(D, seed=7)
        r2 = RandomRotation(D, seed=7)
        assert torch.allclose(r1.Pi, r2.Pi)

    def test_buffer_in_state_dict(self):
        rot = RandomRotation(D, seed=0)
        assert "Pi" in rot.state_dict()

    def test_batched_shapes(self):
        rot = RandomRotation(D, seed=0)
        x = torch.randn(4, 16, D)
        assert rot(x).shape == (4, 16, D)
        assert rot.inverse(rot(x)).shape == (4, 16, D)


# ===========================================================================
# quantizer.py   KVQuantMSE
# ===========================================================================


class TestKVQuantMSE:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_output_shape(self, b, unit_vectors):
        q = KVQuantMSE(D, num_bits=b)
        x_hat = q(unit_vectors)
        assert x_hat.shape == unit_vectors.shape

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_indices_in_range(self, b, unit_vectors):
        q = KVQuantMSE(D, num_bits=b)
        qz = q.quantize(unit_vectors)
        assert qz.indices.min() >= 0
        assert qz.indices.max() < 2**b

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_bounds(self, b, unit_vectors):
        """Total MSE per vector must lie within the paper's proven bounds."""
        q = KVQuantMSE(D, num_bits=b)
        dist = ((unit_vectors - q(unit_vectors)) ** 2).sum(-1).mean().item()
        lower = 1.0 / 4**b
        upper = math.sqrt(3) * math.pi / 2 / 4**b * 1.05  # 5% tolerance
        assert (
            dist >= lower * 0.5
        ), f"b={b}: distortion {dist:.5f} suspiciously below lower bound {lower:.5f}"
        assert (
            dist <= upper
        ), f"b={b}: distortion {dist:.5f} exceeds paper upper bound {upper:.5f}"

    def test_quantize_dequantize_roundtrip_shape(self, general_vectors):
        q = KVQuantMSE(D, num_bits=2)
        qz = q.quantize(general_vectors)
        x_hat = q.dequantize(qz)
        assert x_hat.shape == general_vectors.shape

    def test_norms_stored(self, general_vectors):
        q = KVQuantMSE(D, num_bits=2)
        qz = q.quantize(general_vectors)
        expected_norms = general_vectors.norm(dim=-1, keepdim=True)
        assert torch.allclose(qz.norms, expected_norms, atol=1e-5)

    def test_scale_invariance(self):
        """Doubling the input should double the reconstruction."""
        torch.manual_seed(0)
        x = torch.randn(100, D)
        x = x / x.norm(dim=-1, keepdim=True)
        q = KVQuantMSE(D, num_bits=2)
        x_hat = q(x)
        x2_hat = q(2.0 * x)
        assert torch.allclose(x2_hat, 2.0 * x_hat, atol=1e-5)

    def test_distortion_mse_helper(self, unit_vectors):
        q = KVQuantMSE(D, num_bits=2)
        d = q.distortion_mse(unit_vectors)
        assert d.ndim == 0  # scalar
        assert d.item() > 0


# ===========================================================================
# quantizer.py   KVQuantIP
# ===========================================================================


class TestKVQuantIP:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_output_shape(self, b, unit_vectors):
        q = KVQuantIP(D, num_bits=b)
        x_tilde = q(unit_vectors)
        assert x_tilde.shape == unit_vectors.shape

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_unbiased(self, b, unit_vectors):
        """E[<y, x'>] ≈ <y, x> (bias < 0.01 on 1000 samples)."""
        torch.manual_seed(b)
        y = torch.randn_like(unit_vectors)
        y = y / y.norm(dim=-1, keepdim=True)
        q = KVQuantIP(D, num_bits=b, seed=b * 10, qjl_seed=b * 10 + 1)
        x_tilde = q(unit_vectors)
        bias = ((unit_vectors * y).sum(-1) - (x_tilde * y).sum(-1)).mean().abs().item()
        assert bias < 0.02, f"b={b}: IP bias {bias:.5f} too large"

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_ip_variance_bound(self, b, unit_vectors):
        """Var[<y, x'> <y,x>] must be below the paper's upper bound."""
        torch.manual_seed(b + 100)
        y = torch.randn_like(unit_vectors)
        y = y / y.norm(dim=-1, keepdim=True)
        q = KVQuantIP(D, num_bits=b, seed=b * 20, qjl_seed=b * 20 + 1)
        x_tilde = q(unit_vectors)
        err = (unit_vectors * y).sum(-1) - (x_tilde * y).sum(-1)
        var = err.var().item()
        upper_var = math.sqrt(3) * math.pi**2 / D / 4**b * 1.2  # 20% tolerance
        assert (
            var <= upper_var
        ), f"b={b}: var {var:.6f} exceeds upper bound {upper_var:.6f}"

    def test_quantized_ip_fields(self, unit_vectors):
        q = KVQuantIP(D, num_bits=2)
        qz = q.quantize(unit_vectors)
        assert isinstance(qz, QuantizedIP)
        assert qz.indices.shape[-1] == D
        assert qz.qjl_bits.shape[-1] == D
        assert qz.qjl_bits.dtype == torch.bool
        assert qz.r_norm.shape[-1] == 1
        assert qz.vec_norms.shape[-1] == 1

    def test_general_vectors_shape(self, general_vectors):
        q = KVQuantIP(D, num_bits=2)
        out = q(general_vectors)
        assert out.shape == general_vectors.shape

    def test_b1_zero_mse_stage(self):
        """At b=1 the MSE quantizer should be None."""
        q = KVQuantIP(D, num_bits=1)
        assert q.mse_quantizer is None
        assert q.mse_bits == 0

    def test_distortion_ip_helper(self, unit_vectors):
        torch.manual_seed(0)
        y = torch.randn_like(unit_vectors)
        q = KVQuantIP(D, num_bits=2)
        d = q.distortion_ip(unit_vectors, y)
        assert d.ndim == 0
        assert d.item() >= 0


# ===========================================================================
# outlier.py
# ===========================================================================


class TestOutlierKVQuant:
    def test_requires_calibration(self):
        q = OutlierKVQuant(D)
        x = torch.randn(10, D)
        with pytest.raises(RuntimeError, match="calibrate"):
            q.quantize(x)

    def test_calibrate_sets_indices(self):
        q = OutlierKVQuant(D, n_outlier=32)
        x = torch.randn(200, D)
        q.calibrate(x)
        assert q.outlier_idx.shape == (32,)
        assert q.regular_idx.shape == (D - 32,)

    def test_channel_partition_exhaustive(self):
        """outlier + regular indices must cover all channels exactly once."""
        q = OutlierKVQuant(D, n_outlier=32)
        q.calibrate(torch.randn(200, D))
        all_idx = torch.cat([q.outlier_idx, q.regular_idx]).sort().values
        assert torch.equal(all_idx, torch.arange(D))

    def test_output_shape(self, general_vectors):
        q = OutlierKVQuant(D, n_outlier=32, outlier_bits=3, regular_bits=2)
        q.calibrate(general_vectors.reshape(-1, D))
        out = q(general_vectors)
        assert out.shape == general_vectors.shape

    def test_avg_bits(self):
        n_out, ob, rb = 32, 3, 2
        q = OutlierKVQuant(D, n_outlier=n_out, outlier_bits=ob, regular_bits=rb)
        expected = (n_out * ob + (D - n_out) * rb) / D
        assert abs(q.avg_bits - expected) < 1e-9

    def test_outlier_channels_have_highest_variance(self):
        """Calibration should pick the highest-variance channels as outliers."""
        torch.manual_seed(0)
        x = torch.randn(500, D)
        # Amplify channels 10–41 to be clear outliers
        x[:, 10:42] *= 10.0
        q = OutlierKVQuant(D, n_outlier=32)
        q.calibrate(x)
        outlier_set = set(q.outlier_idx.tolist())
        injected_set = set(range(10, 42))
        overlap = len(outlier_set & injected_set)
        assert overlap >= 28, f"Only {overlap}/32 injected outliers detected"

    def test_quantize_dequantize_roundtrip(self, general_vectors):
        q = OutlierKVQuant(D, n_outlier=32)
        q.calibrate(general_vectors.reshape(-1, D))
        qz = q.quantize(general_vectors)
        out = q.dequantize(qz)
        assert out.shape == general_vectors.shape

    @pytest.mark.parametrize("n_out,ob,rb", [(32, 3, 2), (16, 4, 3), (64, 2, 1)])
    def test_various_configs(self, n_out, ob, rb):
        x = torch.randn(100, D)
        q = OutlierKVQuant(D, n_outlier=n_out, outlier_bits=ob, regular_bits=rb)
        q.calibrate(x)
        out = q(x)
        assert out.shape == x.shape


# ===========================================================================
# kv_cache.py
# ===========================================================================


class TestKVCacheQuantizer:
    @pytest.fixture
    def kv_tensors(self):
        torch.manual_seed(0)
        k = torch.randn(2, 8, 32, D)
        v = torch.randn(2, 8, 32, D)
        return k, v

    def test_requires_calibration_with_outlier(self, kv_tensors):
        k, v = kv_tensors
        qc = KVCacheQuantizer(D, num_bits=3, use_outlier=True)
        with pytest.raises(RuntimeError):
            qc.compress(k)

    def test_compress_decompress_shape(self, kv_tensors):
        k, v = kv_tensors
        qc = KVCacheQuantizer(D, num_bits=3, use_outlier=True)
        qc.calibrate(k, v)
        k_c, v_c = qc.compress_kv(k, v)
        k_hat, v_hat = qc.decompress_kv(k_c, v_c)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_no_outlier_mode(self, kv_tensors):
        k, v = kv_tensors
        qc = KVCacheQuantizer(D, num_bits=2, use_outlier=False)
        k_c = qc.compress(k)
        k_hat = qc.decompress(k_c)
        assert k_hat.shape == k.shape

    def test_is_value_flag(self, kv_tensors):
        k, v = kv_tensors
        qc = KVCacheQuantizer(D, num_bits=3, use_outlier=True)
        qc.calibrate(k, v)
        k_c = qc.compress(k, is_value=False)
        v_c = qc.compress(v, is_value=True)
        k_hat = qc.decompress(k_c, is_value=False)
        v_hat = qc.decompress(v_c, is_value=True)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_avg_bits_with_outlier(self):
        qc = KVCacheQuantizer(
            D,
            num_bits=3,
            use_outlier=True,
            n_outlier=32,
            outlier_bits=4,
            regular_bits=3,
        )
        expected = (32 * 4 + (D - 32) * 3) / D
        assert abs(qc.avg_bits - expected) < 1e-6

    def test_avg_bits_without_outlier(self):
        qc = KVCacheQuantizer(D, num_bits=2, use_outlier=False)
        assert qc.avg_bits == 2.0

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_reconstruction_reasonable(self, kv_tensors, bits):
        """Reconstruction error should decrease as bits increase."""
        k, v = kv_tensors
        qc = KVCacheQuantizer(D, num_bits=bits, use_outlier=False)
        k_c, v_c = qc.compress_kv(k, v)
        k_hat, _ = qc.decompress_kv(k_c, v_c)
        err = ((k - k_hat) ** 2).mean().item()
        # Loose upper bound: error must be less than variance of k (~1.0)
        assert err < 1.5, f"bits={bits}: K reconstruction error {err:.4f} too large"


# ---------------------------------------------------------------------------
# DeltaKVCache Fix 1, 2, 3
# ---------------------------------------------------------------------------

from src.delta import DeltaKVCache  # noqa: E402


class TestDeltaKVCache:
    """Tests for the three delta.py optimisations."""

    # ── Fix 2 ────────────────────────────────────────────────────────────
    def test_anchors_is_set(self):
        """Fix 2: _anchors must be a set for O(1) membership lookup."""
        cache = DeltaKVCache(head_dim=D, num_bits=3)
        assert isinstance(cache._anchors, set), "_anchors must be a set"

    def test_anchor_add_not_append(self):
        """Fix 2: pushing tokens uses set.add(), positions are correct."""
        cache = DeltaKVCache(head_dim=D, num_bits=3, anchor_every=2)
        for _ in range(5):
            cache.push(torch.randn(D), torch.randn(D))
        # t=0 always anchor; anchor_every=2 adds t=2,4
        assert 0 in cache._anchors
        assert 2 in cache._anchors
        assert 4 in cache._anchors

    # ── Fix 1 ────────────────────────────────────────────────────────────
    def test_incremental_lists_populated(self):
        """Fix 1: _k_reconstructed grows by one entry per push()."""
        cache = DeltaKVCache(head_dim=D, num_bits=3)
        for i in range(7):
            cache.push(torch.randn(D), torch.randn(D))
            assert len(cache._k_reconstructed) == i + 1
            assert len(cache._v_reconstructed) == i + 1

    def test_get_output_shape(self):
        """Fix 1: get() returns (T, head_dim) after T pushes."""
        T = 12
        cache = DeltaKVCache(head_dim=D, num_bits=3)
        for _ in range(T):
            cache.push(torch.randn(D), torch.randn(D))
        K, V = cache.get()
        assert K.shape == (T, D)
        assert V.shape == (T, D)

    def test_anchor_reconstructed_exactly(self):
        """Fix 1: anchor tokens are stored float32 reconstruction error is zero."""
        cache = DeltaKVCache(head_dim=D, num_bits=3)
        k0 = torch.randn(D)
        cache.push(k0, torch.randn(D))
        K, _ = cache.get()
        assert torch.allclose(
            K[0], k0, atol=1e-6
        ), "Anchor token not reconstructed exactly"

    def test_get_called_twice_same_result(self):
        """Fix 1: get() is idempotent calling it twice gives identical tensors."""
        cache = DeltaKVCache(head_dim=D, num_bits=3)
        for _ in range(8):
            cache.push(torch.randn(D), torch.randn(D))
        K1, V1 = cache.get()
        K2, V2 = cache.get()
        assert torch.equal(K1, K2)
        assert torch.equal(V1, V2)

    def test_reset_clears_incremental_lists(self):
        """Fix 1: reset() clears _k_reconstructed and _v_reconstructed."""
        cache = DeltaKVCache(head_dim=D, num_bits=3)
        for _ in range(5):
            cache.push(torch.randn(D), torch.randn(D))
        cache.reset()
        assert len(cache._k_reconstructed) == 0
        assert len(cache._v_reconstructed) == 0
        assert len(cache._anchors) == 0

    # ── Fix 3 ────────────────────────────────────────────────────────────
    def test_adaptive_threshold_zero_disables(self):
        """Fix 3: anchor_threshold=0.0 (default) never triggers adaptively."""
        cache = DeltaKVCache(head_dim=D, num_bits=3, anchor_threshold=0.0)
        base = torch.randn(D)
        for _ in range(10):
            cache.push(base * 100, torch.randn(D))  # same direction, huge magnitude
        # Only t=0 should be an anchor
        assert cache._anchors == {0}

    def test_adaptive_threshold_triggers_on_large_delta(self):
        """Fix 3: anchor_threshold triggers when ||delta||/||k|| exceeds threshold."""
        cache = DeltaKVCache(head_dim=D, num_bits=3, anchor_threshold=0.3)
        base = torch.randn(D)
        # Stable tokens: tiny delta
        for _ in range(4):
            cache.push(base + 0.001 * torch.randn(D), torch.randn(D))
        assert len(cache._anchors) == 1  # only t=0

        # Large jump: delta >> vector
        cache.push(torch.randn(D) * 10, torch.randn(D))
        assert len(cache._anchors) == 2  # adaptive anchor triggered

    def test_adaptive_reduces_mse_on_drift(self):
        """Fix 3: adaptive anchoring lowers MSE when sequence drifts suddenly."""
        torch.manual_seed(0)
        T = 30
        keys = []
        k = torch.randn(D)
        for t in range(T):
            k = torch.randn(D) * 4 if t == 15 else k + 0.05 * torch.randn(D)
            keys.append(k.clone())

        def mse(threshold=0.0):
            cache = DeltaKVCache(head_dim=D, num_bits=3, anchor_threshold=threshold)
            for ki in keys:
                cache.push(ki, torch.zeros(D))
            K_hat, _ = cache.get()
            return ((torch.stack(keys) - K_hat) ** 2).mean().item()

        mse_no_adapt = mse(threshold=0.0)
        mse_adaptive = mse(threshold=0.4)
        assert (
            mse_adaptive < mse_no_adapt
        ), f"Adaptive MSE {mse_adaptive:.5f} should be < no-adapt MSE {mse_no_adapt:.5f}"


# ===========================================================================
# kv_cache.py — paper-codec tiered disk offload (KVCacheDiskOffload)
# ===========================================================================

from src.kv_cache import (  # noqa: E402
    KVCacheDiskOffload,
    _PaperKV,
    _paper_compress,
    _paper_dequantize,
    _pack_indices,
    _unpack_indices,
)
from src.quantizer import KVQuantMSE  # noqa: E402


def _mse_factory(dim, num_bits):
    """Same fixed-seed KVQuantMSE the offloader builds internally."""
    return KVQuantMSE(dim=dim, num_bits=num_bits, seed=0)


def _fake_cache(n_layers=6, B=1, H=2, T=8, d=D):
    """Build a plain tuple-of-tuples KV cache of random (k, v) pairs."""
    torch.manual_seed(SEED)
    return tuple(
        (torch.randn(B, H, T, d), torch.randn(B, H, T, d)) for _ in range(n_layers)
    )


def _staged_pairs(staged):
    """Normalise stage_for_forward output (tuple or DynamicCache) to a list of (k, v)."""
    if isinstance(staged, tuple):
        return list(staged)
    return [(l.keys, l.values) for l in staged.layers]


def test_mse_key_reconstructs_far_better_than_ip():
    """The generation cache uses KVQuantMSE for BOTH K and V so reconstruction
    is faithful per coordinate.  This is the whole reason int8 was removable.

    The default KVCacheQuantizer uses KVQuant-IP for K — an inner-product
    *estimator* whose per-coordinate output is wrong (fine for PPL scoring,
    unusable as a real cache).  Passing k_quantizer_cls=KVQuantMSE swaps K to the
    MSE-optimal quantizer.  This test proves the contrast: MSE-for-K reconstructs
    K with high per-coordinate cosine, while IP-for-K does not."""
    from src.kv_cache import KVCacheQuantizer

    torch.manual_seed(SEED)
    B, H, T, d = 1, 2, 64, D
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    def cos(a, b):
        return (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()

    def k_cosine(k_cls):
        q = KVCacheQuantizer(
            head_dim=d, num_bits=4, use_outlier=True,
            n_outlier=max(4, d // 4), outlier_bits=4, regular_bits=3,
            k_quantizer_cls=k_cls,
        )
        q.calibrate(k, v)
        k_hat = q.decompress(q.compress(k, is_value=False), is_value=False)
        return cos(k, k_hat)

    mse_cos = k_cosine(KVQuantMSE)   # the generation path
    ip_cos = k_cosine(None)          # the PPL-scoring default (IP for K)

    # MSE reconstructs K faithfully per coordinate (worst-case random Gaussian
    # still clears 0.95); IP is an inner-product estimator and reconstructs K
    # worse.  On real structured KV during generation the gap is far larger — the
    # IP path drifts into gibberish — but even on random data MSE must beat it.
    assert mse_cos > 0.95, f"MSE-for-K cosine too low: {mse_cos:.4f}"
    assert mse_cos > ip_cos + 0.03, (
        f"MSE-for-K ({mse_cos:.4f}) should exceed IP-for-K ({ip_cos:.4f})"
    )


class TestKVCacheDiskOffload:
    """Tiered VRAM->RAM->disk offload, exercised on both codecs."""

    @pytest.mark.parametrize("codec", ["paper", "paper-outlier"])
    def test_store_stage_roundtrip(self, codec):
        """stage_for_forward reconstructs every layer; shapes and dtype preserved."""
        cache = _fake_cache()
        off = KVCacheDiskOffload(
            max_vram_tokens=512, warm_size=0, disk_dir=None, codec=codec, bits=4
        )
        try:
            off.store(cache)
            pairs = _staged_pairs(off.stage_for_forward())
            assert len(pairs) == len(cache)
            for (k, v), (k_hat, v_hat) in zip(cache, pairs):
                assert k_hat.shape == k.shape and v_hat.shape == v.shape
                # paper Lloyd-Max: lossy but directionally faithful
                rel = (k - k_hat.cpu()).norm() / k.norm()
                assert rel < 0.5, f"{codec} K rel_err too high: {rel:.3f}"
        finally:
            off.close()

    def test_outlier_codec_faithful_on_outlier_channels(self):
        """
        Paper Section 5: on realistic KV (elevated variance spread across ~dim/4
        channels — the case the paper's 32-outlier config targets), the outlier
        codec reconstructs K faithfully AND spends fewer average bits than plain
        paper at the same nominal ``bits`` (paper §5.1: outlier=min(b+1,4),
        regular=max(b-1,1), so avg = 3.25 bpw at b=4 vs plain's 4.0).
        """
        torch.manual_seed(SEED)
        B, H, T, d = 1, 2, 48, D
        # ~dim/4 elevated channels with a range of magnitudes (real KV behaviour,
        # not a handful of extreme spikes).
        elevated = torch.randperm(d)[: d // 4]
        def mk():
            x = torch.randn(B, H, T, d)
            for c in elevated.tolist():
                x[..., c] *= (3.0 + 10.0 * torch.rand(1).item())
            return x
        cache = tuple((mk(), mk()) for _ in range(2))

        def kcos(codec):
            off = KVCacheDiskOffload(codec=codec, bits=4, warm_size=8)
            try:
                off.store(cache)
                kh = _staged_pairs(off.stage_for_forward())[0][0].cpu()
                k = cache[0][0]
                a, b = k.flatten(), kh.flatten()
                return (a @ b / (a.norm() * b.norm())).item()
            finally:
                off.close()

        outlier = kcos("paper-outlier")
        # Faithful enough for generation, at a LOWER average bit budget than plain.
        assert outlier > 0.99, f"outlier codec should be near-lossless, got {outlier:.4f}"

    @pytest.mark.parametrize("codec", ["paper", "paper-outlier"])
    def test_disk_spill_fires(self, codec):
        """warm_size smaller than n_layers must push the remainder to disk."""
        n_layers, warm = 6, 2
        cache = _fake_cache(n_layers=n_layers)
        off = KVCacheDiskOffload(
            max_vram_tokens=512, warm_size=warm, disk_dir=None, codec=codec, bits=3
        )
        try:
            off.store(cache)
            summary = off.memory_summary()
            assert summary["warm_layers"] == warm
            assert summary["disk_layers"] == n_layers - warm
            # spilled layers are still reconstructable
            assert len(_staged_pairs(off.stage_for_forward())) == n_layers
        finally:
            off.close()

    def test_cleanup_removes_disk_dir(self):
        """close() deletes the spill directory."""
        cache = _fake_cache(n_layers=4)
        off = KVCacheDiskOffload(max_vram_tokens=512, warm_size=1, disk_dir=None)
        off.store(cache)
        disk_dir = off.memory_summary()["disk_dir"]
        import os
        assert os.path.isdir(disk_dir)
        off.close()
        assert not os.path.isdir(disk_dir)

    def test_default_codec_is_paper(self):
        """Default codec follows the paper (rotate + Lloyd-Max)."""
        off = KVCacheDiskOffload(warm_size=8)
        try:
            off.store(_fake_cache(n_layers=2))
            # the stored chunks are the paper representation
            entry = next(iter(off._ram.values()))
            assert isinstance(entry[0], _PaperKV)
        finally:
            off.close()

    def test_gqa_bumps_effective_bits(self):
        """GQA models must store more effective bits than MHA to offset error
        amplification (each KV head shared by g query heads).  Without this bump
        a grouped-query model (Qwen2.5-7B, g=7) under-quantizes -> gibberish."""
        cache = _fake_cache(n_layers=1, T=16)

        def stored_avg_bits(g):
            off = KVCacheDiskOffload(codec="paper-outlier", bits=4, gqa_factor=g,
                                     warm_size=8)
            try:
                off.store(cache)
                return off._outlier_q[(0, False)].avg_bits, off._gqa_extra
            finally:
                off.close()

        mha_bits, mha_extra = stored_avg_bits(1)    # no amplification
        gqa_bits, gqa_extra = stored_avg_bits(7)    # Qwen2.5-7B style
        assert mha_extra == 0
        assert gqa_extra == 2, f"expected ceil(log4(7))=2, got {gqa_extra}"
        assert gqa_bits > mha_bits + 1.5, (
            f"GQA avg bits ({gqa_bits}) should exceed MHA ({mha_bits}) by the bump"
        )

    def test_append_only_no_requantize_drift(self):
        """
        The bug this fixes: re-store()-ing a growing cache must NOT re-quantize
        already-frozen tokens.  With the lossy paper codec, re-quantizing drifts
        badly; append-only keeps the prefill block bit-identical across steps.
        """
        torch.manual_seed(SEED)
        B, H, d = 1, 2, D
        # Simulate autoregressive growth: prefill of 8, then append 1 token x5.
        k = torch.randn(B, H, 8, d)
        v = torch.randn(B, H, 8, d)
        off = KVCacheDiskOffload(warm_size=64, codec="paper", bits=3)
        try:
            off.store(((k, v),))
            first = _staged_pairs(off.stage_for_forward())[0][0][..., :8, :].cpu().clone()
            for _ in range(5):
                k = torch.cat([k, torch.randn(B, H, 1, d)], dim=-2)
                v = torch.cat([v, torch.randn(B, H, 1, d)], dim=-2)
                off.store(((k, v),))
            # The first 8 positions must be byte-identical to their first staging
            # (they were compressed once and frozen — never re-quantized).
            later = _staged_pairs(off.stage_for_forward())[0][0][..., :8, :].cpu()
            assert torch.equal(first, later), "prefill block drifted — tokens were re-quantized"
            # And the full length must have grown to 13.
            assert off.memory_summary()["stored_len"] == 13
        finally:
            off.close()

    def test_bad_codec_rejected(self):
        with pytest.raises(ValueError):
            KVCacheDiskOffload(codec="nope")

    def test_ignores_quantizer_arg(self):
        """A passed-in quantizer is accepted but ignored (both codecs are parameter-free)."""
        off = KVCacheDiskOffload(quantizer=object(), warm_size=0)
        try:
            off.store(_fake_cache(n_layers=2))
            assert len(_staged_pairs(off.stage_for_forward())) == 2
        finally:
            off.close()

    def test_paper_file_smaller_than_float16(self):
        """Paper codec (bit-packed 3-bit indices) spills far smaller than the
        raw float16 cache it replaces."""
        import os
        cache = _fake_cache(n_layers=4, T=64)

        off = KVCacheDiskOffload(warm_size=0, disk_dir=None, codec="paper", bits=3)
        off.store(cache)  # warm_size=0 forces everything to disk
        d = off.memory_summary()["disk_dir"]
        paper = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        off.close()

        # float16 baseline: 2 bytes/coord across all K and V entries.
        float16_bytes = sum(
            (k.numel() + v.numel()) * 2 for k, v in cache
        )
        assert paper < float16_bytes, (
            f"paper {paper} should be < float16 {float16_bytes}"
        )


class TestPaperCodec:
    """The paper's rotate + Lloyd-Max offload codec and its bit-packing."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_pack_unpack_exact(self, bits):
        """Bit-packing indices then unpacking is a lossless round-trip."""
        torch.manual_seed(SEED)
        idx = torch.randint(0, 2 ** bits, (1000,))
        packed = _pack_indices(idx, bits)
        assert packed.dtype == torch.uint8
        back = _unpack_indices(packed, bits, idx.numel())
        assert torch.equal(idx, back)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_roundtrip_fidelity_and_shape(self, bits):
        """Higher bits -> lower error; dtype and shape preserved."""
        torch.manual_seed(SEED)
        t = torch.randn(1, 4, 32, D, dtype=torch.float16)
        q = _paper_compress(t, bits, _mse_factory)
        assert isinstance(q, _PaperKV)
        t_hat = _paper_dequantize(q, _mse_factory)
        assert t_hat.shape == t.shape
        assert t_hat.dtype == torch.float16

    def test_more_bits_more_faithful(self):
        """4-bit reconstruction must beat 2-bit (paper MSE monotonicity)."""
        torch.manual_seed(SEED)
        t = torch.randn(1, 4, 32, D)

        def rel(bits):
            q = _paper_compress(t, bits, _mse_factory)
            t_hat = _paper_dequantize(q, _mse_factory).float()
            return ((t - t_hat).norm() / t.norm()).item()

        assert rel(4) < rel(2)


# ---------------------------------------------------------------------------
# Entropy coding  (paper Section 2.2.5)
# ---------------------------------------------------------------------------


class TestEntropyCoding:
    """
    Locks the entropy/Huffman figures quoted in the paper so they cannot
    silently drift.  The paper previously reported the Huffman average
    (3.815) as the Shannon entropy; these are distinct quantities and both
    are asserted here.
    """

    @pytest.mark.parametrize(
        "bits,expected_h",
        [(2, 1.9129), (3, 2.8281), (4, 3.7715)],
    )
    def test_shannon_entropy_matches_paper(self, bits, expected_h):
        from src.entropy import entropy_bits

        assert entropy_bits(bits, D) == pytest.approx(expected_h, abs=1e-3)

    def test_huffman_average_above_entropy_below_raw(self):
        """
        Huffman must sit strictly between the Shannon lower bound and the raw
        bit-width: it emits whole bits per symbol so it cannot reach H, but it
        must still beat the uncoded representation.  Paper quotes 3.815 at b=4.
        """
        from src.entropy import HuffmanCodec, entropy_bits

        c = HuffmanCodec(num_bits=4, dim=D)
        h = entropy_bits(4, D)
        assert h < c.avg_bits < 4.0
        assert c.avg_bits == pytest.approx(3.8151, abs=1e-3)
        # ~4.6% saving vs raw 4-bit, as quoted in the paper
        assert (4.0 - c.avg_bits) / 4.0 == pytest.approx(0.046, abs=5e-3)

    def test_huffman_roundtrip_is_lossless(self):
        from src.entropy import HuffmanCodec

        torch.manual_seed(SEED)
        c = HuffmanCodec(num_bits=4, dim=D)
        idx = torch.randint(0, 16, (500,))
        assert torch.equal(c.decode(c.encode(idx), idx.numel()), idx)


# ---------------------------------------------------------------------------
# Adaptive tier changes are NOT lossless  (paper Section 3.3)
# ---------------------------------------------------------------------------


class TestAdaptiveTierChangeIsLossy:
    def test_promotion_does_not_recover_original(self):
        """
        Recompression runs on the already-dequantized k_hat, not the original
        k, so a demote->promote cycle compounds error instead of undoing it.
        The paper's Section 3.3 text is explicit about this; this test pins the
        direction of the inequality so the claim cannot regress to "reversible".
        """
        torch.manual_seed(SEED)
        x = torch.randn(64, D)
        q4 = KVQuantMSE(dim=D, num_bits=4, seed=0)
        q3 = KVQuantMSE(dim=D, num_bits=3, seed=0)

        def rt(qz, t):
            return qz.dequantize(qz.quantize(t))

        def mse(a, b):
            return float(((a - b) ** 2).mean())

        fresh4 = mse(x, rt(q4, x))
        demoted3 = rt(q3, x)
        promoted = mse(x, rt(q4, demoted3))

        # Promotion is strictly worse than a fresh 4-bit encode ...
        assert promoted > fresh4
        # ... and worse even than the 3-bit state it was promoted from.
        assert promoted > mse(x, demoted3)
