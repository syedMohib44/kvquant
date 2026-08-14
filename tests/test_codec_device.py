"""
Phase 1: the codec must be exact, and must stay on its input's device.

The old numpy `packbits` path forced a host round-trip on every call.  That is
correct but ruinous on the generation path: ~1.9 ms per 128-token block, times
28 layers times 2 tensors, is ~106 ms per decoded token before the model does
any work.  These tests pin both properties that make the torch rewrite usable:
bit-exactness at every supported width, and no silent device transfer.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.kv_cache import (
    _pack_indices,
    _paper_compress,
    _paper_dequantize,
    _unpack_indices,
)
from src.quantizer import KVQuantMSE

D = 64


def _mse_factory(dim: int, bits: int) -> KVQuantMSE:
    return KVQuantMSE(dim=dim, num_bits=bits, seed=0)


class TestBitPacking:
    @pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_roundtrip_exact_cpu(self, num_bits):
        """Bit-packing is lossless at every supported width."""
        torch.manual_seed(0)
        idx = torch.randint(0, 2**num_bits, (5000,))
        back = _unpack_indices(_pack_indices(idx, num_bits), num_bits, idx.numel())
        assert torch.equal(back, idx.to(torch.int64))

    @pytest.mark.cuda
    @pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_roundtrip_exact_cuda_and_stays_on_device(self, num_bits):
        """
        Exact on GPU *and* never leaves it.  The device assertions are the
        point: a silent `.cpu()` would still pass an equality check while
        reintroducing the round-trip this rewrite exists to remove.
        """
        torch.manual_seed(0)
        idx = torch.randint(0, 2**num_bits, (5000,), device="cuda")
        packed = _pack_indices(idx, num_bits)
        assert packed.is_cuda, "packed codes must stay on the input device"

        back = _unpack_indices(packed, num_bits, idx.numel())
        assert back.is_cuda, "unpacked indices must stay on the input device"
        assert torch.equal(back, idx.to(torch.int64))

    def test_packed_dtype_is_uint8(self):
        idx = torch.randint(0, 16, (100,))
        assert _pack_indices(idx, 4).dtype == torch.uint8

    @pytest.mark.parametrize(
        "count,num_bits", [(1, 1), (7, 3), (100, 4), (1000, 5), (333, 8)]
    )
    def test_packed_size_matches_bit_width(self, count, num_bits):
        """Byte count must be exactly ceil(count * num_bits / 8) — no slack."""
        idx = torch.randint(0, 2**num_bits, (count,))
        packed = _pack_indices(idx, num_bits)
        assert packed.numel() == math.ceil(count * num_bits / 8)

    def test_empty_input_roundtrips(self):
        idx = torch.zeros(0, dtype=torch.int64)
        packed = _pack_indices(idx, 4)
        assert packed.numel() == 0
        assert _unpack_indices(packed, 4, 0).numel() == 0

    def test_boundary_values_survive(self):
        """Min and max of the alphabet are where an off-by-one would show."""
        for num_bits in (1, 3, 4, 8):
            hi = 2**num_bits - 1
            idx = torch.tensor([0, hi, 0, hi, hi, 0])
            back = _unpack_indices(
                _pack_indices(idx, num_bits), num_bits, idx.numel()
            )
            assert torch.equal(back, idx.to(torch.int64))

    def test_rejects_out_of_range_bit_width(self):
        idx = torch.zeros(4, dtype=torch.int64)
        with pytest.raises(AssertionError, match=r"num_bits must be in \[1, 8\]"):
            _pack_indices(idx, 9)
        with pytest.raises(AssertionError, match=r"num_bits must be in \[1, 8\]"):
            _unpack_indices(torch.zeros(4, dtype=torch.uint8), 0, 4)

    def test_stream_layout_matches_numpy_packbits(self):
        """
        The torch implementation must produce a byte-identical stream to the
        numpy one it replaced, so codes written by the previous version (e.g.
        already spilled to disk) remain readable.
        """
        import numpy as np

        torch.manual_seed(0)
        num_bits = 3
        idx = torch.randint(0, 2**num_bits, (777,))

        flat = idx.reshape(-1).to(torch.int64).numpy()
        shifts = np.arange(num_bits - 1, -1, -1, dtype=np.int64)
        bits = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
        expected = torch.from_numpy(np.packbits(bits.reshape(-1)))

        assert torch.equal(_pack_indices(idx, num_bits), expected)


class TestPaperCodecDevice:
    @pytest.mark.cuda
    def test_compress_keeps_codes_on_gpu(self):
        """
        `_paper_compress` used to force `.cpu()`. A VRAM-resident code cache
        depends on it not doing that any more.
        """
        torch.manual_seed(0)
        t = torch.randn(1, 4, 32, D, device="cuda")
        q = _paper_compress(t, 4, _mse_factory)
        assert q.packed.is_cuda
        assert q.norms.is_cuda

    @pytest.mark.cuda
    def test_full_roundtrip_on_gpu_without_host_transfer(self):
        torch.manual_seed(0)
        t = torch.randn(1, 4, 32, D, device="cuda")
        t_hat = _paper_dequantize(_paper_compress(t, 4, _mse_factory), _mse_factory)
        assert t_hat.is_cuda
        assert t_hat.shape == t.shape
        rel = ((t - t_hat).norm() / t.norm()).item()
        assert rel < 0.5

    @pytest.mark.cuda
    def test_gpu_and_cpu_agree(self):
        """
        Same input, same seed, both devices — reconstructions must match.
        Guards against a device-dependent codebook or rotation buffer.
        """
        torch.manual_seed(0)
        t_cpu = torch.randn(1, 2, 16, D)
        t_gpu = t_cpu.cuda()

        hat_cpu = _paper_dequantize(
            _paper_compress(t_cpu, 4, _mse_factory), _mse_factory
        )
        hat_gpu = _paper_dequantize(
            _paper_compress(t_gpu, 4, _mse_factory), _mse_factory
        ).cpu()

        assert torch.allclose(hat_cpu, hat_gpu, atol=1e-5)

    def test_cpu_compress_still_works(self):
        """The offload tier compresses on CPU; that path must be unaffected."""
        torch.manual_seed(0)
        t = torch.randn(1, 4, 32, D)
        q = _paper_compress(t, 3, _mse_factory)
        assert not q.packed.is_cuda
        assert _paper_dequantize(q, _mse_factory).shape == t.shape
