"""
Entropy coding for KVQuant codebook indices (paper Section 6).

After Lloyd-Max quantization each coordinate is assigned an index in
{0, ..., 2^b - 1}.  These indices are NOT uniformly distributed - the
probabilities follow the area under each Voronoi cell of the codebook.
Huffman coding exploits this non-uniformity to reduce the average
bit-width toward the Shannon entropy.

At b=4 the entropy is ~3.8 bits/coordinate (paper Table 1), giving ~4%
compression on top of the raw 4-bit representation.

Classes
-------
  HuffmanCodec   - builds a Huffman tree from codebook probabilities and
                   encodes/decodes integer index tensors to/from packed bits.

Functions
---------
  codebook_probs(num_bits, dim)  - compute the symbol probability for each
                                   codebook index under the rotated-coordinate
                                   distribution (approx. N(0, 1/d)).
  entropy_bits(num_bits, dim)    - Shannon entropy H in bits/coordinate.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from .codebook import build_codebook


# ---------------------------------------------------------------------------
# Symbol probabilities
# ---------------------------------------------------------------------------


def codebook_probs(num_bits: int, dim: int) -> Tensor:
    """
    Compute the probability of each codebook symbol under N(0, 1/d).

    Each probability is the integral of N(0, 1/d) over the Voronoi cell
    of the corresponding centroid (i.e. the area between adjacent decision
    boundaries).

    Args:
        num_bits: bits per coordinate.
        dim:      vector dimension d (used for the N(0,1/d) variance).

    Returns:
        probs: Tensor of shape (2**num_bits,) summing to 1.
    """
    centroids = build_codebook(num_bits, dim)  # (k,) scaled by 1/sqrt(d)
    k = len(centroids)
    std = 1.0 / math.sqrt(dim)

    # Decision boundaries: midpoints between adjacent centroids
    # Add -inf and +inf at the ends
    bounds = torch.full((k + 1,), float("inf"))
    bounds[0] = float("-inf")
    bounds[-1] = float("inf")
    bounds[1:-1] = (centroids[:-1] + centroids[1:]) / 2.0

    # P(symbol i) = Phi((b_{i+1} - 0) / std) - Phi((b_i - 0) / std)
    # where Phi is the standard normal CDF
    from torch.distributions import Normal

    normal = Normal(0.0, std)
    lo = normal.cdf(bounds[:-1])  # (k,)
    hi = normal.cdf(bounds[1:])  # (k,)
    probs = (hi - lo).clamp(min=1e-12)
    return probs / probs.sum()  # normalise for numerical safety


def entropy_bits(num_bits: int, dim: int) -> float:
    """
    Shannon entropy H of the codebook symbol distribution, in bits/coordinate.

    H = -sum_i p_i * log2(p_i)

    The raw representation uses exactly `num_bits` bits per coordinate.
    The gap (num_bits - H) is the potential saving from entropy coding.
    """
    p = codebook_probs(num_bits, dim)
    h = -(p * p.log2()).sum().item()
    return h


# ---------------------------------------------------------------------------
# Huffman codec
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Node:
    prob: float
    symbol: int = field(default=-1, compare=False)
    left: "_Node | None" = field(default=None, compare=False)
    right: "_Node | None" = field(default=None, compare=False)


class HuffmanCodec:
    """
    Huffman codec for TurboQuant codebook indices.

    Build once per (num_bits, dim) pair; then use encode() / decode().

    Args:
        num_bits: bits per coordinate (determines alphabet size = 2**num_bits).
        dim:      vector dimension (determines symbol probabilities).
    """

    def __init__(self, num_bits: int, dim: int) -> None:
        self.num_bits = num_bits
        self.dim = dim
        probs = codebook_probs(num_bits, dim).tolist()
        self._codes, self._lengths = _build_huffman(probs)
        # Decoding: flat map from bit-string prefix -> symbol
        self._decode_table = _build_decode_table(self._codes)

    # ------------------------------------------------------------------
    @property
    def avg_bits(self) -> float:
        """Expected bits/symbol under the codebook distribution."""
        probs = codebook_probs(self.num_bits, self.dim).tolist()
        return sum(p * self._lengths[i] for i, p in enumerate(probs))

    @property
    def entropy(self) -> float:
        """Shannon entropy in bits (lower bound on avg_bits)."""
        return entropy_bits(self.num_bits, self.dim)

    # ------------------------------------------------------------------
    def encode(self, indices: Tensor) -> list[int]:
        """
        Huffman-encode a flat sequence of codebook indices.

        Args:
            indices: LongTensor of any shape.

        Returns:
            Packed list of ints where each int is one bit (0 or 1).
            (A real implementation would pack into bytes/uint64 for efficiency.)
        """
        bits: list[int] = []
        for idx in indices.flatten().tolist():
            bits.extend(self._codes[int(idx)])
        return bits

    def decode(self, bits: list[int], length: int) -> Tensor:
        """
        Huffman-decode a bit stream back to codebook indices.

        Args:
            bits:   Bit list produced by encode().
            length: Number of symbols expected in output.

        Returns:
            LongTensor of shape (length,).
        """
        symbols: list[int] = []
        node = ""
        for bit in bits:
            node += str(bit)
            if node in self._decode_table:
                symbols.append(self._decode_table[node])
                node = ""
            if len(symbols) == length:
                break
        return torch.tensor(symbols, dtype=torch.long)


class EntropyStats(NamedTuple):
    raw_bits: float  # naive: num_bits per coordinate
    entropy: float  # Shannon lower bound
    huffman_avg: float  # actual Huffman avg bits/coord
    saving_pct: float  # % reduction vs raw


def analyse(num_bits: int, dim: int) -> EntropyStats:
    """
    Report entropy coding savings for a given (num_bits, dim) config.

    Example::

        >>> from kvquant.entropy import analyse
        >>> analyse(4, 128)
        EntropyStats(raw_bits=4, entropy=3.816, huffman_avg=3.847, saving_pct=3.83)
    """
    codec = HuffmanCodec(num_bits, dim)
    raw = float(num_bits)
    h = codec.entropy
    avg = codec.avg_bits
    pct = (raw - avg) / raw * 100.0
    return EntropyStats(raw, h, avg, pct)


# ---------------------------------------------------------------------------
# Internal Huffman tree builder
# ---------------------------------------------------------------------------


def _build_huffman(probs: list[float]) -> tuple[list[list[int]], list[int]]:
    """
    Build a Huffman tree from symbol probabilities.

    Returns:
        codes:   list of bit-lists, one per symbol.
        lengths: list of code lengths, one per symbol.
    """
    k = len(probs)
    heap = [_Node(prob=p, symbol=i) for i, p in enumerate(probs)]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, _Node(prob=lo.prob + hi.prob, left=lo, right=hi))

    root = heap[0]
    codes: list[list[int]] = [[] for _ in range(k)]
    _assign_codes(root, [], codes)
    lengths = [len(c) for c in codes]
    return codes, lengths


def _assign_codes(node: _Node, prefix: list[int], codes: list[list[int]]) -> None:
    if node.symbol >= 0:  # leaf
        codes[node.symbol] = prefix[:]
        return
    if node.left:
        prefix.append(0)
        _assign_codes(node.left, prefix, codes)
        prefix.pop()
    if node.right:
        prefix.append(1)
        _assign_codes(node.right, prefix, codes)
        prefix.pop()


def _build_decode_table(codes: list[list[int]]) -> dict[str, int]:
    return {"".join(map(str, bits)): sym for sym, bits in enumerate(codes)}
