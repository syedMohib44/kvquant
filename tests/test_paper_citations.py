"""
Guard against citing sections of the TurboQuant paper that do not exist.

Every one of the patterns below was actually present in this repository and had
to be removed.  They were not typos: each attributed a real engineering decision
of ours (per-layer calibration, the GQA bit allowance, Hadamard rotations, the
low-rank correction, product quantization) to a paper that says nothing about
it.  That is the failure mode this file exists to prevent, because a false
citation is far more damaging in review than an uncited choice — it invites a
reader to check a source that will not back the claim.

The paper (arXiv:2504.19874v1, the only version) is a 25-page theory paper whose
complete structure is:

    1  Introduction            1.1 Problem Definition
                               1.2 Related Work
                               1.3 Overview of Techniques and Contributions
    2  Preliminaries           2.1 Shannon Lower Bound on Distortion
                               2.2 QJL: 1-bit inner product quantization
    3  TurboQuant              3.1 MSE Optimal TurboQuant
                               3.2 Inner-product Optimal TurboQuant
                               3.3 Lower Bounds
    4  Experiments             4.1 Empirical Validation
                               4.2 Needle-In-A-Haystack
                               4.3 End-to-end Generation on LongBench
                               4.4 Near Neighbour Search Experiments
    References

There is no section 5 or beyond, no appendix, no third level of numbering
(no §2.2.1), and no line numbers.  Numbered environments are Lemmas 1-4,
Definition 1, Theorems 1-3, Algorithms 1-2, and no Corollaries.

Scope note: this checks `src/` and `tests/` only.  PAPER.md and README.md are
excluded on purpose — PAPER.md has its own "## 5. Experiments" and cross-refers
to it constantly, so a text-level scan there produces nothing but false
positives.  PAPER.md §2.1 states the citation convention it follows instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC_DIRS = ("src", "tests")

# Each entry: (compiled pattern, why it is wrong).
# The message is the point of the test — a bare match tells a future reader
# nothing about which real section, if any, they should have cited.
_FORBIDDEN: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?:paper|Paper)\s*(?:§|Section\s*)\s*(?:[5-9]|1\d)\b"),
        "the paper ends at section 4 (Experiments); there is no section 5+. "
        "The outlier split is a three-sentence aside in its §4.3.",
    ),
    (
        re.compile(r"§\s*2\.2\.\d"),
        "the paper has no third-level numbering; §2 has only §2.1 and §2.2. "
        "The coordinate marginal is Lemma 1; QJL is Definition 1 / Lemma 4.",
    ),
    (
        re.compile(r"§\s*3\.[4-9]"),
        "§3 has only §3.1 (MSE), §3.2 (inner-product), §3.3 (lower bounds). "
        "The low-rank correction is ours and is not in the paper at all.",
    ),
    (
        re.compile(r"§\s*\d{3,}"),
        "the paper has no line numbers, so '§399-412'-style references address "
        "nothing. These came from line offsets in a text dump.",
    ),
    (
        re.compile(r"§\s*Per-layer\s+calibration", re.IGNORECASE),
        "there is no such section. The paper is explicitly data-oblivious "
        "(§1.2) and prescribes no calibration; per-layer calibration is ours.",
    ),
    (
        re.compile(r"(?:Section|§)\s*3 of the paper"),
        "ambiguous: §3 has three subsections. The MSE quantizer is §3.1 "
        "(Algorithm 1, Theorem 1).",
    ),
    (
        re.compile(r"(?:Section|§)\s*4 of the paper"),
        "§4 is Experiments, not a method section. The inner-product quantizer "
        "is §3.2 (Algorithm 2, Theorem 2).",
    ),
]


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for d in _SRC_DIRS:
        files.extend(sorted((_REPO / d).rglob("*.py")))
    # `build/` is a packaging artifact holding a stale copy of `src/`; linting it
    # would fail on citations already fixed in the real tree.
    return [f for f in files if "build" not in f.parts and "__pycache__" not in f.parts]


def test_source_tree_is_not_empty():
    """A glob that silently matches nothing would make every test below vacuous."""
    files = _python_sources()
    assert len(files) > 10, f"expected the source tree, found {len(files)} files"


@pytest.mark.parametrize(
    "pattern,reason", _FORBIDDEN, ids=[p.pattern[:28] for p, _ in _FORBIDDEN]
)
def test_no_fabricated_paper_citations(pattern: re.Pattern[str], reason: str):
    hits: list[str] = []
    for path in _python_sources():
        # This file quotes the forbidden patterns in order to define them.
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = path.relative_to(_REPO).as_posix()
                hits.append(f"  {rel}:{lineno}: {line.strip()}")

    assert not hits, (
        f"Fabricated citation to arXiv:2504.19874 — {reason}\n"
        + "\n".join(hits)
        + "\n\nCite the real section, or state the claim as ours. See the "
        "docstring of this file for the paper's actual structure."
    )
