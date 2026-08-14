"""
Pin the transformers behaviours this library silently depends on.

These are not tests of our code.  They are tripwires on the library underneath
it: each one asserts a contract that, if a future transformers release changes
it, would otherwise break us *quietly* — producing plausible numbers that are
wrong, rather than an exception anyone would notice.

Verified against transformers 4.57.6.
"""

from __future__ import annotations

import torch
import transformers
from transformers import AutoModelForCausalLM


def _tiny():
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    model.eval()
    return model


def test_use_cache_false_still_attends_to_past():
    """
    ``use_cache=False`` must mean "do not grow the cache", not "ignore it".

    ``eval_ppl.py`` scores target tokens with
    ``model(tgt, past_key_values=cache, use_cache=False)`` and relies on the
    prefill context still being attended to.  If a release changed this to drop
    ``past_key_values``, every logit would be computed against a truncated
    context: no error, no warning, just a perplexity number that is quietly
    meaningless — and quantization would take the blame for the degradation.

    The comparison is against a full-context forward, which is the only thing
    that distinguishes "attended to the past" from "silently ignored it".
    """
    model = _tiny()
    ids = torch.arange(1, 17).unsqueeze(0)
    split = 9

    with torch.no_grad():
        prefill = model(ids[:, :split], use_cache=True)
        scored = model(
            ids[:, split:], past_key_values=prefill.past_key_values, use_cache=False
        ).logits
        full = model(ids).logits[:, split:]

    err = (scored - full).abs().max().item()
    assert err < 1e-4, (
        f"use_cache=False no longer attends to past_key_values "
        f"(transformers {transformers.__version__}): max logit error {err:.3e} "
        f"vs a full-context forward. eval_ppl.py's scores are invalid until "
        f"that call is restructured — see the comment at its model() call."
    )


def test_use_cache_false_still_mutates_the_cache_in_place():
    """
    Documents a genuine surprise: ``use_cache=False`` still *grows* the cache
    object that was passed in.

    I initially assumed the opposite and wrote a test asserting it — the name
    plainly suggests "do not cache" — and it failed: the cache went from 9 to 16
    positions.  So ``use_cache=False`` controls only whether the updated cache is
    *returned*, not whether the passed-in object is mutated.

    ``eval_ppl.compute_ppl`` is safe from this because it builds a fresh prefill
    cache inside the per-chunk loop and never reuses one across chunks.  This
    test records the behaviour so nobody "optimises" that prefill out of the loop
    on the reasonable-but-wrong assumption that the cache is left alone; doing so
    would leak each chunk's tokens into the next one's context and quietly
    deflate the reported perplexity.
    """
    model = _tiny()
    ids = torch.arange(1, 17).unsqueeze(0)
    split = 9

    with torch.no_grad():
        prefill = model(ids[:, :split], use_cache=True)
        cache = prefill.past_key_values
        before = cache.get_seq_length()
        model(ids[:, split:], past_key_values=cache, use_cache=False)
        after = cache.get_seq_length()

    assert before == split
    assert after == ids.shape[1], (
        f"use_cache=False no longer mutates the passed-in cache "
        f"({before} -> {after}, transformers {transformers.__version__}). "
        f"That is the safer behaviour, so this is not a bug — but the warning in "
        f"eval_ppl.compute_ppl about hoisting the prefill out of the loop can be "
        f"relaxed, and this test should be updated to match."
    )
