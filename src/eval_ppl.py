"""
eval_ppl.py - KV-cache quantization perplexity evaluation.

Evaluates the perplexity impact of KV cache quantization on WikiText-2 (or
a built-in fallback corpus if the `datasets` library is not installed).

Method
------
For each text chunk:
  1. Tokenise and take the first `context_len` tokens as the KV cache prefill.
  2. Run a full-precision forward pass to populate the KV cache.
  3. Quantize the KV cache with KVCacheQuantizer (or leave it as-is for the
     baseline).
  4. Continue forward over the next `target_len` tokens using the (possibly
     quantized) cache and record the NLL at each position.

This directly measures the degradation KV cache quantization causes during
generation - the scenario the method is designed for - rather than the
teacher-forcing PPL which ignores cache errors entirely.

Usage
-----
    python -m kvquant.eval_ppl                          # distilgpt2, WikiText-2
    python -m kvquant.eval_ppl --model gpt2-medium      # larger base model
    python -m kvquant.eval_ppl --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python -m kvquant.eval_ppl --context-len 256        # longer context
    python -m kvquant.eval_ppl --correction-rank 4      # with low-rank correction
    python -m kvquant.eval_ppl --correction-rank 4 --model gpt2-medium
"""

from __future__ import annotations

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import KVCacheQuantizer, kvs_from_cache, quantize_model_cache
from kvquant.demo_llm import load_model, get_model_dims, sep

BITS_LIST = [2, 3, 4]

# ---------------------------------------------------------------------------
# Built-in fallback corpus (used when `datasets` is not installed)
# ---------------------------------------------------------------------------
FALLBACK_TEXTS = [
    "The history of the Roman Empire is a fascinating study in the rise and fall "
    "of one of the greatest civilizations in human history. Rome began as a small "
    "city-state in central Italy and grew to control much of Europe, North Africa, "
    "and the Middle East. The Roman Republic was founded in 509 BC after the "
    "overthrow of the Roman Kingdom. Julius Caesar crossed the Rubicon in 49 BC, "
    "triggering a civil war that ended the Republic. His adopted son Augustus became "
    "the first Roman Emperor in 27 BC. At its height under Emperor Trajan around "
    "117 AD, the empire covered approximately five million square kilometres. "
    "The Western Roman Empire fell in 476 AD when Odoacer deposed Romulus Augustulus. "
    "The Eastern Roman Empire, the Byzantine Empire, continued until 1453. "
    "Roman contributions to law, engineering, and language endure to this day.",
    "Artificial intelligence has undergone a remarkable transformation in the past "
    "decade, moving from a niche academic discipline to one of the most consequential "
    "technologies of our time. The breakthrough came with deep learning, a technique "
    "that uses neural networks with many layers to learn representations of data. "
    "In 2012, AlexNet won the ImageNet competition by a large margin, demonstrating "
    "that neural networks could outperform hand-crafted computer vision systems. "
    "Since then, deep learning has achieved superhuman performance in tasks ranging "
    "from image recognition and speech transcription to game playing and protein "
    "structure prediction. Large language models trained on vast text corpora have "
    "shown an extraordinary ability to generate coherent and contextually appropriate "
    "text across virtually any domain. These models learn statistical patterns in "
    "language and can complete sentences, answer questions, write code, and engage "
    "in nuanced dialogue. Ensuring that AI systems remain safe and aligned with "
    "human values remains one of the central challenges of the field.",
    "The physics of quantum mechanics describes the behavior of matter and energy "
    "at the smallest scales, where the rules of classical physics break down. "
    "At the quantum scale, particles do not have definite positions or momenta "
    "until measured; instead they exist in superpositions of multiple states. "
    "This was demonstrated by the double-slit experiment, in which individual "
    "electrons sent through two slits create an interference pattern, as if each "
    "electron passes through both slits simultaneously. The mathematical framework "
    "of quantum mechanics, developed by Heisenberg, Schrödinger, and Dirac in "
    "the 1920s, makes extraordinarily precise predictions about atomic phenomena. "
    "Quantum entanglement means that two particles can be correlated so that "
    "measuring one instantly determines the state of the other, regardless of "
    "distance. Einstein called this spooky action at a distance. Quantum mechanics "
    "underpins transistors, lasers, magnetic resonance imaging, and quantum computing.",
    "Climate change refers to long-term shifts in global temperatures and weather "
    "patterns. While natural factors have historically driven climate variations, "
    "since the Industrial Revolution human activities have been the primary driver "
    "of climate change through the burning of fossil fuels such as coal, oil, and "
    "natural gas. These activities release greenhouse gases, primarily carbon dioxide "
    "and methane, which trap heat in the atmosphere. The Intergovernmental Panel on "
    "Climate Change has documented how global average temperatures have risen by "
    "approximately 1.1 degrees Celsius since the pre-industrial period. This warming "
    "is causing more frequent and intense extreme weather events, rising sea levels, "
    "melting of polar ice, and disruptions to ecosystems worldwide. International "
    "efforts to address climate change include the Paris Agreement, signed in 2015, "
    "which aims to limit warming to 1.5 degrees Celsius above pre-industrial levels.",
    "The development of the internet has fundamentally transformed how human beings "
    "communicate, access information, conduct commerce, and organize society. "
    "The origins of the internet trace back to ARPANET, a research network funded "
    "by the United States Department of Defense in the 1960s. Tim Berners-Lee "
    "invented the World Wide Web in 1989, creating the system of hyperlinked "
    "documents that made the internet accessible to the general public. In the "
    "1990s, the commercialization of the internet led to the dot-com boom, as "
    "entrepreneurs and investors rushed to capitalize on the new medium. "
    "Search engines such as Google, founded in 1998, made it possible to navigate "
    "the rapidly expanding web. Social media platforms emerged in the 2000s, "
    "creating new forms of communication and community. Today the internet connects "
    "billions of people worldwide and supports a global digital economy.",
    "The human brain is the most complex organ in the known universe, containing "
    "approximately eighty-six billion neurons, each connected to thousands of others "
    "through synapses. Neurons communicate via electrochemical signals, and the "
    "collective activity of these signals gives rise to perception, thought, emotion, "
    "and consciousness. The cerebral cortex, the outermost layer, is responsible for "
    "higher cognitive functions including language, reasoning, and voluntary movement. "
    "The limbic system, deeper in the brain, governs emotion and memory formation. "
    "The cerebellum coordinates movement and balance, while the brainstem regulates "
    "basic life functions such as breathing and heart rate. Neuroplasticity means "
    "the brain can reorganise its connections in response to experience and injury, "
    "a property that underlies learning and recovery from stroke. Advances in "
    "functional magnetic resonance imaging have allowed scientists to observe the "
    "brain at work, revealing the neural correlates of attention, decision-making, "
    "and social cognition with unprecedented precision.",
    "The theory of evolution by natural selection, proposed independently by "
    "Charles Darwin and Alfred Russel Wallace in 1858, is the unifying framework "
    "of modern biology. It holds that heritable variation exists within populations, "
    "that individuals with traits better suited to their environment survive and "
    "reproduce more successfully, and that over many generations this process "
    "produces cumulative adaptation. Darwin's observations of finches in the "
    "Galapagos Islands illustrated how populations diverge when isolated in "
    "different environments. The modern evolutionary synthesis combined Darwin's "
    "theory with Mendelian genetics, explaining how variation is generated and "
    "inherited. The discovery of DNA's structure by Watson, Crick, Franklin, and "
    "Wilkins provided the molecular basis for heredity. Genome sequencing has "
    "since confirmed common ancestry across all life on Earth and revealed the "
    "evolutionary relationships among species with extraordinary detail.",
    "The Renaissance was a cultural and intellectual movement that began in Italy "
    "in the fourteenth century and spread across Europe over the following two "
    "centuries, marking the transition from the medieval period to the modern era. "
    "It was characterised by renewed interest in the art, literature, and philosophy "
    "of ancient Greece and Rome, a perspective known as humanism that placed human "
    "experience and reason at the centre of enquiry. Artists such as Leonardo da "
    "Vinci and Michelangelo elevated painting and sculpture to new heights of "
    "technical mastery and psychological depth. The invention of the printing press "
    "by Johannes Gutenberg around 1440 accelerated the diffusion of Renaissance "
    "ideas by making books affordable and widely available for the first time. "
    "The period also saw major advances in astronomy, anatomy, and mathematics, "
    "laying the intellectual groundwork for the scientific revolution that followed.",
    "Music theory is the study of the principles and practices that govern how "
    "music is constructed and perceived. At its core are concepts such as pitch, "
    "rhythm, harmony, and form. Pitch classes are organised into scales, of which "
    "the diatonic major and minor scales are most common in Western music. Chords "
    "are formed by stacking intervals, and the relationships between chords create "
    "harmonic progressions that generate tension and resolution. Rhythm organises "
    "events in time through patterns of strong and weak beats, metre, and tempo. "
    "Counterpoint describes how independent melodic lines combine to create texture. "
    "The circle of fifths is a diagram showing the relationships between the twelve "
    "major and minor keys. In the twentieth century, composers such as Schoenberg "
    "developed atonal and twelve-tone systems that abandoned traditional tonality. "
    "Jazz introduced new harmonic vocabulary including extended chords and modal "
    "improvisation, profoundly influencing subsequent popular and classical music.",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_chunks(
    tokenizer,
    context_len: int,
    target_len: int,
    max_chunks: int,
    skip: int = 0,
):
    """
    Return a list of (context_ids, target_ids) tensors from WikiText-2 or
    the built-in fallback corpus.

    skip: number of chunks to skip from the start (used to separate
          calibration data from evaluation data).
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in dataset["text"] if len(t.strip()) > 100)
        print("  Dataset : WikiText-2 test set (via `datasets`)")
    except Exception:
        # Tile the corpus so we have enough tokens for skip + max_chunks.
        # English averages ~5 chars/token; add +2 repeats as safety margin.
        chunk_tokens = context_len + target_len
        base = " ".join(FALLBACK_TEXTS)
        chars_needed = (skip + max_chunks + 2) * chunk_tokens * 5
        repeats = math.ceil(chars_needed / max(len(base), 1)) + 1
        text = (" " + base) * max(repeats, 2)
        print(
            "  Dataset : built-in fallback corpus  "
            "(pip install datasets  for WikiText-2)"
        )

    import logging

    _hf_log = logging.getLogger("transformers")
    _prev = _hf_log.level
    _hf_log.setLevel(logging.ERROR)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    _hf_log.setLevel(_prev)
    chunk = context_len + target_len
    chunks = []
    start_tok = skip * chunk  # token offset corresponding to `skip` chunks
    for i in range(start_tok, len(tokens) - chunk, chunk):
        ctx = tokens[i : i + context_len].unsqueeze(0)
        tgt = tokens[i + context_len : i + chunk].unsqueeze(0)
        chunks.append((ctx, tgt))
        if len(chunks) >= max_chunks:
            break
    return chunks


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------


def compute_ppl(
    model, chunks, kvc=None, correction_rank: int = 0, label: str = ""
) -> float:
    """
    Compute perplexity under (optionally quantized) KV cache.

    For each chunk: prefill context -> (quantize cache) -> score target tokens.
    """
    total_nll = 0.0
    total_tok = 0
    n = len(chunks)

    import sys

    tty = sys.stdout.isatty()
    for i, (ctx, tgt) in enumerate(chunks):
        if tty:
            print(f"\r  {label or 'eval'} [{i+1}/{n}]", end="", flush=True)
        # Prefill: full-precision KV cache from context tokens
        with torch.no_grad():
            prefill = model(ctx, use_cache=True)
        cache = prefill.past_key_values

        # Optionally quantize the cache
        if kvc is not None:
            cache = quantize_model_cache(cache, kvc, correction_rank=correction_rank)

        # Score target tokens against the (possibly quantized) cache.
        # prefill.logits[:, -1, :] predicts tgt[:, 0]  (first generated token).
        # out.logits[:, i, :]      predicts tgt[:, i+1] so we shift by -1.
        with torch.no_grad():
            out = model(tgt, past_key_values=cache, use_cache=False)

        first_lp = F.log_softmax(prefill.logits[:, -1:, :], dim=-1)  # (1,1,V)
        rest_lp = F.log_softmax(out.logits[:, :-1, :], dim=-1)  # (1,T-1,V)
        all_lp = torch.cat([first_lp, rest_lp], dim=1)  # (1,T,V)
        nll = -all_lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (1,T)
        total_nll += nll.sum().item()
        total_tok += tgt.numel()

    if tty:
        print(f"\r{' ' * 40}\r", end="", flush=True)  # clear progress line
    return math.exp(total_nll / max(total_tok, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="KV-cache quantization perplexity benchmark"
    )
    parser.add_argument(
        "--model",
        default="distilgpt2",
        help="HuggingFace model name (default: distilgpt2)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=128,
        help="Tokens used as KV cache context (default: 128)",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=64,
        help="Tokens scored against the cache (default: 64)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        help="Number of text chunks to evaluate (default: 50)",
    )
    parser.add_argument(
        "--correction-rank",
        type=int,
        default=0,
        help="Low-rank correction rank applied to quantized cache. "
        "0 = disabled, 4 = recommended  (default: 0)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Load model in float16 (faster, less memory for large models)",
    )
    args = parser.parse_args()

    model, tok = load_model(args.model)
    if args.half:
        model = model.half()
    _, _, _, head_dim = get_model_dims(model)
    n_outlier = max(4, head_dim // 4)

    sep(f"KV-cache quantization PPL  -  {args.model}")
    print(
        f"  context : {args.context_len} tokens  "
        f"target : {args.target_len} tokens  "
        f"chunks : {args.max_chunks}"
    )
    if args.correction_rank:
        print(f"  correction-rank : {args.correction_rank}")

    chunks = load_chunks(tok, args.context_len, args.target_len, args.max_chunks)
    print(f"  Loaded  : {len(chunks)} chunks\n")

    # Calibration pool - skip past the eval window so no token overlap.
    cal_chunks = load_chunks(
        tok, args.context_len, args.target_len, max_chunks=8, skip=args.max_chunks
    )
    cal_ids = torch.cat([c for c, _ in cal_chunks], dim=0)
    with torch.no_grad():
        cal_out = model(cal_ids, use_cache=True)
    cal_kvs = kvs_from_cache(cal_out.past_key_values)
    T_cal = cal_ids.shape[1]
    # Per-layer calibration data: keep each layer's K/V separate.  Pooling across
    # layers averages out the layer-specific outlier channels and mis-identifies
    # them for every individual layer.
    #
    # This is OUR design decision, not the paper's.  TurboQuant is explicitly
    # data-oblivious ("apply instantly without needing data-specific tuning or
    # calibrations", §1.2) and says nothing about layers.  Calibration is forced
    # on us by the outlier split, which the paper introduces in a §4.3 aside
    # without giving any selection criterion; once you must choose outlier
    # channels from data, doing it per layer is what the measurements support.
    per_layer_kv = [
        (kv[0].reshape(-1, T_cal, head_dim), kv[1].reshape(-1, T_cal, head_dim))
        for kv in cal_kvs
    ]

    # -----------------------------------------------------------------------
    sep("Results")
    W = 34
    print(f"  {'Configuration':<{W}} {'PPL':>8}  {'delta PPL':>10}")
    sep()

    ppl_fp32 = compute_ppl(model, chunks, label="fp32")
    print(f"  {'Float32 (unquant)':<{W}} {ppl_fp32:>8.2f}  {'-':>10}")

    for bits in BITS_LIST:
        # One KVCacheQuantizer PER LAYER, each calibrated on its own layer's KV
        # (see the note above — our choice, not the paper's).  quantize_model_cache()
        # accepts this list and applies the matching quantizer to each attention
        # layer in order.
        kvc = []
        for lk, lv in per_layer_kv:
            q = KVCacheQuantizer(
                head_dim=head_dim,
                num_bits=bits,
                use_outlier=True,
                n_outlier=n_outlier,
                outlier_bits=min(bits + 1, 4),
                regular_bits=max(bits - 1, 1),
            )
            q.calibrate(lk, lv)
            kvc.append(q)

        # Without correction
        ppl_q = compute_ppl(
            model, chunks, kvc=kvc, correction_rank=0, label=f"{bits}-bit"
        )
        d = ppl_q - ppl_fp32
        print(
            f"  {f'{bits}-bit  (avg {kvc[0].avg_bits:.2f} bpw)':<{W}} "
            f"{ppl_q:>8.2f}  {d:>+10.2f}"
        )

        # With low-rank correction (if requested)
        if args.correction_rank > 0:
            ppl_c = compute_ppl(
                model,
                chunks,
                kvc=kvc,
                correction_rank=args.correction_rank,
                label=f"{bits}-bit+rank-{args.correction_rank}",
            )
            d_c = ppl_c - ppl_fp32
            label = f"{bits}-bit + rank-{args.correction_rank} correction"
            print(f"  {label:<{W}} {ppl_c:>8.2f}  {d_c:>+10.2f}")

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
