"""
Run KVQuant++ generation with KV-cache offload to SSD (and optional weight offload).

The KV-cache offload (`offload_to_disk`) is NOT exposed by demo_llm.py's CLI, so this
script is the convenient way to drive it.  It spills the KV cache across
VRAM -> CPU RAM -> SSD using the paper's rotate + Lloyd-Max codec (bit-packed to
2/3/4 bits) so long contexts don't OOM.

Examples
--------
  # Small model, paper codec, 3-bit KV cache spilled to SSD
  python run_offload.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --prompt "What is machine learning?" --bits 3

  # 30B model on a small GPU: offload BOTH weights and KV cache
  python run_offload.py --model Qwen/Qwen3-30B-A3B \
      --weights offload --max-gpu-mem 6GiB --weights-disk-dir D:/kv_weights \
      --offload-to-disk --offload-codec paper --bits 3 \
      --disk-dir D:/kv_cache --prompt "What is machine learning?"

  # Summarise a long document (chunked prefill keeps prefill VRAM small)
  python run_offload.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --file contract.txt --suffix "\n\nSummarise the key points:" \
      --offload-to-disk --bits 2 --prefill-chunk-size 256 --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate with KV-cache offload to SSD (paper Lloyd-Max codec by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- prompt ----
    p.add_argument("--prompt", default="What is machine learning?",
                   help="Prompt text (ignored if --file is given).")
    p.add_argument("--file", default=None,
                   help="Read the prompt from this file instead of --prompt.")
    p.add_argument("--suffix", default="",
                   help="Text appended after --file content (e.g. an instruction).")

    # ---- model ----
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model id.")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4],
                   help="KV precision / paper-codec pack width (2-4).")
    p.add_argument("--prefill-chunk-size", type=int, default=256,
                   help="Chunk long prompts so prefill attention stays small.")

    # ---- KV-cache offload (the point of this script) ----
    p.add_argument("--offload-to-disk", action="store_true", default=True,
                   help="Spill the KV cache VRAM->RAM->SSD.")
    p.add_argument("--no-offload-to-disk", dest="offload_to_disk", action="store_false",
                   help="Disable KV-cache offload (keep it in VRAM).")
    p.add_argument("--offload-codec", default="paper-outlier",
                   choices=["paper-outlier", "paper", "int8"],
                   help="'paper-outlier' = paper Section 5 outlier-aware Lloyd-Max "
                        "(best fidelity on real KV, default); "
                        "'paper' = plain Lloyd-Max, bit-packed (smallest); "
                        "'int8' = near-lossless per-vector uint8 (largest).")
    p.add_argument("--max-vram-tokens", type=int, default=512,
                   help="Token positions kept dequantized (hot) in VRAM.")
    p.add_argument("--warm-size", type=int, default=8,
                   help="Compressed layer entries kept in CPU RAM before spilling to SSD.")
    p.add_argument("--disk-dir", default=None,
                   help="SSD folder for spilled KV cache (auto-cleaned temp dir if unset).")

    # ---- optional weight offload (needed for big models on small GPUs) ----
    p.add_argument("--weights", default=None, choices=[None, "full", "4bit", "8bit", "offload"],
                   help="Weight placement: 4bit/8bit (bitsandbytes) or offload (GPU->RAM->SSD).")
    p.add_argument("--max-gpu-mem", default=None, help='VRAM cap for weights=offload, e.g. "6GiB".')
    p.add_argument("--max-cpu-mem", default=None, help='RAM cap for weights=offload, e.g. "12GiB".')
    p.add_argument("--weights-disk-dir", default=None, help="SSD folder for offloaded weight shards.")

    # ---- environment convenience ----
    p.add_argument("--hf-home", default=None,
                   help="Set HF_HOME to point at an existing model cache (e.g. D:/huggingface-models).")

    args = p.parse_args()

    # Point HuggingFace at an existing cache and quiet the Windows symlink warning.
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # UTF-8 console so model output with non-ASCII doesn't crash on Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Import after env vars are set so HF_HOME takes effect.
    from kvquant import generate

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            prompt = fh.read() + args.suffix
    else:
        prompt = args.prompt

    print(f"Model      : {args.model}")
    print(f"Weights    : {args.weights or 'full'}")
    print(f"KV offload : {args.offload_to_disk} "
          f"(codec={args.offload_codec}, bits={args.bits}, "
          f"warm_size={args.warm_size}, disk_dir={args.disk_dir or 'temp'})")
    print(f"Prompt     : {len(prompt)} chars")
    print("Generating ...\n")

    t0 = time.time()
    out = generate(
        prompt,
        model=args.model,
        bits=args.bits,
        max_new_tokens=args.max_new_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        # KV-cache offload
        offload_to_disk=args.offload_to_disk,
        offload_codec=args.offload_codec,
        max_vram_tokens=args.max_vram_tokens,
        warm_size=args.warm_size,
        disk_dir=args.disk_dir,
        # weight placement (only used when --weights is set)
        weights=args.weights,
        max_gpu_mem=args.max_gpu_mem,
        max_cpu_mem=args.max_cpu_mem,
        weights_disk_dir=args.weights_disk_dir,
    )
    dt = time.time() - t0

    print("=" * 70)
    print(out.text)
    print("=" * 70)
    print(f"\n{args.max_new_tokens} tokens in {dt:.1f}s  "
          f"({args.max_new_tokens / dt:.1f} tok/s)")
    ratio = getattr(out, "compression_ratio", None)
    if ratio:
        print(f"KV compression: {ratio:.1f}x  (avg {getattr(out, 'avg_bits_per_dim', args.bits):.2f} bits/dim)")


if __name__ == "__main__":
    main()
