"""validate_l6_cache.py

Validate that the existing L6 saliency cache (produced by precompute_l6_saliency.py)
contains usable VIDEO scores for VideoZip. Runs on CPU only — no model load — so it's
cheap to run on the remote instance to verify the cache before kicking off any eval.

Why this is the right next step (not a new precompute script):
    The existing audio-side precompute already dumps video saliency in the same JSONL.
    See `precompute_l6_saliency.py` lines ~276 and ~351:
        result["video"] = _cross(video_pos_gpu)
        ...
        "raw_video_scores": {str(args.layer): [video_arr.tolist()] ...}
    No re-precompute is needed. We just need to confirm the JSONL is populated for every
    video we plan to evaluate on, then point VideoZip at it.

Usage on remote:
    python videozip/precompute/validate_l6_cache.py \\
        --cache vizzing/layer_depth_all_full.jsonl \\
        --layer 6 \\
        --metadata /data/armaan/purs/videos/metadata.json

Exit codes:
    0 = cache is fully populated for the metadata
    1 = cache is missing >0 videos referenced in metadata
    2 = cache file not found / unreadable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from videozip.cache.loader import cache_summary, load_video_scores, _normalize_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="vizzing/layer_depth_all_full.jsonl",
        help="Path to the L6 saliency JSONL cache.",
    )
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata.json. If given, we cross-check that every entry with "
             "questions has a video score in the cache.",
    )
    parser.add_argument(
        "--dataset", default=None, help="Restrict the metadata cross-check to one dataset."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-record listings."
    )
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        print(f"FATAL: cache file not found: {args.cache}", file=sys.stderr)
        return 2

    summary = cache_summary(args.cache, layer=args.layer)
    print(f"\nCache: {summary['cache_path']}")
    print(f"Layer: {summary['layer']}")
    print(f"Total records: {summary['n_total']}\n")

    print(f"{'Dataset':<20} {'Records':>8} {'WithAudio':>10} {'WithVideo':>10} "
          f"{'AudioMin':>9} {'AudioMax':>9} {'VideoMin':>9} {'VideoMax':>9}")
    print("-" * 96)
    for ds, slot in sorted(summary["by_dataset"].items()):
        a_lens = slot["audio_token_lens"]
        v_lens = slot["video_token_lens"]
        a_min = min(a_lens) if a_lens else 0
        a_max = max(a_lens) if a_lens else 0
        v_min = min(v_lens) if v_lens else 0
        v_max = max(v_lens) if v_lens else 0
        print(f"{ds:<20} {slot['n_records']:>8} {slot['n_with_audio']:>10} "
              f"{slot['n_with_video']:>10} {a_min:>9} {a_max:>9} {v_min:>9} {v_max:>9}")

    video_scores = load_video_scores(args.cache, layer=args.layer, key="file")
    print(f"\nVideo scores loaded: {len(video_scores)} records")

    if not video_scores:
        print("FATAL: no video scores in cache. Re-run precompute_l6_saliency.py.",
              file=sys.stderr)
        return 1

    if args.metadata is None:
        print("\nDone (no metadata cross-check requested).")
        return 0

    if not os.path.exists(args.metadata):
        print(f"WARN: metadata not found: {args.metadata}", file=sys.stderr)
        return 0

    with open(args.metadata, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = list(raw.values()) if isinstance(raw, dict) else raw
    if args.dataset:
        canon = args.dataset.strip().lower().replace("_", "-").replace(" ", "-")
        entries = [
            e for e in entries
            if e.get("dataset", "").strip().lower().replace("_", "-") == canon
        ]
    relevant = [e for e in entries if e.get("questions")]

    missing = []
    present = 0
    for e in relevant:
        k = _normalize_path(e["file"])
        if k in video_scores:
            present += 1
        else:
            missing.append(e["file"])

    print(f"\nMetadata cross-check ({args.dataset or 'all datasets'}):")
    print(f"  Relevant entries (with questions): {len(relevant)}")
    print(f"  Present in cache:                  {present}")
    print(f"  Missing from cache:                {len(missing)}")

    if missing and not args.quiet:
        print("\nMissing files (first 20):")
        for f in missing[:20]:
            print(f"  {f}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
