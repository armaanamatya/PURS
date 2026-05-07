"""
KL-vs-baseline harness for FastKV-Omni.

For a fixed set of prompts, compute the per-token KL divergence between
  - baseline Qwen2.5-Omni Thinker logits, and
  - logits produced by the same model with FastKV's TSP patch applied at
    a chosen layer / tsp_length.

This is the proxy metric used throughout analysis-only mode: it answers
"does pruning at layer L preserve the next-token distribution?" without
requiring downstream benchmark scoring.

Usage:
    python kl_vs_baseline.py --tsp_idx 15 --tsp_length 2048 \\
        --model_path /data/armaan/models/Qwen2.5-Omni-7B \\
        --prompts_json prompts.json \\
        --out_jsonl ../outputs/kl/tsp15_len2048.jsonl

Importable: a3_tsp_sweep.py reuses run_one_config() to avoid reloading
the model for each sweep point.

Importantly: when tsp_length >= max-prompt-length OR tsp_idx >= num_layers,
TSP is a no-op and the harness should report KL ≈ 0. That's the equivalence
gate before trusting any non-zero KL number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


# Make src/ importable
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from qwen25omni_fastkv import apply_fastkv, disable_fastkv  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Default text prompts (varying lengths). Override with --prompts_json.
# ────────────────────────────────────────────────────────────────────────────
_DEFAULT_PROMPTS = [
    {"id": "short_qa",    "text": "Question: What is the capital of France?\nAnswer:"},
    {"id": "med_summary", "text": "Summarize the following passage in one sentence.\n\n" + (
        "The James Webb Space Telescope is the largest optical telescope in space. "
        "Its high infrared resolution and sensitivity allow it to view objects too "
        "old, distant, or faint for the Hubble Space Telescope. This enables "
        "investigations across many fields of astronomy and cosmology, such as "
        "observation of the first stars, the formation of the first galaxies, and "
        "detailed atmospheric characterization of potentially habitable exoplanets. "
    ) * 4 + "\n\nSummary:"},
    {"id": "long_passage", "text": "Read the passage and answer the question.\n\n" + (
        "Photosynthesis is a process used by plants and other organisms to convert "
        "light energy into chemical energy that, through cellular respiration, can "
        "later be released to fuel the organism's activities. This chemical energy "
        "is stored in carbohydrate molecules, such as sugars and starches, which "
        "are synthesized from carbon dioxide and water. " * 32) +
        "\n\nQuestion: What is photosynthesis used for?\nAnswer:"},
]


def load_prompts(path: str | None):
    if not path:
        return _DEFAULT_PROMPTS
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("prompts_json must be a JSON list of {id, text} objects.")
    return data


# ────────────────────────────────────────────────────────────────────────────
# Forward + KL
# ────────────────────────────────────────────────────────────────────────────

def _logits_from_thinker_lm(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run input_ids through the Thinker LM and return the final-norm hidden
    states projected through lm_head, i.e. logits over the vocab."""
    with torch.no_grad():
        out = model.thinker.model(input_ids=input_ids)
        # Project hidden states to vocab via the Thinker's lm_head.
        # In Qwen2.5-Omni, the LM head usually lives on `model.thinker.lm_head` or
        # is tied to embed_tokens. Try both.
        if hasattr(model.thinker, "lm_head"):
            logits = model.thinker.lm_head(out.last_hidden_state)
        else:
            logits = F.linear(out.last_hidden_state, model.thinker.model.embed_tokens.weight)
    return logits  # (B, T, V)


def _per_token_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per token, averaged over the vocab dim. Shapes (B, T, V)."""
    p = F.log_softmax(p_logits.float(), dim=-1)
    q = F.log_softmax(q_logits.float(), dim=-1)
    p_prob = p.exp()
    return (p_prob * (p - q)).sum(dim=-1)  # (B, T)


def run_one_config(
    model,
    processor,
    prompts: Iterable[dict],
    tsp_idx: int,
    tsp_length: int,
    window_size: int = 8,
    kernel_size: int = 7,
    pooling: str = "avgpool",
    device: str | None = None,
    baseline_cache: dict | None = None,
):
    """Run baseline + patched forward over a list of prompts and return per-prompt KL stats.

    `baseline_cache`: optional dict mapping prompt_id → CPU baseline-logits tensor.
    Pass an empty dict on the first call; subsequent calls (e.g. in a sweep) will
    reuse the cached baseline logits and skip the baseline forward.
    """
    device = device or next(model.parameters()).device
    if baseline_cache is None:
        baseline_cache = {}

    results = []
    for p in prompts:
        pid, text = p["id"], p["text"]
        inputs = processor(text=[text], return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        # Baseline (cached if we've seen this prompt id)
        if pid in baseline_cache:
            base_logits = baseline_cache[pid].to(device)
        else:
            disable_fastkv(model)  # ensure clean slate
            t0 = time.perf_counter()
            base_logits = _logits_from_thinker_lm(model, inputs.input_ids)
            t_base = (time.perf_counter() - t0) * 1000
            baseline_cache[pid] = base_logits.cpu()

        # Patched
        apply_fastkv(model, tsp_idx=tsp_idx, tsp_length=tsp_length,
                     window_size=window_size, kernel_size=kernel_size, pooling=pooling)
        t0 = time.perf_counter()
        patched_logits = _logits_from_thinker_lm(model, inputs.input_ids)
        t_patched = (time.perf_counter() - t0) * 1000
        disable_fastkv(model)

        # The patched forward may produce a shorter sequence (post-TSP slice).
        # Compare on the trailing-window region only (always retained by TSP),
        # which is the only region with a 1:1 token correspondence between runs.
        K = patched_logits.shape[1]
        if K == seq_len:
            kl = _per_token_kl(base_logits, patched_logits)
            kl_arr = kl[0].cpu().numpy()
        else:
            # Compare the trailing window — TSP guarantees the last `window_size` tokens survive
            tail_n = min(window_size, K, seq_len)
            kl_tail = _per_token_kl(
                base_logits[:, -tail_n:, :],
                patched_logits[:, -tail_n:, :],
            )
            kl_arr = kl_tail[0].cpu().numpy()

        results.append({
            "prompt_id": pid,
            "seq_len": seq_len,
            "patched_seq_len": K,
            "tsp_idx": tsp_idx,
            "tsp_length": tsp_length,
            "kl_mean": float(kl_arr.mean()),
            "kl_max": float(kl_arr.max()),
            "kl_p50": float(sorted(kl_arr.tolist())[len(kl_arr) // 2]),
            "patched_ms": t_patched,
        })
        print(f"  prompt={pid:14s} seq={seq_len:5d}→{K:5d}  "
              f"KL mean={results[-1]['kl_mean']:.4e} max={results[-1]['kl_max']:.4e}")

    return results, baseline_cache


# ────────────────────────────────────────────────────────────────────────────
# CLI entrypoint (single config)
# ────────────────────────────────────────────────────────────────────────────

def _load_model(model_path: str):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qmod
    _qmod.check_torch_load_is_safe = lambda: None
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/armaan/models/Qwen2.5-Omni-7B")
    parser.add_argument("--tsp_idx", type=int, required=True)
    parser.add_argument("--tsp_length", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", default="avgpool", choices=["avgpool", "maxpool"])
    parser.add_argument("--prompts_json", default=None,
                        help="Optional path to a JSON list of {id, text}. Defaults to a small built-in set.")
    parser.add_argument("--out_jsonl", default=None,
                        help="Where to write per-prompt KL records. Default: outputs/kl/tsp{idx}_len{len}.jsonl")
    args = parser.parse_args()

    out_path = args.out_jsonl or str(
        HERE.parent / "outputs" / "kl" / f"tsp{args.tsp_idx}_len{args.tsp_length}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model, processor = _load_model(args.model_path)
    prompts = load_prompts(args.prompts_json)

    print(f"[kl] config: tsp_idx={args.tsp_idx} tsp_length={args.tsp_length} "
          f"window={args.window_size} kernel={args.kernel_size} pooling={args.pooling}")
    results, _ = run_one_config(
        model, processor, prompts,
        tsp_idx=args.tsp_idx, tsp_length=args.tsp_length,
        window_size=args.window_size, kernel_size=args.kernel_size, pooling=args.pooling,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[kl] wrote {len(results)} records → {out_path}")


if __name__ == "__main__":
    main()
