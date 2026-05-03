"""
2D attention heatmap visualization for OmniZip-compressed Qwen2.5-Omni.

This script is the OmniZip analogue of `viz_attention_heatmap_qwen.py`.
It:
  - captures OmniZip's `global_mask` (which tokens survive compression)
  - remaps audio/video token positions into the compressed index space
  - computes attention heatmaps on the compressed sequence
  - draws a magnified inset for the strongest intra-window audio<->video attention
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch  # type: ignore

import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore

OMNIZIP_DIR = os.path.join(os.path.dirname(__file__), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))
from qwen_omni_utils import process_mm_info  # type: ignore  # noqa: E402

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration  # type: ignore
from transformers import Qwen2_5OmniProcessor  # type: ignore

# OmniZip's forward-time monkeypatch needs omnizip_units module reference
import omnizip.omnizip_units as omnizip_units_module  # type: ignore


def parse_layers(raw: str) -> list[int]:
    if not raw.strip():
        raise ValueError("`--layers` cannot be empty")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _mean_attention_heads(attn: torch.Tensor) -> torch.Tensor:
    """
    attn: (num_heads, q_len, k_len)
    returns: (q_len, k_len)
    """
    return attn.mean(dim=0)


def _safe_mean(a: np.ndarray) -> float:
    if a.size == 0:
        return float("-inf")
    return float(a.mean())


def _choose_best_window(
    attn_crop: np.ndarray,
    audio_pos_rel: np.ndarray,
    video_pos_rel: np.ndarray,
    window_ids_abs: np.ndarray,
    abs_pos: np.ndarray,
) -> int:
    if audio_pos_rel.size == 0 or video_pos_rel.size == 0:
        return int(np.min(window_ids_abs)) if window_ids_abs.size > 0 else 0

    abs_set = set(int(x) for x in abs_pos.tolist())
    if not abs_set:
        return 0

    audio_abs = np.array([int(abs_pos[i]) for i in audio_pos_rel.tolist()], dtype=np.int64)
    video_abs = np.array([int(abs_pos[i]) for i in video_pos_rel.tolist()], dtype=np.int64)

    audio_w = window_ids_abs[audio_abs]
    video_w = window_ids_abs[video_abs]

    candidate_windows = sorted(set(audio_w.tolist()).intersection(set(video_w.tolist())))
    if not candidate_windows:
        candidate_windows = sorted(set(audio_w.tolist())) or [0]

    best_win = candidate_windows[0]
    best_score = float("-inf")

    for w in candidate_windows:
        audio_rel_w = np.array([i for i, wid in zip(audio_pos_rel.tolist(), audio_w.tolist()) if wid == w], dtype=int)
        video_rel_w = np.array([i for i, wid in zip(video_pos_rel.tolist(), video_w.tolist()) if wid == w], dtype=int)
        if audio_rel_w.size == 0 or video_rel_w.size == 0:
            continue

        a_to_v = attn_crop[np.ix_(audio_rel_w, video_rel_w)]
        v_to_a = attn_crop[np.ix_(video_rel_w, audio_rel_w)]
        score = 0.5 * (_safe_mean(a_to_v) + _safe_mean(v_to_a))
        if score > best_score:
            best_score = score
            best_win = int(w)

    return int(best_win)


def plot_layer_heatmap(
    ax: plt.Axes,
    attn_crop: np.ndarray,  # (q_len, k_len)
    audio_pos_rel: np.ndarray,  # positions in crop coordinates
    video_pos_rel: np.ndarray,
    window_ids_abs: np.ndarray,  # per absolute token position in the compressed seq
    crop_abs_pos: np.ndarray,  # mapping from crop index -> absolute token position
    layer_title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "magma",
    inset_frac: Tuple[float, float, float, float] = (0.52, 0.20, 0.42, 0.30),
) -> matplotlib.image.AxesImage:
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(attn_crop, origin="upper", aspect="auto", cmap=cmap, norm=norm)

    best_win = _choose_best_window(
        attn_crop=attn_crop,
        audio_pos_rel=audio_pos_rel,
        video_pos_rel=video_pos_rel,
        window_ids_abs=window_ids_abs,
        abs_pos=crop_abs_pos,
    )

    audio_abs = crop_abs_pos[audio_pos_rel]
    video_abs = crop_abs_pos[video_pos_rel]

    audio_rel_best = audio_pos_rel[window_ids_abs[audio_abs] == best_win]
    video_rel_best = video_pos_rel[window_ids_abs[video_abs] == best_win]

    window_rel = np.sort(np.concatenate([audio_rel_best, video_rel_best]))
    if window_rel.size == 0:
        window_rel = np.arange(min(128, attn_crop.shape[0]), dtype=int)

    q_idx = window_rel
    k_idx = window_rel
    inset = attn_crop[np.ix_(q_idx, k_idx)]

    y0 = int(q_idx.min())
    y1 = int(q_idx.max()) + 1
    x0 = int(k_idx.min())
    x1 = int(k_idx.max()) + 1
    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor="#e74c3c",
        linewidth=1.2,
        linestyle="--",
    )
    ax.add_patch(rect)

    iax = ax.inset_axes(inset_frac)
    iax.imshow(inset, origin="upper", aspect="auto", cmap=cmap, norm=norm)
    iax.set_xticks([])
    iax.set_yticks([])

    ax.set_title(layer_title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    return im


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OmniZip attention heatmap figure (Fig.2 style).")
    parser.add_argument("--video_path", type=str, default=os.path.join("videos", "worldsense", "attribute_reasoning", "video.mp4"))
    parser.add_argument("--question", type=str, default="What is the profession of the man with a beard wearing a suit in the video?")
    parser.add_argument("--model_path", type=str, default="/workspace/model")
    parser.add_argument("--layers", type=str, default="4,20")
    parser.add_argument("--pad", type=int, default=80)
    parser.add_argument("--vmin", type=float, default=1e-4)
    parser.add_argument("--vmax", type=float, default=1e-2)
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention_viz_omnizip_heatmap"))

    parser.add_argument("--rho_audio", type=float, default=0.4)
    parser.add_argument("--rho_video", type=float, default=0.7)
    parser.add_argument("--g", type=int, default=3)
    parser.add_argument("--contextual_ratio", type=float, default=0.05)

    args = parser.parse_args()

    layers = parse_layers(args.layers)
    os.makedirs(args.out_dir, exist_ok=True)

    # bypass CVE-2025-32434 (torch < 2.6), consistent with your other scripts
    import omnizip.modeling_qwen2_5_omni as _omni_mod  # type: ignore

    _omni_mod.check_torch_load_is_safe = lambda: None

    print("Loading OmniZip Qwen2.5-Omni with eager attention …")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    model.thinker.omnizip_config = {
        "rho_audio": args.rho_audio,
        "rho_video": args.rho_video,
        "g": args.g,
        "contextual_ratio": args.contextual_ratio,
    }

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "video", "video": args.video_path}, {"type": "text", "text": args.question}],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    num_input_frames = videos[0].shape[0]
    model.thinker.nframes = num_input_frames
    print(f"Video frames sampled: {num_input_frames}")

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to("cuda:0")

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(torch.float16)

    input_ids = inputs["input_ids"]  # (1, original_seq_len)
    ids_cpu = input_ids[0].cpu()
    orig_seq_len = int(input_ids.shape[1])

    audio_token_id = int(model.thinker.config.audio_token_id)
    video_token_id = int(model.thinker.config.video_token_id)

    audio_orig_pos = (ids_cpu == audio_token_id).nonzero(as_tuple=True)[0].numpy()
    video_orig_pos = (ids_cpu == video_token_id).nonzero(as_tuple=True)[0].numpy()
    if audio_orig_pos.size == 0 or video_orig_pos.size == 0:
        raise ValueError(
            f"Expected both audio and video tokens. Got audio={audio_orig_pos.size}, video={video_orig_pos.size}."
        )

    # Compute temporal position ids for the ORIGINAL sequence; later we subset using global_mask.
    position_ids_orig, _ = model.thinker.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
        use_audio_in_video=True,
        audio_seqlens=inputs.get("audio_feature_lengths"),
        second_per_grids=inputs.get("video_second_per_grid"),
    )
    temporal_pos_orig = position_ids_orig[0, 0].detach().cpu().numpy()  # (orig_seq_len,)

    # Monkey-patch omnizip() to capture global_mask
    original_omnizip = omnizip_units_module.omnizip
    captured_mask: list[torch.Tensor] = [None]  # will hold (orig_seq_len,) bool tensor

    def _patched_omnizip(*p_args, **p_kwargs):
        result = original_omnizip(*p_args, **p_kwargs)
        captured_mask[0] = result[1].cpu()
        return result

    omnizip_units_module.omnizip = _patched_omnizip

    THINKER_KEYS = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        "video_second_per_grid",
        "inputs_embeds",
        "position_ids",
        "past_key_values",
        "rope_deltas",
        "use_cache",
        "cache_position",
        "labels",
    }
    thinker_inputs = {k: v for k, v in inputs.items() if k in THINKER_KEYS}

    print("Running prefill forward pass (OmniZip + eager attn – may take a while) …")
    with torch.no_grad():
        output = model.thinker(
            **thinker_inputs,
            output_attentions=True,
            use_audio_in_video=True,
            use_cache=False,
            return_dict=True,
        )

    omnizip_units_module.omnizip = original_omnizip
    global_mask = captured_mask[0]
    if global_mask is None:
        raise RuntimeError("Failed to capture OmniZip global_mask via monkey-patch.")

    compressed_len = int(global_mask.sum().item())
    print(f"Compressed sequence: {compressed_len} tokens (dropped {orig_seq_len - compressed_len}).")

    # Remap original token indices -> compressed token indices
    orig_to_comp = torch.full((orig_seq_len,), -1, dtype=torch.long)
    comp_idx = 0
    for orig_i in range(orig_seq_len):
        if bool(global_mask[orig_i].item()):
            orig_to_comp[orig_i] = comp_idx
            comp_idx += 1

    audio_comp_pos = orig_to_comp[torch.tensor(audio_orig_pos, dtype=torch.long)].numpy()
    video_comp_pos = orig_to_comp[torch.tensor(video_orig_pos, dtype=torch.long)].numpy()
    audio_comp_pos = audio_comp_pos[audio_comp_pos >= 0]
    video_comp_pos = video_comp_pos[video_comp_pos >= 0]

    temporal_pos_comp = temporal_pos_orig[global_mask.numpy()]

    # Compute time windows in the compressed index space
    seconds_per_chunk = float(model.thinker.config.seconds_per_chunk)
    position_id_per_seconds = float(model.thinker.config.position_id_per_seconds)
    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
    if t_ntoken_per_chunk <= 0:
        raise ValueError(f"Computed invalid t_ntoken_per_chunk={t_ntoken_per_chunk}.")

    modal_comp_pos = np.sort(np.concatenate([audio_comp_pos, video_comp_pos]))
    q_start = max(0, int(modal_comp_pos.min()) - args.pad)
    q_end = min(compressed_len, int(modal_comp_pos.max()) + args.pad + 1)

    ref = float(np.min(temporal_pos_comp[modal_comp_pos]))
    window_ids_comp = ((temporal_pos_comp - ref) // t_ntoken_per_chunk).astype(np.int64)

    crop_abs_pos = np.arange(q_start, q_end, dtype=int)
    audio_pos_rel = audio_comp_pos[(audio_comp_pos >= q_start) & (audio_comp_pos < q_end)] - q_start
    video_pos_rel = video_comp_pos[(video_comp_pos >= q_start) & (video_comp_pos < q_end)] - q_start

    # Plot one panel per requested layer
    fig, axes = plt.subplots(1, len(layers), figsize=(6.3 * len(layers), 5.2))
    if len(layers) == 1:
        axes = [axes]

    for ax, li in zip(axes, layers):
        attn = output.attentions[li]  # (1, heads, seq_comp, seq_comp)
        attn_crop_t = attn[0, :, q_start:q_end, q_start:q_end]  # (heads, q, k)
        attn_crop = _mean_attention_heads(attn_crop_t).float().cpu().numpy()

        layer_title = f"OmniZip Qwen2.5-Omni Layer {li}"
        im = plot_layer_heatmap(
            ax=ax,
            attn_crop=attn_crop,
            audio_pos_rel=audio_pos_rel,
            video_pos_rel=video_pos_rel,
            window_ids_abs=window_ids_comp,
            crop_abs_pos=crop_abs_pos,
            layer_title=layer_title,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.12)
        cbar.set_ticks([args.vmin, args.vmin * 10, args.vmin * 100] if args.vmax >= args.vmin * 100 else [args.vmin, args.vmax])

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "attention_heatmap_omnizip_layers.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

