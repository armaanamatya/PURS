"""
enrich_metadata.py
Populates row_data Q&A into metadata.json so the eval script needs no HuggingFace.

For each video entry with empty row_data:
  1. Tries to extract the YouTube ID from the MP4 file via ffprobe (yt-dlp embeds it)
  2. If found, searches the corresponding dataset for Q&A matching that video_id
  3. Normalises Q&A into a standard `questions` list

Standard question format stored per entry:
  "questions": [
    {
      "question": "...",
      "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A",
      "task_type": "..."
    }
  ]

Usage:
    pip install yt-dlp datasets huggingface_hub requests
    python enrich_metadata.py
    # optionally pass HF_TOKEN for OmniVideoBench
    HF_TOKEN=hf_xxx python enrich_metadata.py
"""

import json
import os
import subprocess
from pathlib import Path

META_FILE = Path("videos/metadata.json")
HF_TOKEN  = os.environ.get("HF_TOKEN")

# ── ffprobe: extract YouTube ID embedded by yt-dlp ───────────────────────────

def ffprobe_youtube_id(path: str) -> str | None:
    """yt-dlp embeds the YouTube URL in the 'comment' or 'description' tag."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        tags = data.get("format", {}).get("tags", {})
        for key in ("comment", "description", "COMMENT", "DESCRIPTION", "purl", "PURL"):
            val = tags.get(key, "")
            if "youtube.com/watch?v=" in val:
                return val.split("v=")[-1].split("&")[0][:11]
            if "youtu.be/" in val:
                return val.split("youtu.be/")[-1].split("?")[0][:11]
    except Exception:
        pass
    return None

# ── Dataset Q&A loaders ───────────────────────────────────────────────────────

def load_daily_omni():
    from datasets import load_dataset
    print("  Loading Daily-Omni...")
    ds = load_dataset("liarliar/Daily-Omni", split="train")
    # Build index: video_id -> list of QA rows
    idx = {}
    for row in ds:
        vid = row.get("video_id", "")
        if vid:
            idx.setdefault(vid, []).append(row)
    return idx

def load_videomme():
    from datasets import load_dataset
    print("  Loading Video-MME...")
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    idx = {}
    for row in ds:
        vid = row.get("videoID") or row.get("video_id", "")
        if vid:
            idx.setdefault(vid, []).append(row)
    return idx

def load_worldsense_qa() -> dict:
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    print("  Loading WorldSense QA JSON...")
    url = "https://huggingface.co/datasets/honglyhly/WorldSense/resolve/main/worldsense_qa.json"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

# ── Normalise Q&A into standard format ───────────────────────────────────────

def normalise_daily_omni(rows: list) -> list:
    out = []
    for row in rows:
        choices = row.get("Choice", [])
        out.append({
            "question":  row.get("Question", ""),
            "choices":   [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)],
            "answer":    row.get("Answer", ""),
            "task_type": row.get("Type", ""),
        })
    return out

def normalise_videomme(rows: list) -> list:
    out = []
    for row in rows:
        options = row.get("options", [])
        out.append({
            "question":  row.get("question", ""),
            "choices":   options if options and options[0].startswith("A") else [f"{chr(65+i)}. {c}" for i, c in enumerate(options)],
            "answer":    row.get("answer", ""),
            "task_type": row.get("task_type", ""),
        })
    return out

def normalise_worldsense(video_data: dict) -> list:
    out = []
    for key, task in video_data.items():
        if not key.startswith("task"):
            continue
        candidates = task.get("candidates", [])
        out.append({
            "question":  task.get("question", ""),
            "choices":   candidates if candidates and candidates[0].startswith("A") else [f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)],
            "answer":    task.get("answer", ""),
            "task_type": task.get("task_type", ""),
        })
    return out

# ── Enrich a single entry ─────────────────────────────────────────────────────

def enrich_entry(entry: dict, ds_indexes: dict) -> dict:
    """Adds a `questions` list to entry. Returns modified entry."""

    # Already has questions — skip
    if entry.get("questions"):
        return entry

    # WorldSense with all_tasks already stored
    if entry.get("row_data", {}).get("all_tasks"):
        entry["questions"] = normalise_worldsense(entry["row_data"]["all_tasks"])
        print(f"  {entry['category']}_{entry['index']}: WorldSense Q&A from existing row_data ({len(entry['questions'])} questions)")
        return entry

    source = entry.get("dataset_source", "")
    file_path = entry.get("file", "")

    # Try to get YouTube ID from file or existing row_data
    yt_id = (
        entry.get("row_data", {}).get("youtube_id")
        or entry.get("youtube_id")
        or ffprobe_youtube_id(file_path)
    )

    if source == "daily_omni":
        idx = ds_indexes.get("daily_omni")
        if idx is None:
            ds_indexes["daily_omni"] = load_daily_omni()
            idx = ds_indexes["daily_omni"]
        if yt_id and yt_id in idx:
            entry["questions"] = normalise_daily_omni(idx[yt_id])
            print(f"  {entry['category']}_{entry['index']}: Daily-Omni Q&A found for {yt_id} ({len(entry['questions'])} questions)")
        else:
            print(f"  {entry['category']}_{entry['index']}: Daily-Omni — YouTube ID not found (yt_id={yt_id})")

    elif source == "videomme":
        idx = ds_indexes.get("videomme")
        if idx is None:
            ds_indexes["videomme"] = load_videomme()
            idx = ds_indexes["videomme"]
        if yt_id and yt_id in idx:
            entry["questions"] = normalise_videomme(idx[yt_id])
            print(f"  {entry['category']}_{entry['index']}: Video-MME Q&A found for {yt_id} ({len(entry['questions'])} questions)")
        else:
            print(f"  {entry['category']}_{entry['index']}: Video-MME — YouTube ID not found (yt_id={yt_id})")

    elif source == "worldsense":
        ws_qa = ds_indexes.get("worldsense_qa")
        if ws_qa is None:
            ds_indexes["worldsense_qa"] = load_worldsense_qa()
            ws_qa = ds_indexes["worldsense_qa"]
        vid_id = entry.get("row_data", {}).get("video_id")
        if vid_id and vid_id in ws_qa:
            entry["questions"] = normalise_worldsense(ws_qa[vid_id])
            print(f"  {entry['category']}_{entry['index']}: WorldSense Q&A found for {vid_id} ({len(entry['questions'])} questions)")
        else:
            print(f"  {entry['category']}_{entry['index']}: WorldSense — video_id not found (vid_id={vid_id})")

    elif source == "yt_search":
        print(f"  {entry['category']}_{entry['index']}: yt_search — no dataset Q&A available, skipping")

    else:
        print(f"  {entry['category']}_{entry['index']}: unknown source '{source}', skipping")

    return entry

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    meta = json.loads(META_FILE.read_text())
    print(f"Loaded {len(meta)} entries from {META_FILE}\n")

    ds_indexes = {}  # lazily loaded dataset indexes

    for i, entry in enumerate(meta):
        meta[i] = enrich_entry(entry, ds_indexes)

    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved enriched metadata -> {META_FILE}")

    # Summary
    with_qa  = sum(1 for e in meta if e.get("questions"))
    total_qs = sum(len(e.get("questions", [])) for e in meta)
    print(f"Entries with Q&A: {with_qa}/{len(meta)}")
    print(f"Total questions:  {total_qs}")

if __name__ == "__main__":
    main()
