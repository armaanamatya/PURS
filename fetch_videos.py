"""
fetch_videos.py
Downloads up to MAX_PER_CATEGORY videos per question-type/task-type category from 4 benchmarks.

Structure: videos/{dataset}/{category_slug}/video.mp4  (first video, legacy name)
                                             video_1.mp4, video_2.mp4, ... (additional)
Metadata:  videos/metadata.json  — Q&A embedded directly, no separate enrich step.

Datasets:
  video-mme       — 12 task_types, YouTube, duration="short" filter
  daily-omni      — 6 Types, YouTube, video_duration field filter
  worldsense      — 26 task_types (auto-discovered), HF zip range-requests
  omnivideobench  — 13 question_types, HF direct download (gated — needs HF_TOKEN)

Usage:
    pip install yt-dlp datasets huggingface_hub requests pyarrow pandas
    HF_TOKEN=hf_xxx python fetch_videos.py   # all datasets
    python fetch_videos.py                   # skips OmniVideoBench
"""

import io
import json
import os
import re
import struct
import subprocess
import zlib
from collections import Counter
from pathlib import Path

import requests
import yt_dlp

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("WARNING: pandas/pyarrow not installed — OmniVideoBench disabled")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: `datasets` not installed — Video-MME, Daily-Omni disabled")

# ── Config ────────────────────────────────────────────────────────────────────

MAX_DURATION     = 90   # seconds; videos longer than this are skipped
MAX_PER_CATEGORY = 20   # target number of videos per task/question category
OUT_DIR          = Path("videos")
META_FILE        = OUT_DIR / "metadata.json"
HF_TOKEN         = os.environ.get("HF_TOKEN")

HF_HEADERS = {"User-Agent": "Mozilla/5.0"}
if HF_TOKEN:
    HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def parse_duration(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if re.match(r"^\d+:\d{2}(:\d{2})?$", s):
        parts = list(map(int, s.split(":")))
        return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
    m = re.match(r"^(\d+(?:\.\d+)?)\s*s?$", s, re.IGNORECASE)
    return float(m.group(1)) if m else None


def ffprobe_duration(path: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def post_check(path: Path) -> bool:
    dur = ffprobe_duration(path)
    if dur is None:
        print(f"    [post-check] Could not verify {path.name} — keeping")
        return True
    if dur > MAX_DURATION:
        print(f"    [post-check] FAIL {path.name}: {dur:.1f}s > {MAX_DURATION}s — deleting")
        path.unlink(missing_ok=True)
        return False
    print(f"    [post-check] OK {path.name}: {dur:.1f}s")
    return True


def ensure_prefixed(choices: list) -> list:
    if choices and isinstance(choices[0], str) and len(choices[0]) >= 2 and choices[0][1] in (". ", "."):
        return choices
    return [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]


def make_question(question: str, choices: list, answer: str, task_type: str) -> dict:
    return {
        "question":  question,
        "choices":   ensure_prefixed(choices),
        "answer":    answer.strip().upper(),
        "task_type": task_type,
    }


def next_video_path(out_dir: Path) -> Path:
    """Path for the next video file to write (video.mp4 first, then video_N.mp4)."""
    count = len(sorted(out_dir.glob("video*.mp4")))
    return out_dir / "video.mp4" if count == 0 else out_dir / f"video_{count}.mp4"

# ── YouTube downloader ────────────────────────────────────────────────────────

def _match_filter(info, *, incomplete):
    dur = info.get("duration")
    if dur and dur > MAX_DURATION:
        return f"Duration {dur}s > {MAX_DURATION}s"
    return None


def download_youtube(video_id: str, out_path: Path) -> bool:
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "match_filter": _match_filter,
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
        "ignoreerrors": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return False
            dur = info.get("duration")
            if dur and dur > MAX_DURATION:
                print(f"    Skip {video_id}: {dur}s > {MAX_DURATION}s")
                return False
            ydl.download([url])
        return out_path.exists()
    except Exception as e:
        print(f"    yt-dlp error for {video_id}: {e}")
        return False

# ── HuggingFace downloader ────────────────────────────────────────────────────

def hf_get(url: str, stream=False) -> requests.Response:
    r = requests.get(url, headers=HF_HEADERS, stream=stream, timeout=120)
    r.raise_for_status()
    return r


def download_hf_video(url: str, out_path: Path) -> bool:
    try:
        r = hf_get(url, stream=True)
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    HF download error: {e}")
        return False

# ── WorldSense ZIP range-request extractor ────────────────────────────────────

WS_REPO  = "honglyhly/WorldSense"
WS_ZIPS  = [f"worldsense_videos_{i}.zip" for i in range(11)]
_ws_index: dict = {}


def _ws_url(fname):
    return f"https://huggingface.co/datasets/{WS_REPO}/resolve/main/{fname}"


def _range_get(url, start, end):
    r = requests.get(url, headers={**HF_HEADERS, "Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    return r.content


def _build_ws_index():
    global _ws_index
    if _ws_index:
        return
    print("  Building WorldSense zip index (one-time scan)...")
    for zip_name in WS_ZIPS:
        url = _ws_url(zip_name)
        try:
            r = requests.head(url, headers=HF_HEADERS, timeout=15, allow_redirects=True)
            size = int(r.headers["Content-Length"])
            tail_size = min(65536 + 22, size)
            tail = _range_get(url, size - tail_size, size - 1)
            eocd_idx = tail.rfind(b"PK\x05\x06")
            if eocd_idx == -1:
                continue
            eocd = tail[eocd_idx:]
            _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from("<4sHHHHIIH", eocd)
            cd_data = _range_get(url, cd_offset, cd_offset + cd_size - 1)
            pos = 0
            while pos < len(cd_data):
                if cd_data[pos:pos+4] != b"PK\x01\x02":
                    break
                (_, _, _, _, comp, _, _, _, comp_size, uncomp_size,
                 fn_len, ex_len, cm_len, _, _, _, lh_offset) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd_data, pos)
                pos += 46
                fname = cd_data[pos:pos+fn_len].decode("utf-8", errors="replace")
                pos += fn_len + ex_len + cm_len
                _ws_index[fname] = (zip_name, lh_offset, comp_size, uncomp_size, comp)
            print(f"    Indexed {zip_name} ({len(_ws_index)} files so far)")
        except Exception as e:
            print(f"    Could not index {zip_name}: {e}")


def download_worldsense_video(video_id: str, out_path: Path) -> bool:
    _build_ws_index()
    fname = f"{video_id}.mp4"
    if fname not in _ws_index:
        print(f"    {fname} not found in WorldSense zips")
        return False
    zip_name, lh_offset, comp_size, uncomp_size, compression = _ws_index[fname]
    print(f"    Extracting {fname} from {zip_name} ({uncomp_size/1e6:.1f} MB)...")
    try:
        url = _ws_url(zip_name)
        hdr = _range_get(url, lh_offset, lh_offset + 30)
        _, _, _, _, _, _, _, _, _, fn_len, ex_len = struct.unpack_from("<4sHHHHHIIIHH", hdr)
        data_start = lh_offset + 30 + fn_len + ex_len
        compressed = _range_get(url, data_start, data_start + comp_size - 1)
        data = compressed if compression == 0 else zlib.decompress(compressed, -15)
        out_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"    WorldSense extraction error: {e}")
        return False

# ── Dataset caches ────────────────────────────────────────────────────────────

_cache: dict = {}


def _load_hf(key, *args, **kwargs):
    if key not in _cache:
        if not HAS_DATASETS:
            _cache[key] = None
        else:
            try:
                print(f"  Loading {key} from HuggingFace...")
                _cache[key] = load_dataset(*args, **kwargs)
            except Exception as e:
                print(f"  Could not load {key}: {e}")
                _cache[key] = None
    return _cache[key]

# ── 1. Video-MME ──────────────────────────────────────────────────────────────

VIDEOMME_DATASET = "video-mme"

def fetch_videomme(existing: list[dict]) -> list[dict]:
    """Up to MAX_PER_CATEGORY videos per task_type, duration=short only.

    existing: already-complete result dicts from metadata.json for this dataset.
    Returns existing entries (unchanged) + any newly downloaded entries.
    """
    ds = _load_hf("videomme", "lmms-lab/Video-MME", split="test")
    if ds is None:
        return existing

    by_type: dict[str, list] = {}
    for row in ds:
        if str(row.get("duration", "")).lower() != "short":
            continue
        tt = row.get("task_type", "").strip()
        if tt:
            by_type.setdefault(tt, []).append(row)

    print(f"  Video-MME task_types ({len(by_type)}): {sorted(by_type)}")

    # Count existing entries per category (only those whose file still exists)
    existing_by_type: dict[str, list] = {}
    for r in existing:
        if Path(r["file"]).exists() and r.get("questions"):
            existing_by_type.setdefault(r["task_type"], []).append(r)

    results = list(existing)  # preserve all existing entries
    done_videos: set[str] = {r.get("youtube_id", "") for r in existing if r.get("youtube_id")}

    for task_type, rows in sorted(by_type.items()):
        slug = slugify(task_type)
        out_dir = OUT_DIR / VIDEOMME_DATASET / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        already = len(existing_by_type.get(task_type, []))
        if already >= MAX_PER_CATEGORY:
            print(f"  [{VIDEOMME_DATASET}] {task_type}: {already}/{MAX_PER_CATEGORY} — complete")
            continue

        new_count = 0
        for row in rows:
            if already + new_count >= MAX_PER_CATEGORY:
                break
            vid_id = row.get("videoID") or row.get("video_id", "")
            if not vid_id or vid_id in done_videos:
                continue

            out_path = next_video_path(out_dir)
            print(f"  [{VIDEOMME_DATASET}/{slug}] trying {vid_id} (#{already + new_count + 1}/{MAX_PER_CATEGORY})...")
            if not download_youtube(vid_id, out_path):
                continue
            if not post_check(out_path):
                continue

            dur = ffprobe_duration(out_path)
            questions = [
                make_question(r["question"], r.get("options", []), r["answer"], r["task_type"])
                for r in rows if (r.get("videoID") or r.get("video_id")) == vid_id
            ]
            results.append({
                "dataset":    VIDEOMME_DATASET,
                "task_type":  task_type,
                "file":       str(out_path),
                "duration_s": round(dur, 2) if dur else None,
                "youtube_id": vid_id,
                "questions":  questions,
            })
            done_videos.add(vid_id)
            new_count += 1

        total = already + new_count
        if new_count == 0 and already < MAX_PER_CATEGORY:
            print(f"  [{VIDEOMME_DATASET}] {task_type}: no additional videos found (have {already}/{MAX_PER_CATEGORY})")
        elif new_count > 0:
            print(f"  [{VIDEOMME_DATASET}] {task_type}: {total}/{MAX_PER_CATEGORY} total (+{new_count} new)")

    return results

# ── 2. Daily-Omni ─────────────────────────────────────────────────────────────

DAILY_OMNI_DATASET = "daily-omni"

def fetch_daily_omni(existing: list[dict]) -> list[dict]:
    """Up to MAX_PER_CATEGORY videos per Type, video_duration <= MAX_DURATION."""
    ds = _load_hf("daily_omni", "liarliar/Daily-Omni", split="train")
    if ds is None:
        return existing

    by_type: dict[str, list] = {}
    for row in ds:
        raw_dur = parse_duration(row.get("video_duration"))
        if raw_dur is not None and raw_dur > MAX_DURATION:
            continue
        tt = row.get("Type", "").strip()
        if tt:
            by_type.setdefault(tt, []).append(row)

    print(f"  Daily-Omni Types ({len(by_type)}): {sorted(by_type)}")

    existing_by_type: dict[str, list] = {}
    for r in existing:
        if Path(r["file"]).exists() and r.get("questions"):
            existing_by_type.setdefault(r["task_type"], []).append(r)

    results = list(existing)
    done_videos: set[str] = {r.get("youtube_id", "") for r in existing if r.get("youtube_id")}

    for q_type, rows in sorted(by_type.items()):
        slug = slugify(q_type)
        out_dir = OUT_DIR / DAILY_OMNI_DATASET / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        already = len(existing_by_type.get(q_type, []))
        if already >= MAX_PER_CATEGORY:
            print(f"  [{DAILY_OMNI_DATASET}] {q_type}: {already}/{MAX_PER_CATEGORY} — complete")
            continue

        new_count = 0
        for row in rows:
            if already + new_count >= MAX_PER_CATEGORY:
                break
            vid_id = row.get("video_id", "")
            if not vid_id or vid_id in done_videos:
                continue

            out_path = next_video_path(out_dir)
            print(f"  [{DAILY_OMNI_DATASET}/{slug}] trying {vid_id} [{row.get('video_duration')}] (#{already + new_count + 1}/{MAX_PER_CATEGORY})...")
            if not download_youtube(vid_id, out_path):
                continue
            if not post_check(out_path):
                continue

            dur = ffprobe_duration(out_path)
            questions = [
                make_question(r["Question"], r.get("Choice", []), r["Answer"], r["Type"])
                for r in rows if r.get("video_id") == vid_id
            ]
            results.append({
                "dataset":    DAILY_OMNI_DATASET,
                "task_type":  q_type,
                "file":       str(out_path),
                "duration_s": round(dur, 2) if dur else None,
                "youtube_id": vid_id,
                "questions":  questions,
            })
            done_videos.add(vid_id)
            new_count += 1

        total = already + new_count
        if new_count == 0 and already < MAX_PER_CATEGORY:
            print(f"  [{DAILY_OMNI_DATASET}] {q_type}: no additional videos found (have {already}/{MAX_PER_CATEGORY})")
        elif new_count > 0:
            print(f"  [{DAILY_OMNI_DATASET}] {q_type}: {total}/{MAX_PER_CATEGORY} total (+{new_count} new)")

    return results

# ── 3. WorldSense ─────────────────────────────────────────────────────────────

WORLDSENSE_DATASET = "worldsense"

def fetch_worldsense(existing: list[dict]) -> list[dict]:
    """Up to MAX_PER_CATEGORY videos per task_type (auto-discovered from QA JSON)."""
    url = f"https://huggingface.co/datasets/{WS_REPO}/resolve/main/worldsense_qa.json"
    try:
        print("  Loading WorldSense QA JSON...")
        r = hf_get(url)
        qa: dict = r.json()
    except Exception as e:
        print(f"  Could not load WorldSense QA: {e}")
        return existing

    by_type: dict[str, list[tuple]] = {}
    for video_id, video_data in qa.items():
        for key, task in video_data.items():
            if not key.startswith("task"):
                continue
            tt = task.get("task_type", "").strip()
            if tt:
                by_type.setdefault(tt, []).append((video_id, task, video_data))

    print(f"  WorldSense task_types ({len(by_type)}): {sorted(by_type)}")

    existing_by_type: dict[str, list] = {}
    for r in existing:
        if Path(r["file"]).exists() and r.get("questions"):
            existing_by_type.setdefault(r["task_type"], []).append(r)

    results = list(existing)
    done_videos: set[str] = {r.get("video_id", "") for r in existing if r.get("video_id")}

    for task_type, entries in sorted(by_type.items()):
        slug = slugify(task_type)
        out_dir = OUT_DIR / WORLDSENSE_DATASET / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        already = len(existing_by_type.get(task_type, []))
        if already >= MAX_PER_CATEGORY:
            print(f"  [{WORLDSENSE_DATASET}] {task_type}: {already}/{MAX_PER_CATEGORY} — complete")
            continue

        new_count = 0
        for video_id, task, video_data in entries:
            if already + new_count >= MAX_PER_CATEGORY:
                break
            if video_id in done_videos:
                continue

            out_path = next_video_path(out_dir)
            print(f"  [{WORLDSENSE_DATASET}/{slug}] trying {video_id} (#{already + new_count + 1}/{MAX_PER_CATEGORY})...")
            if not download_worldsense_video(video_id, out_path):
                continue
            if not post_check(out_path):
                continue

            dur = ffprobe_duration(out_path)
            questions = [
                make_question(t.get("question", ""), t.get("candidates", []), t.get("answer", ""), t.get("task_type", ""))
                for k, t in video_data.items() if k.startswith("task")
            ]
            results.append({
                "dataset":       WORLDSENSE_DATASET,
                "task_type":     task_type,
                "file":          str(out_path),
                "duration_s":    round(dur, 2) if dur else None,
                "video_id":      video_id,
                "domain":        video_data.get("domain"),
                "video_caption": video_data.get("video_caption"),
                "questions":     questions,
            })
            done_videos.add(video_id)
            new_count += 1

        total = already + new_count
        if new_count == 0 and already < MAX_PER_CATEGORY:
            print(f"  [{WORLDSENSE_DATASET}] {task_type}: no additional videos found (have {already}/{MAX_PER_CATEGORY})")
        elif new_count > 0:
            print(f"  [{WORLDSENSE_DATASET}] {task_type}: {total}/{MAX_PER_CATEGORY} total (+{new_count} new)")

    return results

# ── 4. OmniVideoBench ─────────────────────────────────────────────────────────

OMNIVIDEO_DATASET = "omnivideobench"

def fetch_omnivideobench(existing: list[dict]) -> list[dict]:
    """Up to MAX_PER_CATEGORY videos per question_type. Gated — requires HF_TOKEN."""
    if not HF_TOKEN:
        print("  OmniVideoBench: HF_TOKEN not set — skipping")
        return existing
    if not HAS_PARQUET:
        print("  OmniVideoBench: pyarrow/pandas not installed — skipping")
        return existing

    url = "https://huggingface.co/datasets/NJU-LINK/OmniVideoBench/resolve/main/data.parquet"
    try:
        print("  Loading OmniVideoBench parquet...")
        r = hf_get(url)
        df = pq.read_table(io.BytesIO(r.content)).to_pandas()
    except Exception as e:
        print(f"  Could not load OmniVideoBench: {e}")
        return existing

    by_type: dict[str, list] = {}
    for _, row in df.iterrows():
        raw_dur = parse_duration(row.get("duration"))
        if raw_dur is not None and raw_dur > MAX_DURATION:
            continue
        qt = str(row.get("question_type", "")).strip()
        if qt:
            by_type.setdefault(qt, []).append(row)

    print(f"  OmniVideoBench question_types ({len(by_type)}): {sorted(by_type)}")

    existing_by_type: dict[str, list] = {}
    for r in existing:
        if Path(r["file"]).exists() and r.get("questions"):
            existing_by_type.setdefault(r["task_type"], []).append(r)

    results = list(existing)
    done_videos: set[str] = {r.get("video_name", "") for r in existing if r.get("video_name")}

    for q_type, rows in sorted(by_type.items()):
        slug = slugify(q_type)
        out_dir = OUT_DIR / OMNIVIDEO_DATASET / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        already = len(existing_by_type.get(q_type, []))
        if already >= MAX_PER_CATEGORY:
            print(f"  [{OMNIVIDEO_DATASET}] {q_type}: {already}/{MAX_PER_CATEGORY} — complete")
            continue

        new_count = 0
        for row in rows:
            if already + new_count >= MAX_PER_CATEGORY:
                break
            vid_name = str(row.get("video", "")).strip()
            if not vid_name or vid_name in done_videos:
                continue

            vid_url = f"https://huggingface.co/datasets/NJU-LINK/OmniVideoBench/resolve/main/videos/{vid_name}.mp4"
            out_path = next_video_path(out_dir)
            print(f"  [{OMNIVIDEO_DATASET}/{slug}] trying {vid_name} [{row.get('duration')}] (#{already + new_count + 1}/{MAX_PER_CATEGORY})...")
            if not download_hf_video(vid_url, out_path):
                continue
            if not post_check(out_path):
                continue

            dur = ffprobe_duration(out_path)
            vid_rows = [r for r in rows if str(r.get("video", "")).strip() == vid_name]
            questions = [
                make_question(str(r["question"]), list(r.get("options", [])), str(r["correct_option"]), str(r["question_type"]))
                for r in vid_rows
            ]
            results.append({
                "dataset":    OMNIVIDEO_DATASET,
                "task_type":  q_type,
                "file":       str(out_path),
                "duration_s": round(dur, 2) if dur else None,
                "video_name": vid_name,
                "video_type": str(row.get("video_type", "")),
                "questions":  questions,
            })
            done_videos.add(vid_name)
            new_count += 1

        total = already + new_count
        if new_count == 0 and already < MAX_PER_CATEGORY:
            print(f"  [{OMNIVIDEO_DATASET}] {q_type}: no additional videos found (have {already}/{MAX_PER_CATEGORY})")
        elif new_count > 0:
            print(f"  [{OMNIVIDEO_DATASET}] {q_type}: {total}/{MAX_PER_CATEGORY} total (+{new_count} new)")

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET (OmniVideoBench will be skipped)'}")
    print(f"MAX_PER_CATEGORY: {MAX_PER_CATEGORY}\n")

    all_results: list[dict] = []
    if META_FILE.exists():
        all_results = json.loads(META_FILE.read_text())
        print(f"Loaded {len(all_results)} existing entries from {META_FILE}")

    datasets = [
        ("Video-MME",      "video-mme",       fetch_videomme),
        ("Daily-Omni",     "daily-omni",      fetch_daily_omni),
        ("WorldSense",     "worldsense",      fetch_worldsense),
        ("OmniVideoBench", "omnivideobench",  fetch_omnivideobench),
    ]

    for name, ds_key, fetcher in datasets:
        existing = [r for r in all_results if r["dataset"] == ds_key]
        counts = Counter(
            r["task_type"] for r in existing
            if Path(r["file"]).exists() and r.get("questions")
        )
        if counts and all(c >= MAX_PER_CATEGORY for c in counts.values()):
            print(f"\nSkipping {name} — {MAX_PER_CATEGORY}/category already complete ({len(existing)} entries)")
            continue

        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        others = [r for r in all_results if r["dataset"] != ds_key]
        try:
            results = fetcher(existing)
            all_results = others + results
            new = len(results) - len(existing)
            print(f"  -> {len(results)} total entries ({new:+d} new)")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results = others + existing  # restore on failure

        META_FILE.write_text(json.dumps(all_results, indent=2))

    print(f"\nMetadata saved -> {META_FILE}")

    # ── Validation ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nValidation\n{'='*60}")
    by_dataset: dict[str, list] = {}
    for r in all_results:
        by_dataset.setdefault(r["dataset"], []).append(r)

    total_ok = total_missing_qa = total_missing_video = 0
    for ds_name, entries in sorted(by_dataset.items()):
        counts = Counter(r["task_type"] for r in entries if Path(r["file"]).exists() and r.get("questions"))
        print(f"\n  {ds_name} ({len(entries)} entries, {len(counts)} categories):")
        by_type: dict[str, list] = {}
        for e in entries:
            by_type.setdefault(e["task_type"], []).append(e)
        for tt, tt_entries in sorted(by_type.items()):
            ok  = [e for e in tt_entries if Path(e["file"]).exists() and e.get("questions")]
            bad = [e for e in tt_entries if not Path(e["file"]).exists() or not e.get("questions")]
            print(f"    {tt:<40} {len(ok)}/{MAX_PER_CATEGORY} OK" + (f"  ({len(bad)} missing)" if bad else ""))
            total_ok           += len(ok)
            total_missing_video += sum(1 for e in tt_entries if not Path(e["file"]).exists())
            total_missing_qa    += sum(1 for e in tt_entries if not e.get("questions"))

    total = len(all_results)
    print(f"\n  Total: {total}  |  OK: {total_ok}  |  Missing video: {total_missing_video}  |  Missing Q&A: {total_missing_qa}")


if __name__ == "__main__":
    main()
