"""
fetch_dataset_videos.py
Downloads 2 videos + full row metadata per category from dataset sources.

Source priority per category:
  YouTube-backed (Video-MME, Daily-Omni)  → downloaded via yt-dlp
  HuggingFace zip (WorldSense)            → range-request ZIP extraction
  HuggingFace gated (OmniVideoBench)      → requires HF_TOKEN + accepted terms
  HuggingFace parquet (ShortVid-Bench)    → video bytes extracted from parquet
  YouTube search fallback                 → for notebooklm + anything unfilled

Duration is enforced:
  - Pre-filter:  metadata field (video_duration, duration col, or yt-dlp info)
  - Post-check:  ffprobe reads actual encoded duration, deletes file if > MAX_DURATION

Usage:
    pip install yt-dlp datasets huggingface_hub requests pyarrow pandas
    HF_TOKEN=hf_xxx python fetch_dataset_videos.py
"""

import io
import json
import os
import re
import struct
import subprocess
import zlib
from pathlib import Path

import requests
import yt_dlp

try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("WARNING: pandas/pyarrow not installed — ShortVid-Bench and OmniVideoBench disabled")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: `datasets` not installed — Video-MME and Daily-Omni disabled")

# ── Config ───────────────────────────────────────────────────────────────────

MAX_DURATION    = 90        # seconds, strict upper bound
TARGET_PER_CAT  = 2
SEARCH_POOL     = 25        # YouTube fallback: results to scan per query

OUT_DIR   = Path("videos")
META_FILE = OUT_DIR / "metadata.json"
HF_TOKEN  = os.environ.get("HF_TOKEN")

HF_HEADERS = {"User-Agent": "Mozilla/5.0"}
if HF_TOKEN:
    HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

# ── Category → source definitions ────────────────────────────────────────────
# Each source is tried in order until TARGET_PER_CAT videos are collected.
# "filter" values are checked with a case-insensitive substring match.

CATEGORY_SOURCES = {
    "soccer_matches": [
        {"dataset": "daily_omni",      "field": "video_category",        "values": ["Sports"],            "note": "filter fine_category for soccer/football"},
        {"dataset": "worldsense",      "field": "domain",                 "values": ["Sports"]},
        {"dataset": "omnivideobench",  "field": "video_type",             "values": ["Sports"]},
        {"dataset": "yt_search",       "queries": ["soccer goal highlight short clip", "football goal 60 seconds"]},
    ],
    "tennis_matches": [
        {"dataset": "daily_omni",      "field": "video_category",        "values": ["Sports"]},
        {"dataset": "worldsense",      "field": "domain",                 "values": ["Sports"]},
        {"dataset": "omnivideobench",  "field": "video_type",             "values": ["Sports"]},
        {"dataset": "yt_search",       "queries": ["tennis match point short clip", "tennis rally short"]},
    ],
    "tv_show": [
        {"dataset": "videomme",        "field": "sub_category",           "values": ["Film & Television"]},
        {"dataset": "worldsense",      "field": "domain",                 "values": ["Film & TV"]},
        {"dataset": "omnivideobench",  "field": "video_type",             "values": ["TV"]},
        {"dataset": "daily_omni",      "field": "video_category",        "values": ["Film & Animation", "Entertainment"]},
        {"dataset": "yt_search",       "queries": ["tv show scene short clip", "sitcom funny moment clip"]},
    ],
    "lecture": [
        {"dataset": "videomme",        "field": "domain",                 "values": ["Knowledge"]},
        {"dataset": "daily_omni",      "field": "video_category",        "values": ["Education"]},
        {"dataset": "yt_search",       "queries": ["university lecture short excerpt", "professor lecture 60 seconds"]},
    ],
    "podcast": [
        {"dataset": "daily_omni",      "field": "video_category",        "values": ["People & Blogs", "Howto & Style"]},
        {"dataset": "yt_search",       "queries": ["podcast clip short conversation", "podcast soundbite 60 seconds"]},
    ],
    "notebooklm": [
        {"dataset": "yt_search",       "queries": ["notebooklm audio overview demo", "notebooklm podcast example short", "google notebooklm demo clip"]},
    ],
    "vlog": [
        {"dataset": "videomme",        "field": "sub_category",           "values": ["Life Record"]},
        {"dataset": "daily_omni",      "field": "video_category",        "values": ["People & Blogs", "Howto & Style"]},
        {"dataset": "omnivideobench",  "field": "video_type",             "values": ["Vlog"]},
        {"dataset": "yt_search",       "queries": ["daily vlog short clip", "vlog 60 seconds"]},
    ],
    "yt_shorts": [
        {"dataset": "shortvid_bench",  "field": None,                     "values": None},
        {"dataset": "yt_search",       "queries": ["#shorts satisfying clip", "#shorts interesting viral"]},
    ],
    "movie_clips": [
        {"dataset": "videomme",        "field": "sub_category",           "values": ["Film & Television"]},
        {"dataset": "worldsense",      "field": "domain",                 "values": ["Film & TV"]},
        {"dataset": "yt_search",       "queries": ["famous movie scene short clip", "film iconic scene clip"]},
    ],
}

# ── Duration helpers ──────────────────────────────────────────────────────────

def parse_duration(raw) -> float | None:
    """Parse duration from various formats to seconds. Returns None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    # "1:30" or "01:30" or "1:30:00"
    if re.match(r"^\d+:\d{2}(:\d{2})?$", s):
        parts = list(map(int, s.split(":")))
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    # "90s" or "90"
    m = re.match(r"^(\d+(?:\.\d+)?)\s*s?$", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def ffprobe_duration(path: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def post_check(path: Path) -> bool:
    """Delete and return False if file exceeds MAX_DURATION."""
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

# ── YouTube downloader (yt-dlp) ───────────────────────────────────────────────

def _match_filter(info, *, incomplete):
    dur = info.get("duration")
    if dur and dur > MAX_DURATION:
        return f"Duration {dur}s > {MAX_DURATION}s"
    return None


def download_youtube(video_id: str, out_path: Path) -> bool:
    """Download a YouTube video by ID. Returns True on success."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "match_filter": _match_filter,
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_path.with_suffix(".%(ext)s")),
        "ignoreerrors": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return False
            dur = info.get("duration")
            if dur and dur > MAX_DURATION:
                print(f"    Skipping {video_id}: reported {dur}s > {MAX_DURATION}s")
                return False
            ydl.download([url])
        return out_path.with_suffix(".mp4").exists() or any(out_path.parent.glob(f"{out_path.stem}.*"))
    except Exception as e:
        print(f"    yt-dlp error for {video_id}: {e}")
        return False


def search_and_download_youtube(query: str, out_path: Path) -> tuple[bool, dict]:
    """Search YouTube, download first result ≤ MAX_DURATION. Returns (success, info)."""
    search_url = f"ytsearch{SEARCH_POOL}:{query}"
    info_opts = {"quiet": True, "no_warnings": True, "ignoreerrors": True}
    try:
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            results = ydl.extract_info(search_url, download=False)
    except Exception as e:
        print(f"    Search error: {e}")
        return False, {}

    for entry in (results.get("entries") or []):
        if not entry:
            continue
        dur = entry.get("duration")
        if dur and dur > MAX_DURATION:
            continue
        vid_id = entry.get("id", "")
        print(f"    Found: [{dur}s] {entry.get('title','')[:60]}")
        success = download_youtube(vid_id, out_path)
        if success:
            return True, {
                "youtube_id": vid_id,
                "title": entry.get("title", ""),
                "url": entry.get("webpage_url", ""),
                "uploader": entry.get("uploader", ""),
                "reported_duration_s": dur,
            }
    return False, {}

# ── HuggingFace file downloader ───────────────────────────────────────────────

def hf_get(url: str, stream=False) -> requests.Response:
    r = requests.get(url, headers=HF_HEADERS, stream=stream, timeout=60)
    r.raise_for_status()
    return r


def download_hf_video(url: str, out_path: Path) -> bool:
    """Download a file from HuggingFace to out_path."""
    try:
        r = hf_get(url, stream=True)
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    HF download error: {e}")
        return False

# ── WorldSense ZIP range-request extractor ────────────────────────────────────

WS_REPO   = "honglyhly/WorldSense"
WS_ZIPS   = [f"worldsense_videos_{i}.zip" for i in range(11)]
_ws_index: dict = {}   # populated lazily: {video_id.mp4: (zip_name, offset, comp_size, uncomp_size, compression)}


def _ws_url(fname): return f"https://huggingface.co/datasets/{WS_REPO}/resolve/main/{fname}"

def _range_get(url, start, end):
    r = requests.get(url, headers={**HF_HEADERS, "Range": f"bytes={start}-{end}"}, timeout=30)
    r.raise_for_status()
    return r.content

def _build_ws_index():
    """Scan all WorldSense zips and build a filename → location index."""
    global _ws_index
    if _ws_index:
        return
    print("  Building WorldSense zip index (one-time scan)...")
    for zip_name in WS_ZIPS:
        url = _ws_url(zip_name)
        try:
            r = requests.head(url, headers=HF_HEADERS, timeout=10, allow_redirects=True)
            size = int(r.headers["Content-Length"])
            tail_size = min(65536 + 22, size)
            tail = _range_get(url, size - tail_size, size - 1)
            eocd_idx = tail.rfind(b"PK\x05\x06")
            if eocd_idx == -1: continue
            eocd = tail[eocd_idx:]
            _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from("<4sHHHHIIH", eocd)
            cd_data = _range_get(url, cd_offset, cd_offset + cd_size - 1)
            pos = 0
            while pos < len(cd_data):
                if cd_data[pos:pos+4] != b"PK\x01\x02": break
                (_, _, _, _, comp, _, _, _, comp_size, uncomp_size,
                 fn_len, ex_len, cm_len, _, _, _, lh_offset) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd_data, pos)
                pos += 46
                fname = cd_data[pos:pos+fn_len].decode("utf-8", errors="replace")
                pos += fn_len + ex_len + cm_len
                _ws_index[fname] = (zip_name, lh_offset, comp_size, uncomp_size, comp)
            print(f"    Indexed {zip_name}")
        except Exception as e:
            print(f"    Could not index {zip_name}: {e}")


def _extract_from_ws_zip(zip_name, lh_offset, comp_size, uncomp_size, compression) -> bytes:
    url = _ws_url(zip_name)
    hdr = _range_get(url, lh_offset, lh_offset + 30)
    _, _, _, _, _, _, _, _, _, fn_len, ex_len = struct.unpack_from("<4sHHHHHIIIHH", hdr)
    data_start = lh_offset + 30 + fn_len + ex_len
    compressed = _range_get(url, data_start, data_start + comp_size - 1)
    if compression == 0: return compressed
    if compression == 8: return zlib.decompress(compressed, -15)
    raise ValueError(f"Unsupported compression: {compression}")


def download_worldsense_video(video_id: str, out_path: Path) -> bool:
    _build_ws_index()
    fname = f"{video_id}.mp4"
    if fname not in _ws_index:
        print(f"    {fname} not found in WorldSense zips")
        return False
    zip_name, lh_offset, comp_size, uncomp_size, compression = _ws_index[fname]
    print(f"    Extracting {fname} from {zip_name} ({uncomp_size/1e6:.1f} MB)...")
    try:
        data = _extract_from_ws_zip(zip_name, lh_offset, comp_size, uncomp_size, compression)
        out_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"    WorldSense extraction error: {e}")
        return False

# ── Dataset loaders ───────────────────────────────────────────────────────────

_cache = {}

def _load(name, *args, **kwargs):
    if name not in _cache:
        if not HAS_DATASETS:
            return None
        try:
            from datasets import load_dataset
            print(f"  Loading {name} from HuggingFace...")
            _cache[name] = load_dataset(*args, **kwargs)
        except Exception as e:
            print(f"  Could not load {name}: {e}")
            _cache[name] = None
    return _cache[name]


def get_videomme():
    ds = _load("videomme", "lmms-lab/Video-MME", split="test")
    return ds


def get_daily_omni():
    ds = _load("daily_omni", "liarliar/Daily-Omni", split="train")
    return ds


def get_worldsense_qa() -> dict:
    if "worldsense_qa" not in _cache:
        url = f"https://huggingface.co/datasets/{WS_REPO}/resolve/main/worldsense_qa.json"
        try:
            print("  Loading WorldSense QA JSON...")
            r = hf_get(url)
            _cache["worldsense_qa"] = r.json()
        except Exception as e:
            print(f"  Could not load WorldSense QA: {e}")
            _cache["worldsense_qa"] = {}
    return _cache["worldsense_qa"]


def get_omnivideobench_df():
    if not HF_TOKEN:
        print("  OmniVideoBench: HF_TOKEN not set — skipping")
        return None
    if not HAS_PARQUET:
        return None
    if "omnivideobench" not in _cache:
        url = "https://huggingface.co/datasets/NJU-LINK/OmniVideoBench/resolve/main/data.parquet"
        try:
            print("  Loading OmniVideoBench parquet...")
            r = hf_get(url)
            _cache["omnivideobench"] = pq.read_table(io.BytesIO(r.content)).to_pandas()
        except Exception as e:
            print(f"  Could not load OmniVideoBench: {e}")
            _cache["omnivideobench"] = None
    return _cache["omnivideobench"]


def get_shortvid_bench():
    ds = _load("shortvid_bench", "TencentARC/ShortVid-Bench", split="train", trust_remote_code=True)
    return ds

# ── Row → metadata dict ────────────────────────────────────────────────────────

def row_to_meta(row, dataset_name: str, category: str, index: int, file_path: str, actual_dur: float | None) -> dict:
    base = {
        "category":          category,
        "index":             index,
        "dataset_source":    dataset_name,
        "file":              file_path,
        "actual_duration_s": round(actual_dur, 2) if actual_dur else None,
    }
    base["row_data"] = {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in dict(row).items() if k != "video"}
    return base

# ── Per-dataset fetchers ───────────────────────────────────────────────────────

def _find_file(out_path: Path) -> Path | None:
    for p in out_path.parent.glob(f"{out_path.stem}.*"):
        return p
    return None


def fetch_from_videomme(category: str, src: dict, out_dir: Path, needed: int) -> list[dict]:
    ds = get_videomme()
    if ds is None: return []
    field, values = src["field"], [v.lower() for v in src["values"]]

    results = []
    for row in ds:
        if len(results) >= needed: break
        cell = str(row.get(field, "")).lower()
        if not any(v in cell for v in values): continue
        if row.get("duration", "").lower() not in ("short",): continue  # only short videos

        vid_id = row.get("videoID") or row.get("video_id", "")
        if not vid_id: continue

        idx = len(results) + 1
        out_path = out_dir / f"{category}_{idx}"
        print(f"    [{row.get('duration')}] {row.get('question','')[:50]}... (videoID={vid_id})")

        success = download_youtube(vid_id, out_path)
        if not success: continue
        real_path = _find_file(out_path)
        if not real_path or not post_check(real_path): continue

        dur = ffprobe_duration(real_path)
        meta = row_to_meta(row, "videomme", category, idx, str(real_path), dur)
        meta["youtube_id"] = vid_id
        results.append(meta)
    return results


def fetch_from_daily_omni(category: str, src: dict, out_dir: Path, needed: int) -> list[dict]:
    ds = get_daily_omni()
    if ds is None: return []
    field, values = src["field"], [v.lower() for v in src["values"]]

    results = []
    for row in ds:
        if len(results) >= needed: break
        cell = str(row.get(field, "")).lower()
        if not any(v in cell for v in values): continue

        # Enforce duration from metadata
        raw_dur = parse_duration(row.get("video_duration"))
        if raw_dur is not None and raw_dur > MAX_DURATION: continue

        vid_id = row.get("video_id", "")
        if not vid_id: continue

        idx = len(results) + 1
        out_path = out_dir / f"{category}_{idx}"
        print(f"    [{row.get('video_duration')}] {row.get('Question','')[:50]} (video_id={vid_id})")

        success = download_youtube(vid_id, out_path)
        if not success: continue
        real_path = _find_file(out_path)
        if not real_path or not post_check(real_path): continue

        dur = ffprobe_duration(real_path)
        meta = row_to_meta(row, "daily_omni", category, idx, str(real_path), dur)
        meta["youtube_id"] = vid_id
        results.append(meta)
    return results


def fetch_from_worldsense(category: str, src: dict, out_dir: Path, needed: int) -> list[dict]:
    qa = get_worldsense_qa()
    if not qa: return []
    field, values = src["field"], [v.lower() for v in src["values"]]

    results = []
    for video_id, video_data in qa.items():
        if len(results) >= needed: break
        domain = str(video_data.get("domain", "")).lower()
        if not any(v in domain for v in values): continue

        idx = len(results) + 1
        out_path = out_dir / f"{category}_{idx}.mp4"

        success = download_worldsense_video(video_id, out_path)
        if not success: continue
        if not post_check(out_path): continue

        dur = ffprobe_duration(out_path)
        # Flatten first task's QA data as representative row
        first_task = next(iter({k: v for k, v in video_data.items() if k.startswith("task")}.values()), {})
        meta = {
            "category":          category,
            "index":             idx,
            "dataset_source":    "worldsense",
            "file":              str(out_path),
            "actual_duration_s": round(dur, 2) if dur else None,
            "row_data": {
                "video_id":      video_id,
                "domain":        video_data.get("domain"),
                "video_caption": video_data.get("video_caption"),
                "all_tasks":     {k: v for k, v in video_data.items() if k.startswith("task")},
            },
        }
        results.append(meta)
    return results


def fetch_from_omnivideobench(category: str, src: dict, out_dir: Path, needed: int) -> list[dict]:
    df = get_omnivideobench_df()
    if df is None: return []
    field, values = src["field"], [v.lower() for v in src["values"]]

    results = []
    for _, row in df.iterrows():
        if len(results) >= needed: break
        cell = str(row.get(field, "")).lower()
        if not any(v in cell for v in values): continue

        raw_dur = parse_duration(row.get("duration"))
        if raw_dur is not None and raw_dur > MAX_DURATION: continue

        vid_name = str(row.get("video", "")).strip()
        if not vid_name: continue

        # video files are at videos/{video_name}.mp4 in the repo
        vid_url = f"https://huggingface.co/datasets/NJU-LINK/OmniVideoBench/resolve/main/videos/{vid_name}.mp4"
        idx = len(results) + 1
        out_path = out_dir / f"{category}_{idx}.mp4"
        print(f"    [{row.get('duration')}] {str(row.get('question',''))[:50]} ({vid_name})")

        success = download_hf_video(vid_url, out_path)
        if not success: continue
        if not post_check(out_path): continue

        dur = ffprobe_duration(out_path)
        meta = {
            "category":          category,
            "index":             idx,
            "dataset_source":    "omnivideobench",
            "file":              str(out_path),
            "actual_duration_s": round(dur, 2) if dur else None,
            "row_data":          row.drop(labels=["video"], errors="ignore").to_dict(),
        }
        results.append(meta)
    return results


def fetch_from_shortvid_bench(category: str, src: dict, out_dir: Path, needed: int) -> list[dict]:
    ds = get_shortvid_bench()
    if ds is None: return []

    # Use Arrow format to get raw bytes without triggering torchcodec decode
    try:
        import pyarrow as pa
        raw_ds = ds.with_format("arrow")
    except Exception as e:
        print(f"  ShortVid-Bench: could not use arrow format: {e}")
        return []

    results = []
    i = 0
    while len(results) < needed and i < len(ds):
        try:
            # Fetch one row at a time as a raw Arrow table
            batch = raw_ds[i:i+1]
            i += 1

            # Extract video struct from Arrow column
            video_col = batch.column("video")
            if video_col is None or len(video_col) == 0:
                continue

            video_scalar = video_col[0]
            # Arrow struct: access 'bytes' child
            if hasattr(video_scalar, "as_py"):
                video_dict = video_scalar.as_py()
            else:
                video_dict = video_scalar

            if isinstance(video_dict, dict):
                raw_bytes = video_dict.get("bytes")
            elif isinstance(video_dict, (bytes, bytearray)):
                raw_bytes = video_dict
            else:
                continue

            if not raw_bytes:
                continue

            idx = len(results) + 1
            out_path = out_dir / f"{category}_{idx}.mp4"
            out_path.write_bytes(raw_bytes)

            if not post_check(out_path):
                continue

            dur = ffprobe_duration(out_path)

            # Get metadata columns (excluding video)
            row_clean = {}
            for col_name in batch.schema.names:
                if col_name == "video":
                    continue
                col_val = batch.column(col_name)[0]
                row_clean[col_name] = col_val.as_py() if hasattr(col_val, "as_py") else str(col_val)

            meta = {
                "category":          category,
                "index":             idx,
                "dataset_source":    "shortvid_bench",
                "file":              str(out_path),
                "actual_duration_s": round(dur, 2) if dur else None,
                "row_data":          row_clean,
            }
            results.append(meta)

        except Exception as e:
            print(f"  ShortVid-Bench row {i} error: {e}")
            i += 1
            continue

    return results


def fetch_from_yt_search(category: str, src: dict, out_dir: Path, needed: int, offset: int = 0) -> list[dict]:
    results = []
    for query in src["queries"]:
        if len(results) >= needed: break
        print(f"    YouTube search: '{query}'")
        idx = offset + len(results) + 1
        out_path = out_dir / f"{category}_{idx}"
        success, info = search_and_download_youtube(query, out_path)
        if not success: continue
        real_path = _find_file(out_path)
        if not real_path or not post_check(real_path): continue
        dur = ffprobe_duration(real_path)
        meta = {
            "category":          category,
            "index":             idx,
            "dataset_source":    "yt_search",
            "file":              str(real_path),
            "actual_duration_s": round(dur, 2) if dur else None,
            "row_data":          info,
        }
        results.append(meta)
    return results

# ── Dispatcher ────────────────────────────────────────────────────────────────

FETCHERS = {
    "videomme":      fetch_from_videomme,
    "daily_omni":    fetch_from_daily_omni,
    "worldsense":    fetch_from_worldsense,
    "omnivideobench":fetch_from_omnivideobench,
    "shortvid_bench":fetch_from_shortvid_bench,
    "yt_search":     fetch_from_yt_search,
}


def fetch_category(category: str, sources: list[dict]) -> list[dict]:
    out_dir = OUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    collected: list[dict] = []

    for src in sources:
        if len(collected) >= TARGET_PER_CAT:
            break
        needed = TARGET_PER_CAT - len(collected)
        ds_name = src["dataset"]
        print(f"  -> trying {ds_name} (need {needed} more)")

        fetcher = FETCHERS.get(ds_name)
        if fetcher is None:
            continue

        if ds_name == "yt_search":
            new = fetcher(category, src, out_dir, needed, offset=len(collected))
        else:
            new = fetcher(category, src, out_dir, needed)

        # Re-index to avoid filename collisions from different sources
        for item in new:
            item["index"] = len(collected) + 1
        collected.extend(new)

    if len(collected) < TARGET_PER_CAT:
        print(f"  WARNING: only {len(collected)}/{TARGET_PER_CAT} collected for '{category}'")
    return collected

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if HF_TOKEN:
        print(f"HF_TOKEN detected — OmniVideoBench enabled")
    else:
        print(f"No HF_TOKEN — OmniVideoBench disabled (set HF_TOKEN env var to enable)")

    all_meta: list[dict] = []

    RESUME_FROM = None  # set to None to run all
    resume = RESUME_FROM is None

    for category, sources in CATEGORY_SOURCES.items():
        if not resume:
            if category == RESUME_FROM:
                resume = True
            else:
                print(f"Skipping {category} (already done)")
                continue
        print(f"\n{'='*60}\nCategory: {category}\n{'='*60}")
        results = fetch_category(category, sources)
        all_meta.extend(results)

    META_FILE.write_text(json.dumps(all_meta, indent=2))
    print(f"\nMetadata -> {META_FILE}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for cat in CATEGORY_SOURCES:
        items = [m for m in all_meta if m["category"] == cat]
        status = "✓" if len(items) == TARGET_PER_CAT else f"PARTIAL {len(items)}/{TARGET_PER_CAT}"
        print(f"  {cat:<20} {status}")
        for m in items:
            src = m["dataset_source"]
            dur = m["actual_duration_s"]
            title = m["row_data"].get("title") or m["row_data"].get("Question") or m["row_data"].get("question") or ""
            print(f"    [{dur}s] [{src}] {str(title)[:50]}")


if __name__ == "__main__":
    main()
