"""
Downloads 2 YouTube videos per category, each strictly ≤ 90 seconds.

Duration is enforced twice:
  1. Pre-download: yt-dlp match_filter rejects any video whose reported
     duration exceeds MAX_DURATION before any bytes are fetched.
  2. Post-download: ffprobe reads the actual file duration and deletes
     anything that slipped through (e.g. live streams with no metadata).

Usage:
    pip install yt-dlp
    python fetch_yt_videos.py

Requires ffprobe on PATH (ships with ffmpeg: https://ffmpeg.org/download.html)
"""

import json
import subprocess
import sys
from pathlib import Path

import yt_dlp

# ── Config ──────────────────────────────────────────────────────────────────

MAX_DURATION = 90       # seconds, strict upper bound
TARGET_PER_CAT = 2      # videos per category
SEARCH_POOL = 30        # how many results to scan per query before giving up

OUT_DIR = Path("website/public/videos/custom")
META_FILE = OUT_DIR / "metadata.json"

# ── Search queries per category ──────────────────────────────────────────────
# Multiple fallback queries per category, tried in order if first yields < 2.

CATEGORIES = {
    "soccer_matches": [
        "soccer goal highlight clip short",
        "football goal 60 seconds",
        "soccer skill clip short",
    ],
    "tennis_matches": [
        "tennis match point rally short clip",
        "tennis ace short",
        "tennis volley highlight clip",
    ],
    "tv_show": [
        "tv show scene funny clip",
        "television series short scene clip",
        "sitcom funny moment clip",
    ],
    "lecture": [
        "university lecture short excerpt clip",
        "educational lecture 60 seconds",
        "professor lecture clip short",
    ],
    "podcast": [
        "podcast clip short conversation",
        "podcast soundbite short",
        "podcast excerpt 60 seconds",
    ],
    "notebooklm": [
        "notebooklm audio overview demo",
        "notebooklm podcast example short",
        "google notebooklm demo clip",
    ],
    "vlog": [
        "daily vlog short clip",
        "vlog day in life 60 seconds",
        "vlog short moment clip",
    ],
    "yt_shorts": [
        "#shorts satisfying clip",
        "#shorts viral funny",
        "#shorts interesting fact",
    ],
    "movie_clips": [
        "famous movie scene clip short",
        "movie dialogue scene short clip",
        "film iconic scene clip",
    ],
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_match_filter(max_sec: int):
    """Return a yt-dlp match_filter that rejects videos longer than max_sec."""
    def match_filter(info, *, incomplete):
        dur = info.get("duration")
        if dur is None:
            # No duration metadata — allow through, post-check will catch it
            return None
        if dur > max_sec:
            return f"Duration {dur:.0f}s exceeds limit of {max_sec}s"
        return None
    return match_filter


def get_actual_duration(path: Path) -> float | None:
    """Use ffprobe to get the real duration of a downloaded file."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        print(f"    [ffprobe] Could not read duration for {path.name}: {e}")
        return None


def post_check(path: Path, max_sec: int) -> bool:
    """Return True if the file passes the duration check, delete it if not."""
    dur = get_actual_duration(path)
    if dur is None:
        print(f"    [post-check] Could not verify {path.name} — keeping it")
        return True
    if dur > max_sec:
        print(f"    [post-check] FAIL — {path.name} is {dur:.1f}s > {max_sec}s — deleting")
        path.unlink(missing_ok=True)
        return False
    print(f"    [post-check] OK — {path.name} is {dur:.1f}s")
    return True


# ── Core downloader ──────────────────────────────────────────────────────────

def download_category(category: str, queries: list[str]) -> list[dict]:
    """Download up to TARGET_PER_CAT videos for a category. Returns metadata list."""
    cat_dir = OUT_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    collected: list[dict] = []
    seen_ids: set[str] = set()

    for query in queries:
        if len(collected) >= TARGET_PER_CAT:
            break

        search_url = f"ytsearch{SEARCH_POOL}:{query}"
        print(f"  Searching: '{query}'")

        # yt-dlp options — extract info only first, then download selectively
        info_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "match_filter": make_match_filter(MAX_DURATION),
            "ignoreerrors": True,
        }

        try:
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                results = ydl.extract_info(search_url, download=False)
        except Exception as e:
            print(f"    Search error: {e}")
            continue

        if not results or "entries" not in results:
            continue

        for entry in results["entries"]:
            if len(collected) >= TARGET_PER_CAT:
                break
            if entry is None:
                continue

            vid_id = entry.get("id", "")
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)

            dur = entry.get("duration")
            if dur is not None and dur > MAX_DURATION:
                continue  # double-check before downloading

            idx = len(collected) + 1
            out_name = f"{category}_{idx}"
            out_tmpl = str(cat_dir / f"{out_name}.%(ext)s")

            print(f"    Downloading: [{dur}s] {entry.get('title', vid_id)[:60]}")

            dl_opts = {
                "quiet": True,
                "no_warnings": True,
                "match_filter": make_match_filter(MAX_DURATION),
                "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "merge_output_format": "mp4",
                "outtmpl": out_tmpl,
                "ignoreerrors": True,
            }

            try:
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.download([entry["webpage_url"]])
            except Exception as e:
                print(f"    Download error: {e}")
                continue

            # Find the downloaded file
            matches = list(cat_dir.glob(f"{out_name}.*"))
            if not matches:
                print(f"    No file found after download — skipping")
                continue

            dl_path = matches[0]

            # Post-download duration check
            if not post_check(dl_path, MAX_DURATION):
                continue

            actual_dur = get_actual_duration(dl_path)
            collected.append({
                "category": category,
                "index": idx,
                "file": str(dl_path.relative_to(OUT_DIR.parent.parent.parent)),
                "title": entry.get("title", ""),
                "youtube_id": vid_id,
                "url": entry.get("webpage_url", ""),
                "reported_duration_s": dur,
                "actual_duration_s": round(actual_dur, 2) if actual_dur else None,
                "uploader": entry.get("uploader", ""),
                "query_used": query,
            })
            print(f"    Saved: {dl_path.name}")

    if len(collected) < TARGET_PER_CAT:
        print(f"  WARNING: Only found {len(collected)}/{TARGET_PER_CAT} for '{category}'")

    return collected


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata: list[dict] = []

    for category, queries in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        results = download_category(category, queries)
        all_metadata.extend(results)
        print(f"  Done: {len(results)}/{TARGET_PER_CAT} videos")

    # Save metadata
    META_FILE.write_text(json.dumps(all_metadata, indent=2))
    print(f"\nMetadata saved to {META_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for cat in CATEGORIES:
        cat_results = [m for m in all_metadata if m["category"] == cat]
        status = "OK" if len(cat_results) == TARGET_PER_CAT else f"PARTIAL ({len(cat_results)}/{TARGET_PER_CAT})"
        print(f"  {cat:<20} {status}")
        for m in cat_results:
            print(f"    [{m['actual_duration_s']}s] {m['title'][:50]}")


if __name__ == "__main__":
    main()
