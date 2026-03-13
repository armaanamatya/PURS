"""
Downloads 2 WorldSense videos efficiently using HTTP range requests.
Only downloads the ZIP central directory + the specific video bytes — not the full 2 GB zip.

Usage:
    python download_videos.py
"""

import io
import os
import struct
import zlib
from pathlib import Path

import requests

OUT_DIR = Path(__file__).parent / "website" / "public" / "videos" / "worldsense"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
WORLDSENSE_REPO = "honglyhly/WorldSense"
TARGET_IDS = ["dvOkwKAs", "UYkFSXsh"]
ZIPS = [f"worldsense_videos_{i}.zip" for i in range(11)]


def headers(extra=None):
    h = {"User-Agent": "Mozilla/5.0"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    if extra:
        h.update(extra)
    return h


def zip_url(zip_name):
    return f"https://huggingface.co/datasets/{WORLDSENSE_REPO}/resolve/main/{zip_name}"


def range_get(url, start, end):
    """Fetch bytes [start, end] inclusive via HTTP range request."""
    r = requests.get(url, headers=headers({"Range": f"bytes={start}-{end}"}), timeout=30)
    r.raise_for_status()
    return r.content


def get_file_size(url):
    r = requests.head(url, headers=headers(), timeout=10, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def find_eocd(data):
    """Find End of Central Directory signature in tail bytes."""
    sig = b"PK\x05\x06"
    idx = data.rfind(sig)
    if idx == -1:
        raise ValueError("EOCD not found")
    return idx, data[idx:]


def parse_eocd(eocd_data):
    # EOCD: sig(4) disk(2) start_disk(2) entries_on_disk(2) total_entries(2)
    #       cd_size(4) cd_offset(4) comment_len(2)
    sig, disk, start_disk, entries_disk, total_entries, cd_size, cd_offset, comment_len = \
        struct.unpack_from("<4sHHHHIIH", eocd_data)
    assert sig == b"PK\x05\x06"
    return cd_offset, cd_size, total_entries


def parse_central_directory(cd_data):
    """Return dict of {filename: (local_header_offset, compressed_size, uncompressed_size, compression)}."""
    entries = {}
    pos = 0
    while pos < len(cd_data):
        if cd_data[pos:pos+4] != b"PK\x01\x02":
            break
        # Central directory file header
        (sig, ver_made, ver_needed, flags, compression,
         mod_time, mod_date, crc32, compressed_size, uncompressed_size,
         fname_len, extra_len, comment_len, disk_start, int_attr, ext_attr,
         local_header_offset) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd_data, pos)
        pos += 46
        fname = cd_data[pos:pos+fname_len].decode("utf-8", errors="replace")
        pos += fname_len + extra_len + comment_len
        entries[fname] = (local_header_offset, compressed_size, uncompressed_size, compression)
    return entries


def extract_file_from_zip_url(url, local_header_offset, compressed_size, uncompressed_size, compression):
    """Download and decompress a single file from a remote zip."""
    # Local file header: sig(4) ver(2) flags(2) comp(2) time(2) date(2) crc(4)
    #                    comp_size(4) uncomp_size(4) fname_len(2) extra_len(2)
    header_data = range_get(url, local_header_offset, local_header_offset + 30)
    sig, ver, flags, comp, mod_time, mod_date, crc, cs, us, fname_len, extra_len = \
        struct.unpack_from("<4sHHHHHIIIHH", header_data)
    assert sig == b"PK\x03\x04", f"Bad local header sig: {sig}"

    data_start = local_header_offset + 30 + fname_len + extra_len
    data_end = data_start + compressed_size - 1

    compressed = range_get(url, data_start, data_end)

    if compression == 0:  # stored
        return compressed
    elif compression == 8:  # deflated
        return zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"Unsupported compression method: {compression}")


def fetch_videos_from_zip(url, target_filenames):
    """Use range requests to extract only the target files from a remote zip."""
    print(f"  Getting file size...")
    try:
        size = get_file_size(url)
    except Exception as e:
        print(f"  Could not get file size: {e}")
        return {}

    print(f"  Zip size: {size / 1e9:.2f} GB — reading central directory...")

    # Read last 65 KB to find EOCD (max comment size is 65535 bytes)
    tail_size = min(65536 + 22, size)
    tail = range_get(url, size - tail_size, size - 1)

    eocd_idx, eocd_data = find_eocd(tail)
    cd_offset, cd_size, total_entries = parse_eocd(eocd_data)
    print(f"  Central directory: {total_entries} entries, {cd_size / 1024:.1f} KB at offset {cd_offset}")

    # Download central directory
    cd_data = range_get(url, cd_offset, cd_offset + cd_size - 1)
    entries = parse_central_directory(cd_data)

    found = {}
    for fname in target_filenames:
        if fname not in entries:
            continue
        local_offset, comp_size, uncomp_size, compression = entries[fname]
        print(f"  Found {fname} ({uncomp_size / 1e6:.1f} MB) — downloading...")
        data = extract_file_from_zip_url(url, local_offset, comp_size, uncomp_size, compression)
        dest = OUT_DIR / fname
        dest.write_bytes(data)
        print(f"  ✓ Saved to {dest}")
        found[fname] = str(dest)

    return found


# ── Main ───────────────────────────────────────────────────────────────────

print("=== Downloading WorldSense videos ===")
target_filenames = [f"{vid}.mp4" for vid in TARGET_IDS]
remaining = set(target_filenames)

for zip_name in ZIPS:
    if not remaining:
        break
    url = zip_url(zip_name)
    print(f"\nChecking {zip_name}...")
    try:
        found = fetch_videos_from_zip(url, list(remaining))
        remaining -= set(found.keys())
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== Results ===")
for fname in target_filenames:
    dest = OUT_DIR / fname
    if dest.exists():
        print(f"✓ {fname} ({dest.stat().st_size / 1e6:.1f} MB) -> {dest}")
    else:
        print(f"✗ {fname} not found")
