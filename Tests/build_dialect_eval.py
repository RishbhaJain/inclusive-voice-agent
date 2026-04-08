"""
build_dialect_eval.py
─────────────────────
Builds a labeled audio evaluation set for the dialect classifier by streaming
Common Voice Scripted Speech 25.0 (English) from Mozilla Data Collective (MDC).

Why streaming instead of downloading the full archive?
  The MDC archive is ~88 GB. We only need 150 clips (~50 per dialect).
  Common Voice archives store TSV metadata files before the audio clips, so
  we can:
    1. Parse the TSV early in the stream to identify which clip filenames we want.
    2. Extract only those MP3s as they appear.
    3. Stop streaming the moment all three dialects are full.
  In practice this downloads only a small fraction of the total archive.

Setup (one-time):
  1. Add to your .env file:
       MDC_API_KEY=35fe7bdc71ef3087d286b046f44c74c797a22ac0c090d068b0d93597328c69ad
  2. Accept the dataset terms at:
       https://datacollective.mozillafoundation.org/datasets/cmndapwry02jnmh07dyo46mot
     (Click "Download" or "Agree to terms" on that page — required once per account)
  3. Run: python Tests/build_dialect_eval.py

Output:
  Tests/dialect_eval_data/
    southern/   clip_000.npy  clip_001.npy  ...
    sandiego/   clip_000.npy  ...
    neutral/    clip_000.npy  ...
    manifest.csv
"""

import csv
import io
import os
import sys
import tarfile

import librosa
import numpy as np
import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# ── Config ──────────────────────────────────────────────────────────────────────

MDC_DATASET_ID = "cmndapwry02jnmh07dyo46mot"  # Common Voice Scripted Speech 25.0 - English
MDC_API_BASE = "https://datacollective.mozillafoundation.org/api"
MDC_DATASET_URL = "https://datacollective.mozillafoundation.org/datasets/cmndapwry02jnmh07dyo46mot"

TARGET_SR = 16_000  # Hz — required by the ECAPA-TDNN model

# Neutral: exact match on "United States English" with no pipe (no regional sub-label).
# Exact matching ensures we don't accidentally pull in regional clips like
# "United States English|southern United States" into the neutral bucket.
NEUTRAL_LABEL = "United States English"

# US regional sub-accents: ordered list of (lowercase substring, dialect) rules.
# CV25 accent labels are self-reported free-text, pipe-delimited, e.g.:
#   "United States English|southern United States|New Orleans dialect"
# First match wins — more specific rules are placed before broader ones.
#
# Five regional groups:
#   southern   — Deep South, Texas, Appalachian, New Orleans
#   california — California, West Coast, Pacific Northwest
#   midwest    — Midwestern, Minnesotan, Ohio, Chicago, Great Lakes
#   northeast  — Boston, New York, Philadelphia, New England
#   neutral    — plain "United States English" (exact match, no pipe)
REGIONAL_RULES: list[tuple[str, str]] = [
    # Southern
    ("southern united states", "southern"),
    ("southern u.s.",          "southern"),
    ("new orleans",            "southern"),
    ("appalachia",             "southern"),
    ("appalachian",            "southern"),
    ("texas",                  "southern"),
    ("tennessee",              "southern"),
    ("mississippi",            "southern"),
    ("alabama",                "southern"),
    ("southern draw",          "southern"),  # catches both "draw" and "drawl"
    ("southern ohio",          "southern"),
    # California / West Coast
    ("california",             "california"),
    ("west coast",             "california"),
    ("pacific northwest",      "california"),
    ("pacific north west",     "california"),
    ("valley girl",            "california"),
    ("surffer",                "california"),  # self-reported "surffer dude"
    # Midwest
    ("minnesotan",             "midwest"),
    ("midwestern",             "midwest"),
    ("midwest",                "midwest"),
    ("unite states midwest",   "midwest"),   # typo present in CV data
    ("ohio",                   "midwest"),
    ("chicago",                "midwest"),
    ("michigan",               "midwest"),
    ("indiana",                "midwest"),
    # Northeast
    ("boston",                 "northeast"),
    ("new york",               "northeast"),
    ("new yorker",             "northeast"),
    ("philadelphia",           "northeast"),
    ("northeastern",           "northeast"),
    ("new england",            "northeast"),
    ("mid-atlantic",           "northeast"),
    ("upstate new york",       "northeast"),
]

DIALECT_NAMES: frozenset[str] = frozenset({"southern", "california", "midwest", "northeast", "neutral"})

# Per-dialect clip caps. Regional groups: collect everything available.
# Neutral: cap at 300 — 447K available clips would require streaming most of the
# 94 GB archive. 300 is more than enough for a balanced eval baseline.
DIALECT_CAPS: dict[str, float] = {
    "southern":   float("inf"),
    "california": float("inf"),
    "midwest":    float("inf"),
    "northeast":  float("inf"),
    "neutral":    300,
}
# Common Voice TSV column names (CV25 format)
# The `accents` column is the self-reported accent field used for ground truth.
TSV_PATH_COL = "path"
TSV_ACCENT_COL = "accents"
TSV_CLIENT_COL = "client_id"

OUT_DIR = os.path.join(os.path.dirname(__file__), "dialect_eval_data")


# ── MDC helpers ─────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("MDC_API_KEY", "").strip()
    if not key:
        print(
            "\nERROR: MDC_API_KEY not set.\n"
            "Add this line to your .env file:\n"
            "  MDC_API_KEY=<your key from https://datacollective.mozillafoundation.org/profile/api>\n"
        )
        sys.exit(1)
    return key


def _get_download_url(api_key: str) -> str:
    """
    POST to /download to get a presigned S3 URL for the dataset archive.
    This URL is short-lived (~hours) and doesn't require auth to download.

    Why a presigned URL?
      MDC stores data on S3. Rather than proxying 88 GB through their API
      servers, they give you a temporary direct-to-S3 URL. This is standard
      practice for large file downloads.
    """
    url = f"{MDC_API_BASE}/datasets/{MDC_DATASET_ID}/download"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "voice-agent-inclusive/1.0",
    }
    resp = requests.post(url, headers=headers, timeout=30)

    if resp.status_code == 403:
        err = resp.json().get("error", "")
        if "Terms must be accepted" in err:
            print(
                "\nERROR: You must accept the dataset terms before downloading.\n"
                f"\n  1. Visit: {MDC_DATASET_URL}\n"
                "  2. Click 'Download' or 'Agree to terms' on that page.\n"
                "  3. Re-run: python Tests/build_dialect_eval.py\n"
            )
        else:
            print(f"\nERROR: Access denied — {err}\n")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"\nERROR: Unexpected API response {resp.status_code}: {resp.text[:200]}\n")
        sys.exit(1)

    data = resp.json()
    download_url = data.get("downloadUrl")
    if not download_url:
        print(f"\nERROR: No downloadUrl in response: {data}\n")
        sys.exit(1)

    size_gb = int(data.get("sizeBytes", 0)) / 1e9
    print(f"Got presigned download URL (archive size: {size_gb:.1f} GB)")
    print("Streaming — will stop as soon as all dialects are filled.\n")
    return download_url


# ── Streaming tar parser ────────────────────────────────────────────────────────

class _ByteLimitExceeded(Exception):
    """Raised by _StreamingResponse when the byte cap is hit."""


class _StreamingResponse:
    """
    Wraps a requests streaming response as a file-like object so that
    tarfile.open() can read from it incrementally without loading the
    entire archive into memory.

    Why not just pass the response directly?
      tarfile.open() expects a file-like object with a .read() method.
      requests' streaming response is an iterator over chunks; this adapter
      bridges the two interfaces.

    max_bytes:
      If set, raises _ByteLimitExceeded once that many bytes have been read.
      This lets you cap how much of a multi-GB archive you stream, e.g. for
      a quick sample run: --max-gb 10 stops after the first 10 GB.
    """

    def __init__(
        self,
        resp: requests.Response,
        chunk_size: int = 1 << 20,
        max_bytes: int | None = None,
    ):
        self._iter = resp.iter_content(chunk_size=chunk_size)
        self._buf = b""
        self._max_bytes = max_bytes
        self._bytes_read = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunks = [self._buf]
            for chunk in self._iter:
                chunks.append(chunk)
            self._buf = b""
            return b"".join(chunks)

        while len(self._buf) < n:
            try:
                self._buf += next(self._iter)
            except StopIteration:
                break

        out, self._buf = self._buf[:n], self._buf[n:]
        self._bytes_read += len(out)
        if self._max_bytes is not None and self._bytes_read >= self._max_bytes:
            raise _ByteLimitExceeded(f"Reached {self._bytes_read / 1e9:.1f} GB byte cap")
        return out

    def readable(self) -> bool:
        return True



def _resample_mp3(mp3_bytes: bytes) -> np.ndarray:
    """
    Decode MP3 bytes → float32 mono numpy array at TARGET_SR (16 kHz).

    librosa handles:
      - MP3 decoding (via ffmpeg/audioread under the hood)
      - Channel mixing (stereo → mono)
      - Resampling with anti-aliasing
    """
    return librosa.load(io.BytesIO(mp3_bytes), sr=TARGET_SR, mono=True)[0]


# ── Main ────────────────────────────────────────────────────────────────────────

def _stream_archive(download_url: str, discover_only: bool = False) -> tuple[dict, list]:
    """
    Stream the CV tar.gz archive and either:
      - discover_only=True:  print all TSV filenames + columns, then exit (no MP3s downloaded)
      - discover_only=False: extract matching MP3s and return (counts, manifest_rows)

    Why streaming (r|gz) instead of downloading first?
      tarfile's pipe mode reads one member at a time. We never hold more than
      one file in memory. For a 94 GB archive, this means we only download
      as far as needed — typically 5–15 GB before all 150 clips are found.
    """
    counts: dict[str, int] = {d: 0 for d in DIALECT_NAMES}
    manifest_rows: list[dict] = []
    target_clips: dict[str, tuple[str, str, str]] = {}
    tsv_loaded = False
    tsv_count = 0  # how many TSVs we've inspected

    http_resp = requests.get(download_url, stream=True, timeout=(10, 3600))
    http_resp.raise_for_status()
    stream = _StreamingResponse(http_resp)

    if discover_only:
        print("DISCOVER MODE — scanning TSV files only, no audio downloaded.\n")
    else:
        print("Scanning archive stream...")
        print("  (TSV metadata files appear before audio clips in Common Voice archives)\n")

    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                name = member.name
                basename = os.path.basename(name)

                # ── TSV files: inspect for accent column ──────────────────────
                if name.endswith(".tsv"):
                    tsv_count += 1
                    f = tf.extractfile(member)
                    if f is None:
                        continue

                    # Peek at the header row only (don't parse all rows yet)
                    header_line = f.readline().decode("utf-8", errors="replace").strip()
                    cols = header_line.split("\t")
                    print(f"  TSV [{tsv_count}]: {name}")
                    print(f"    Columns: {cols}")

                    if discover_only:
                        # In discover mode, stop after inspecting the first
                        # 10 TSVs — that's enough to see the archive structure
                        if tsv_count >= 10:
                            print("\nDiscover mode complete — re-run without --discover to extract clips.")
                            break
                        continue

                    # In normal mode, only parse validated.tsv.
                    # Why validated.tsv specifically?
                    #   - dev.tsv / test.tsv are small splits (~few thousand clips each)
                    #   - validated.tsv contains ALL quality-checked clips — the largest pool
                    #   - More clips = better chance of finding 50 per dialect accent label
                    # The archive order is: clip_durations → dev → invalidated → other →
                    # reported → test → train → validated. We stream past all of them until
                    # we hit validated.tsv, parse it, then extract MP3s.
                    if basename != "validated.tsv":
                        print(f"    Skipping (using validated.tsv for maximum clip pool)")
                        continue

                    accent_col = next(
                        (c for c in ("accents", "accent", "dialect", "region") if c in cols),
                        None,
                    )
                    if accent_col is None:
                        print(f"    No accent column — skipping")
                        continue

                    print(f"    Accent column: '{accent_col}' — parsing...")
                    # Re-read the full file (already consumed the header; re-extract)
                    # tarfile doesn't support seeking in pipe mode, so we need the
                    # full content — read everything after the header
                    rest = f.read().decode("utf-8", errors="replace")
                    full_text = header_line + "\n" + rest
                    parsed = _parse_tsv_text(full_text, accent_col)
                    if parsed:
                        target_clips.update(parsed)
                        tsv_loaded = True
                        print(f"    Loaded {len(parsed)} target clips from this TSV")
                    else:
                        print(f"    No clips with target accent labels — skipping")
                    continue

                # ── MP3 files: extract matching clips ─────────────────────────
                if discover_only or not name.endswith(".mp3"):
                    continue
                if not tsv_loaded:
                    continue

                clip_info = target_clips.get(basename)
                if clip_info is None:
                    continue

                dialect, accent_label, client_id = clip_info
                if counts[dialect] >= DIALECT_CAPS[dialect]:
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue

                try:
                    mp3_bytes = f.read()
                    y = _resample_mp3(mp3_bytes)
                except Exception as exc:
                    print(f"  WARNING: Could not decode {basename}: {exc}")
                    continue

                duration_s = len(y) / TARGET_SR
                clip_idx = counts[dialect]
                filename = f"clip_{clip_idx:03d}.npy"
                out_path = os.path.join(OUT_DIR, dialect, filename)

                np.save(out_path, y)
                counts[dialect] += 1

                manifest_rows.append({
                    "path": os.path.relpath(out_path, OUT_DIR),
                    "true_dialect": dialect,
                    "accent_label": accent_label,
                    "duration_s": f"{duration_s:.2f}",
                    "common_voice_id": client_id,
                })

                cap = DIALECT_CAPS[dialect]
                cap_str = str(int(cap)) if cap != float("inf") else "all"
                remaining = len(target_clips)
                print(
                    f"  [{dialect:>10}] {counts[dialect]:>4}/{cap_str}"
                    f"  {filename}  ({duration_s:.1f}s)"
                    f"  [{remaining} target clips left in archive]"
                )

                # Remove this clip from target_clips now that it's been collected.
                # When target_clips is empty, every clip we wanted has been seen —
                # either collected or skipped (dialect over cap). Stop streaming.
                target_clips.pop(basename, None)
                if not target_clips:
                    print("\nAll target clips accounted for — closing stream early.")
                    break

                # Also evict clips whose dialect has hit its cap — no point waiting for them.
                over_cap = {d for d in DIALECT_NAMES if counts[d] >= DIALECT_CAPS[d]}
                if over_cap:
                    to_remove = [fn for fn, info in target_clips.items() if info[0] in over_cap]
                    for fn in to_remove:
                        target_clips.pop(fn)

    except Exception as exc:
        if "pipe" not in str(exc).lower() and "truncated" not in str(exc).lower():
            raise

    return counts, manifest_rows


def _classify_accent(raw: str) -> str | None:
    """Return dialect name for a CV25 accent string, or None if not a target."""
    if raw == NEUTRAL_LABEL:
        return "neutral"
    lower = raw.lower()
    for substring, dialect in REGIONAL_RULES:
        if substring in lower:
            return dialect
    return None


def _parse_tsv_text(text: str, accent_col: str) -> dict[str, tuple[str, str, str]]:
    """
    Parse a full TSV string (with known accent column name) into a mapping of
    clip_filename → (dialect, accent_label, client_id).
    """
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    target: dict[str, tuple[str, str, str]] = {}
    for row in reader:
        raw_accent = row.get(accent_col, "").strip()
        dialect = _classify_accent(raw_accent)
        if dialect is None:
            continue
        clip_filename = row.get(TSV_PATH_COL, "").strip()
        if not clip_filename:
            continue
        if not clip_filename.lower().endswith(".mp3"):
            clip_filename += ".mp3"
        client_id = row.get(TSV_CLIENT_COL, "")
        target[clip_filename] = (dialect, raw_accent, client_id)
    return target


RAW_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "raw_sample")


def _extract_raw_sample(download_url: str, max_gb: float = 10.0) -> None:
    """
    Stream the first `max_gb` gigabytes of the CV archive and extract
    everything — all TSVs and all MP3s — without any accent filtering.

    Why no filtering?
      This is a one-time data pull so you can explore the archive locally.
      Once you have the raw files on disk you can filter, grep the TSVs,
      inspect accent labels, and re-run classification with zero network cost.

    Output layout:
      Tests/raw_sample/
        validated.tsv          (and any other TSVs encountered)
        clips/
          common_voice_en_12345678.mp3
          ...

    Usage:
      python Tests/build_dialect_eval.py --extract-all
      python Tests/build_dialect_eval.py --extract-all --max-gb 5
    """
    clips_dir = os.path.join(RAW_SAMPLE_DIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    max_bytes = int(max_gb * 1e9)
    print(f"RAW EXTRACT MODE — streaming first {max_gb:.0f} GB, saving everything.\n"
          f"  TSVs  → {RAW_SAMPLE_DIR}/\n"
          f"  MP3s  → {clips_dir}/\n")

    http_resp = requests.get(download_url, stream=True, timeout=(10, 3600))
    http_resp.raise_for_status()
    stream = _StreamingResponse(http_resp, max_bytes=max_bytes)

    mp3_count = 0
    tsv_count = 0

    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                name = member.name
                basename = os.path.basename(name)

                f = tf.extractfile(member)
                if f is None:
                    continue

                if name.endswith(".tsv"):
                    out_path = os.path.join(RAW_SAMPLE_DIR, basename)
                    with open(out_path, "wb") as out:
                        out.write(f.read())
                    tsv_count += 1
                    print(f"  TSV saved:  {basename}")

                elif name.endswith(".mp3"):
                    out_path = os.path.join(clips_dir, basename)
                    with open(out_path, "wb") as out:
                        out.write(f.read())
                    mp3_count += 1
                    if mp3_count % 500 == 0:
                        gb_read = stream._bytes_read / 1e9
                        print(f"  {mp3_count} MP3s extracted  ({gb_read:.2f} GB streamed)")

    except _ByteLimitExceeded as e:
        print(f"\nByte cap reached: {e}")
    except Exception as exc:
        if "pipe" not in str(exc).lower() and "truncated" not in str(exc).lower():
            raise

    gb_read = stream._bytes_read / 1e9
    print(f"\nDone. Extracted {mp3_count} MP3s + {tsv_count} TSVs ({gb_read:.2f} GB streamed).")
    print(f"Files are in: {RAW_SAMPLE_DIR}/")
    print("\nNext steps:")
    print("  grep the TSV:  grep -i 'southern\\|california\\|midwest' raw_sample/validated.tsv | head -20")
    print("  count accents: cut -f8 raw_sample/validated.tsv | sort | uniq -c | sort -rn | head -30")


def main() -> None:
    discover_only = "--discover" in sys.argv
    extract_all   = "--extract-all" in sys.argv

    # Parse --max-gb N (float), default 10.0
    max_gb = 10.0
    if "--max-gb" in sys.argv:
        idx = sys.argv.index("--max-gb")
        try:
            max_gb = float(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("ERROR: --max-gb requires a number, e.g. --max-gb 5")
            sys.exit(1)

    api_key = _get_api_key()
    download_url = _get_download_url(api_key)

    if extract_all:
        _extract_raw_sample(download_url, max_gb=max_gb)
        return

    if not discover_only:
        for dialect in DIALECT_NAMES:
            os.makedirs(os.path.join(OUT_DIR, dialect), exist_ok=True)

    counts, manifest_rows = _stream_archive(download_url, discover_only=discover_only)

    if discover_only:
        return

    # Write manifest CSV
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "true_dialect", "accent_label", "duration_s", "common_voice_id"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Saved {len(manifest_rows)} clips to {OUT_DIR}/")
    print(f"Manifest written to {manifest_path}")
    for dialect, count in counts.items():
        print(f"  {dialect}: {count} clips")

    under = {d: int(DIALECT_CAPS[d]) - v for d, v in counts.items()
             if DIALECT_CAPS[d] != float("inf") and v < DIALECT_CAPS[d]}
    if under:
        print(
            f"\nWARNING: Some capped dialects are under-filled: {under}\n"
            "The test_dialect_classifier.py will still run with however many clips are available."
        )


if __name__ == "__main__":
    main()
