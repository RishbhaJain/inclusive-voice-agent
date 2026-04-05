"""
build_dialect_eval.py
─────────────────────
One-time script that builds a labeled audio evaluation set for the dialect
classifier by pulling clips from Mozilla Common Voice 13.0.

What it does, step by step:
  1. Streams the Common Voice English test split from HuggingFace.
     (~500MB on first run; HuggingFace caches it at ~/.cache/huggingface/)
  2. Filters for clips whose `accent` field matches our three dialect labels.
  3. Resamples each clip from 48 kHz (Common Voice native) → 16 kHz float32.
     Why 16 kHz? That's the sample rate the ECAPA model was trained on.
     Feeding it 48 kHz audio would distort its frequency perception.
  4. Saves each clip as a .npy file (numpy array) instead of an MP3.
     Why .npy? Loading a numpy file takes microseconds; decoding an MP3
     takes 20–100 ms. When you run the classifier benchmark many times
     during development, this difference adds up to minutes of wasted I/O.
  5. Writes a manifest.csv that maps each file path to its ground-truth
     dialect label, accent string, duration, and Common Voice clip ID.

Run once:
    python Tests/build_dialect_eval.py

Output layout:
    Tests/dialect_eval_data/
        southern/   clip_000.npy  clip_001.npy  ...
        sandiego/   clip_000.npy  ...
        neutral/    clip_000.npy  ...
        manifest.csv
"""

import csv
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import io

TARGET_PER_DIALECT = 50
TARGET_SR = 16_000  # Hz — required by the ECAPA-TDNN model

# Map Common Voice accent labels → our three dialect profiles.
# "Western United States" is the closest proxy for San Diego/California
# in the Common Voice taxonomy — there's no "California" label.
ACCENT_MAP = {
    "Southern United States": "southern",
    "Western United States": "sandiego",
    "General American English": "neutral",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "dialect_eval_data")


def resample_audio_bytes(audio_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Decode raw audio bytes (MP3/OGG from Common Voice) into a float32
    mono numpy array at `target_sr` Hz.

    librosa.load handles:
      - Format decoding (calls ffmpeg/audioread under the hood)
      - Channel mixing (stereo → mono via averaging)
      - Resampling with anti-aliasing (avoids aliasing artifacts that
        naive upsampling would introduce)
    """
    audio_file = io.BytesIO(audio_bytes)
    y, _ = librosa.load(audio_file, sr=target_sr, mono=True)
    return y


def main() -> None:
    # Lazy import: datasets is only needed here, not in the classifier itself.
    # This avoids pulling in the heavy HuggingFace datasets library at import
    # time in production code.
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets soundfile")
        sys.exit(1)

    print("Loading Common Voice 13.0 English test split...")
    print("(First run downloads ~500MB — subsequent runs use the cache.)\n")

    # trust_remote_code=False is safe here; Common Voice is a well-known dataset.
    # split="test" is intentional: we want held-out clips, not training data,
    # so our evaluation reflects real-world generalization.
    ds = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "en",
        split="test",
        trust_remote_code=False,
    )

    # Tally how many clips we've collected per dialect.
    counts: dict[str, int] = {d: 0 for d in ACCENT_MAP.values()}
    manifest_rows: list[dict] = []

    # Create output directories.
    for dialect in ACCENT_MAP.values():
        os.makedirs(os.path.join(OUT_DIR, dialect), exist_ok=True)

    print(f"Scanning dataset for accent labels: {list(ACCENT_MAP.keys())}")
    print(f"Target: {TARGET_PER_DIALECT} clips per dialect\n")

    for row in ds:
        accent = row.get("accent", "").strip()
        dialect = ACCENT_MAP.get(accent)

        if dialect is None:
            continue  # accent not in our map — skip
        if counts[dialect] >= TARGET_PER_DIALECT:
            continue  # already have enough for this dialect

        # Common Voice rows expose audio as a dict with "array" (float32 at
        # the dataset's native sample rate) and "sampling_rate".
        audio_dict = row["audio"]
        y_native = np.array(audio_dict["array"], dtype=np.float32)
        native_sr = audio_dict["sampling_rate"]

        # Resample to 16 kHz if needed.
        if native_sr != TARGET_SR:
            y = librosa.resample(y_native, orig_sr=native_sr, target_sr=TARGET_SR)
        else:
            y = y_native

        duration_s = len(y) / TARGET_SR
        clip_idx = counts[dialect]
        filename = f"clip_{clip_idx:03d}.npy"
        out_path = os.path.join(OUT_DIR, dialect, filename)

        np.save(out_path, y)
        counts[dialect] += 1

        manifest_rows.append({
            "path": os.path.relpath(out_path, OUT_DIR),
            "true_dialect": dialect,
            "accent_label": accent,
            "duration_s": f"{duration_s:.2f}",
            "common_voice_id": row.get("client_id", ""),
        })

        print(f"  [{dialect:>8}] {counts[dialect]:>3}/{TARGET_PER_DIALECT}  {filename}  ({duration_s:.1f}s)")

        # Stop early if all dialects are filled.
        if all(v >= TARGET_PER_DIALECT for v in counts.values()):
            break

    # Write manifest CSV.
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_dialect", "accent_label", "duration_s", "common_voice_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Saved {len(manifest_rows)} clips to {OUT_DIR}/")
    print(f"Manifest written to {manifest_path}")
    for dialect, count in counts.items():
        print(f"  {dialect}: {count} clips")


if __name__ == "__main__":
    main()
