"""
test_dialect_classifier.py
───────────────────────────
Benchmark the DialectClassifier against the labeled audio set built by
build_dialect_eval.py.

What it does:
  1. Reads manifest.csv from Tests/dialect_eval_data/
  2. Loads each .npy clip (preprocessed 16 kHz float32 audio)
  3. Runs DialectClassifier.classify_audio_chunk() on each clip
  4. Compares the predicted dialect to the ground-truth label
  5. Prints:
       - A confusion matrix (rows = true labels, cols = predicted labels)
       - Per-dialect accuracy
       - Overall accuracy
       - Latency percentiles (avg, p90)

Run:
    python Tests/test_dialect_classifier.py

Prerequisites:
    - Run Tests/build_dialect_eval.py first to generate the eval data
    - voicebot/dialect_classifier.py and voicebot/dialect_profiles.py must exist

Reading the confusion matrix:
    Each row is the true dialect; each column is what the model predicted.
    The diagonal (true == predicted) is correct; off-diagonal cells are errors.
    Example: if row=southern col=neutral has a high count, the model is
    misclassifying Southern speakers as neutral — a specific, fixable problem.
"""

import csv
import os
import sys
import time

import numpy as np
from tabulate import tabulate

EVAL_DIR = os.path.join(os.path.dirname(__file__), "dialect_eval_data")
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.csv")
DIALECTS = ["neutral", "southern", "sandiego"]


def load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run `python Tests/build_dialect_eval.py` first to generate eval data.")
        sys.exit(1)
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    # Import here so the module can be imported without triggering model load.
    try:
        from voicebot.dialect_classifier import DialectClassifier
    except ImportError as e:
        print(f"ERROR: Could not import DialectClassifier — {e}")
        print("Make sure voicebot/dialect_classifier.py exists.")
        sys.exit(1)

    manifest = load_manifest()
    print(f"Loaded {len(manifest)} clips from manifest.\n")
    print("Initializing DialectClassifier (downloads ECAPA model on first run)...")
    classifier = DialectClassifier()
    print("Model ready.\n")

    # confusion[true][predicted] = count
    confusion: dict[str, dict[str, int]] = {
        d: {p: 0 for p in DIALECTS} for d in DIALECTS
    }
    latencies: list[float] = []

    for row in manifest:
        npy_path = os.path.join(EVAL_DIR, row["path"])
        true_dialect = row["true_dialect"]

        if not os.path.exists(npy_path):
            print(f"  WARNING: Missing clip {npy_path} — skipping")
            continue

        audio = np.load(npy_path)

        start = time.perf_counter()
        profile = classifier.classify_audio_chunk(audio, sr=16_000)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        predicted = profile.name
        if predicted not in confusion[true_dialect]:
            # Gracefully handle unexpected labels
            predicted = "neutral"
        confusion[true_dialect][predicted] += 1

    if not latencies:
        print("No clips were evaluated. Check that the eval data directory is populated.")
        sys.exit(1)

    # ── Confusion matrix ──
    print("=== DIALECT CLASSIFIER ACCURACY ===\n")
    header = ["True \\ Pred"] + DIALECTS + ["Accuracy"]
    matrix_rows = []
    per_dialect_correct = {}

    for true_d in DIALECTS:
        row_counts = [confusion[true_d][pred_d] for pred_d in DIALECTS]
        total = sum(row_counts)
        correct = confusion[true_d][true_d]
        per_dialect_correct[true_d] = (correct, total)
        acc = f"{100 * correct / total:.1f}%" if total else "N/A"
        matrix_rows.append([true_d] + row_counts + [acc])

    print(tabulate(matrix_rows, headers=header, tablefmt="grid"))

    # ── Overall accuracy ──
    total_correct = sum(confusion[d][d] for d in DIALECTS)
    total_clips = sum(sum(confusion[d].values()) for d in DIALECTS)
    overall_acc = 100 * total_correct / total_clips if total_clips else 0

    # ── Latency stats ──
    latencies.sort()
    avg_lat = sum(latencies) / len(latencies)
    p90_idx = int(0.9 * len(latencies))
    p90_lat = latencies[p90_idx]

    print(f"\nOverall accuracy: {overall_acc:.1f}%  |  Avg latency: {avg_lat:.0f}ms  |  p90: {p90_lat:.0f}ms")

    # ── Decision gate (from plan) ──
    if overall_acc < 70:
        print("\n⚠️  Accuracy < 70% — consider increasing reliance on keyword second-pass")
        print("   or fine-tuning the ECAPA model on a Texas/California-specific dataset.")
    else:
        print("\n✅ Accuracy >= 70% — ECAPA classifier is viable as a first-pass filter.")


if __name__ == "__main__":
    main()
