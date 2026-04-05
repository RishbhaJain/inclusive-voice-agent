"""
test_tts.py
────────────
Generates MP3 audio files for every combination of (TTS config × response)
so you can listen and choose which voice/speed feels right for each dialect
profile. Also measures time-to-first-chunk (TTFC) latency, which determines
how long a caller waits before they hear the agent start speaking.

What it does:
  - For each of 6 TTS configs and 20 dealership responses, calls OpenAI TTS
    with streaming enabled, records TTFC and total synthesis time, and saves
    the full audio as an MP3 file.
  - Prints a per-config latency summary table.
  - Writes a tts_summary.csv for further analysis.

Output layout:
    tts_output/
        r01_alloy_s1.0.mp3
        r01_onyx_s0.9.mp3
        ...
        r20_shimmer_s1.1.mp3
        tts_summary.csv

Why TTFC matters:
    On a phone call, the caller hears silence until the first audio chunk
    arrives. Total synthesis time is less important than TTFC — a voice that
    starts speaking after 200ms and streams smoothly feels more responsive
    than one that buffers for 600ms before playing.

Run:
    python Tests/test_tts.py
    python Tests/test_tts.py --out-dir my_output_dir

Then listen:
    open tts_output/   # macOS — opens Finder; click any .mp3 to play
    # Compare r01_onyx_s0.85.mp3 vs r01_shimmer_s1.1.mp3 for the same sentence
"""

import argparse
import asyncio
import csv
import os
import time

import aiofiles
from tabulate import tabulate
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── Config matrix ──────────────────────────────────────────────────────────────
# 6 configs × 20 responses = 120 audio files.
# Candidate assignments (to be confirmed by listening after running this script):
#   Southern profile → onyx s0.9 or fable s0.85 (warm, unhurried)
#   SD profile       → shimmer s1.1 or echo s1.15 (upbeat, faster)
#   Neutral          → alloy s1.0 or nova s1.0
CONFIGS = [
    {"voice": "alloy",   "speed": 1.0},   # neutral baseline — OpenAI default
    {"voice": "nova",    "speed": 1.0},   # female, friendly
    {"voice": "onyx",    "speed": 0.9},   # deep/warm, slowed — Southern candidate
    {"voice": "fable",   "speed": 0.85},  # storytelling quality, slower — Southern candidate
    {"voice": "shimmer", "speed": 1.1},   # lighter/upbeat, faster — SD candidate
    {"voice": "echo",    "speed": 1.15},  # male, faster — SD candidate
]

# ── Dealership response corpus ─────────────────────────────────────────────────
# 20 representative agent responses covering the range of sentence lengths and
# phonetic complexity the agent will actually produce.
# Includes Southern-toned and SD-toned variants so you can hear how the same
# voice handles different registers.
CORPUS = [
    # Short confirmations
    "Sure thing, let me pull that up for you.",
    "Absolutely, I can help you with that.",
    "Got it. One moment while I check.",

    # Price and inventory
    "The 2024 F-150 XLT starts at thirty-eight thousand nine hundred dollars and we have three in stock right now.",
    "We have the Silverado in black, white, and silver. The crew cab is the most popular.",
    "That Tacoma has forty-two thousand miles and it's priced at twenty-nine thousand.",

    # Scheduling
    "I can get you in for a test drive this Saturday at ten AM. Does that work?",
    "Our service department is open Monday through Saturday, seven AM to six PM.",

    # Warranty and financing
    "Our certified pre-owned program covers any vehicle under sixty thousand miles, includes a one-hundred-and-seventy-two point inspection, and comes with a twelve-month bumper-to-bumper warranty.",
    "We're currently offering zero percent financing for sixty months on select models.",
    "With your trade-in, your monthly payment would come out to around four hundred and twenty dollars.",

    # Trade-in and deals
    "I'd love to take a look at your trade-in. Can you bring it by this week?",
    "We're running a summer clearance event right now — you could save up to three thousand dollars off MSRP.",

    # Empathetic / conversational
    "That's a great question. Let me get the exact answer from our finance team.",
    "I completely understand. Take all the time you need.",

    # Southern-toned (warm, patient register — tests how voices handle this style)
    "Well, I appreciate you calling in today. Let me see what we can do for ya.",
    "We'd love to have you come on in and take a look. We've got some real nice options right now.",

    # SD-toned (casual, clipped — tests faster, lighter voices)
    "Totally, I can check that out for you right now.",
    "Yeah for sure, that one's still available. Want me to hold it?",

    # Longer closing
    "Is there anything else I can help you with today? We want to make sure you have everything you need before you come in.",
]


async def generate_and_save(
    client: AsyncOpenAI,
    text: str,
    voice: str,
    speed: float,
    out_dir: str,
    response_id: int,
) -> dict:
    """
    Stream TTS audio for one (text, voice, speed) combination.

    Returns a dict with timing data and file metadata.

    Why stream instead of a single request?
        OpenAI's TTS streaming endpoint begins sending audio chunks as soon
        as the first few hundred milliseconds of audio are synthesized.
        Recording TTFC (time to first chunk) lets us distinguish voices that
        start fast from ones that buffer longer before responding — a critical
        difference for phone call UX.
    """
    filename = f"r{response_id:02d}_{voice}_s{speed}.mp3"
    out_path = os.path.join(out_dir, filename)

    chunk_count = 0
    total_bytes = 0
    ttfc_ms: float | None = None

    request_start = time.perf_counter()

    async with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=speed,
        response_format="mp3",
    ) as response:
        async with aiofiles.open(out_path, "wb") as f:
            async for chunk in response.iter_bytes(chunk_size=4096):
                if chunk_count == 0:
                    # Record wall-clock time until first byte of audio data.
                    ttfc_ms = (time.perf_counter() - request_start) * 1000
                await f.write(chunk)
                total_bytes += len(chunk)
                chunk_count += 1

    total_ms = (time.perf_counter() - request_start) * 1000

    return {
        "response_id": response_id,
        "voice": voice,
        "speed": speed,
        "filename": filename,
        "ttfc_ms": round(ttfc_ms or 0, 1),
        "total_ms": round(total_ms, 1),
        "bytes": total_bytes,
        "text_preview": text[:60],
    }


async def run_tts_benchmark(out_dir: str = "tts_output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    client = AsyncOpenAI()

    print(f"Generating {len(CONFIGS)} configs × {len(CORPUS)} responses = {len(CONFIGS) * len(CORPUS)} MP3 files")
    print(f"Output directory: {out_dir}/\n")

    all_results: list[dict] = []

    for config in CONFIGS:
        voice = config["voice"]
        speed = config["speed"]
        config_results: list[dict] = []

        print(f"Voice: {voice} | Speed: {speed}")
        for i, text in enumerate(CORPUS, start=1):
            result = await generate_and_save(client, text, voice, speed, out_dir, i)
            config_results.append(result)
            all_results.append(result)
            print(f"  r{i:02d}  TTFC: {result['ttfc_ms']:.0f}ms  Total: {result['total_ms']:.0f}ms  {result['filename']}")

        avg_ttfc = sum(r["ttfc_ms"] for r in config_results) / len(config_results)
        print(f"  → Avg TTFC: {avg_ttfc:.0f}ms\n")

    # ── Per-config latency summary ──
    summary_rows = []
    for config in CONFIGS:
        v, s = config["voice"], config["speed"]
        rows = [r for r in all_results if r["voice"] == v and r["speed"] == s]
        ttfc_vals = sorted(r["ttfc_ms"] for r in rows)
        avg_ttfc = sum(ttfc_vals) / len(ttfc_vals)
        p90_ttfc = ttfc_vals[int(0.9 * len(ttfc_vals))]
        avg_total = sum(r["total_ms"] for r in rows) / len(rows)
        summary_rows.append([f"{v} s{s}", f"{avg_ttfc:.0f}ms", f"{p90_ttfc:.0f}ms", f"{avg_total:.0f}ms"])

    print("=== TTS LATENCY SUMMARY (averaged over all responses) ===")
    print(tabulate(
        summary_rows,
        headers=["Config", "Avg TTFC", "p90 TTFC", "Avg Total"],
        tablefmt="grid",
    ))

    # ── Write CSV ──
    csv_path = os.path.join(out_dir, "tts_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["response_id", "voice", "speed", "filename", "ttfc_ms", "total_ms", "bytes", "text_preview"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nFull results written to {csv_path}")
    print(f"\nNext step: open {out_dir}/ and listen to the files.")
    print("Compare r01_onyx_s0.9.mp3 vs r01_shimmer_s1.1.mp3 for the same sentence,")
    print("then choose which voice/speed to assign to each dialect profile.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tts_output", help="Directory for MP3 files and CSV")
    args = parser.parse_args()
    asyncio.run(run_tts_benchmark(out_dir=args.out_dir))
