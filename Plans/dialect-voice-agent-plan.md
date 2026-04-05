# Plan: Evaluation-First Dialect-Aware Voice Agent

## Context
The voice agent needs to sound natural for two distinct caller profiles — Southern/Texas (slower speech, longer mid-sentence pauses, warm fillers) and San Diego/California (faster, clipped, casual). Rather than guessing at TTS voices and silence thresholds, we build the evaluation framework first, use it to validate each dial, then wire the full pipeline.

The `TurnDetector` is the only complete component. `agent.py`, `server.py` are empty stubs. The TTS provider is OpenAI (`tts-1`/`tts-1-hd`), transport is Twilio Media Streams.

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `voicebot/turn_detector.py` | **Modify** — parameterize `min_silence` / `hard_limit` in `__init__` |
| `voicebot/dialect_profiles.py` | **Create** — `DialectProfile` dataclass + profile registry |
| `voicebot/dialect_classifier.py` | **Create** — ECAPA-TDNN accent classifier (HuggingFace), maps accent → `DialectProfile` |
| `voicebot/client.py` | **Create** — shared `AsyncOpenAI` singleton (avoid duplicate HTTP pools) |
| `voicebot/agent.py` | **Fill** — `VoiceAgent` class: transcript buffer, LLM call, TTS stream |
| `voicebot/server.py` | **Fill** — FastAPI + Twilio WebSocket server |
| `Tests/eval.py` | **Modify** — add `dialect` tag + `min_silence_override` field, new cases, per-dialect summary table |
| `Tests/test_tts.py` | **Create** — TTS latency benchmark + audio file output |
| `requirements.txt` | **Modify** — remove stdlib entries, add new packages |

---

## Phase 1 — Parameterize `TurnDetector` *(do this first — unblocks everything else)*

**File:** `voicebot/turn_detector.py`

Change `__init__` signature only — no other method changes needed:
```python
def __init__(self, min_silence: int = 800, hard_limit: int = 2000):
    self.nlp = spacy.load("en_core_web_sm")
    self.min_silence = min_silence
    self.hard_limit = hard_limit
```

**Why default values?** Python default arguments mean existing code that calls `TurnDetector()` with no arguments still works exactly as before. You're making the class *configurable* without breaking anything that already uses it. This is the principle of backwards-compatible interfaces — add flexibility while preserving existing behavior.

**Why does this unblock everything?** Every downstream piece — the dialect eval, the dialect profiles, the agent — needs to create a `TurnDetector` with *different* thresholds per dialect. Right now that's impossible because the values are hardcoded. This one 2-line change is the keystone.

---

## Phase 2 — Expand `Tests/eval.py` for Dialect Coverage

Add two new fields to each test case dict:
```python
{
    "text": "...",
    "silence": 1050,
    "expected": TurnDecision.WAIT,
    "dialect": "southern",         # "southern" | "sandiego" | "neutral"
    "min_silence_override": 1200   # optional — creates a detector with this threshold instead of 800
}
```

**Why `min_silence_override`?** The point of this eval is to test whether *different thresholds* produce correct decisions for different dialects. A Southern speaker who pauses for 1050ms mid-sentence should get WAIT — but only if the detector is set to 1200ms. With the default 800ms, that same 1050ms pause would trigger TALK (a bug). The override lets you test the *combination* of input + threshold in one place.

**Why `dialect` as a tag?** So the runner can group results and compute per-dialect pass rates. You want to know "my Southern cases pass at 96%" independently from "my neutral cases pass at 99%". This turns one big eval into a per-profile report.

**New test cases (~40 additional):**

*Southern (min_silence_override=1200):*
- Incomplete sentences at 1050–1150ms silence → WAIT (validates 1200ms threshold holds)
- Complete sentences at 1300–1500ms silence → TALK
- Multi-turn conversation sequences with `conversation_id` field

*San Diego (min_silence_override=650):*
- Complete clipped sentences at 680–700ms → TALK
- Mid-sentence fragments at 580–620ms → WAIT (validates 650ms threshold doesn't fire early)

**Runner changes:**
- Accept optional `--dialect` CLI flag to filter cases
- When `min_silence_override` present, instantiate a `TurnDetector` with that value
- Print three tables: aggregate, per-dialect summary, failures only

**Target output:**
```
=== BY DIALECT ===
Dialect   | Passed | Failed | Total | Pass Rate
----------|--------|--------|-------|----------
neutral   | 92     | 1      | 93    | 98.9%
southern  | 28     | 1      | 29    | 96.6%
sandiego  | 18     | 0      | 18    | 100.0%
```

---

## Phase 3 — `Tests/build_dialect_eval.py` + `Tests/test_dialect_classifier.py`

### Where the labeled audio comes from

**Mozilla Common Voice 13.0** (HuggingFace: `mozilla-foundation/common_voice_13_0`, `"en"` config) has an `accent` field per clip:
- `"Southern United States"` → label as `"southern"`
- `"Western United States"` → label as `"sandiego"` (closest available proxy)
- `"General American English"` → label as `"neutral"`

**Why Common Voice?** It's a free, openly licensed dataset of real human speech donated by volunteers worldwide. The `accent` field was self-reported by contributors, so it's a reliable ground-truth label. Crucially, you don't have to record anyone or pay for data — you just download a filtered slice programmatically.

### `Tests/build_dialect_eval.py` — one-time dataset builder

```python
# What this script does:
# 1. Streams the Common Voice English test split from HuggingFace (~500MB, cached after first run)
# 2. Filters for clips whose `accent` field matches our ACCENT_MAP
# 3. Resamples each clip from its native 48kHz MP3 → 16kHz float32 numpy array
#    (16kHz is what the ECAPA model was trained on — using 48kHz would give garbage results)
# 4. Saves each clip as a .npy file so we don't have to re-download or re-resample in future runs
# 5. Writes a manifest.csv that maps each file path to its ground-truth dialect label

from datasets import load_dataset
import librosa, numpy as np

TARGET_PER_DIALECT = 50
ACCENT_MAP = {
    "Southern United States": "southern",
    "Western United States": "sandiego",
    "General American English": "neutral",
}
```

Run once: `python Tests/build_dialect_eval.py`

**Why save as .npy instead of keeping the MP3?** Numpy arrays load in microseconds. MP3 decoding takes 20–100ms per file. When you run the classifier eval 100 times during development, that adds up to minutes of wasted I/O. Save the preprocessed form, not the raw audio.

### `Tests/test_dialect_classifier.py` — classifier accuracy benchmark

```python
# What this script does:
# 1. Reads manifest.csv (built by build_dialect_eval.py)
# 2. Loads each .npy clip and feeds it to DialectClassifier.classify_audio_chunk()
# 3. Compares predicted dialect to true dialect label
# 4. Prints confusion matrix + per-dialect accuracy + latency percentiles
```

**Target output:**
```
=== DIALECT CLASSIFIER ACCURACY ===
True \ Pred  | neutral | southern | sandiego |
-------------|---------|----------|----------|
neutral      |   42    |    5     |    3     |  84%
southern     |    4    |   41     |    5     |  82%
sandiego     |    6    |    7     |   37     |  74%

Overall accuracy: 80.0%  |  Avg latency: 112ms  |  p90: 148ms
```

**Why a confusion matrix?** It tells you *which* errors are happening, not just how many. If the model confuses "sandiego" with "neutral" 90% of the time, that's a fixable problem (maybe the keyword second-pass can resolve it). If it confuses "southern" with "sandiego", that's a different problem entirely — the model can't separate the two US accents.

> **Decision gate:** If overall accuracy < 70%, we rely more heavily on the keyword second-pass and defer fine-tuning to a later iteration.

**New packages for this phase:** `datasets>=2.20` (streams HuggingFace datasets), `soundfile>=0.12` (audio decoding)

---

## Phase 4 — `Tests/test_tts.py`: TTS Voice Evaluation Framework

**Purpose:** Pick the right agent `voice` + `speed` for each dialect *profile* by listening, not guessing.

> **What the MP3 files are:** These are the *agent's synthesized voice* speaking 20 dealership responses in 6 different OpenAI TTS configs. You listen to them and decide "this config sounds right when speaking *to* a Southern caller." There is no caller audio here — this is purely evaluating TTS output quality.

**Config matrix (6 configs × 20 responses = 120 audio files):**
```python
CONFIGS = [
    {"voice": "alloy",   "speed": 1.0},   # neutral baseline — the OpenAI default
    {"voice": "nova",    "speed": 1.0},   # female, friendly — neutral candidate
    {"voice": "onyx",    "speed": 0.9},   # deep/warm, slowed slightly — Southern candidate
    {"voice": "fable",   "speed": 0.85},  # storytelling quality, slower — Southern candidate
    {"voice": "shimmer", "speed": 1.1},   # lighter, upbeat, faster — SD candidate
    {"voice": "echo",    "speed": 1.15},  # male, faster — SD candidate
]
```

**Why this approach instead of just picking a voice?** OpenAI gives you 6 voices and a speed knob — that's a small enough grid to exhaustively test by ear. Human listening is the gold standard for TTS quality. There's no automated metric that reliably predicts whether a voice feels natural for a specific regional audience.

**Key functions:**
```python
async def generate_and_save(client, text, voice, speed, out_dir, response_id) -> dict:
    # `streaming=True` on the TTS call — this means we start receiving audio bytes
    # before the whole sentence is synthesized. We record time_to_first_chunk_ms
    # (the pause before audio starts) separately from total_ms (full synthesis time).
    # TTFC is the number that matters most for perceived latency on a phone call.
    # Saves as: tts_output/r01_onyx_s0.85.mp3

async def run_tts_benchmark(out_dir: str = "tts_output") -> None:
    # Iterates CONFIGS × corpus, collects timing data, prints latency table, saves tts_summary.csv
```

**Dealership response corpus (20 sentences):** short confirmations, price quotes, warranty explanations, one Southern-toned, one SD-toned response.

**Output:** `tts_output/` with 120 `.mp3` files + `tts_summary.csv`. Console table: avg TTFC, p90 TTFC per config.

> **Blocker:** Do not commit `tts_voice`/`tts_speed` values into `dialect_profiles.py` until you've listened to the Phase 4 output and chosen your preferred configs.

---

## Phase 5 — `voicebot/dialect_profiles.py` + `voicebot/dialect_classifier.py`

### 5a — `dialect_profiles.py` (pure data — no logic)

```python
from dataclasses import dataclass

@dataclass
class DialectProfile:
    name: str
    min_silence: int      # how long to wait before triggering NLP check (ms)
    hard_limit: int       # silence duration that forces TALK unconditionally (ms)
    tts_voice: str        # which OpenAI voice to use when speaking to this caller
    tts_speed: float      # speaking rate multiplier (1.0 = normal)
    system_prompt_tone: str  # injected into LLM system prompt to match caller's register

# Values filled in after Phase 2 (silence thresholds) and Phase 4 (voice/speed) results
PROFILES = {
    "neutral":  DialectProfile(min_silence=800,  hard_limit=2000, tts_voice="alloy",   tts_speed=1.0, ...),
    "southern": DialectProfile(min_silence=1200, hard_limit=3000, tts_voice="onyx",    tts_speed=0.9, ...),
    "sandiego": DialectProfile(min_silence=650,  hard_limit=1600, tts_voice="shimmer", tts_speed=1.1, ...),
}
```

**Why a dataclass?** It's a plain Python data container with no logic. Keeping data separate from logic (the classifier, the agent) makes it easy to change one without touching the other. If you want to add a new dialect, you add one entry here — nothing else needs to change.

**Why `system_prompt_tone`?** The LLM response should mirror the caller's register. If a Southern caller is relaxed and unhurried, the agent saying "Certainly! I'd be happy to assist you with that." feels robotic. "Sure thing, let me look that up for ya." feels human. The prompt tone encodes that expected register.

### 5b — `voicebot/dialect_classifier.py` (ML-based dialect detection)

**Model:** `Jzuluaga/accent-id-commonaccent_ecapa`
- Architecture: ECAPA-TDNN — originally designed for speaker *verification* (who is speaking?) adapted for accent *classification* (where is the speaker from?)
- 16 English accent classes
- **~50–150ms inference on CPU** — fast enough for real-time
- ~20 MB model

**Why ECAPA over the larger Wav2Vec2 or Whisper models?** Two reasons. First, latency: ECAPA runs in 50–150ms; Wav2Vec2-large runs in 300–800ms; Whisper-large is 1000ms+. For a phone call, adding 1 second to the first turn is very noticeable. Second, size: 20MB loads fast on startup; 1.3GB does not. For an agent that needs to be ready when the call connects, startup time matters.

**The two-stage detection system:**
1. ECAPA classifies the caller's accent into one of 16 classes
2. Within the "US" class, keyword scan on the first 20-word transcript (y'all/fixin → southern, like/totally/hella → sandiego) acts as a second signal

**Why two stages?** No pre-trained HuggingFace model separates Texas from California — both cluster under a generic "US" label in every available dataset. ECAPA gives you a reliable first filter (is this caller US-accented? British? Indian?), and the keyword scan does the intra-US disambiguation that the model can't do alone.

**mulaw → model input pipeline:**
```
Twilio sends: base64-encoded mulaw at 8kHz

Step 1: base64.b64decode(data) → raw mulaw bytes
  Why: Twilio encodes audio as base64 for JSON transport. Decode it first.

Step 2: audioop.ulaw2lin(mulaw_bytes, 2) → PCM int16
  Why: mulaw is a compressed audio format (G.711). ulaw2lin expands it back to
  linear PCM — the raw amplitude values the model expects.

Step 3: librosa.resample(pcm_float32, orig_sr=8000, target_sr=16000)
  Why: The ECAPA model was trained on 16kHz audio. Feeding it 8kHz audio would
  be like showing a face-recognition model a blurry photo — technically valid
  input but the model would perform poorly. librosa handles the anti-aliasing
  filter automatically (naive upsampling without filtering creates artifacts).

Step 4: model inference → softmax → top label → map to DialectProfile
```

**Why run this in a thread executor?**
```python
async def classify_async(self, mulaw_bytes: bytes) -> DialectProfile:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.classify_audio_chunk, ...)
```
PyTorch inference is CPU-bound (blocking). In an async program, a blocking operation on the event loop freezes *everything* — no WebSocket messages can be processed, no audio can be streamed. `run_in_executor` moves the blocking work to a thread pool, letting the event loop continue handling other I/O while the model runs.

**When classification fires:** After the first ~3 seconds of caller audio. After 2 consistent predictions, the profile locks for the rest of the call.

---

## Phase 6 — `voicebot/agent.py`

```python
class VoiceAgent:
    def __init__(self, profile: DialectProfile = PROFILES["neutral"]):
        self.profile = profile
        self.detector = TurnDetector(min_silence=profile.min_silence, hard_limit=profile.hard_limit)
        self.history: list[dict] = [{"role": "system", "content": profile.system_prompt_tone}]
        self.transcript_buffer: str = ""

    async def on_transcript_update(self, new_text: str, silence_ms: int) -> AsyncIterator[bytes] | None:
        # Appends to buffer, runs TurnDetector
        # If TALK: calls _get_llm_response → normalize_text → _synthesize_speech, yields audio chunks

    async def on_audio_chunk(self, mulaw_bytes: bytes) -> None:
        # Accumulates audio; after ~3s calls DialectClassifier.classify_async()
        # If returned profile differs, re-initializes self.detector with new thresholds
        # This means the agent can "correct itself" mid-call if dialect becomes clear

    async def _get_llm_response(self, user_text: str) -> str:
        # Appends user turn to self.history, calls GPT, appends assistant turn
        # self.history acts as the conversation memory — sent in full on each API call
        # so GPT has context of the whole conversation, not just the latest utterance

    async def _synthesize_speech(self, text: str) -> AsyncIterator[bytes]:
        # Streams TTS using profile.tts_voice and profile.tts_speed
        # `stream=True` means audio chunks arrive as they're synthesized
        # This reduces perceived latency — caller hears first syllable faster

    @staticmethod
    def normalize_text(text: str) -> str:
        # "$38,900" -> "thirty-eight thousand nine hundred dollars"
        # "F-150" -> "F one fifty", "10am" -> "ten AM"
        # Why: TTS reads "$38,900" as "dollar sign thirty-eight comma nine hundred"
        # Simple regex replacements, ~25 lines inline. Extract only if it grows beyond that.
```

**Shared client — `voicebot/client.py`:**
```python
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_client = AsyncOpenAI()
```
**Why a singleton?** `AsyncOpenAI()` opens an HTTP connection pool. If both `agent.py` and `turn_detector.py` each call `AsyncOpenAI()`, you get two connection pools to the same service — wasted resources. A shared singleton means one pool, imported wherever needed.

---

## Phase 7 — `voicebot/server.py` (Twilio WebSocket)

**Why FastAPI over raw websockets?** Twilio requires *both* an HTTP POST endpoint (to receive the incoming call webhook and return TwiML) and a WebSocket endpoint (to stream audio). FastAPI handles both cleanly in one app. Raw `websockets` library only handles WebSockets — you'd need a second framework for the HTTP side.

```python
@app.post("/incoming-call")
# Returns TwiML XML that tells Twilio: "connect this call to our WebSocket at /media-stream"
# TwiML is Twilio's XML-based instruction language — it's how you program Twilio's behavior

@app.websocket("/media-stream")
# Receives JSON frames from Twilio. Each frame has an "event" field:
#   "connected" — WebSocket handshake done
#   "media"     — contains base64 mulaw audio chunk
#   "stop"      — call ended
# For each "media" event: pass audio to agent.on_audio_chunk()
# For each transcript update: call agent.on_transcript_update()
# When agent yields TTS audio chunks: re-encode and send back to Twilio
```

**mulaw pipeline (outbound TTS → Twilio):**
```
OpenAI TTS → MP3 bytes (44.1kHz stereo)
           → pydub.AudioSegment.from_mp3() — decode MP3 to PCM
           → .set_channels(1).set_frame_rate(8000) — resample to 8kHz mono
             (Why 8kHz? That's what the phone network supports. Higher sample rates
             get downsampled by the network anyway — you'd just be wasting bandwidth.)
           → audioop.lin2ulaw(pcm_bytes, 2) — compress back to mulaw
           → base64.b64encode() — encode for JSON transport
           → send as Twilio "media" event JSON frame
```

> **Open question:** STT (speech-to-text) is not yet in scope. `server.py` scaffolds around Twilio's built-in `<Gather input="speech">` or a Deepgram streaming sidecar. This decision needs to be made before `server.py` can be fully wired. Both options have trade-offs (latency vs. cost vs. accuracy).

---

## Execution Order

```
Phase 1 (parameterize TurnDetector) — do this first
  ├─→ Phase 2 (turn detector eval expansion — tests dialect-specific thresholds)
  └─→ Phase 5a (dialect_profiles — silence thresholds informed by Phase 2 results)
         ├─→ Phase 5b (dialect_classifier — ECAPA model, imports profiles)
         └─→ Phase 6 (agent.py — imports profiles + classifier + detector)
                └─→ Phase 7 (server.py — feeds raw audio to on_audio_chunk)

Phase 3 (build_dialect_eval → test_dialect_classifier) — independent, run in parallel
  Uses Common Voice audio to measure ECAPA accuracy before committing to it

Phase 4 (test_tts.py) — independent, run in parallel with Phase 2+3
  Generates 120 MP3 files; you listen and pick voice/speed per dialect
  → informs tts_voice/tts_speed values committed into Phase 5a
```

---

## New Packages Required

```
# existing — keep
spacy>=3.7
openai>=1.30
python-dotenv>=1.0
tabulate>=0.9

# TTS benchmark (async file writes)
aiofiles>=23.0

# Dialect eval set builder (streams HuggingFace Common Voice dataset)
datasets>=2.20
soundfile>=0.12       # decodes MP3 audio from Common Voice

# Dialect classifier (HuggingFace ECAPA-TDNN model)
speechbrain>=1.0
torch>=2.0
transformers>=4.30
librosa>=0.10         # high-quality audio resampling (8kHz → 16kHz)

# Server + Twilio
fastapi>=0.111
uvicorn[standard]>=0.30
websockets>=12.0
twilio>=9.0
pydub>=0.25           # MP3 → PCM resampling for Twilio outbound audio
audioop-lts>=0.2.1    # Python 3.12+ shim for stdlib audioop (mulaw codec)
```

Remove from `requirements.txt`: `asyncio`, `time`, `enum` — these are Python stdlib, they should never be in a requirements file (pip can't install them and the entries just cause confusion).

Post-install: `python -m spacy download en_core_web_sm` — this downloads the spaCy language model (~12MB). It's a data download, not a pip package, so it can't go in requirements.txt.

The ECAPA model (~20MB) auto-downloads to `~/.cache/huggingface/` on first `DialectClassifier()` instantiation.

---

## Verification (run in order)

```bash
# 1. Turn detector eval — should pass all 100 existing neutral cases
python -m Tests.eval

# 2. Dialect eval data — downloads Common Voice, saves ~150 .npy clips
python Tests/build_dialect_eval.py

# 3. Classifier accuracy — runs ECAPA on the 150 labeled clips
python Tests/test_dialect_classifier.py
# Expected: ~80% overall. If < 70%, we increase reliance on keyword second-pass.

# 4. TTS voice comparison — generates 120 MP3s, print latency table
python Tests/test_tts.py --out-dir tts_output
open tts_output/   # listen: r01_onyx_s0.85.mp3 vs r01_shimmer_s1.1.mp3

# 5. Smoke test dialect classifier directly
python -c "
import numpy as np
from voicebot.dialect_classifier import DialectClassifier
dc = DialectClassifier()
dummy = np.zeros(16000 * 3, dtype=np.float32)  # 3s of silence
print(dc.classify_audio_chunk(dummy))           # should return neutral profile
"

# 6. Smoke test full agent pipeline (no server, no audio — just text in, audio chunks out)
python -c "
import asyncio
from voicebot.agent import VoiceAgent
from voicebot.dialect_profiles import PROFILES
async def t():
    agent = VoiceAgent(profile=PROFILES['southern'])
    result = await agent.on_transcript_update('I want to buy a truck today.', 1500)
    async for chunk in result:
        print(f'received audio chunk: {len(chunk)} bytes')
asyncio.run(t())
"

# 7. Start server (needs ngrok or a public URL for Twilio webhook)
uvicorn voicebot.server:app --host 0.0.0.0 --port 8000
```
