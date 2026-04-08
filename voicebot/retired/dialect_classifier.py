"""
dialect_classifier.py
──────────────────────
Audio-based dialect classifier using the ECAPA-TDNN model from HuggingFace.

Model: Jzuluaga/accent-id-commonaccent_ecapa
  - Architecture: ECAPA-TDNN, originally designed for speaker verification,
    adapted here for accent classification.
  - 16 English accent classes (US, British, Australian, Indian, etc.)
  - ~20 MB — loads quickly on startup, unlike Wav2Vec2-large (~1.3 GB)
  - ~50–150ms inference on CPU — within budget for real-time phone calls

Two-stage detection:
  Stage 1 — ECAPA classifies the caller's accent from raw audio.
             This reliably separates US from non-US callers.
  Stage 2 — Keyword scan on the transcript provides intra-US disambiguation.
             No pre-trained model cleanly separates Texas from California;
             both fall under a generic "US" label. The keyword scan bridges
             this gap using dialect markers in the transcript.

Why run inference in a thread executor?
  PyTorch inference is CPU-bound (blocking). Calling it directly on the
  asyncio event loop freezes all WebSocket I/O until it completes. Moving
  it to run_in_executor lets the event loop keep processing Twilio frames
  while the model runs in a background thread.

Audio input format:
  - Sample rate: 16,000 Hz (model requirement)
  - Encoding: float32, mono
  - Minimum recommended length: ~3 seconds (longer = more stable result)

Twilio sends mulaw at 8 kHz. The conversion pipeline is:
    base64 → mulaw bytes → audioop.ulaw2lin → PCM int16
    → numpy float32 → librosa.resample(8000 → 16000) → classify
"""

import asyncio
import audioop
import base64
from functools import lru_cache

import librosa
import numpy as np

from voicebot.dialect_profiles import PROFILES, DialectProfile, get_profile

# ── Accent label → dialect profile mapping ─────────────────────────────────────
# ECAPA's 16-class output labels come from CommonAccent.
# Both "Southern United States" and "Western United States" map to "us" in
# most CommonAccent taxonomy versions. The keyword second-pass below handles
# intra-US disambiguation.
ECAPA_TO_DIALECT: dict[str, str] = {
    "us":          "neutral",   # disambiguated further by keyword scan
    "england":     "neutral",   # default non-US to neutral for this deployment
    "australia":   "neutral",
    "canada":      "neutral",
    "ireland":     "neutral",
    "scotland":    "neutral",
    "wales":       "neutral",
    "newzealand":  "neutral",
    "india":       "neutral",
    "bermuda":     "neutral",
    "philippines": "neutral",
    "hongkong":    "neutral",
    "malaysia":    "neutral",
    "singapore":   "neutral",
    "southatlandtic": "neutral",
    "african":     "neutral",
}

# ── Keyword markers for intra-US dialect disambiguation ────────────────────────
# Applied as a second-pass when ECAPA returns "us" to narrow down to
# southern or sandiego within the US accent class.
# Thresholds are intentionally conservative:
#   Southern requires 2 hits (y'all, fixin, etc. are rare in non-Southern speech)
#   SD requires 3 hits (like/totally appear in neutral speech frequently)
SOUTHERN_MARKERS = frozenset({
    "y'all", "yall", "fixin", "reckon", "bless", "yonder", "ain't",
    "holler", "fixin'", "dadgum", "howdy", "doggone",
})
SANDIEGO_MARKERS = frozenset({
    "like", "totally", "dude", "stoked", "super", "hella",
    "basically", "legit", "gnarly", "bro",
})
SOUTHERN_THRESHOLD = 2
SANDIEGO_THRESHOLD = 3


@lru_cache(maxsize=1)
def _load_model():
    """
    Load the ECAPA-TDNN model from SpeechBrain/HuggingFace.
    Result is cached so the model is only loaded once per process.
    Downloads ~20 MB to ~/.cache/huggingface/ on first call.
    """
    try:
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        raise ImportError(
            "speechbrain is required for dialect classification. "
            "Install it with: pip install speechbrain"
        )
    return EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
    )


class DialectClassifier:
    """
    Classifies the dialect/accent of a caller from raw audio.

    Usage (sync, from a test script):
        dc = DialectClassifier()
        audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s silence
        profile = dc.classify_audio_chunk(audio, sr=16000)

    Usage (async, from the agent/server):
        dc = DialectClassifier()
        profile = await dc.classify_async(mulaw_bytes)
    """

    def __init__(self) -> None:
        # Trigger model load at construction time so the first classify call
        # doesn't incur the ~2s download/load latency mid-conversation.
        self._model = _load_model()

    def classify_audio_chunk(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        transcript: str = "",
    ) -> DialectProfile:
        """
        Classify dialect from a float32 audio array.

        Args:
            audio:      float32 numpy array at `sr` Hz (mono).
            sr:         sample rate of `audio`. Resampled to 16 kHz if needed.
            transcript: optional transcript text for keyword second-pass.

        Returns:
            DialectProfile for the detected dialect.
        """
        if sr != 16_000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)

        # SpeechBrain expects a torch tensor. We convert via numpy to avoid
        # importing torch at the top level (keeps import time fast when the
        # classifier is not used).
        import torch
        waveform = torch.tensor(audio).unsqueeze(0)  # shape: (1, time)

        # classify_batch returns (out_prob, score, index, label)
        _, _, _, labels = self._model.classify_batch(waveform)
        ecapa_label = labels[0].lower().strip()

        # Stage 1: map ECAPA label to a dialect bucket
        dialect_name = ECAPA_TO_DIALECT.get(ecapa_label, "neutral")

        # Stage 2: keyword scan to split "us" into southern / sandiego
        if dialect_name == "neutral" and transcript:
            dialect_name = self._keyword_disambiguate(transcript)

        return get_profile(dialect_name)

    def _keyword_disambiguate(self, transcript: str) -> str:
        """
        Scan transcript for dialect markers and return the best-fit dialect name.
        Returns "neutral" if no clear signal is found.
        """
        words = transcript.lower().split()
        southern_hits = sum(1 for w in words if w in SOUTHERN_MARKERS)
        sandiego_hits = sum(1 for w in words if w in SANDIEGO_MARKERS)

        if southern_hits >= SOUTHERN_THRESHOLD:
            return "southern"
        if sandiego_hits >= SANDIEGO_THRESHOLD:
            return "sandiego"
        return "neutral"

    async def classify_async(
        self,
        mulaw_bytes: bytes,
        transcript: str = "",
    ) -> DialectProfile:
        """
        Async wrapper for classify_audio_chunk that decodes Twilio mulaw audio.

        The blocking classification work runs in a thread executor so the
        asyncio event loop is not blocked while the model runs.

        Args:
            mulaw_bytes: raw mulaw-encoded audio at 8 kHz (as received from Twilio).
            transcript:  accumulated transcript so far (for keyword second-pass).
        """
        loop = asyncio.get_event_loop()
        audio = _mulaw_to_float32(mulaw_bytes)
        return await loop.run_in_executor(
            None,
            self.classify_audio_chunk,
            audio,
            8_000,   # mulaw from Twilio is always 8 kHz; librosa will resample
            transcript,
        )


def _mulaw_to_float32(mulaw_bytes: bytes) -> np.ndarray:
    """
    Convert raw mulaw bytes (G.711 u-law) to a float32 numpy array.

    Steps:
      1. audioop.ulaw2lin: decompresses mulaw → linear PCM int16
         (the '2' means 2 bytes per sample = 16-bit)
      2. np.frombuffer: wraps the bytes as a numpy int16 array (no copy)
      3. astype(float32) / 32768.0: normalises int16 range [-32768, 32767]
         to float32 range [-1.0, 1.0], which is what librosa/torch expect
    """
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


def decode_base64_mulaw(b64_payload: str) -> bytes:
    """
    Decode a base64-encoded mulaw payload from a Twilio media event JSON frame.
    Twilio wraps audio in base64 for JSON transport; this undoes that.
    """
    return base64.b64decode(b64_payload)
