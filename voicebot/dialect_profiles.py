"""
dialect_profiles.py
────────────────────
Pure data module — no logic, no imports beyond dataclasses.

A DialectProfile bundles every per-dialect tunable into one object so that
all configurable knobs live in one place. Adding a new dialect (or changing
a threshold) requires editing only this file — nothing else needs to change.

Profile values are intentionally left as reasonable starting points.
They should be updated after running:
  - Tests/eval.py (to validate silence thresholds)
  - Tests/test_tts.py (to pick tts_voice / tts_speed by listening)
"""

from dataclasses import dataclass


@dataclass
class DialectProfile:
    name: str

    # Turn-detection thresholds (milliseconds).
    # min_silence: how long the caller must be quiet before the NLP check runs.
    #   Lower  → respond faster (good for quick SD speakers).
    #   Higher → wait longer (avoids cutting off slow Southern mid-sentence pauses).
    # hard_limit: if silence exceeds this, TALK unconditionally — no NLP needed.
    #   Longer for Southern callers who may trail off for several seconds.
    min_silence: int
    hard_limit: int

    # OpenAI TTS settings.
    # tts_voice: one of alloy | nova | onyx | fable | shimmer | echo.
    #   Chosen by listening to Tests/test_tts.py output — not by guessing.
    #   Placeholder values below; update after running the TTS benchmark.
    # tts_speed: float in [0.25, 4.0]. 1.0 = normal rate.
    tts_voice: str
    tts_speed: float

    # Injected as the system message in every LLM call for this profile.
    # Shapes the vocabulary and register of the agent's responses to match
    # the caller's expectations without changing the core prompt logic.
    system_prompt_tone: str


# ── Profile registry ───────────────────────────────────────────────────────────
# Thresholds: informed by Tests/eval.py dialect pass rates.
# Voice/speed: placeholders — update after running Tests/test_tts.py and listening.

PROFILES: dict[str, DialectProfile] = {
    "neutral": DialectProfile(
        name="neutral",
        min_silence=800,
        hard_limit=2000,
        tts_voice="alloy",    # TODO: confirm after TTS benchmark
        tts_speed=1.0,
        system_prompt_tone=(
            "You are a professional and helpful car dealership assistant. "
            "Be concise, clear, and friendly."
        ),
    ),
    "southern": DialectProfile(
        name="southern",
        min_silence=1200,
        hard_limit=3000,
        tts_voice="onyx",     # TODO: confirm after TTS benchmark (fable is the alt)
        tts_speed=0.9,
        system_prompt_tone=(
            "You are a warm, patient car dealership assistant speaking with a Southern customer. "
            "Be personable and unhurried. Use plain, approachable language. "
            "Never rush or interrupt."
        ),
    ),
    "sandiego": DialectProfile(
        name="sandiego",
        min_silence=650,
        hard_limit=1600,
        tts_voice="shimmer",  # TODO: confirm after TTS benchmark (echo is the alt)
        tts_speed=1.1,
        system_prompt_tone=(
            "You are an upbeat, casual car dealership assistant speaking with a California customer. "
            "Keep responses concise and conversational. Match the caller's relaxed energy."
        ),
    ),
}


def get_profile(name: str) -> DialectProfile:
    """Return the named profile, defaulting to neutral if unrecognised."""
    return PROFILES.get(name, PROFILES["neutral"])
