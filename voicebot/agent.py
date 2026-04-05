"""
agent.py
─────────
VoiceAgent orchestrates one complete phone call session.

Data flow per turn:
  1. Twilio sends audio chunks → on_audio_chunk() accumulates them.
     After ~3 seconds, the DialectClassifier runs and may update the
     active DialectProfile (and re-tune the TurnDetector thresholds).
  2. The STT layer (Twilio Gather / Deepgram) delivers transcript text
     + silence_ms → on_transcript_update() evaluates the turn.
  3. If TurnDecision is TALK:
       a. normalize_text() cleans the LLM response before TTS.
       b. _get_llm_response() calls GPT with the full conversation history.
       c. _synthesize_speech() streams the response back as MP3 chunks.
  4. server.py re-encodes those chunks to mulaw and sends them to Twilio.

Thread safety note:
  on_audio_chunk and on_transcript_update may both be called from the same
  asyncio event loop; no concurrent access issues. The DialectClassifier
  runs in a thread executor (see dialect_classifier.py), so self.profile
  updates are sequenced through the event loop — no locks needed.
"""

import asyncio
import re
from typing import AsyncIterator

from voicebot.client import openai_client
from voicebot.dialect_classifier import DialectClassifier
from voicebot.dialect_profiles import PROFILES, DialectProfile
from voicebot.turn_detector import TurnDecision, TurnDetector

# How many bytes of mulaw audio to accumulate before running the classifier.
# At 8 kHz, 1 byte = 1 sample = 0.125 ms.
# 24,000 bytes ≈ 3 seconds — enough audio for a stable ECAPA result.
_CLASSIFY_AFTER_BYTES = 24_000

# GPT model used for response generation.
_LLM_MODEL = "gpt-4o-mini"


class VoiceAgent:
    """
    Stateful agent for a single phone call session.

    Create one instance per call; discard it when the call ends.
    """

    def __init__(self, profile: DialectProfile = PROFILES["neutral"]) -> None:
        self.profile = profile
        self.detector = TurnDetector(
            min_silence=profile.min_silence,
            hard_limit=profile.hard_limit,
        )
        # Conversation history sent to GPT on every turn.
        # Starts with the dialect-appropriate system prompt.
        self.history: list[dict] = [
            {"role": "system", "content": profile.system_prompt_tone}
        ]
        self.transcript_buffer: str = ""

        self._classifier = DialectClassifier()
        self._audio_buffer: bytes = b""
        self._dialect_locked: bool = False   # True after 2 consistent detections
        self._detection_count: int = 0
        self._last_detected: str = profile.name

    # ── Public interface called by server.py ───────────────────────────────────

    async def on_audio_chunk(self, mulaw_bytes: bytes) -> None:
        """
        Accept a raw mulaw audio chunk from Twilio.

        Accumulates audio until _CLASSIFY_AFTER_BYTES is reached, then runs
        the dialect classifier. If the detected dialect differs from the
        current profile, the TurnDetector is re-tuned to the new thresholds.

        After 2 consistent detections, the profile locks for the call.
        """
        if self._dialect_locked:
            return

        self._audio_buffer += mulaw_bytes
        if len(self._audio_buffer) < _CLASSIFY_AFTER_BYTES:
            return

        detected_profile = await self._classifier.classify_async(
            self._audio_buffer,
            transcript=self.transcript_buffer,
        )
        self._audio_buffer = b""  # reset for next classification window

        if detected_profile.name == self._last_detected:
            self._detection_count += 1
        else:
            self._detection_count = 1
            self._last_detected = detected_profile.name

        if self._detection_count >= 2:
            self._dialect_locked = True

        if detected_profile.name != self.profile.name:
            self._update_profile(detected_profile)

    async def on_transcript_update(
        self,
        new_text: str,
        silence_ms: int,
    ) -> AsyncIterator[bytes] | None:
        """
        Called each time the STT layer delivers a new transcript segment.

        Appends text to the buffer, evaluates the turn, and — if the decision
        is TALK — calls the LLM and streams TTS audio chunks back.

        Returns:
            An async iterator of MP3 bytes if the agent should speak,
            or None if the agent should wait.
        """
        self.transcript_buffer = (self.transcript_buffer + " " + new_text).strip()
        decision = self.detector.evaluate(self.transcript_buffer, silence_ms)

        if decision != TurnDecision.TALK:
            return None

        user_text = self.transcript_buffer
        self.transcript_buffer = ""   # reset for next turn
        return self._respond(user_text)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _update_profile(self, new_profile: DialectProfile) -> None:
        """
        Switch to a new dialect profile mid-call.

        Updates the TurnDetector thresholds immediately; conversation history
        is not reset (the caller's previous turns remain in context).
        """
        self.profile = new_profile
        self.detector = TurnDetector(
            min_silence=new_profile.min_silence,
            hard_limit=new_profile.hard_limit,
        )
        # Replace only the system message (index 0) with the new tone prompt.
        self.history[0] = {"role": "system", "content": new_profile.system_prompt_tone}

    async def _respond(self, user_text: str) -> AsyncIterator[bytes]:
        """
        Full response pipeline: LLM → text normalization → TTS streaming.
        Yields raw MP3 audio chunks as they arrive from OpenAI.
        """
        llm_text = await self._get_llm_response(user_text)
        tts_text = self.normalize_text(llm_text)
        async for chunk in self._synthesize_speech(tts_text):
            yield chunk

    async def _get_llm_response(self, user_text: str) -> str:
        """
        Append the caller's utterance to history, call GPT, append the reply.

        self.history is sent in full on every API call so GPT has the entire
        conversation in context — not just the latest utterance. This is
        standard "chat completions with memory" pattern.
        """
        self.history.append({"role": "user", "content": user_text})
        response = await openai_client.chat.completions.create(
            model=_LLM_MODEL,
            messages=self.history,
        )
        reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply

    async def _synthesize_speech(self, text: str) -> AsyncIterator[bytes]:
        """
        Stream TTS audio from OpenAI using the dialect profile's voice/speed.

        stream=True (via with_streaming_response) sends the first audio chunk
        before the full sentence is synthesized — this reduces perceived
        latency on the call. The caller hears the first syllable faster.
        """
        async with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=self.profile.tts_voice,
            input=text,
            speed=self.profile.tts_speed,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Clean LLM output before sending to TTS so numbers and symbols are
        spoken naturally rather than character-by-character.

        Why this matters:
          TTS reads "$38,900" as "dollar sign thirty-eight comma nine hundred".
          After normalization it reads "thirty-eight thousand nine hundred dollars".

        Covers the most common dealership output patterns. If more cases
        arise, extract to voicebot/text_utils.py.
        """
        # Currency: $38,900 → thirty-eight thousand nine hundred dollars
        text = re.sub(
            r"\$([0-9,]+)",
            lambda m: _dollars_to_words(m.group(1)) + " dollars",
            text,
        )
        # Vehicle model numbers: F-150 → F one fifty
        text = re.sub(r"\bF-(\d+)\b", lambda m: f"F {_number_to_words(int(m.group(1)))}", text)
        # Times: 10am → ten AM, 3:30pm → three thirty PM
        text = re.sub(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
            lambda m: _time_to_words(m),
            text,
            flags=re.IGNORECASE,
        )
        # Plain numbers with commas: 42,000 → forty-two thousand
        text = re.sub(r"\b(\d{1,3}(?:,\d{3})+)\b", lambda m: _number_to_words(int(m.group(1).replace(",", ""))), text)
        # Percent: 0% → zero percent
        text = re.sub(r"\b(\d+)%", lambda m: f"{_number_to_words(int(m.group(1)))} percent", text)
        return text


# ── Text normalisation helpers ─────────────────────────────────────────────────
# Intentionally minimal — only patterns that appear in dealership responses.
# Extend as needed; keep them pure functions for easy testing.

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _number_to_words(n: int) -> str:
    """Convert a non-negative integer ≤ 999,999 to English words."""
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _number_to_words(-n)

    parts: list[str] = []
    if n >= 1_000_000:
        parts.append(_number_to_words(n // 1_000_000) + " million")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(_number_to_words(n // 1_000) + " thousand")
        n %= 1_000
    if n >= 100:
        parts.append(_ones_word(n // 100) + " hundred")
        n %= 100
    if n >= 20:
        tail = ("-" + _ones_word(n % 10)) if n % 10 else ""
        parts.append(_TENS[n // 10] + tail)
    elif n > 0:
        parts.append(_ONES[n])

    return " ".join(parts)


def _ones_word(n: int) -> str:
    return _ONES[n] if 0 < n < 20 else _TENS[n // 10] + (("-" + _ONES[n % 10]) if n % 10 else "")


def _dollars_to_words(amount_str: str) -> str:
    return _number_to_words(int(amount_str.replace(",", "")))


def _time_to_words(m: re.Match) -> str:
    hour = int(m.group(1))
    minutes = m.group(2)
    period = m.group(3).upper()
    result = _number_to_words(hour)
    if minutes and minutes != "00":
        result += " " + _number_to_words(int(minutes))
    return result + " " + period
