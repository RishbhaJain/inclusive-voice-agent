"""
mock_call.py
─────────────
Interactive mock of a phone call — no Twilio, no ngrok, no real audio.

You type transcript lines and silence durations in the terminal.
VoiceAgent processes each input exactly as it would during a real call,
including the CallAdapter EMA updates and TurnDetector decisions.

TTS response text is printed to the terminal (not spoken aloud).
ECAPA dialect classification is skipped — the profile stays neutral
unless you pass --profile southern|sandiego.

Usage:
    python Tests/mock_call.py
    python Tests/mock_call.py --profile southern
    python Tests/mock_call.py --silence 1500   # default silence_ms per turn
    python Tests/mock_call.py --no-llm         # skip OpenAI, print "[LLM skipped]"
"""

import argparse
import asyncio
import sys
import os

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voicebot.dialect_profiles import PROFILES, get_profile
from voicebot.turn_detector import TurnDecision


# ── Minimal agent subclass that skips ECAPA and optionally skips LLM ──────────

class MockVoiceAgent:
    """
    Wraps VoiceAgent but:
      - Skips ECAPA classification (no model load, no audio needed)
      - Optionally stubs out OpenAI calls (--no-llm)
    """

    def __init__(self, profile_name: str = "neutral", skip_llm: bool = False):
        from voicebot.call_adapter import CallAdapter
        from voicebot.turn_detector import TurnDetector

        self.profile = get_profile(profile_name)
        self.detector = TurnDetector(
            min_silence=self.profile.min_silence,
            hard_limit=self.profile.hard_limit,
        )
        self.history = [{"role": "system", "content": self.profile.system_prompt_tone}]
        self.transcript_buffer = ""
        self._adapter = CallAdapter(self.profile)
        self._bytes_this_turn = 0
        self._skip_llm = skip_llm

        # Simulate ~3s of audio per turn (24000 mulaw bytes at 8kHz)
        # This makes the WPM estimate meaningful
        self._simulated_audio_bytes_per_turn = 24_000

    async def on_transcript_update(self, text: str, silence_ms: int) -> str | None:
        """Returns agent reply text, or None if WAIT."""
        self.transcript_buffer = (self.transcript_buffer + " " + text).strip()

        # Simulate audio bytes accumulated this turn
        self._bytes_this_turn = self._simulated_audio_bytes_per_turn

        decision = self.detector.evaluate(self.transcript_buffer, silence_ms)

        # Adapt pause thresholds
        self._adapter.observe_pause(silence_ms, was_end_of_turn=(decision == TurnDecision.TALK))
        self.detector.min_silence = self._adapter.min_silence
        self.detector.hard_limit = self._adapter.hard_limit

        _print_adapter_state(self._adapter, decision)

        if decision != TurnDecision.TALK:
            return None

        # Adapt speech rate
        duration_s = self._bytes_this_turn / 8_000
        words = len(self.transcript_buffer.split())
        self._adapter.observe_speech_rate(words, duration_s)
        self.profile.tts_speed = self._adapter.tts_speed

        user_text = self.transcript_buffer
        self.transcript_buffer = ""

        if self._skip_llm:
            return f"[LLM skipped — would respond to: '{user_text}']"

        return await self._get_llm_response(user_text)

    async def _get_llm_response(self, user_text: str) -> str:
        from voicebot.client import openai_client
        self.history.append({"role": "user", "content": user_text})
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
        )
        reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ── Display helpers ────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _print_adapter_state(adapter, decision) -> None:
    decision_color = GREEN if decision == TurnDecision.TALK else YELLOW
    decision_str = f"{decision_color}{decision.name}{RESET}"
    print(
        f"  {DIM}adapter:{RESET} "
        f"min_silence={CYAN}{adapter.min_silence}ms{RESET}  "
        f"hard_limit={CYAN}{adapter.hard_limit}ms{RESET}  "
        f"tts_speed={CYAN}{adapter.tts_speed:.3f}x{RESET}  "
        f"→ {decision_str}"
    )


def _print_separator() -> None:
    print(f"{DIM}{'─' * 60}{RESET}")


# ── Main loop ──────────────────────────────────────────────────────────────────

async def run(profile_name: str, default_silence: int, skip_llm: bool) -> None:
    agent = MockVoiceAgent(profile_name=profile_name, skip_llm=skip_llm)

    print(f"\n{BOLD}Mock Call — profile: {profile_name}{RESET}")
    print(f"Starting thresholds: min_silence={agent.profile.min_silence}ms  "
          f"hard_limit={agent.profile.hard_limit}ms  "
          f"tts_speed={agent.profile.tts_speed}x")
    print(f"Type a transcript line and press Enter. Optionally append :<silence_ms>")
    print(f"  e.g.  'I want to buy a truck'         (uses default {default_silence}ms)")
    print(f"  e.g.  'I want to buy a truck':2500    (uses 2500ms)")
    print(f"Type 'quit' to exit.\n")

    turn = 0
    while True:
        try:
            raw = input(f"{BOLD}You [{turn+1}]:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCall ended.")
            break

        if not raw or raw.lower() == "quit":
            print("Call ended.")
            break

        # Parse optional :<silence_ms> suffix
        if ":" in raw:
            parts = raw.rsplit(":", 1)
            text = parts[0].strip()
            try:
                silence_ms = int(parts[1].strip())
            except ValueError:
                text = raw
                silence_ms = default_silence
        else:
            text = raw
            silence_ms = default_silence

        _print_separator()
        print(f"  {DIM}silence={silence_ms}ms{RESET}")

        reply = await agent.on_transcript_update(text, silence_ms)

        if reply is not None:
            print(f"\n{BOLD}Agent:{RESET} {GREEN}{reply}{RESET}\n")
            turn += 1
        else:
            print(f"  {YELLOW}(agent waiting for more input){RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive mock phone call")
    parser.add_argument("--profile", default="neutral",
                        choices=list(PROFILES.keys()),
                        help="Starting dialect profile")
    parser.add_argument("--silence", type=int, default=1200,
                        help="Default silence_ms per turn (default: 1200)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip OpenAI API calls — just print what would be sent")
    args = parser.parse_args()

    asyncio.run(run(args.profile, args.silence, args.no_llm))


if __name__ == "__main__":
    main()
