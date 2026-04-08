"""
call_adapter.py
────────────────
Adapts endpointing thresholds to the caller's speaking rhythm using WPM.

How it works:
  - Seeded with starting min_silence / hard_limit from the config.
  - After each committed user turn, observe_speech_rate() updates a WPM EMA.
  - Endpointing thresholds scale from the base values using WPM ratio:
      fast talker (high WPM) → thresholds shrink (tighter windows)
      slow talker  (low WPM) → thresholds grow  (more patience)
  - EMA with α=0.2 converges ~67% after 5 turns, ~89% after 10 turns.
"""

from dataclasses import dataclass


@dataclass
class AdapterConfig:
    """Starting thresholds for CallAdapter. Pass one of these at construction."""
    min_silence: int   # ms — minimum silence before spaCy check runs
    hard_limit: int    # ms — maximum silence before force-commit
    tts_speed: float   # initial TTS speed (not yet applied dynamically)


class CallAdapter:
    # EMA weight: fraction of the gap closed per observation.
    # 0.2 = slow/stable. 0.5 = fast but noisy.
    ALPHA: float = 0.2

    # Hard bounds on adapted thresholds.
    # These prevent runaway adaptation from pathological inputs
    # (e.g. caller leaves the phone for 30 seconds, inflating hard_limit).
    MIN_SILENCE_FLOOR: int = 400    # ms — below this, agent is too trigger-happy
    MIN_SILENCE_CEIL: int  = 2500   # ms — above this, caller experience feels laggy
    HARD_LIMIT_FLOOR: int  = 1000    # ms
    HARD_LIMIT_CEIL: int   = 20000  # ms — raised for testing

    # WPM baseline and scale for tts_speed mapping.
    # At WPM_BASELINE, tts_speed = 1.0.
    # Speed change = (observed_wpm - baseline) / scale.
    # With scale=500: a 100 WPM deviation → 0.2 speed change (clamped before that).
    WPM_BASELINE: float = 150.0
    WPM_SCALE: float    = 500.0
    TTS_SPEED_FLOOR: float = 0.8
    TTS_SPEED_CEIL: float  = 1.2

    # Minimum utterance length to trust for WPM estimation.
    # Very short utterances (< 0.5s or < 3 words) are too noisy to use.
    MIN_DURATION_S: float = 0.5
    MIN_WORDS: int        = 3

    def __init__(self, config: AdapterConfig) -> None:
        # _base_* are fixed reference points; WPM scaling multiplies against them.
        self._base_min_silence: float = float(config.min_silence)
        self._base_hard_limit: float  = float(config.hard_limit)
        self._min_silence: float = float(config.min_silence)
        self._hard_limit: float  = float(config.hard_limit)
        self._tts_speed: float   = config.tts_speed
        self._wpm_ema: float | None = None  # None until first speech rate observation

    # ── Observation methods ────────────────────────────────────────────────────

    def observe_pause(self, silence_ms: int, was_end_of_turn: bool) -> None:
        """
        No-op: silence-based threshold updates are disabled.

        Pause durations are unreliable when the agent interrupts the caller
        mid-turn — the measured silence includes agent speech time, which
        corrupts the EMA. Endpointing thresholds are now derived from WPM
        via observe_speech_rate() instead.
        """
        pass

    def observe_speech_rate(self, words: int, duration_s: float) -> None:
        """
        Update TTS speed based on the caller's observed speaking rate.

        Args:
            words:      Number of words in the transcript for this turn.
            duration_s: Audio duration for this turn (seconds).

        The mapping is:
          WPM = words / duration_s * 60
          speed_delta = (WPM - WPM_BASELINE) / WPM_SCALE
          tts_speed = 1.0 + speed_delta  (clamped to [0.8, 1.2])

        Examples:
          96 WPM  → delta = (96-150)/500 = -0.108 → speed ~0.89 (slow caller)
          150 WPM → delta = 0 → speed 1.0 (average)
          300 WPM → delta = (300-150)/500 = 0.3 → speed 1.2 (fast caller, clamped)
        """
        if duration_s < self.MIN_DURATION_S or words < self.MIN_WORDS:
            return  # utterance too short to trust

        wpm = (words / duration_s) * 60.0

        if self._wpm_ema is None:
            self._wpm_ema = wpm
        else:
            self._wpm_ema += self.ALPHA * (wpm - self._wpm_ema)

        speed = 1.0 + (self._wpm_ema - self.WPM_BASELINE) / self.WPM_SCALE
        self._tts_speed = max(self.TTS_SPEED_FLOOR, min(self.TTS_SPEED_CEIL, speed))

        # Scale endpointing thresholds from the profile's base values using
        # WPM ratio. Fast speakers (high WPM) produce shorter natural pauses
        # so we shrink the thresholds; slow speakers (low WPM) need more time.
        # scale > 1 for slow speakers, < 1 for fast speakers.
        # Clamped to [0.5, 2.0] to prevent extreme drift.
        scale = max(0.5, min(2.0, self.WPM_BASELINE / self._wpm_ema))
        self._min_silence = max(
            self.MIN_SILENCE_FLOOR,
            min(self.MIN_SILENCE_CEIL, self._base_min_silence * scale),
        )
        self._hard_limit = max(
            self.HARD_LIMIT_FLOOR,
            min(self.HARD_LIMIT_CEIL, self._base_hard_limit * scale),
        )
        if self._hard_limit <= self._min_silence:
            self._hard_limit = self._min_silence + 300

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def min_silence(self) -> int:
        return int(round(self._min_silence))

    @property
    def hard_limit(self) -> int:
        return int(round(self._hard_limit))

    @property
    def tts_speed(self) -> float:
        return round(self._tts_speed, 3)

    def __repr__(self) -> str:
        wpm = f"{self._wpm_ema:.0f}" if self._wpm_ema is not None else "n/a"
        return (
            f"CallAdapter(min_silence={self.min_silence}, "
            f"hard_limit={self.hard_limit}, "
            f"tts_speed={self.tts_speed}, "
            f"wpm_ema={wpm})"
        )
