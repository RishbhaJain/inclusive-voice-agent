"""
lk_agent.py
────────────
LiveKit Agents 1.x worker for the voice agent.

Architecture:
  Twilio number → Twilio SIP Trunk → LiveKit SIP Bridge → this worker
  (see docs/livekit_setup.md for SIP configuration steps)

What runs per call:
  1. AgentSession handles bidirectional audio over WebRTC/SIP.
  2. Deepgram transcribes in real-time; Silero VAD detects pauses.
  3. SpacyTurnDetector holds the turn when the user's last utterance is a
     syntactic fragment — LiveKit waits up to max_delay before force-committing.
  4. CallAdapter tracks the caller's WPM and scales min/max endpointing delays
     so fast talkers get tighter windows and slow talkers get more room.

Run (dev mode — connects to LiveKit Cloud, logs locally):
    python -m voicebot.lk_agent dev

Run (production — persistent worker):
    python -m voicebot.lk_agent start

Required .env keys:
    LIVEKIT_URL          wss://your-project.livekit.cloud
    LIVEKIT_API_KEY      APIxxxx
    LIVEKIT_API_SECRET   your-secret
    DEEPGRAM_API_KEY     your-deepgram-key
    OPENAI_API_KEY       your-openai-key
"""

import logging
import time

from dotenv import load_dotenv, find_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent,
    JobContext,
    UserStateChangedEvent,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, openai, silero

from voicebot.call_adapter import AdapterConfig, CallAdapter
from voicebot.turn_detector import SpacyTurnDetector
from voicebot.vehicle_context import fetch_context_by_vehicle, format_vehicle_context

load_dotenv(find_dotenv())

log = logging.getLogger("voicebot")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)

GREETING = (
    "Hello, welcome to the service center. "
    "What can I help you with today, and what vehicle are you calling about?"
)
LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a professional, helpful service assistant for a car dealership. "
    "Be concise and friendly — callers are on the phone, so keep responses brief.\n\n"
    "When a caller mentions their vehicle make, model, and year, immediately call the "
    "`lookup_vehicle_recalls` tool to check for open safety recalls and owner complaints. "
    "Do not wait — call it as soon as you have all three values.\n\n"
    "When presenting recall results:\n"
    "- Recalls found: state each one clearly, note repairs are free at any authorized dealer, "
    "and offer to help schedule service.\n"
    "- No recalls: say so explicitly (e.g. 'Good news — no open recalls on file for your vehicle.').\n"
    "- Tool fails: say you're having trouble reaching the recall database and offer to connect "
    "them with a service advisor.\n\n"
    "Never speculate about recalls from your training knowledge — only report what the tool returns."
)


class DealershipAgent(Agent):
    """
    Agent subclass that adds a lookup_vehicle_recalls tool.
    The LLM calls this tool as soon as the caller identifies their vehicle.
    """

    def __init__(self, tts_speed: float = 1.0) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
            tts=openai.TTS(voice="alloy", speed=tts_speed),
        )

    @llm.function_tool
    async def lookup_vehicle_recalls(
        self,
        make: str,
        model: str,
        year: str,
    ) -> str:
        """
        Look up open safety recalls and owner complaints for the caller's vehicle
        from the NHTSA database. Call this as soon as the caller provides their
        vehicle make, model, and year.

        Args:
            make:  Vehicle manufacturer, e.g. Toyota, Ford, Honda.
            model: Vehicle model name, e.g. RAV4, F-150, Civic.
            year:  Four-digit model year, e.g. 2021.
        """
        log.info("Fetching NHTSA context for %s %s %s", year, make, model)
        ctx = await fetch_context_by_vehicle(make, model, year)
        result = format_vehicle_context(ctx)
        log.info("NHTSA context fetched — urgency=%s recalls=%d", ctx["urgency"], len(ctx["recalls"]))
        return result


async def entrypoint(ctx: JobContext) -> None:
    """
    Entry point called by the LiveKit worker for each incoming phone call.
    One entrypoint invocation = one complete phone call session.
    """
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    config = AdapterConfig(min_silence=800, hard_limit=20000, tts_speed=1.0)
    adapter = CallAdapter(config)

    # Timestamps for WPM measurement: track when the caller starts and stops speaking.
    speech_started_at: float | None = None
    speech_stopped_at: float | None = None

    session = AgentSession(
        stt=deepgram.STT(),
        vad=silero.VAD.load(),
        llm=openai.LLM(model=LLM_MODEL),
        turn_handling={
            "turn_detection": SpacyTurnDetector(),
            "endpointing": {
                "min_delay": config.min_silence / 1000.0,
                "max_delay": config.hard_limit / 1000.0,
            },
        },
    )

    agent = DealershipAgent(tts_speed=config.tts_speed)

    @session.on("user_state_changed")
    def on_user_state_changed(event: UserStateChangedEvent) -> None:
        nonlocal speech_started_at, speech_stopped_at

        if event.old_state == "listening" and event.new_state == "speaking":
            speech_started_at = time.monotonic()
            speech_stopped_at = None

        elif event.old_state == "speaking" and event.new_state == "listening":
            speech_stopped_at = time.monotonic()

    @session.on("conversation_item_added")
    def on_item_added(event: ConversationItemAddedEvent) -> None:
        nonlocal speech_started_at, speech_stopped_at
        item = event.item
        if not hasattr(item, "role") or item.role != "user":
            return

        if speech_started_at is not None and speech_stopped_at is not None:
            duration_s = speech_stopped_at - speech_started_at
            content = getattr(item, "content", None)
            if isinstance(content, str):
                transcript = content
            elif isinstance(content, list):
                transcript = " ".join(
                    c if isinstance(c, str) else getattr(c, "text", "")
                    for c in content
                )
            else:
                transcript = ""
            words = len(transcript.split())
            adapter.observe_speech_rate(words, duration_s)
            _apply_adapter(session, adapter)

        speech_started_at = None
        speech_stopped_at = None

    participant = await ctx.wait_for_participant()
    log.info("Participant joined: %s", participant.identity)

    await session.start(agent, room=ctx.room)
    await session.say(GREETING, allow_interruptions=True)


def _apply_adapter(session: AgentSession, adapter: CallAdapter) -> None:
    session.update_options(
        min_endpointing_delay=adapter.min_silence / 1000.0,
        max_endpointing_delay=adapter.hard_limit / 1000.0,
    )
    log.info("Adapter update: %s", adapter)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="dealership-agent"))
