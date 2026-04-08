"""
server.py
──────────
FastAPI application that connects an incoming Twilio phone call to the
VoiceAgent pipeline using a Gather-only architecture.

Call flow (one turn):
  1. Twilio calls POST /incoming-call → we return TwiML with <Say> greeting
     and <Gather input="speech"> to collect the caller's first utterance.
  2. Caller speaks → Twilio STT transcribes and POSTs to /transcription.
  3. We run the VoiceAgent (TurnDetector + CallAdapter + GPT) and get reply text.
  4. We return TwiML: <Say> the reply, then another <Gather> for the next turn.
  5. Repeat from step 2 until the caller hangs up.

Why Gather-only instead of Media Streams?
  Media Streams + Gather is a common source of confusion:
  - Twilio does not play WebSocket audio back during an active <Gather>.
  - After <Gather> redirects to the action URL, the call leaves the original
    TwiML context — a second <Gather> is needed to keep the conversation going.
  Gather-only avoids both issues: each /transcription response simply contains
  the agent's reply (<Say>) and a fresh <Gather> for the next turn.

  Trade-off: TTS voice is Twilio's built-in Polly (Joanna) rather than
  OpenAI TTS. The adapter still runs fully — pause length and WPM are observed
  on every turn — you just can't hear tts_speed changes yet.

Run:
    uvicorn voicebot.server:app --host 0.0.0.0 --port 8000 --reload

Expose with ngrok:
    ngrok http 8000

Configure Twilio number webhook (Voice > A call comes in):
    https://<ngrok-url>/incoming-call   [HTTP POST]
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

from voicebot.agent import VoiceAgent
from voicebot.dialect_classifier import _load_model
from voicebot.dialect_profiles import PROFILES

log = logging.getLogger("voicebot")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Load ECAPA at startup in a thread so it doesn't block the first request.
    Without this, the first call triggers a ~2s model load that risks
    hitting Twilio's 5s webhook timeout.
    """
    log.info("Pre-loading ECAPA accent model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)
    log.info("✅ ECAPA ready. Server accepting calls.")
    yield
    log.info("Server shutting down.")


app = FastAPI(lifespan=lifespan)

# In-memory session store: CallSid → VoiceAgent
# One agent per call; discarded when Twilio sends no further requests.
_active_sessions: dict[str, VoiceAgent] = {}

# Twilio Polly voice for agent responses.
# Joanna (US English, neural) is the closest to a natural dealership voice
# available without OpenAI TTS. Change to any Twilio-supported Polly voice.
_TTS_VOICE = "Polly.Joanna"

# Greeting spoken when a new call arrives.
_GREETING = "Hello, welcome to the dealership. How can I help you today?"


def _error_twiml(msg: str) -> Response:
    """Return a TwiML <Say> with a safe fallback message instead of a 500.
    Twilio reads this aloud to the caller rather than saying 'application error'."""
    r = VoiceResponse()
    r.say(msg)
    return Response(content=str(r), media_type="application/xml")


@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    """
    Twilio calls this endpoint when a call arrives.

    Returns TwiML that:
      1. Says the greeting.
      2. Opens a <Gather> to collect the caller's first utterance via STT.

    The action="/transcription" tells Twilio where to POST the transcript
    after the caller finishes speaking (speech_timeout="auto").
    """
    try:
        form = await request.form()
        call_sid = str(form.get("CallSid", "unknown"))
        log.info("📞 Incoming call — CallSid %s", call_sid[:8])

        # Create session eagerly so /transcription always finds one
        if call_sid not in _active_sessions:
            _active_sessions[call_sid] = VoiceAgent(profile=PROFILES["neutral"])

        response = VoiceResponse()
        response.say(_GREETING, voice=_TTS_VOICE)
        response.gather(
            input="speech",
            action="/transcription",
            method="POST",
            speech_timeout="auto",
            language="en-US",
        )
        twiml = str(response)
        log.info("    Returning TwiML (%d bytes)", len(twiml))
        return Response(content=twiml, media_type="application/xml")

    except Exception:
        log.exception("❌ /incoming-call crashed")
        return _error_twiml("Sorry, we encountered a technical issue. Please call back.")


@app.post("/transcription")
async def transcription(request: Request) -> Response:
    """
    Twilio POSTs here after each <Gather> captures an utterance.

    We:
      1. Extract transcript + speech duration from the form data.
      2. Run VoiceAgent.respond_text() — adapter + TurnDetector + GPT.
      3. Return TwiML: <Say> the reply (if TALK) + a fresh <Gather> for
         the next turn.

    Why always append a fresh <Gather>?
      Without it Twilio ends the call after reading the response. The fresh
      <Gather> keeps the call alive and listens for the next utterance.
    """
    try:
        form = await request.form()
        call_sid   = str(form.get("CallSid", ""))
        transcript = str(form.get("SpeechResult", "")).strip()
        duration_s = float(form.get("SpeechDuration", 0) or 0)
        silence_ms = int(duration_s * 1000)

        log.info("🗣  [%s] %r  (duration=%.1fs → %dms)",
                 call_sid[:8], transcript, duration_s, silence_ms)

        if call_sid not in _active_sessions:
            log.warning("    No session for %s — creating one", call_sid[:8])
            _active_sessions[call_sid] = VoiceAgent(profile=PROFILES["neutral"])

        agent = _active_sessions[call_sid]
        reply_text: str | None = None

        if transcript:
            reply_text = await agent.respond_text(transcript, silence_ms)
        else:
            log.info("    (empty transcript — skipping LLM)")

        response = VoiceResponse()
        if reply_text:
            log.info("🤖  Agent: %r", reply_text[:120])
            response.say(reply_text, voice=_TTS_VOICE)
        else:
            log.info("    (agent waiting — TurnDetector said WAIT)")

        # Always re-open Gather so the call keeps going
        response.gather(
            input="speech",
            action="/transcription",
            method="POST",
            speech_timeout="auto",
            language="en-US",
        )

        return Response(content=str(response), media_type="application/xml")

    except Exception:
        log.exception("❌ /transcription crashed")
        return _error_twiml("Sorry, I had trouble processing that. Could you repeat?")
