"""
server.py
──────────
FastAPI application that connects an incoming Twilio phone call to the
VoiceAgent pipeline.

Why FastAPI over raw websockets?
  Twilio requires both an HTTP POST endpoint (to receive the incoming call
  webhook and return TwiML) and a WebSocket endpoint (to stream audio).
  FastAPI handles both in one app with clean routing.

Call flow:
  1. Twilio calls POST /incoming-call (configured in your Twilio number settings).
  2. We return TwiML that tells Twilio to open a Media Stream WebSocket to
     /media-stream on this server.
  3. Twilio connects the WebSocket and streams base64-encoded mulaw audio
     frames as JSON events.
  4. We:
       a. Decode each audio frame and pass it to VoiceAgent.on_audio_chunk()
          (dialect classification).
       b. When Twilio delivers a transcription (via Gather), call
          VoiceAgent.on_transcript_update() to evaluate the turn.
       c. When the agent responds, re-encode MP3 → mulaw and send back over
          the WebSocket as a Twilio media event.
  5. When Twilio sends "stop", we close the session.

STT note:
  Speech-to-text is handled by Twilio's built-in <Gather input="speech">
  in the TwiML. This means Twilio transcribes the audio server-side and
  posts the result to /transcription. For lower latency, this can be
  replaced with a Deepgram streaming sidecar — that decision is tracked
  in the implementation plan.

Audio format:
  Inbound  (Twilio → us): mulaw, 8 kHz, mono, base64-encoded in JSON
  Outbound (us → Twilio): mulaw, 8 kHz, mono, base64-encoded in JSON

Run:
    uvicorn voicebot.server:app --host 0.0.0.0 --port 8000

Then expose with ngrok and configure your Twilio number webhook to:
    https://<ngrok-url>/incoming-call
"""

import audioop
import base64
import io
import json

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydub import AudioSegment
from twilio.twiml.voice_response import Start, Stream, VoiceResponse

from voicebot.agent import VoiceAgent
from voicebot.dialect_profiles import PROFILES

app = FastAPI()


@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    """
    Twilio calls this endpoint when a call arrives.

    We return TwiML that:
      1. Opens a Media Stream WebSocket to /media-stream (for raw audio).
      2. Uses <Gather> to transcribe speech and POST results to /transcription.

    The `track="inbound_track"` tells Twilio to stream only the caller's
    audio, not the agent's outbound audio (which we send ourselves).
    """
    host = request.headers.get("host", "localhost")
    ws_url = f"wss://{host}/media-stream"

    response = VoiceResponse()

    # Start streaming audio to our WebSocket
    start = Start()
    stream = Stream(url=ws_url)
    stream.parameter(name="track", value="inbound_track")
    start.append(stream)
    response.append(start)

    # Gather transcribes speech and POSTs to /transcription
    # timeout=1: wait up to 1s of silence before sending partial result
    # speechTimeout="auto": let Twilio decide when the utterance ends
    # Note: we do NOT rely solely on Twilio's end-of-speech detection —
    # VoiceAgent.on_transcript_update() applies our own TurnDetector logic.
    response.gather(
        input="speech",
        action="/transcription",
        method="POST",
        timeout=1,
        speech_timeout="auto",
        language="en-US",
    )

    return Response(content=str(response), media_type="application/xml")


@app.post("/transcription")
async def transcription(request: Request) -> Response:
    """
    Twilio POSTs transcribed speech here after each Gather completes.

    We extract the transcript + speech duration, then look up the active
    agent session for this CallSid and drive the turn evaluation.

    Note: this endpoint is stateless per request; the session state lives
    in the VoiceAgent instance stored in _active_sessions.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "")
    transcript = form.get("SpeechResult", "").strip()
    # SpeechDuration is in seconds; convert to ms for TurnDetector
    speech_duration_ms = int(float(form.get("SpeechDuration", 0)) * 1000)

    if not transcript or call_sid not in _active_sessions:
        return Response(content="<Response/>", media_type="application/xml")

    agent: VoiceAgent = _active_sessions[call_sid]["agent"]
    ws: WebSocket | None = _active_sessions[call_sid].get("ws")

    # The silence after speaking is approximately total silence observed by
    # Twilio's speech_timeout. We use speech_duration_ms as a proxy.
    audio_gen = await agent.on_transcript_update(transcript, speech_duration_ms)

    if audio_gen is not None and ws is not None:
        # Stream TTS audio back to Twilio over the active WebSocket
        import asyncio
        asyncio.create_task(_stream_tts_to_twilio(audio_gen, ws, call_sid))

    return Response(content="<Response/>", media_type="application/xml")


# In-memory session store: CallSid → {agent, ws}
# For production, replace with Redis or another shared store if running
# multiple server replicas. Single-instance deployments are fine with this.
_active_sessions: dict[str, dict] = {}


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    """
    WebSocket endpoint for Twilio Media Streams.

    Twilio sends JSON frames with an "event" field:
      "connected"  — WebSocket handshake complete; Twilio sends metadata
      "start"      — Stream is starting; contains CallSid and stream metadata
      "media"      — Contains base64-encoded mulaw audio chunk
      "stop"       — Call has ended

    We:
      - Create a VoiceAgent on "start"
      - Pass each audio chunk to agent.on_audio_chunk() for dialect detection
      - Clean up the session on "stop"
    """
    await ws.accept()
    call_sid: str | None = None

    try:
        async for raw_message in ws.iter_text():
            message = json.loads(raw_message)
            event = message.get("event")

            if event == "connected":
                # Twilio confirms the WebSocket is open. No action needed.
                pass

            elif event == "start":
                # Extract CallSid and create a fresh agent for this call.
                # All calls start with the neutral profile; the DialectClassifier
                # will update it after ~3 seconds of audio.
                call_sid = message["start"]["callSid"]
                _active_sessions[call_sid] = {
                    "agent": VoiceAgent(profile=PROFILES["neutral"]),
                    "ws": ws,
                }

            elif event == "media" and call_sid:
                # Decode base64 → raw mulaw bytes and pass to the agent.
                # The agent accumulates audio and runs dialect classification
                # after enough bytes have arrived.
                b64_payload = message["media"]["payload"]
                mulaw_bytes = base64.b64decode(b64_payload)
                agent: VoiceAgent = _active_sessions[call_sid]["agent"]
                await agent.on_audio_chunk(mulaw_bytes)

            elif event == "stop":
                # Call ended — clean up session.
                if call_sid and call_sid in _active_sessions:
                    del _active_sessions[call_sid]
                break

    except WebSocketDisconnect:
        if call_sid and call_sid in _active_sessions:
            del _active_sessions[call_sid]


async def _stream_tts_to_twilio(
    audio_gen,
    ws: WebSocket,
    call_sid: str,
) -> None:
    """
    Re-encode MP3 TTS audio chunks → mulaw 8 kHz → base64 → Twilio JSON frame
    and send over the active WebSocket.

    Why this re-encoding is necessary:
      OpenAI TTS outputs MP3 at 44.1 kHz stereo.
      Twilio's Media Streams only accept mulaw at 8 kHz mono.
      Steps:
        1. pydub decodes MP3 → raw PCM
        2. .set_frame_rate(8000).set_channels(1) resamples and mixes to mono
           (pydub calls ffmpeg/libav under the hood for the resampling)
        3. audioop.lin2ulaw compresses PCM int16 → mulaw
        4. base64.b64encode wraps for JSON transport

    Resampling runs on the event loop thread here (via pydub, which is
    synchronous). For very long responses, consider moving this to an
    executor to avoid blocking. For typical dealership responses (<15s),
    the blocking time is acceptably short.
    """
    mp3_buffer = bytearray()

    async for chunk in audio_gen:
        mp3_buffer.extend(chunk)

    if not mp3_buffer:
        return

    # Decode MP3 and resample to 8 kHz mono mulaw
    segment = AudioSegment.from_file(io.BytesIO(bytes(mp3_buffer)), format="mp3")
    segment = segment.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm_bytes = segment.raw_data
    mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    b64_audio = base64.b64encode(mulaw_bytes).decode("utf-8")

    # Twilio media event format
    payload = json.dumps({
        "event": "media",
        "streamSid": call_sid,   # Twilio uses streamSid to route audio
        "media": {"payload": b64_audio},
    })

    try:
        await ws.send_text(payload)
    except Exception:
        # WebSocket may have closed between response generation and send.
        # Silently discard — the call has already ended.
        pass
