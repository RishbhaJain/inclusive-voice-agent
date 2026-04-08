# LiveKit + Twilio SIP Setup

## Step 1: LiveKit Cloud account

1. Sign up at https://cloud.livekit.io (free tier: 50 hours/month)
2. Create a project
3. Go to Settings → Keys → Create new key pair
4. Add to your `.env`:
   ```
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=APIxxxx
   LIVEKIT_API_SECRET=your-secret
   ```

## Step 2: Deepgram API key

1. Sign up at https://deepgram.com (free: $200 credit, no credit card required)
2. Dashboard → API Keys → Create a key
3. Add to `.env`:
   ```
   DEEPGRAM_API_KEY=your-deepgram-key
   ```

## Step 3: Connect Twilio to LiveKit via SIP

LiveKit Cloud has a built-in SIP server. Twilio will forward calls to it.

### 3a. Create a LiveKit SIP Inbound Trunk

In the LiveKit Cloud dashboard:
- Go to SIP → Inbound Trunks → Create
- Note the SIP URI LiveKit gives you (looks like `xxxxx.sip.livekit.cloud`)

### 3b. Create a SIP Dispatch Rule

The dispatch rule tells LiveKit which agent worker to send calls to.
Run this once (requires livekit-server-sdk or use the Cloud dashboard):

```bash
# Using LiveKit CLI (install with: brew install livekit-io/homebrew-livekit/lk)
lk sip dispatch create \
  --type individual-dispatch \
  --room-prefix "call-" \
  --metadata '{"type": "dealership"}'
```

Or create it in the LiveKit Cloud dashboard under SIP → Dispatch Rules.

### 3c. Configure Twilio to forward calls to LiveKit

In Twilio Console:
1. Phone Numbers → Manage → your number
2. Voice Configuration → "A call comes in"
3. Select **SIP Endpoint** (not Webhook)
4. Enter: `sip:your-trunk-id@xxxxx.sip.livekit.cloud`

## Step 4: Run the agent worker

```bash
# Dev mode (connects to LiveKit Cloud, logs locally, restarts on file changes)
python -m voicebot.lk_agent dev

# Production (persistent, no auto-restart)
python -m voicebot.lk_agent start
```

The worker connects to LiveKit Cloud and waits for jobs. When a call arrives via
the SIP trunk, LiveKit dispatches it to your worker and calls `entrypoint()`.

## Step 5: Test without a real phone call

Use the LiveKit CLI to simulate a participant joining a room:

```bash
# In terminal 1: start the worker
python -m voicebot.lk_agent dev

# In terminal 2: simulate a room join (triggers entrypoint)
lk room join --room lk-call-test --identity test-caller
```

The worker will run the entrypoint, say the greeting, and wait for audio.
You can use LiveKit Meet or the CLI to send audio to test the STT pipeline.

## What the old files (server.py, agent.py) are for

`voicebot/server.py` and `voicebot/agent.py` are the old Gather-only Twilio
architecture (3–6s latency). They are **not used** by the new LiveKit worker.
Keep them around until LiveKit is confirmed working, then delete them.
