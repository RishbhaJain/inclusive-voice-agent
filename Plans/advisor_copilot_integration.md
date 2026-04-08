# Integration Plan: Advisor AI Upsell Copilot → Voice Agent

## Context

The voicebot (`voicebot/lk_agent.py`) is a LiveKit phone agent for a car dealership — it currently has a generic greeting and a blank-slate LLM with no customer or vehicle context. The Advisor AI Upsell Copilot (`advisor_ai_upsell_copilo/src/app/services.py`) generates personalized briefings per VIN using NHTSA recall/complaint data + customer pitch profiles.

Goal: When a caller mentions their vehicle (make/model/year), fetch NHTSA recall and complaint data and inject it as context into the LLM — so the agent can answer recall questions and surface relevant service recommendations.

**No VIN collection needed.** NHTSA's `recallsByVehicle` API works with make/model/year alone. VIN-specific filtering only adds value if you need to check whether a specific unit already had recall work done, which requires dealer system access — out of scope.

---

## Integration Architecture (Single Phase)

### Flow

```
Caller: "I drive a 2021 Toyota RAV4"
    ↓
LLM extracts: make=Toyota, model=RAV4, year=2021
    ↓
[Background async call]
fetch_context_by_vehicle(make, model, year)
    ├─ GET /recalls/recallsByVehicle?make=Toyota&model=RAV4&modelYear=2021
    └─ GET /complaints/complaintsByVehicle?make=Toyota&model=RAV4&modelYear=2021
    ↓
Inject structured context into session.chat_ctx as a system message:
    - Open recalls (up to 3): campaign, component, consequence
    - Top complaints (up to 3): component, frequency
    - Urgency: HIGH (recall present) / LOW (none)
    ↓
LLM now answers recall questions from context:
Caller: "Are there any open recalls?"
Agent: "Yes, I found one open recall on the 2021 RAV4 — it involves the 
        fuel pump and could cause the engine to stall. Toyota will fix 
        this free of charge at any authorized dealer. Want me to help 
        you schedule that?"
```

---

## Concrete Code Changes

### New module: `voicebot/vehicle_context.py`
Single async function:
```python
async def fetch_context_by_vehicle(
    make: str, model: str, year: str, http_client: httpx.AsyncClient
) -> dict:
    """
    Returns:
    {
        "vehicle": {"make": ..., "model": ..., "year": ...},
        "recalls": [{"campaign": ..., "component": ..., "consequence": ...}, ...],  # up to 3
        "complaints": [{"component": ..., "count": ...}, ...],  # top 3 by frequency
        "urgency": "HIGH" | "MEDIUM" | "LOW"
    }
    """
```

Internally reuses `fetch_recalls()` and `fetch_complaints()` from `advisor_ai_upsell_copilo/src/app/services.py` — no duplication.

**Urgency logic** (mirrors `services.py`):
- HIGH: any open recall present
- MEDIUM: complaints present, no recall
- LOW: nothing found

### Changes to `voicebot/lk_agent.py`
1. **Update GREETING**: "Hello, welcome to the service center. What can I help you with today, and what vehicle are you calling about?"
2. **Add system prompt** instructing LLM to:
   - Extract make/model/year from conversation when mentioned
   - After vehicle is identified, signal via a structured tool call or marker so the agent can trigger `fetch_context_by_vehicle()`
   - Answer recall questions from injected context, not from LLM training knowledge (stale)
   - If no recalls found: say so explicitly ("Good news — I don't see any open recalls for that vehicle")
   - If NHTSA call fails: say "I'm having trouble reaching the recall database right now — let me connect you to a service advisor"
3. **Add `on_conversation_item_added` handler extension**: after each user turn, check if vehicle make/model/year has been extracted → trigger background NHTSA lookup → inject result into `session.chat_ctx`

### System message injected after vehicle identification
```
[VEHICLE CONTEXT — {year} {make} {model}]
Open Recalls ({count}):
  1. Campaign {id}: {component} — {consequence}
  2. ...
Top Complaints:
  1. {component}: reported {count} times
  2. ...
Urgency: HIGH

Use this data to answer recall questions. Do not speculate beyond it.
```

---

## Critical Files

| File | Change |
|------|--------|
| `voicebot/lk_agent.py` | Update GREETING, add system prompt, extend conversation_item_added handler |
| `voicebot/vehicle_context.py` | New module — `fetch_context_by_vehicle()` |
| `advisor_ai_upsell_copilo/src/app/services.py` | Import and reuse `fetch_recalls()`, `fetch_complaints()` (lines 63-167) |
| `advisor_ai_upsell_copilo/src/app/dependencies.py` | Reuse `get_nhtsa_client()` for httpx client |
| `Tests/mock_call.py` | Extend with `--vehicle` flag to simulate vehicle identification flow |

---

## Edge Cases

| Case | Handling |
|------|---------|
| Make/model not found in NHTSA | Return empty recalls/complaints, log warning, LLM falls back gracefully |
| NHTSA timeout (10s) | Return empty context, agent informs caller |
| No recalls found | Explicitly tell caller: "No open recalls found for your {year} {make} {model}" |
| LLM misparses vehicle name | Low confidence → agent asks for clarification: "Did you say a 2021 RAV4?" |
| Caller has multiple vehicles | Agent asks: "Which vehicle are you calling about?" |

---

## Verification

1. `mock_call.py` — type "I have a 2021 Toyota RAV4" → confirm NHTSA context injected into LLM
2. `mock_call.py` — type "are there any recalls?" → confirm LLM responds from injected data, not hallucination
3. `mock_call.py` — simulate NHTSA timeout → confirm graceful fallback message
4. `mock_call.py` — use a model with no recalls → confirm "no recalls found" response
5. Add pytest unit tests in `Tests/` for `vehicle_context.py`: valid lookup, NHTSA failure, empty results
6. End-to-end LiveKit test: `python -m voicebot.lk_agent dev` — run through full flow with a real vehicle name
