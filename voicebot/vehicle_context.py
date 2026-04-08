"""
vehicle_context.py
──────────────────
Fetches live NHTSA recall and complaint data for a vehicle by make/model/year.
Used by the voicebot to inject accurate recall context into the LLM so it can
answer callers' recall questions without hallucinating stale training data.

No VIN needed — NHTSA's recallsByVehicle API works with make/model/year alone.
"""

import logging
from typing import Any

import httpx

log = logging.getLogger("voicebot")

NHTSA_RECALLS_URL    = "https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}"
NHTSA_COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={year}"
NHTSA_TIMEOUT        = 10.0
MAX_RECALLS          = 3
MAX_COMPLAINTS       = 3


async def fetch_context_by_vehicle(
    make: str,
    model: str,
    year: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """
    Fetch open recalls and top complaints for a vehicle from NHTSA.

    Creates (and closes) its own httpx client if none is provided.

    Returns a dict with keys:
        vehicle    – {"make": str, "model": str, "year": str}
        recalls    – list of up to MAX_RECALLS dicts (campaign, component, consequence, summary)
        complaints – list of up to MAX_COMPLAINTS dicts (component, count)
        urgency    – "HIGH" | "MEDIUM" | "LOW"
    """
    own_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=NHTSA_TIMEOUT)

    recalls: list[dict] = []
    complaints: list[dict] = []

    try:
        try:
            recalls = await _fetch_recalls(make, model, year, client)
        except Exception as exc:
            log.warning("NHTSA recalls fetch failed for %s %s %s: %s", year, make, model, exc)

        try:
            complaints = await _fetch_complaints(make, model, year, client)
        except Exception as exc:
            log.warning("NHTSA complaints fetch failed for %s %s %s: %s", year, make, model, exc)
    finally:
        if own_client:
            await client.aclose()

    if recalls:
        urgency = "HIGH"
    elif complaints:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    return {
        "vehicle":    {"make": make, "model": model, "year": year},
        "recalls":    recalls[:MAX_RECALLS],
        "complaints": complaints[:MAX_COMPLAINTS],
        "urgency":    urgency,
    }


async def _fetch_recalls(
    make: str, model: str, year: str, client: httpx.AsyncClient
) -> list[dict]:
    url = NHTSA_RECALLS_URL.format(make=make, model=model, year=year)
    response = await client.get(url)
    response.raise_for_status()
    results = response.json().get("results", [])
    return [
        {
            "campaign":    r.get("NHTSACampaignNumber", ""),
            "component":   r.get("Component", ""),
            "consequence": r.get("Consequence", ""),
            "summary":     r.get("Summary", ""),
        }
        for r in results
    ]


async def _fetch_complaints(
    make: str, model: str, year: str, client: httpx.AsyncClient
) -> list[dict]:
    url = NHTSA_COMPLAINTS_URL.format(make=make, model=model, year=year)
    response = await client.get(url)
    response.raise_for_status()
    results = response.json().get("results", [])
    sorted_results = sorted(
        results, key=lambda r: r.get("numberOfComplaints", 0), reverse=True
    )
    return [
        {
            "component": r.get("components", ""),
            "count":     r.get("numberOfComplaints", 0),
        }
        for r in sorted_results
    ]


def format_vehicle_context(ctx: dict[str, Any]) -> str:
    """
    Format the NHTSA context dict into a structured string for the LLM.
    This string is returned from the tool call and injected into the chat context.
    """
    v = ctx["vehicle"]
    lines = [f"[VEHICLE CONTEXT — {v['year']} {v['make']} {v['model']}]"]
    lines.append(f"Urgency: {ctx['urgency']}")
    lines.append("")

    recalls = ctx["recalls"]
    if recalls:
        lines.append(f"Open Recalls ({len(recalls)}):")
        for i, r in enumerate(recalls, 1):
            line = f"  {i}. {r['component']}"
            if r["consequence"]:
                line += f" — {r['consequence']}"
            lines.append(line)
    else:
        lines.append("Open Recalls: None found.")

    lines.append("")

    complaints = ctx["complaints"]
    if complaints:
        lines.append(f"Top Owner Complaints ({len(complaints)}):")
        for i, c in enumerate(complaints, 1):
            lines.append(f"  {i}. {c['component']}: reported {c['count']} times")
    else:
        lines.append("Top Complaints: None found.")

    lines.append("")
    lines.append(
        "Use only this data when discussing recalls or complaints. "
        "Do not speculate beyond what is listed above."
    )
    return "\n".join(lines)
