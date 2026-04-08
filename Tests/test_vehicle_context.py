"""
Tests for voicebot/vehicle_context.py

Covers:
  - Successful recall + complaint fetch
  - NHTSA recall failure (graceful degradation)
  - NHTSA complaints 400 error (graceful degradation — common for recent model years)
  - Empty results (no recalls, no complaints)
  - Urgency logic (HIGH / MEDIUM / LOW)
  - format_vehicle_context output structure
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voicebot.vehicle_context import (
    fetch_context_by_vehicle,
    format_vehicle_context,
    MAX_RECALLS,
    MAX_COMPLAINTS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_response(status_code: int, json_body: dict) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_body
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _recalls_payload(n: int) -> dict:
    return {
        "results": [
            {
                "NHTSACampaignNumber": f"24V{i:06d}",
                "Component": f"BRAKES:{i}",
                "Consequence": f"Crash risk {i}",
                "Summary": f"Summary {i}",
            }
            for i in range(n)
        ]
    }


def _complaints_payload(counts: list[int]) -> dict:
    return {
        "results": [
            {
                "components": f"COMPONENT_{i}",
                "numberOfComplaints": c,
                "summary": f"Summary {i}",
            }
            for i, c in enumerate(counts)
        ]
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFetchContextByVehicle:

    @pytest.mark.asyncio
    async def test_recalls_and_complaints_returned(self):
        """Fetches both recalls and complaints and returns them in the result."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, _recalls_payload(2)),
            _mock_response(200, _complaints_payload([10, 5, 1])),
        ]

        ctx = await fetch_context_by_vehicle("Toyota", "RAV4", "2021", client)

        assert ctx["vehicle"] == {"make": "Toyota", "model": "RAV4", "year": "2021"}
        assert len(ctx["recalls"]) == 2
        assert ctx["recalls"][0]["campaign"] == "24V000000"
        assert len(ctx["complaints"]) == 3
        assert ctx["complaints"][0]["component"] == "COMPONENT_0"
        assert ctx["complaints"][0]["count"] == 10

    @pytest.mark.asyncio
    async def test_urgency_high_when_recalls_present(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, _recalls_payload(1)),
            _mock_response(200, _complaints_payload([])),
        ]
        ctx = await fetch_context_by_vehicle("Toyota", "RAV4", "2021", client)
        assert ctx["urgency"] == "HIGH"

    @pytest.mark.asyncio
    async def test_urgency_medium_when_only_complaints(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, _complaints_payload([3])),
        ]
        ctx = await fetch_context_by_vehicle("Toyota", "RAV4", "2021", client)
        assert ctx["urgency"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_urgency_low_when_nothing_found(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        ctx = await fetch_context_by_vehicle("Honda", "Civic", "2023", client)
        assert ctx["urgency"] == "LOW"
        assert ctx["recalls"] == []
        assert ctx["complaints"] == []

    @pytest.mark.asyncio
    async def test_recalls_capped_at_max(self):
        """Returns at most MAX_RECALLS recalls even if NHTSA returns more."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, _recalls_payload(MAX_RECALLS + 5)),
            _mock_response(200, _complaints_payload([])),
        ]
        ctx = await fetch_context_by_vehicle("Ford", "F-150", "2020", client)
        assert len(ctx["recalls"]) == MAX_RECALLS

    @pytest.mark.asyncio
    async def test_complaints_sorted_by_count_descending(self):
        """Complaints should be sorted highest-count first."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, _complaints_payload([1, 99, 5])),
        ]
        ctx = await fetch_context_by_vehicle("Honda", "Civic", "2022", client)
        counts = [c["count"] for c in ctx["complaints"]]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.asyncio
    async def test_recalls_fetch_failure_degrades_gracefully(self):
        """If NHTSA recalls call fails, returns empty recalls and still fetches complaints."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(500, {}),                         # recalls fail
            _mock_response(200, _complaints_payload([7])),   # complaints succeed
        ]
        ctx = await fetch_context_by_vehicle("Ford", "F-150", "2020", client)
        assert ctx["recalls"] == []
        assert len(ctx["complaints"]) == 1
        assert ctx["urgency"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_complaints_400_degrades_gracefully(self):
        """NHTSA returns 400 for some model years — should not crash, should return empty."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(200, _recalls_payload(1)),   # recalls succeed
            _mock_response(400, {}),                     # complaints 400
        ]
        ctx = await fetch_context_by_vehicle("Honda", "Civic", "2023", client)
        assert len(ctx["recalls"]) == 1
        assert ctx["complaints"] == []
        assert ctx["urgency"] == "HIGH"

    @pytest.mark.asyncio
    async def test_both_calls_fail_returns_empty_context(self):
        """If both NHTSA calls fail, returns empty context with LOW urgency."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [
            _mock_response(500, {}),
            _mock_response(500, {}),
        ]
        ctx = await fetch_context_by_vehicle("Foo", "Bar", "2020", client)
        assert ctx["recalls"] == []
        assert ctx["complaints"] == []
        assert ctx["urgency"] == "LOW"

    @pytest.mark.asyncio
    async def test_creates_own_client_when_none_provided(self):
        """When no http_client is passed, creates and closes its own client."""
        mock_response_recalls = _mock_response(200, {"results": []})
        mock_response_complaints = _mock_response(200, {"results": []})

        mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
        mock_client_instance.get.side_effect = [mock_response_recalls, mock_response_complaints]
        mock_client_instance.aclose = AsyncMock()

        with patch("voicebot.vehicle_context.httpx.AsyncClient", return_value=mock_client_instance):
            ctx = await fetch_context_by_vehicle("Honda", "Civic", "2023")

        mock_client_instance.aclose.assert_awaited_once()
        assert ctx["urgency"] == "LOW"


class TestFormatVehicleContext:

    def test_shows_vehicle_header(self):
        ctx = {
            "vehicle": {"make": "Toyota", "model": "RAV4", "year": "2021"},
            "recalls": [], "complaints": [], "urgency": "LOW",
        }
        output = format_vehicle_context(ctx)
        assert "[VEHICLE CONTEXT — 2021 Toyota RAV4]" in output

    def test_shows_recall_component_and_consequence_without_campaign_id(self):
        ctx = {
            "vehicle": {"make": "Toyota", "model": "RAV4", "year": "2021"},
            "recalls": [{
                "campaign": "23V865000",
                "component": "AIR BAGS",
                "consequence": "Increased injury risk",
                "summary": "Sensor may fail",
            }],
            "complaints": [],
            "urgency": "HIGH",
        }
        output = format_vehicle_context(ctx)
        assert "23V865000" not in output
        assert "AIR BAGS" in output
        assert "Increased injury risk" in output

    def test_shows_no_recalls_message_when_empty(self):
        ctx = {
            "vehicle": {"make": "Honda", "model": "Civic", "year": "2023"},
            "recalls": [], "complaints": [], "urgency": "LOW",
        }
        output = format_vehicle_context(ctx)
        assert "None found" in output

    def test_shows_complaint_component_and_count(self):
        ctx = {
            "vehicle": {"make": "Ford", "model": "F-150", "year": "2020"},
            "recalls": [],
            "complaints": [{"component": "ELECTRICAL SYSTEM", "count": 42}],
            "urgency": "MEDIUM",
        }
        output = format_vehicle_context(ctx)
        assert "ELECTRICAL SYSTEM" in output
        assert "42" in output

    def test_urgency_appears_in_output(self):
        for urgency in ("HIGH", "MEDIUM", "LOW"):
            ctx = {
                "vehicle": {"make": "X", "model": "Y", "year": "2020"},
                "recalls": [], "complaints": [], "urgency": urgency,
            }
            assert f"Urgency: {urgency}" in format_vehicle_context(ctx)

    def test_no_speculation_disclaimer_present(self):
        ctx = {
            "vehicle": {"make": "X", "model": "Y", "year": "2020"},
            "recalls": [], "complaints": [], "urgency": "LOW",
        }
        output = format_vehicle_context(ctx)
        assert "Do not speculate" in output
