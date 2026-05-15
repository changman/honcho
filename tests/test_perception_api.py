"""Integration tests for the multimodal perception API.

Tests run against the real async PostgreSQL stack via conftest fixtures.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.utils.bq import float_to_bq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fingerprint(seed: int | None = None) -> list[float]:
    """512-dim float vector. Deterministic when seed is given."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(512)]


def _make_shifted_fingerprint(base: list[float], shift: int = 50) -> list[float]:
    v = list(base)
    for i in range(shift):
        v[i] = 1.0 - v[i]
    return v


def _make_opposite_fingerprint(base: list[float]) -> list[float]:
    return [1.0 - x for x in base]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def workspace_and_session(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> tuple[str, str]:
    """Create a Session and return (workspace_name, session_name).

    URL path params in Honcho use session names, not internal UUIDs.
    The router resolves name → UUID before FK insertion.
    """
    workspace, _peer = sample_data
    session = models.Session(
        name="perception-test-session",
        workspace_name=workspace.name,
    )
    db_session.add(session)
    await db_session.commit()
    return workspace.name, session.name


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

class TestIngestPerception:
    def _ingest_payload(self, session_id: str, **overrides: Any) -> dict:
        fp = _make_fingerprint(seed=10)
        base: dict = {
            "session_id": session_id,
            "source_type": "video_1fps",
            "salience_score": 0.75,
            "fingerprint": fp,
            "metadata": {"camera_id": "cam-01"},
        }
        return {**base, **overrides}

    def test_ingest_creates_event(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        payload = self._ingest_payload(sid)
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=payload,
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["session_id"] == sid
        assert data["source_type"] == "video_1fps"
        assert "id" in data
        assert "created_at" in data
        assert "is_state_change" in data

    def test_ingest_auto_computes_bq_and_is_searchable(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """When only a float fingerprint is provided, BQ is computed and stored."""
        ws, sid = workspace_and_session
        fp = _make_fingerprint(seed=11)
        ingest_resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=self._ingest_payload(sid, fingerprint=fp, fingerprint_bq=None),
        )
        assert ingest_resp.status_code == 201, ingest_resp.text
        event_id = ingest_resp.json()["id"]

        search_resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": fp, "top_k": 5},
        )
        assert search_resp.status_code == 200, search_resp.text
        assert any(h["id"] == event_id for h in search_resp.json())

    def test_ingest_unknown_session_returns_404(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, _sid = workspace_and_session
        payload = self._ingest_payload("no-such-session")
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/no-such-session/perception/ingest",
            json=payload,
        )
        assert resp.status_code == 404

    def test_ingest_missing_required_fields_returns_422(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json={"session_id": sid},
        )
        assert resp.status_code == 422

    def test_ingest_invalid_source_type_returns_422(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=self._ingest_payload(sid, source_type="invalid_type"),
        )
        assert resp.status_code == 422

    def test_ingest_salience_out_of_range_returns_422(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=self._ingest_payload(sid, salience_score=1.5),
        )
        assert resp.status_code == 422

    def test_state_change_flag_first_event_is_false(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """First event in a session is never a state change."""
        ws, sid = workspace_and_session
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=self._ingest_payload(sid),
        )
        assert resp.status_code == 201
        assert resp.json()["is_state_change"] is False

    def test_state_change_flag_identical_fingerprint_is_false(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Duplicate fingerprint in the same session is NOT a state change."""
        ws, sid = workspace_and_session
        fp = _make_fingerprint(seed=20)
        payload = self._ingest_payload(sid, fingerprint=fp)
        client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=payload,
        )
        resp2 = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=payload,
        )
        assert resp2.status_code == 201
        assert resp2.json()["is_state_change"] is False

    def test_state_change_flag_opposite_fingerprint_is_true(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Maximally different fingerprint after an existing one → state change."""
        ws, sid = workspace_and_session
        fp_a = _make_fingerprint(seed=30)
        fp_b = _make_opposite_fingerprint(fp_a)

        client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=self._ingest_payload(sid, fingerprint=fp_a),
        )
        resp2 = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=self._ingest_payload(sid, fingerprint=fp_b),
        )
        assert resp2.status_code == 201
        assert resp2.json()["is_state_change"] is True


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

class TestSearchPerception:
    def _ingest(
        self,
        client: TestClient,
        ws: str,
        sid: str,
        fingerprint: list[float],
        salience: float = 0.5,
    ) -> str:
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json={
                "session_id": sid,
                "source_type": "video_1fps",
                "salience_score": salience,
                "fingerprint": fingerprint,
            },
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["id"]

    def test_search_returns_nearest_event_first(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        fp_target = _make_fingerprint(seed=40)
        fp_noise = _make_fingerprint(seed=41)
        fp_opposite = _make_opposite_fingerprint(fp_target)

        id_target = self._ingest(client, ws, sid, fp_target)
        self._ingest(client, ws, sid, fp_noise)
        self._ingest(client, ws, sid, fp_opposite)

        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": fp_target, "top_k": 3},
        )
        assert resp.status_code == 200, resp.text
        results = resp.json()
        assert len(results) > 0
        assert results[0]["id"] == id_target

    def test_search_results_sorted_descending_by_similarity(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        fp_base = _make_fingerprint(seed=50)
        self._ingest(client, ws, sid, _make_shifted_fingerprint(fp_base, shift=5))
        self._ingest(client, ws, sid, _make_shifted_fingerprint(fp_base, shift=200))

        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": fp_base, "top_k": 5},
        )
        assert resp.status_code == 200
        scores = [
            r["similarity_score"]
            for r in resp.json()
            if r["similarity_score"] is not None
        ]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limits_results(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        fp_base = _make_fingerprint(seed=60)
        for i in range(7):
            self._ingest(client, ws, sid, _make_shifted_fingerprint(fp_base, shift=i * 5))

        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": fp_base, "top_k": 3},
        )
        assert resp.status_code == 200
        assert len(resp.json()) <= 3

    def test_search_empty_session_returns_empty_list(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": _make_fingerprint(seed=70), "top_k": 5},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_wrong_fingerprint_length_returns_422(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": [0.5] * 10, "top_k": 5},
        )
        assert resp.status_code == 422

    def test_search_result_contains_similarity_score(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        fp = _make_fingerprint(seed=80)
        self._ingest(client, ws, sid, fp)

        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": fp, "top_k": 1},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        score = results[0]["similarity_score"]
        assert score is not None
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Phase 2: Key-frame interleaving + stream segmentation
# ---------------------------------------------------------------------------

class TestKeyFrameInterleaving:
    """Key-frame interleaving: skip full vector for near-duplicate frames."""

    def _ingest(
        self,
        client: TestClient,
        ws: str,
        sid: str,
        fingerprint: list[float],
        **extra: Any,
    ) -> dict:
        payload = {
            "session_id": sid,
            "source_type": "video_1fps",
            "salience_score": 0.5,
            "fingerprint": fingerprint,
            **extra,
        }
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=payload,
        )
        assert resp.status_code == 201, resp.text
        return resp.json()

    def test_first_event_always_accepted(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """First frame in session is always ingested (no prior to compare)."""
        ws, sid = workspace_and_session
        resp = self._ingest(client, ws, sid, _make_fingerprint(seed=90))
        assert resp["id"] is not None

    def test_identical_frame_is_still_ingested(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Even near-duplicate frames are ingested — just without the full vector.
        The event still gets a valid ID and is searchable via BQ."""
        ws, sid = workspace_and_session
        fp = _make_fingerprint(seed=91)
        self._ingest(client, ws, sid, fp)
        resp2 = self._ingest(client, ws, sid, fp)
        assert resp2["id"] is not None

    def test_significantly_different_frame_accepted(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Frame with large cosine distance from previous is accepted."""
        ws, sid = workspace_and_session
        fp_a = _make_fingerprint(seed=92)
        fp_b = _make_opposite_fingerprint(fp_a)
        self._ingest(client, ws, sid, fp_a)
        resp2 = self._ingest(client, ws, sid, fp_b)
        assert resp2["id"] is not None


class TestStreamSegmentation:
    """Stream segmentation: captured_at, segment_id, is_segment_start/end."""

    def _ingest(
        self,
        client: TestClient,
        ws: str,
        sid: str,
        fingerprint: list[float],
        **extra: Any,
    ) -> dict:
        payload = {
            "session_id": sid,
            "source_type": "video_1fps",
            "salience_score": 0.5,
            "fingerprint": fingerprint,
            **extra,
        }
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/ingest",
            json=payload,
        )
        assert resp.status_code == 201, resp.text
        return resp.json()

    def test_ingest_with_segment_fields(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Segment metadata is stored and echoed back in the response."""
        ws, sid = workspace_and_session
        import datetime

        captured = datetime.datetime.now(datetime.timezone.utc).isoformat()
        resp = self._ingest(
            client, ws, sid,
            _make_fingerprint(seed=100),
            captured_at=captured,
            segment_id="seg-abc",
            is_segment_start=True,
            is_segment_end=False,
        )
        assert resp["segment_id"] == "seg-abc"
        assert resp["is_segment_start"] is True
        assert resp["is_segment_end"] is False
        assert resp["captured_at"] is not None

    def test_ingest_segment_end_event(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        ws, sid = workspace_and_session
        resp = self._ingest(
            client, ws, sid,
            _make_fingerprint(seed=101),
            segment_id="seg-xyz",
            is_segment_end=True,
        )
        assert resp["is_segment_end"] is True
        assert resp["segment_id"] == "seg-xyz"

    def test_search_result_includes_segment_fields(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Search results should include segment metadata."""
        ws, sid = workspace_and_session
        fp = _make_fingerprint(seed=102)
        self._ingest(
            client, ws, sid, fp,
            segment_id="seg-search-test",
            is_segment_start=True,
        )
        resp = client.post(
            f"/v1/workspaces/{ws}/sessions/{sid}/perception/search",
            json={"query_fingerprint": fp, "top_k": 1},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["segment_id"] == "seg-search-test"
        assert results[0]["is_segment_start"] is True

    def test_ingest_without_segment_fields_defaults(
        self,
        client: TestClient,
        workspace_and_session: tuple[str, str],
    ):
        """Optional segment fields default gracefully when omitted."""
        ws, sid = workspace_and_session
        resp = self._ingest(client, ws, sid, _make_fingerprint(seed=103))
        assert resp["segment_id"] is None
        assert resp["captured_at"] is None
        assert resp["is_segment_start"] is False
        assert resp["is_segment_end"] is False
