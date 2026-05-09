"""CRUD operations for multimodal PerceptionEvents."""

import math
from logging import getLogger
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.schemas.perception import PerceptionIngestRequest
from src.utils.bq import rank_by_hamming

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance between two float vectors (0 = identical, 2 = opposite)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Key-frame interleaving gate (Phase 2-A)
# ---------------------------------------------------------------------------

async def should_store_full_fingerprint(
    db: AsyncSession,
    session_id: str,
    new_fingerprint: list[float],
    delta_threshold: float = 0.05,
) -> bool:
    """Return True if this frame differs enough from the last stored one.

    Implements the design doc's "Key-frame Interleaving": only store the full
    512d float vector when the cosine distance from the previous stored
    fingerprint exceeds delta_threshold (default 5%).

    When False, the caller should persist fingerprint_bq only (saving ~2KB
    per frame for identical / near-duplicate captures).
    """
    stmt = (
        select(models.PerceptionEvent)
        .where(
            models.PerceptionEvent.session_id == session_id,
            models.PerceptionEvent.fingerprint.is_not(None),
        )
        .order_by(models.PerceptionEvent.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    last = result.scalar_one_or_none()

    if last is None:
        return True  # First frame: always store

    last_fp = last.fingerprint
    if last_fp is None:
        return True

    # pgvector returns fingerprint as a list-like object
    distance = _cosine_distance(list(last_fp), new_fingerprint)
    return distance > delta_threshold


# ---------------------------------------------------------------------------
# Core CRUD
# ---------------------------------------------------------------------------

async def create_perception_event(
    db: AsyncSession,
    event: PerceptionIngestRequest,
    workspace_name: str,
    store_full_fingerprint: bool = True,
) -> models.PerceptionEvent:
    """Persist a new PerceptionEvent.

    workspace_name is validated upstream (auth middleware). Pass
    store_full_fingerprint=False to apply key-frame interleaving: only the
    BQ string is stored, not the 512d float vector.
    """
    new_event = models.PerceptionEvent(
        session_id=event.session_id,
        source_type=event.source_type,
        salience_score=event.salience_score,
        fingerprint=event.fingerprint if store_full_fingerprint else None,
        fingerprint_bq=event.fingerprint_bq,
        metadata_=event.metadata or {},
        captured_at=event.captured_at,
        segment_id=event.segment_id,
        is_segment_start=event.is_segment_start,
        is_segment_end=event.is_segment_end,
    )
    db.add(new_event)
    await db.flush()
    return new_event


async def get_perception_event(
    db: AsyncSession,
    event_id: str,
) -> models.PerceptionEvent | None:
    """Fetch a single PerceptionEvent by ID."""
    stmt = select(models.PerceptionEvent).where(models.PerceptionEvent.id == event_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

async def search_perception_events(
    db: AsyncSession,
    query_fingerprint_bq: str | None,
    session_id: str | None = None,
    top_k: int = 5,
    min_similarity: float = 0.60,
) -> list[tuple[models.PerceptionEvent, float]]:
    """Search PerceptionEvents by BQ Hamming distance.

    Strategy:
      1. Load all events in the session that have a fingerprint_bq.
      2. Rank by Hamming similarity to the query BQ string.
      3. If no query BQ or no candidates, fall back to most recent top_k events.

    Returns:
        List of (PerceptionEvent, similarity_score) sorted best-first.
    """
    stmt = select(models.PerceptionEvent)
    if session_id:
        stmt = stmt.where(models.PerceptionEvent.session_id == session_id)
    if query_fingerprint_bq:
        stmt = stmt.where(models.PerceptionEvent.fingerprint_bq.is_not(None))

    result = await db.execute(stmt)
    candidates = result.scalars().all()

    if not candidates:
        return []

    if not query_fingerprint_bq:
        recent = sorted(candidates, key=lambda e: e.created_at, reverse=True)[:top_k]
        return [(e, 0.0) for e in recent]

    ranked = rank_by_hamming(
        query_bq=query_fingerprint_bq,
        candidates=[(e.fingerprint_bq, e) for e in candidates if e.fingerprint_bq],
        top_k=top_k,
        min_similarity=min_similarity,
    )
    return [(event, score) for event, score in ranked]


async def get_events_in_segment(
    db: AsyncSession,
    segment_id: str,
    session_id: str | None = None,
) -> list[models.PerceptionEvent]:
    """Retrieve all events belonging to a stream segment, ordered by creation time."""
    stmt = (
        select(models.PerceptionEvent)
        .where(models.PerceptionEvent.segment_id == segment_id)
        .order_by(models.PerceptionEvent.created_at.asc())
    )
    if session_id:
        stmt = stmt.where(models.PerceptionEvent.session_id == session_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# State change detection
# ---------------------------------------------------------------------------

async def find_or_flag_state_change(
    db: AsyncSession,
    session_id: str,
    new_fingerprint_bq: str,
    visual_sim_threshold: float = 0.90,
) -> tuple[models.PerceptionEvent | None, bool]:
    """Detect a physical state change based on BQ fingerprint similarity.

    Implements design doc Threshold-based Branching:
      - Find the most recent event in the session that has a BQ fingerprint.
      - similarity < visual_sim_threshold → STATE_CHANGE.
      - similarity >= threshold → DUPLICATE.
      - No prior event → new scene, not a state change.

    Returns:
        (nearest_event, is_state_change)
    """
    stmt = (
        select(models.PerceptionEvent)
        .where(
            models.PerceptionEvent.session_id == session_id,
            models.PerceptionEvent.fingerprint_bq.is_not(None),
        )
        .order_by(models.PerceptionEvent.created_at.desc())
        .limit(10)
    )
    result = await db.execute(stmt)
    recent = result.scalars().all()

    if not recent:
        return None, False

    ranked = rank_by_hamming(
        query_bq=new_fingerprint_bq,
        candidates=[(e.fingerprint_bq, e) for e in recent if e.fingerprint_bq],
        top_k=1,
    )
    if not ranked:
        return None, False

    best_event, similarity = ranked[0]
    return best_event, similarity < visual_sim_threshold


# ---------------------------------------------------------------------------
# Dream-time pruning
# ---------------------------------------------------------------------------

async def prune_perception_event_fingerprint(
    db: AsyncSession,
    event_id: str,
) -> models.PerceptionEvent | None:
    """Dream-time pruning: set fingerprint to NULL, keep fingerprint_bq.

    Called by MultimodalInductionSpecialist after synthesizing a higher-level
    event, to free vector storage while preserving BQ search capability.
    """
    event = await get_perception_event(db, event_id)
    if event is None:
        return None
    event.fingerprint = None  # type: ignore[assignment]
    await db.flush()
    return event


# ---------------------------------------------------------------------------
# Associated data
# ---------------------------------------------------------------------------

async def get_associated_data(
    db: AsyncSession,
    perception_event_id: str,
) -> dict[str, Any]:
    """Load Documents and Messages linked to a PerceptionEvent."""
    doc_stmt = select(models.Document).where(
        models.Document.perception_event_id == perception_event_id
    )
    msg_stmt = select(models.Message).where(
        models.Message.perception_event_id == perception_event_id
    )
    docs_res = await db.execute(doc_stmt)
    msgs_res = await db.execute(msg_stmt)
    return {
        "documents": list(docs_res.scalars().all()),
        "messages": list(msgs_res.scalars().all()),
    }
