"""CRUD operations for multimodal PerceptionEvents."""

from logging import getLogger
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.schemas.perception import PerceptionIngestRequest
from src.utils.bq import rank_by_hamming

logger = getLogger(__name__)


async def create_perception_event(
    db: AsyncSession,
    event: PerceptionIngestRequest,
    workspace_name: str,
) -> models.PerceptionEvent:
    """Persist a new PerceptionEvent.

    workspace_name is validated upstream (auth middleware); it is not stored
    on the model directly because PerceptionEvent is scoped to a Session which
    already carries the workspace relationship.
    """
    new_event = models.PerceptionEvent(
        session_id=event.session_id,
        source_type=event.source_type,
        salience_score=event.salience_score,
        fingerprint=event.fingerprint,
        fingerprint_bq=event.fingerprint_bq,
        metadata_=event.metadata or {},
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
      3. If no query BQ is provided (or no candidates have a BQ), fall back to
         the most recent top_k events with similarity=0.0.

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


async def find_or_flag_state_change(
    db: AsyncSession,
    session_id: str,
    new_fingerprint_bq: str,
    visual_sim_threshold: float = 0.90,
) -> tuple[models.PerceptionEvent | None, bool]:
    """Detect a physical state change based on BQ fingerprint similarity.

    Implements the design doc's Threshold-based Branching:
      - Find the most recent event in the session that has a BQ fingerprint.
      - If visual similarity < visual_sim_threshold → flag as STATE_CHANGE.
      - If visual similarity >= threshold → DUPLICATE.
      - If no prior event → new scene (not a state change).

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
