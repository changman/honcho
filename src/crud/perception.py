import datetime
from collections.abc import Sequence
from logging import getLogger
from typing import Any, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src import models
from src.dependencies import tracked_db

logger = getLogger(__name__)


async def create_perception_event(
    db: AsyncSession,
    event: "schemas.PerceptionIngestRequest",
    workspace_name: str
) -> models.PerceptionEvent:
    """
    Create a new perception event in the database.
    """
    from src import schemas  # Defer import to break circular dependency

    # Note: workspace_name is needed for multitenancy but not directly on PerceptionEvent model.
    # It's associated via the session. We assume the session's workspace is validated before this call.
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
    event_id: str
) -> Optional[models.PerceptionEvent]:
    """
    Retrieve a perception event by its ID.
    """
    stmt = select(models.PerceptionEvent).where(models.PerceptionEvent.id == event_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def search_perception_events(
    db: AsyncSession,
    query_fingerprint_bq: str,
    session_id: Optional[str] = None,
    top_k: int = 5
) -> List[models.PerceptionEvent]:
    """
    Search for perception events using Hamming distance on BQ fingerprints.
    """
    # Placeholder for native bitwise search.
    # For now, we'll just return recent events for demonstration.
    stmt = select(models.PerceptionEvent)
    
    if session_id:
        stmt = stmt.where(models.PerceptionEvent.session_id == session_id)
        
    stmt = stmt.order_by(models.PerceptionEvent.created_at.desc()).limit(top_k)
    
    result = await db.execute(stmt)
    return list(result.scalars().all())

async def get_associated_data(
    db: AsyncSession,
    perception_event_id: str
) -> dict[str, Any]:
    """
    Retrieve associated documents and messages for a perception event.
    """
    doc_stmt = select(models.Document).where(models.Document.perception_event_id == perception_event_id)
    msg_stmt = select(models.Message).where(models.Message.perception_event_id == perception_event_id)
    
    docs_res = await db.execute(doc_stmt)
    msgs_res = await db.execute(msg_stmt)
    
    return {
        "documents": list(docs_res.scalars().all()),
        "messages": list(msgs_res.scalars().all())
    }
