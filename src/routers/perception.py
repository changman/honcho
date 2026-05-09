import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.security import require_auth
from src.utils.bq import float_to_bq

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions/{session_id}/perception",
    tags=["perception"],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)


@router.post("/ingest", response_model=schemas.PerceptionEventResponse, status_code=201)
async def ingest_perception(
    background_tasks: BackgroundTasks,
    event: schemas.PerceptionIngestRequest,
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """Ingest a new perception event (image/audio fingerprint).

    If the request includes a raw float fingerprint but no fingerprint_bq,
    the BQ string is computed automatically.

    Returns the created event, plus an is_state_change flag comparing it
    against the most recent event in the same session.
    """
    if event.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id in body does not match URL")

    # Auto-compute BQ if caller sent the raw float fingerprint
    if event.fingerprint and not event.fingerprint_bq:
        event = event.model_copy(update={"fingerprint_bq": float_to_bq(event.fingerprint)})

    # State-change detection before persisting
    is_state_change = False
    if event.fingerprint_bq:
        _, is_state_change = await crud.find_or_flag_state_change(
            db, session_id=session_id, new_fingerprint_bq=event.fingerprint_bq
        )

    new_event = await crud.create_perception_event(db, event=event, workspace_name=workspace_id)

    # High-salience events will trigger background reasoning in Phase 4
    # if event.salience_score > 0.7:
    #     background_tasks.add_task(trigger_perception_reasoning, new_event)

    response = schemas.PerceptionEventResponse.model_validate(new_event)
    return response.model_copy(update={"is_state_change": is_state_change})


@router.post("/search", response_model=List[schemas.PerceptionEventOut])
async def search_perception(
    request: schemas.PerceptionSearchRequest,
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """Search for perception events by visual fingerprint similarity.

    Converts the float query_fingerprint to a BQ binary string, then ranks
    stored events by Hamming distance (lower distance = higher similarity).
    """
    query_bq = float_to_bq(request.query_fingerprint)

    ranked = await crud.search_perception_events(
        db,
        query_fingerprint_bq=query_bq,
        session_id=session_id,
        top_k=request.top_k,
    )

    results = []
    for event, similarity in ranked:
        associated = await crud.get_associated_data(db, event.id)
        results.append(
            schemas.PerceptionEventOut(
                id=event.id,
                session_id=event.session_id,
                source_type=event.source_type,
                salience_score=event.salience_score,
                fingerprint_bq=event.fingerprint_bq,
                metadata=event.metadata_,
                created_at=event.created_at,
                similarity_score=round(similarity, 4),
                associated_document_ids=[d.id for d in associated["documents"]],
                associated_message_ids=[m.public_id for m in associated["messages"]],
            )
        )

    return results
