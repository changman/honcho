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

    Key-frame interleaving: if the new fingerprint is within 5% cosine distance
    of the most recent stored fingerprint, only the BQ string is persisted
    (not the full 512d vector), saving ~2KB per near-duplicate frame.

    State-change detection: compares the new BQ fingerprint against recent
    events in the session and sets is_state_change=True when visual similarity
    drops below 0.90.
    """
    if event.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id in body does not match URL")

    # Auto-compute BQ from float fingerprint if not provided
    if event.fingerprint and not event.fingerprint_bq:
        event = event.model_copy(update={"fingerprint_bq": float_to_bq(event.fingerprint)})

    # State-change detection (runs before persist — no extra DB round-trip)
    is_state_change = False
    if event.fingerprint_bq:
        _, is_state_change = await crud.find_or_flag_state_change(
            db, session_id=session_id, new_fingerprint_bq=event.fingerprint_bq
        )

    # Key-frame interleaving: skip full vector storage for near-duplicate frames
    store_full = True
    if event.fingerprint:
        store_full = await crud.should_store_full_fingerprint(
            db, session_id=session_id, new_fingerprint=event.fingerprint
        )

    new_event = await crud.create_perception_event(
        db, event=event, workspace_name=workspace_id, store_full_fingerprint=store_full
    )

    if not store_full:
        logger.debug(
            "Key-frame interleaving: skipped full vector for event %s (near-duplicate)",
            new_event.id,
        )

    # High-salience events will trigger background reasoning in Phase 4
    # if event.salience_score > 0.7:
    #     background_tasks.add_task(trigger_perception_reasoning, new_event)

    return schemas.PerceptionEventResponse(
        id=new_event.id,
        session_id=new_event.session_id,
        source_type=new_event.source_type,
        salience_score=new_event.salience_score,
        fingerprint_bq=new_event.fingerprint_bq,
        metadata=new_event.metadata_,
        created_at=new_event.created_at,
        captured_at=new_event.captured_at,
        segment_id=new_event.segment_id,
        is_segment_start=new_event.is_segment_start,
        is_segment_end=new_event.is_segment_end,
        is_state_change=is_state_change,
    )


@router.post("/search", response_model=List[schemas.PerceptionEventOut])
async def search_perception(
    request: schemas.PerceptionSearchRequest,
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """Search for perception events by visual fingerprint similarity.

    Converts the float query_fingerprint to a BQ binary string, then ranks
    stored events by Hamming distance. Results include similarity_score (0–1).
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
                captured_at=event.captured_at,
                segment_id=event.segment_id,
                is_segment_start=event.is_segment_start,
                is_segment_end=event.is_segment_end,
                similarity_score=round(similarity, 4),
                associated_document_ids=[d.id for d in associated["documents"]],
                associated_message_ids=[m.public_id for m in associated["messages"]],
            )
        )

    return results
