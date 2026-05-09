import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException
from src.security import require_auth

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
    """Ingest a new perception event (image/audio fingerprint)."""
    if event.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id in body does not match URL")

    new_event = await crud.create_perception_event(db, event=event, workspace_name=workspace_id)

    # High-salience events will trigger background reasoning in Phase 4
    # if event.salience_score > 0.7:
    #     background_tasks.add_task(trigger_perception_reasoning, new_event)

    return schemas.PerceptionEventResponse.model_validate(new_event)


@router.post("/search", response_model=List[schemas.PerceptionEventOut])
async def search_perception(
    request: schemas.PerceptionSearchRequest,
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """Search for perception events similar to the query fingerprint (Phase 1: placeholder BQ search)."""
    events = await crud.search_perception_events(
        db,
        query_fingerprint_bq=None,  # Phase 1 will replace with real BQ conversion
        session_id=session_id,
        top_k=request.top_k,
    )

    results = []
    for event in events:
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
                associated_document_ids=[d.id for d in associated["documents"]],
                associated_message_ids=[m.public_id for m in associated["messages"]],
            )
        )

    return results
