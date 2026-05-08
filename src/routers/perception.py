from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.crud import perception as crud_perception
from src.dependencies import get_db
from src.schemas import PerceptionIngestRequest, PerceptionEventOut, PerceptionSearchRequest
from src.models import Document, Message

router = APIRouter()


@router.post(
    "/v1/perception/ingest",
    response_model=PerceptionEventOut,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_perception_event(
    event_data: PerceptionIngestRequest, db: Session = Depends(get_db)
):
    # 1단계: PerceptionEvent 행 생성
    db_event = crud_perception.create_perception_event(db=db, event_data=event_data)

    # 2단계: Salience가 높을 경우 큐(Redis)에 'Active Reasoning' 작업 적재 (TODO: implement Redis queue)
    if db_event.salience_score > 0.7:  # Example threshold
        # Here you would typically add a task to a Redis queue
        # For now, we'll just log this intention
        print(f"High salience event {db_event.id} detected. Adding to Active Reasoning queue.")

    # Retrieve associated documents and messages to populate PerceptionEventOut
    associated_documents_ids = [doc.id for doc in db.query(Document).filter(Document.perception_event_id == db_event.id).all()]
    associated_messages_ids = [msg.public_id for msg in db.query(Message).filter(Message.perception_event_id == db_event.id).all()]

    # Re-fetch the event with relationships if needed, or construct the output schema directly
    # For simplicity, constructing directly using the IDs fetched above
    return PerceptionEventOut(
        id=str(db_event.id),
        session_id=str(db_event.session_id),
        source_type=db_event.source_type,
        salience_score=db_event.salience_score,
        fingerprint_bq=db_event.fingerprint_bq,
        metadata=db_event.metadata_,
        created_at=db_event.created_at,
        associated_documents=associated_documents_ids, # These will be IDs for now
        associated_messages=associated_messages_ids, # These will be IDs for now
    )


@router.post(
    "/v1/perception/search",
    response_model=List[PerceptionEventOut],
)
async def search_perception_events_api(
    search_data: PerceptionSearchRequest, db: Session = Depends(get_db)
):
    perception_events = crud_perception.search_perception_events(
        db=db,
        query_fingerprint=search_data.query_fingerprint,
        session_id=search_data.session_id,
        top_k=search_data.top_k,
    )
    if not perception_events:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching perception events found")
    return perception_events
