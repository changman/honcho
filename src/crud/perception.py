import logging
from typing import Any, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from src.models import Document, Message, PerceptionEvent
from src.schemas import PerceptionIngestRequest, PerceptionEventOut

logger = logging.getLogger(__name__)

def create_perception_event(db: Session, event_data: PerceptionIngestRequest) -> PerceptionEvent:
    db_event = PerceptionEvent(
        session_id=event_data.session_id,
        source_type=event_data.source_type,
        salience_score=event_data.salience_score,
        fingerprint=event_data.fingerprint,
        fingerprint_bq=event_data.fingerprint_bq,
        metadata_=event_data.metadata or {},
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

def get_perception_event(db: Session, event_id: str) -> Optional[PerceptionEvent]:
    return db.query(PerceptionEvent).filter(PerceptionEvent.id == event_id).first()

def search_perception_events(
    db: Session,
    query_fingerprint: List[float],
    session_id: Optional[str] = None,
    top_k: int = 5,
) -> List[PerceptionEventOut]:
    # Use the <# operator for Hamming distance on fingerprint_bq if available and appropriate
    # For now, let's assume cosine similarity on the full fingerprint vector
    # This part will need careful optimization based on the actual BQ implementation and performance
    query = db.query(PerceptionEvent, PerceptionEvent.fingerprint.cosine_distance(query_fingerprint).label("distance"))

    if session_id:
        query = query.filter(PerceptionEvent.session_id == session_id)
    
    # Order by distance (lower is better for cosine distance)
    query = query.order_by(text("distance")).limit(top_k)

    results = query.all()
    
    perception_events_out = []
    for event, distance in results:
        associated_documents_ids = [doc.id for doc in db.query(Document).filter(Document.perception_event_id == event.id).all()]
        associated_messages_ids = [msg.public_id for msg in db.query(Message).filter(Message.perception_event_id == event.id).all()]

        perception_events_out.append(PerceptionEventOut(
            id=event.id,
            session_id=event.session_id,
            source_type=event.source_type,
            salience_score=event.salience_score,
            fingerprint_bq=event.fingerprint_bq,
            metadata=event.metadata_,
            created_at=event.created_at,
            associated_documents=associated_documents_ids,
            associated_messages=associated_messages_ids,
            # For PerceptionEventOut, we need to load the full DocumentOut and MessageOut objects
            # This is a simplification; in a real app, you might fetch these lazily or with specific joins
            # For now, just pass the IDs, and the API layer can fetch full objects if needed.
        ))
    return perception_events_out

