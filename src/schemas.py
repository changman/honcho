import datetime
from typing import Any, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, conlist, validator

from src.utils.types import DocumentLevel, TaskType, VectorSyncState


# --- Perception Event Schemas ---

class PerceptionIngestRequest(BaseModel):
    session_id: str
    source_type: Literal["video_1fps", "audio_vad", "remote_stream"]
    salience_score: float
    fingerprint: Optional[conlist(float, min_length=512, max_length=512)] = None
    fingerprint_bq: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class PerceptionSearchRequest(BaseModel):
    query_fingerprint: conlist(float, min_length=512, max_length=512)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, gt=0, le=100)


class PerceptionEventResponse(BaseModel):
    id: str
    session_id: str
    source_type: Literal["video_1fps", "audio_vad", "remote_stream"]
    salience_score: float
    fingerprint_bq: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime.datetime
    associated_documents: List[str] = Field(default_factory=list)
    associated_messages: List[str] = Field(default_factory=list)


class PerceptionEventOut(PerceptionEventResponse):
    associated_documents: List['DocumentOut'] = Field(default_factory=list)
    associated_messages: List['MessageOut'] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True


# --- Existing Schemas (for context, not part of this task) ---

class HealthResponse(BaseModel):
    status: str
    version: str


class MessageIngest(BaseModel):
    session_name: str
    peer_name: str
    content: str
    h_metadata: Optional[dict[str, Any]] = Field(default_factory=dict, alias="metadata")
    internal_metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    workspace_name: str

    class Config:
        populate_by_name = True


class MessageOut(BaseModel):
    id: int
    public_id: str
    session_name: str
    peer_name: str
    content: str
    token_count: int
    seq_in_session: int
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(alias="metadata")
    internal_metadata: dict[str, Any]
    workspace_name: str
    perception_event_id: Optional[str] = None

    class Config:
        populate_by_name = True


class PeerIn(BaseModel):
    name: str
    workspace_name: str
    h_metadata: Optional[dict[str, Any]] = Field(default_factory=dict, alias="metadata")
    configuration: Optional[dict[str, Any]] = Field(default_factory=dict)
    internal_metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class PeerOut(BaseModel):
    id: str
    name: str
    workspace_name: str
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(alias="metadata")
    configuration: dict[str, Any]
    internal_metadata: dict[str, Any]
    sessions: Optional[List[Any]] = None  # To avoid circular dependency for now

    class Config:
        populate_by_name = True


class SessionIn(BaseModel):
    name: str
    workspace_name: str
    h_metadata: Optional[dict[str, Any]] = Field(default_factory=dict, alias="metadata")
    configuration: Optional[dict[str, Any]] = Field(default_factory=dict)
    internal_metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    peers: Optional[List[str]] = Field(default_factory=list)  # List of peer names

    class Config:
        populate_by_name = True


class SessionOut(BaseModel):
    id: str
    name: str
    workspace_name: str
    is_active: bool
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(alias="metadata")
    configuration: dict[str, Any]
    internal_metadata: dict[str, Any]
    peers: Optional[List[PeerOut]] = None
    messages: Optional[List[MessageOut]] = None  # To avoid circular dependency for now

    class Config:
        populate_by_name = True


class DocumentOut(BaseModel):
    id: str
    content: str
    level: DocumentLevel
    times_derived: int
    embedding: Optional[List[float]] = None
    source_ids: Optional[List[str]] = None
    created_at: datetime.datetime
    observer: str
    observed: str
    workspace_name: str
    session_name: Optional[str] = None
    deleted_at: Optional[datetime.datetime] = None
    perception_event_id: Optional[str] = None # New field
    internal_metadata: dict[str, Any]
    sync_state: VectorSyncState
    last_sync_at: Optional[datetime.datetime] = None
    sync_attempts: int


class WebhookEndpointIn(BaseModel):
    url: str
    workspace_name: str


class WebhookEndpointOut(BaseModel):
    id: str
    workspace_name: str
    url: str
    created_at: datetime.datetime


class EmbeddingRequest(BaseModel):
    text: str
    model: Optional[str] = None


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str


class PeerCardOut(BaseModel):
    peer_name: str
    workspace_name: str
    first_message_at: Optional[datetime.datetime] = None
    last_message_at: Optional[datetime.datetime] = None
    message_count: int
    word_count: int
    token_count: int
    summary: Optional[str] = None


class QueueItemOut(BaseModel):
    id: int
    session_id: Optional[str] = None
    work_unit_key: str
    task_type: TaskType
    payload: dict[str, Any]
    processed: bool
    error: Optional[str] = None
    created_at: datetime.datetime
    workspace_name: Optional[str] = None
    message_id: Optional[int] = None


class UpdateQueueItem(BaseModel):
    processed: bool
    error: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    agentic: bool = False
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    metadata: dict[str, Any]
    message_id: str


class ObservationEvent(BaseModel):
    id: str
    type: str
    description: str
    embedding: List[float]
    metadata: dict[str, Any]
    created_at: datetime.datetime
    derived_from_messages: List[str]
    derived_from_observations: List[str]
    peer_name: str
    workspace_name: str
    session_name: Optional[str] = None


class ObservationIn(BaseModel):
    type: str
    description: str
    embedding: List[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    derived_from_messages: List[str] = Field(default_factory=list)
    derived_from_observations: List[str] = Field(default_factory=list)
    peer_name: str
    workspace_name: str
    session_name: Optional[str] = None


class SearchResults(BaseModel):
    results: List[ObservationEvent]


class MessageWithEvents(MessageOut):
    events: List[ObservationEvent] = Field(default_factory=list)


class SessionWithMessagesAndEvents(SessionOut):
    messages: List[MessageWithEvents] = Field(default_factory=list)


class AgenticChatResponse(BaseModel):
    response: str
    metadata: dict[str, Any]
    message_id: str
    observations_created: List[ObservationEvent] = Field(default_factory=list)


class GraphSearchResults(BaseModel):
    nodes: List[dict[str, Any]]
    edges: List[dict[str, Any]]


class KnowledgeGraphRequest(BaseModel):
    peer_name: str
    workspace_name: str
    limit: int = 100
    offset: int = 0


class MultimodalObservation(BaseModel):
    modality: Literal["image", "audio", "video"]
    # Add more fields as needed for specific modalities, e.g., image_url, audio_transcript, etc.
    content_url: str
    description: Optional[str] = None
    embedding: List[float]
    metadata: Optional[dict[str, Any]] = None


class MultimodalMemorySearchRequest(BaseModel):
    query_embedding: List[float]
    modality_filter: Optional[List[Literal["image", "audio", "video"]]] = None
    top_k: int = Field(default=5, gt=0, le=100)


class MultimodalMemorySearchResult(BaseModel):
    id: str
    modality: Literal["image", "audio", "video"]
    content_url: str
    description: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    score: float


class DocumentSearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = Field(default=5, gt=0, le=100)
    session_id: Optional[str] = None
    observer: Optional[str] = None
    observed: Optional[str] = None
    level: Optional[DocumentLevel] = None


class DocumentSearchResult(BaseModel):
    document: DocumentOut
    score: float


class MessageSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=100)
    session_name: Optional[str] = None
    peer_name: Optional[str] = None
    workspace_name: str


class MessageSearchResult(BaseModel):
    message: MessageOut
    score: float


class SessionContextRequest(BaseModel):
    session_id: str
    workspace_name: str
    current_message_id: Optional[str] = None
    peer_name: Optional[str] = None
    limit_messages: int = 20
    limit_documents: int = 20
    limit_observations: int = 20


class SessionContextResponse(BaseModel):
    messages: List[MessageOut]
    documents: List[DocumentOut]
    observations: List[ObservationEvent]
    peer_card: Optional[PeerCardOut] = None


class TaskRequest(BaseModel):
    session_id: Optional[str] = None
    task_type: TaskType
    payload: dict[str, Any]
    workspace_name: Optional[str] = None


class TaskResponse(BaseModel):
    id: int
    work_unit_key: str
    task_type: TaskType
    payload: dict[str, Any]
    created_at: datetime.datetime


class WorkspaceIn(BaseModel):
    name: str
    h_metadata: Optional[dict[str, Any]] = Field(default_factory=dict, alias="metadata")
    configuration: Optional[dict[str, Any]] = Field(default_factory=dict)
    internal_metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class WorkspaceOut(BaseModel):
    id: str
    name: str
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(alias="metadata")
    configuration: dict[str, Any]
    internal_metadata: dict[str, Any]

    class Config:
        populate_by_name = True
