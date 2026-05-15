"""Pydantic schemas for the multimodal perception API."""

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PerceptionIngestRequest(BaseModel):
    session_id: str | None = None  # ignored; session resolved from URL path
    source_type: Literal["video_1fps", "audio_vad", "remote_stream"]
    salience_score: float = Field(ge=0.0, le=1.0)
    fingerprint: list[float] | None = Field(default=None, min_length=512, max_length=512)
    fingerprint_bq: str | None = None
    metadata: dict[str, Any] | None = None

    # Stream segmentation fields (Phase 2)
    captured_at: datetime.datetime | None = None
    segment_id: str | None = None
    is_segment_start: bool = False
    is_segment_end: bool = False


class PerceptionSearchRequest(BaseModel):
    query_fingerprint: list[float] = Field(min_length=512, max_length=512)
    session_id: str | None = None
    top_k: int = Field(default=5, gt=0, le=100)


class PerceptionEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    source_type: str
    salience_score: float
    fingerprint_bq: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime.datetime
    is_state_change: bool = False

    # Stream segmentation (echoed back)
    captured_at: datetime.datetime | None = None
    segment_id: str | None = None
    is_segment_start: bool = False
    is_segment_end: bool = False


class PerceptionEventOut(PerceptionEventResponse):
    similarity_score: float | None = None
    associated_document_ids: list[str] = Field(default_factory=list)
    associated_message_ids: list[str] = Field(default_factory=list)
