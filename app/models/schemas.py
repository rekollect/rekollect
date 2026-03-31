"""Pydantic models for all requests/responses."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class AddRequest(BaseModel):
    type: str = Field(
        "text",
        description="Content type: 'text' for notes/facts/conversations, 'url' for web pages, 'chat' for full conversation transcripts, 'file' for documents",
    )
    content: str | None = Field(
        None,
        description="The content to store. For chat/session saves, include the full conversation text.",
    )
    content_json: dict[str, Any] | None = Field(
        None,
        description="Structured content as JSON. Use instead of content for structured data.",
    )
    source: str | None = Field(
        None,
        description="Origin of the content: a URL, app name, tool name, or person.",
    )
    title: str | None = Field(
        None,
        description="Optional title or label. For session saves, use the project/task name.",
    )
    collection: str = Field(
        "default",
        pattern=r"^[a-z0-9][a-z0-9_-]{0,49}$",
        description="Collection to organize content into: 'snippets', 'architecture', 'runbooks', etc. Defaults to 'default'.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key-value metadata: tags, project, date, agent name, etc.",
    )


class AddResponse(BaseModel):
    id: UUID
    status: str
    chunks: int
    entities: int
    document: dict[str, Any] | None = None


class RecallRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language search query. Ask like you're looking for something: 'Cooper Flagg stats', 'why did we choose FastAPI', 'Italian restaurant Main Street'.",
    )
    collection: str | None = Field(
        None,
        description="Search only this collection (e.g. 'snippets'). Omit to search all collections.",
    )
    limit: int = Field(
        10,
        description="Max results to return. Default 10. Increase for broader recall.",
    )
    include_raw: bool = False


class RecallResult(BaseModel):
    type: Literal["chunk", "fact"]
    content: str
    score: float | None = None
    source: str | None = None
    entity: str | None = None
    valid_from: str | None = None
    document_id: UUID | None = None
    created_at: str | None = None
    document_title: str | None = None
    document_type: str | None = None  # text, file, url, chat
    collection: str | None = None
    chunk_id: UUID | None = None


class RecallResponse(BaseModel):
    results: list[RecallResult]


class ProcessRequest(BaseModel):
    id: UUID
    reprocess: bool = False


class ProcessResponse(BaseModel):
    id: UUID
    status: str
    chunks: int = 0
    entities: int = 0


class RememberResponse(BaseModel):
    """Response for /v1/remember -- add + process combined."""
    id: UUID
    status: str
    chunks: int
    entities: int
    document: dict[str, Any] | None = None


# --- API Key schemas ---

class ApiKeyCreate(BaseModel):
    name: str | None = Field(
        None,
        description="A label for this key, e.g. 'My Claude Desktop', 'Work Cursor'.",
    )


class ApiKeyResponse(BaseModel):
    """Returned when listing keys. Never includes the full key."""
    id: UUID
    key_prefix: str
    name: str | None
    last_used_at: datetime | None
    created_at: datetime


class ApiKeyCreated(BaseModel):
    """Returned once on creation -- the only time the full key is visible."""
    id: UUID
    key: str
    key_prefix: str
    name: str | None
    created_at: datetime


# --- Collection schemas ---

class CollectionSummary(BaseModel):
    collection: str
    document_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionSummary]
