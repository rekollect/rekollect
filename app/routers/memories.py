"""CRUD endpoints for memories (documents + chunks)."""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.storage import postgres

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Response schemas ---

class MemorySummary(BaseModel):
    id: UUID
    title: str | None = None
    type: str | None = None
    collection: str | None = None
    source: str | None = None
    created_at: str | None = None
    chunk_count: int = 0
    status: str | None = None


class MemoryListResponse(BaseModel):
    memories: list[MemorySummary]
    total: int
    limit: int
    offset: int


class MemoryDetail(BaseModel):
    document: dict[str, Any]
    chunks: list[dict[str, Any]]


class DeleteResponse(BaseModel):
    id: UUID
    deleted: bool = True


class BulkDeleteResponse(BaseModel):
    deleted_count: int


# --- Endpoints ---

@router.get("/memories", response_model=MemoryListResponse, operation_id="list_memories")
async def list_memories(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    type: str | None = Query(None, description="Filter by type: text, url, file, chat"),
    collection: str | None = Query(None, description="Filter by collection"),
    user_id: str = Depends(get_current_user),
):
    """List stored memories with optional filters. Use to browse what has been
    saved, verify storage, or find a memory ID for update/delete operations."""
    pool = request.app.state.pool
    data = await postgres.list_documents(pool, user_id, limit, offset, type, collection)

    memories = [
        MemorySummary(
            id=d["id"],
            title=d.get("user_title"),
            type=d.get("file_type") or (d.get("metadata") or {}).get("type", "text"),
            collection=d.get("collection"),
            source=(d.get("metadata") or {}).get("source"),
            created_at=d.get("created_at"),
            chunk_count=data["chunk_counts"].get(str(d["id"]), 0),
            status=d.get("processing_status"),
        )
        for d in data["rows"]
    ]

    return MemoryListResponse(memories=memories, total=data["total"], limit=limit, offset=offset)


@router.get("/memories/{memory_id}", response_model=MemoryDetail, operation_id="get_memory")
async def get_memory(
    memory_id: UUID,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Retrieve a specific memory by ID including its full content and chunks."""
    pool = request.app.state.pool
    result = await postgres.get_document_with_chunks(pool, str(memory_id), user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryDetail(**result)


@router.delete("/memories/{memory_id}", response_model=DeleteResponse, operation_id="delete_memory")
async def delete_memory(
    memory_id: UUID,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Delete a specific memory permanently. Use when information is outdated,
    incorrect, or no longer needed."""
    pool = request.app.state.pool
    deleted = await postgres.delete_document(pool, str(memory_id), user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return DeleteResponse(id=memory_id)


@router.delete("/memories", response_model=BulkDeleteResponse, operation_id="delete_all_memories")
async def delete_all_memories(
    request: Request,
    confirm: bool = Query(False, description="Must be true to confirm bulk deletion"),
    collection: str | None = Query(None, description="Delete only this collection. Omit to delete all."),
    user_id: str = Depends(get_current_user),
):
    """Delete all stored memories. Requires confirm=true parameter. Use with
    caution -- this permanently removes all memories for the current user.
    Optionally filter by collection to delete only that collection's documents."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Bulk delete requires confirm=true query parameter",
        )
    pool = request.app.state.pool
    count = await postgres.delete_user_documents(pool, user_id, collection)
    return BulkDeleteResponse(deleted_count=count)
