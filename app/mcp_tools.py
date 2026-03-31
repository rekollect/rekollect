"""MCP tool definitions for Rekollect.

Each tool calls the service layer directly -- no HTTP round-trip.
User identity comes from API key in the bearer token.
"""

import logging

import bcrypt
import asyncpg
from fastmcp.server.dependencies import get_access_token

from app.services.memory import store_document, process_document, search_memories
from app.storage import postgres

logger = logging.getLogger(__name__)

# Shared state populated by FastAPI lifespan
_state: dict = {}


def init(settings, pool, graph):
    """Called from FastAPI lifespan to share state with MCP tools."""
    _state["settings"] = settings
    _state["pool"] = pool
    _state["graph"] = graph


async def _get_user_context() -> tuple[str, asyncpg.Pool]:
    """Read API key from bearer token, return (user_id, pool)."""
    if not _state:
        raise RuntimeError("MCP tools not initialized")

    pool: asyncpg.Pool = _state["pool"]

    token = get_access_token()
    if token is None:
        raise ValueError("Not authenticated — provide API key as bearer token")

    # get_access_token() returns a Token object; extract the raw string
    raw_token = token.token if hasattr(token, "token") else str(token)

    # Look up API key by checking bcrypt hash
    rows = await pool.fetch(
        "SELECT user_id, key_hash FROM api_keys WHERE revoked_at IS NULL"
    )
    for row in rows:
        if bcrypt.checkpw(raw_token.encode(), row["key_hash"].encode()):
            return str(row["user_id"]), pool

    raise ValueError("Invalid API key")


def register(mcp):
    """Register all MCP tools on the given FastMCP instance."""

    @mcp.tool
    async def remember(
        content: str,
        title: str | None = None,
        type: str = "text",
        source: str | None = None,
        collection: str = "default",
        metadata: dict | None = None,
    ) -> dict:
        """Save anything worth remembering -- notes, ideas, decisions, code patterns,
        research findings, meeting notes, or full conversations. Call proactively."""
        if not content or not content.strip():
            return {"error": "Content is required"}
        user_id, pool = await _get_user_context()
        doc = await store_document(
            pool, user_id=user_id, content=content,
            doc_type=type, source=source, title=title,
            metadata=metadata or {}, collection=collection,
        )
        result = await process_document(
            pool, _state["graph"], _state["settings"], doc["id"], reprocess=False,
        )
        return {
            "id": result.doc_id, "status": result.status,
            "chunks": result.chunks, "entities": result.entities,
            "document": result.document,
        }

    @mcp.tool
    async def recall(
        query: str,
        limit: int = 10,
        collection: str | None = None,
    ) -> list[dict]:
        """Search your personal memory. Use before starting any task, when answering
        questions about past decisions, or anytime prior context would help."""
        user_id, pool = await _get_user_context()
        return await search_memories(
            pool, _state["graph"], _state["settings"],
            user_id=user_id, query=query, limit=limit, collection=collection,
        )

    @mcp.tool
    async def add(
        content: str,
        title: str | None = None,
        type: str = "text",
        source: str | None = None,
        collection: str = "default",
        metadata: dict | None = None,
    ) -> dict:
        """Store content without processing. Use for large batches or when you
        want to control when processing happens. Prefer remember for most cases."""
        user_id, pool = await _get_user_context()
        doc = await store_document(
            pool, user_id=user_id, content=content,
            doc_type=type, source=source, title=title,
            metadata=metadata or {}, collection=collection,
        )
        return {"id": doc["id"], "status": doc["processing_status"]}

    @mcp.tool
    async def process(id: str, reprocess: bool = False) -> dict:
        """Process a document: chunk, embed, extract to graph, dedup."""
        user_id, pool = await _get_user_context()
        doc = await postgres.get_document(pool, id)
        if not doc:
            return {"error": "Document not found or not authorized"}
        result = await process_document(
            pool, _state["graph"], _state["settings"], id, reprocess=reprocess,
        )
        return {
            "id": result.doc_id, "status": result.status,
            "chunks": result.chunks, "entities": result.entities,
        }

    @mcp.tool
    async def list_memories(
        limit: int = 20,
        offset: int = 0,
        type: str | None = None,
        collection: str | None = None,
    ) -> dict:
        """List stored memories with optional filters."""
        user_id, pool = await _get_user_context()
        result = await postgres.list_documents(
            pool, user_id,
            limit=limit, offset=offset, doc_type=type, collection=collection,
        )
        rows = result["rows"]
        chunk_counts = result["chunk_counts"]
        return {
            "memories": [
                {
                    "id": d["id"], "title": d.get("user_title"),
                    "type": d.get("file_type"), "collection": d.get("collection"),
                    "source": (d.get("metadata") or {}).get("source"),
                    "created_at": d.get("created_at"),
                    "chunk_count": chunk_counts.get(str(d["id"]), 0),
                    "status": d.get("processing_status"),
                }
                for d in rows
            ],
            "total": result["total"], "limit": limit, "offset": offset,
        }

    @mcp.tool
    async def get_memory(memory_id: str) -> dict:
        """Retrieve a specific memory by ID including full content and chunks."""
        user_id, pool = await _get_user_context()
        result = await postgres.get_document_with_chunks(pool, memory_id, user_id)
        if not result:
            return {"error": "Memory not found"}
        return result

    @mcp.tool
    async def delete_memory(memory_id: str) -> dict:
        """Delete a specific memory permanently."""
        user_id, pool = await _get_user_context()
        deleted = await postgres.delete_document(pool, memory_id, user_id)
        if not deleted:
            return {"error": "Memory not found"}
        return {"id": memory_id, "deleted": True}

    @mcp.tool
    async def delete_all_memories(confirm: bool = False, collection: str | None = None) -> dict:
        """Delete all stored memories. Requires confirm=true."""
        if not confirm:
            return {"error": "Must pass confirm=true to delete all memories"}
        user_id, pool = await _get_user_context()
        count = await postgres.delete_user_documents(pool, user_id, collection=collection)
        return {"deleted_count": count}

    @mcp.tool
    async def list_collections() -> list[dict]:
        """List all collections with document counts."""
        user_id, pool = await _get_user_context()
        rows = await postgres.get_user_collections(pool, user_id)
        return [{"collection": r["collection"], "document_count": r["document_count"]} for r in rows]
