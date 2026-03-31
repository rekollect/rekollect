"""Documents table CRUD via asyncpg.

Module-level functions that accept an asyncpg pool as first arg.
Replaces Supabase client calls with direct SQL.
"""

import json
import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


async def create_document(
    pool: asyncpg.Pool,
    user_id: str,
    content: str,
    doc_type: str = "text",
    source: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    collection: str = "default",
) -> dict:
    """Create a text document. Returns the full document record."""
    meta = json.dumps({**(metadata or {}), "source": source, "type": doc_type})
    row = await pool.fetchrow(
        """
        INSERT INTO documents (user_id, user_title, user_text, collection, processing_status, metadata)
        VALUES ($1, $2, $3, $4, 'processing', $5::jsonb)
        RETURNING *
        """,
        user_id,
        title,
        content,
        collection,
        meta,
    )
    return dict(row)


async def get_document(pool: asyncpg.Pool, doc_id: str) -> Optional[dict]:
    row = await pool.fetchrow("SELECT * FROM documents WHERE id = $1", doc_id)
    return dict(row) if row else None


async def update_status(pool: asyncpg.Pool, doc_id: str, status: str, **extra):
    sets = ["processing_status = $2", "updated_at = now()"]
    params: list = [doc_id, status]
    idx = 3
    for key, value in extra.items():
        sets.append(f"{key} = ${idx}")
        params.append(value)
        idx += 1
    query = f"UPDATE documents SET {', '.join(sets)} WHERE id = $1"
    await pool.execute(query, *params)


async def get_pending(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        "SELECT * FROM documents WHERE processing_status = 'pending'"
    )
    return [dict(r) for r in rows]


async def get_user_collections(pool: asyncpg.Pool, user_id: str) -> list[dict]:
    """Return distinct collections for a user with document counts."""
    rows = await pool.fetch(
        """
        SELECT collection, COUNT(*) AS document_count
        FROM documents
        WHERE user_id = $1
        GROUP BY collection
        ORDER BY collection
        """,
        user_id,
    )
    return [dict(r) for r in rows]


async def get_chunk_counts(pool: asyncpg.Pool, doc_ids: list[str]) -> dict[str, int]:
    """Return chunk counts grouped by document_id."""
    if not doc_ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT document_id::text, COUNT(*) AS chunk_count
        FROM chunks
        WHERE document_id = ANY($1::uuid[])
        GROUP BY document_id
        """,
        doc_ids,
    )
    return {r["document_id"]: r["chunk_count"] for r in rows}


async def list_documents(
    pool: asyncpg.Pool,
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    doc_type: Optional[str] = None,
    collection: Optional[str] = None,
) -> dict:
    """List documents with count, pagination, and optional filters.

    Returns {"rows": [...], "total": int, "chunk_counts": {doc_id: int}}.
    """
    conditions = ["user_id = $1"]
    params: list = [user_id]
    idx = 2

    if doc_type:
        conditions.append(f"file_type = ${idx}")
        params.append(doc_type)
        idx += 1
    if collection:
        conditions.append(f"collection = ${idx}")
        params.append(collection)
        idx += 1

    where = " AND ".join(conditions)

    # Count
    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS total FROM documents WHERE {where}", *params
    )
    total = count_row["total"]

    # Data
    params_data = list(params) + [limit, offset]
    rows = await pool.fetch(
        f"""
        SELECT id, user_title, file_type, collection, metadata, created_at, processing_status
        FROM documents
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params_data,
    )
    rows = [dict(r) for r in rows]

    # Serialize metadata (jsonb comes back as dict, but id comes back as UUID)
    for r in rows:
        r["id"] = str(r["id"])
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat()

    doc_ids = [r["id"] for r in rows]
    chunk_counts = await get_chunk_counts(pool, doc_ids)

    return {"rows": rows, "total": total, "chunk_counts": chunk_counts}


async def get_document_with_chunks(
    pool: asyncpg.Pool, doc_id: str, user_id: str
) -> Optional[dict]:
    """Fetch a single document (owned by user_id) with all its chunks."""
    doc = await pool.fetchrow(
        "SELECT * FROM documents WHERE id = $1 AND user_id = $2",
        doc_id,
        user_id,
    )
    if not doc:
        return None

    chunks = await pool.fetch(
        "SELECT * FROM chunks WHERE document_id = $1 ORDER BY chunk_index",
        doc_id,
    )

    return {
        "document": _serialize_row(doc),
        "chunks": [_serialize_row(c) for c in chunks],
    }


async def delete_document(pool: asyncpg.Pool, doc_id: str, user_id: str) -> bool:
    """Delete a document and its chunks. Returns False if not owned / not found."""
    doc = await pool.fetchrow(
        "SELECT id FROM documents WHERE id = $1 AND user_id = $2",
        doc_id,
        user_id,
    )
    if not doc:
        return False

    # Cascading delete handles chunks via FK, but be explicit for embeddings
    await pool.execute("DELETE FROM embeddings WHERE document_id = $1", doc_id)
    await pool.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
    await pool.execute("DELETE FROM documents WHERE id = $1", doc_id)
    return True


async def delete_user_documents(
    pool: asyncpg.Pool, user_id: str, collection: Optional[str] = None
) -> int:
    """Bulk-delete all documents (and their chunks) for a user."""
    conditions = ["user_id = $1"]
    params: list = [user_id]
    if collection:
        conditions.append("collection = $2")
        params.append(collection)

    where = " AND ".join(conditions)

    # Get doc IDs first
    rows = await pool.fetch(
        f"SELECT id FROM documents WHERE {where}", *params
    )
    doc_ids = [str(r["id"]) for r in rows]

    if not doc_ids:
        return 0

    # Delete embeddings, chunks, then documents
    await pool.execute(
        "DELETE FROM embeddings WHERE document_id = ANY($1::uuid[])", doc_ids
    )
    await pool.execute(
        "DELETE FROM chunks WHERE document_id = ANY($1::uuid[])", doc_ids
    )
    await pool.execute(
        f"DELETE FROM documents WHERE {where}", *params
    )

    return len(doc_ids)


def _serialize_row(row: asyncpg.Record) -> dict:
    """Convert an asyncpg Record to a JSON-safe dict."""
    d = dict(row)
    for k, v in d.items():
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
        elif hasattr(v, "hex"):  # UUID
            d[k] = str(v)
    return d
