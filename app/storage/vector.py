"""Vector storage using pgvector — direct SQL via asyncpg."""

import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


async def insert_chunks(
    pool: asyncpg.Pool,
    document_id: str,
    user_id: str,
    chunks: list[dict],
    collection: str = "default",
    **kwargs,
):
    """Insert chunks and embeddings into Postgres."""
    for chunk in chunks:
        # Insert chunk
        chunk_row = await pool.fetchrow(
            """
            INSERT INTO chunks (document_id, content, chunk_index)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            document_id,
            chunk["content"],
            chunk["index"],
        )
        if not chunk_row:
            logger.warning(f"Failed to insert chunk {chunk['index']} for doc {document_id}")
            continue

        chunk_id = chunk_row["id"]

        # Insert embedding
        embedding_str = "[" + ",".join(str(x) for x in chunk["embedding"]) + "]"
        await pool.execute(
            """
            INSERT INTO embeddings (chunk_id, document_id, user_id, embedding)
            VALUES ($1, $2, $3, $4::vector)
            """,
            chunk_id,
            document_id,
            user_id,
            embedding_str,
        )


async def search(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    user_id: Optional[str] = None,
    limit: int = 10,
    threshold: float = 0.1,
    collection: Optional[str] = None,
) -> list[dict]:
    """Semantic search using pgvector cosine similarity."""
    try:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        conditions = []
        params: list = [embedding_str, limit]
        idx = 3

        if user_id:
            conditions.append(f"e.user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if collection:
            conditions.append(f"d.collection = ${idx}")
            params.append(collection)
            idx += 1

        where = (" AND " + " AND ".join(conditions)) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT
                c.id AS chunk_id,
                c.content AS chunk_text,
                c.document_id,
                d.collection,
                d.created_at,
                1 - (e.embedding <=> $1::vector) AS similarity
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.id
            JOIN documents d ON e.document_id = d.id
            WHERE 1=1 {where}
            ORDER BY e.embedding <=> $1::vector
            LIMIT $2
            """,
            *params,
        )

        results = []
        for r in rows:
            if r["similarity"] < threshold:
                continue
            results.append({
                "chunk_id": str(r["chunk_id"]),
                "chunk_text": r["chunk_text"],
                "document_id": str(r["document_id"]),
                "collection": r["collection"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "similarity": float(r["similarity"]),
            })
        return results
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


async def delete_by_document(pool: asyncpg.Pool, document_id: str):
    """Delete all chunks and embeddings for a document."""
    await pool.execute("DELETE FROM embeddings WHERE document_id = $1", document_id)
    await pool.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
