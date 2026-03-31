"""Shared service layer for document storage and processing.

Extracted from add.py and process.py to avoid code duplication.
Used by /v1/add, /v1/process, /v1/remember, and /v1/recall endpoints.
"""

import json
import logging
from typing import Any, Optional
from uuid import UUID

import asyncpg
from fastapi import HTTPException
from openai import AsyncOpenAI

from app.graph.dedup import ensure_vector_index, run_dedup_sweep
from app.graph.engine import GraphEngine
from app.models.schemas import AddRequest
from app.processing.chunker import chunk_text
from app.storage import postgres, vector

logger = logging.getLogger(__name__)


def resolve_content(req: AddRequest) -> str:
    """Extract text content from an AddRequest, raising 400 if empty."""
    content = req.content
    if not content and req.content_json:
        content = json.dumps(req.content_json, indent=2)
    if not content:
        raise HTTPException(status_code=400, detail="content or content_json required")
    return content


async def embed_texts(settings, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await oai.embeddings.create(
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        input=texts,
    )
    return [e.embedding for e in resp.data]


class ProcessingResult:
    """Result of document processing."""

    def __init__(
        self,
        doc_id: UUID,
        status: str,
        chunks: int = 0,
        entities: int = 0,
        document: Optional[dict[str, Any]] = None,
    ):
        self.doc_id = doc_id
        self.status = status
        self.chunks = chunks
        self.entities = entities
        self.document = document


async def store_document(
    pool: asyncpg.Pool,
    user_id: str,
    content: str,
    doc_type: str = "text",
    source: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    collection: str = "default",
) -> dict:
    """Store a document without processing.

    Sets processing_status to 'pending' (not 'processing').
    Returns the full document record.
    """
    doc = await postgres.create_document(
        pool,
        user_id=user_id,
        content=content,
        doc_type=doc_type,
        source=source,
        title=title,
        metadata=metadata,
        collection=collection,
    )

    # Override status to pending (create_document sets to 'processing')
    await postgres.update_status(pool, doc["id"], "pending")

    # Fetch updated document
    updated_doc = await postgres.get_document(pool, doc["id"])
    return updated_doc


async def process_document(
    pool: asyncpg.Pool,
    graph,
    settings,
    doc_id: str,
    reprocess: bool = False,
) -> ProcessingResult:
    """Process a document: chunk, embed, insert to pgvector, extract to graph, dedup."""
    # Fetch document
    doc = await postgres.get_document(pool, str(doc_id))
    if not doc:
        raise ValueError(f"Document {doc_id} not found")

    doc_id_str = str(doc["id"])
    doc_user_id = str(doc["user_id"])

    try:
        # Mark as processing
        await postgres.update_status(pool, doc_id_str, "processing")

        # Delete old chunks if reprocessing
        if reprocess:
            await vector.delete_by_document(pool, doc_id_str)

        # Get content
        content = doc.get("user_text", "")
        if not content:
            await postgres.update_status(pool, doc_id_str, "completed")
            full_doc = await postgres.get_document(pool, doc_id_str)
            return ProcessingResult(
                doc_id=doc_id_str,
                status="completed",
                chunks=0,
                entities=0,
                document=full_doc,
            )

        # Chunk
        chunks = chunk_text(content)
        if not chunks:
            await postgres.update_status(pool, doc_id_str, "completed")
            full_doc = await postgres.get_document(pool, doc_id_str)
            return ProcessingResult(
                doc_id=doc_id_str,
                status="completed",
                chunks=0,
                entities=0,
                document=full_doc,
            )

        # Embed
        embeddings = await embed_texts(settings, chunks)

        # Insert chunks
        chunk_rows = [
            {"content": c, "embedding": emb, "index": i}
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        doc_collection = doc.get("collection", "default")
        await vector.insert_chunks(
            pool, doc_id_str, doc_user_id, chunk_rows,
            collection=doc_collection,
        )

        # Graph extraction
        entity_count = 0
        if graph:
            source = (doc.get("metadata") or {}).get("source") or "rekollect-api"
            entity_count = await graph.add_episodes(
                chunks, source=source,
                user_id=doc_user_id,
                collection=doc_collection,
            )

        # Dedup sweep
        try:
            if graph:
                driver = graph.graphiti.driver
                embedder = graph.graphiti.embedder
                await ensure_vector_index(driver, settings.embedding_dim)
                await run_dedup_sweep(
                    driver, embedder, dedup_model=settings.dedup_model
                )
        except Exception as e:
            logger.warning(f"Dedup sweep failed (non-fatal): {e}")

        # Mark completed
        await postgres.update_status(pool, doc_id_str, "completed")

        # Fetch full document for response
        full_doc = await postgres.get_document(pool, doc_id_str)

        return ProcessingResult(
            doc_id=doc_id_str,
            status="completed",
            chunks=len(chunks),
            entities=entity_count,
            document=full_doc,
        )

    except Exception as e:
        logger.error(f"Processing failed for {doc_id_str}: {e}")
        await postgres.update_status(pool, doc_id_str, "failed")
        raise


async def search_memories(
    pool: asyncpg.Pool,
    graph,
    settings,
    user_id: str,
    query: str,
    limit: int = 10,
    collection: Optional[str] = None,
) -> list[dict]:
    """Search pgvector + graph, merge and deduplicate results.

    Returns plain dicts (not Pydantic models) so both the HTTP route and
    MCP tools can consume the output directly.
    """
    # 1. Embed query
    query_embedding = (await embed_texts(settings, [query]))[0]

    results: list[dict] = []

    # 2. pgvector similarity search
    try:
        chunk_results = await vector.search(
            pool, query_embedding, user_id=user_id, limit=limit,
            collection=collection,
        )

        # Collect document IDs for batch fetching
        doc_ids = list(
            {c.get("document_id") for c in chunk_results if c.get("document_id")}
        )

        # Batch-fetch document metadata
        doc_lookup: dict[str, dict] = {}
        if doc_ids:
            doc_rows = await pool.fetch(
                """
                SELECT id::text, user_title, file_type
                FROM documents
                WHERE id = ANY($1::uuid[])
                """,
                doc_ids,
            )
            for doc in doc_rows:
                doc_lookup[doc["id"]] = {
                    "title": doc.get("user_title"),
                    "type": doc.get("file_type") or "text",
                }

        # Build enriched results
        for c in chunk_results:
            doc_id = c.get("document_id")
            doc_meta = doc_lookup.get(str(doc_id)) if doc_id else {}

            results.append({
                "type": "chunk",
                "content": c.get("chunk_text", c.get("content", "")),
                "score": c.get("similarity"),
                "document_id": doc_id,
                "created_at": c.get("created_at"),
                "chunk_id": c.get("chunk_id"),
                "document_title": (doc_meta or {}).get("title"),
                "document_type": (doc_meta or {}).get("type"),
                "collection": c.get("collection"),
                "entity": None,
                "valid_from": None,
            })
    except Exception as e:
        logger.warning(f"pgvector search failed: {e}")

    # 3. Graphiti search for facts (if available)
    if graph:
        try:
            # Build group_ids for user-scoped graph search
            if collection:
                group_ids = [GraphEngine.make_group_id(user_id, collection)]
            else:
                user_collections = await postgres.get_user_collections(pool, user_id)
                if user_collections:
                    group_ids = GraphEngine.make_group_ids(
                        user_id, [c["collection"] for c in user_collections]
                    )
                else:
                    group_ids = [GraphEngine.make_group_id(user_id, "default")]

            facts = await graph.search(query, limit=limit, group_ids=group_ids)
            for f in facts:
                results.append({
                    "type": "fact",
                    "content": f["content"],
                    "score": None,
                    "document_id": None,
                    "created_at": f.get("created_at"),
                    "chunk_id": None,
                    "document_title": None,
                    "document_type": None,
                    "collection": None,
                    "entity": f.get("entity"),
                    "valid_from": f.get("valid_from"),
                })
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")

    # 4. Deduplicate by content
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results:
        key = r["content"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # 5. Sort: score desc (facts without score go last)
    unique.sort(key=lambda r: (r["score"] or 0), reverse=True)

    return unique[:limit]
