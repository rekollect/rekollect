"""Rekollect REST API — FastAPI wrapper around RekollectMemory."""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rekollect.memory import RekollectMemory

memory: RekollectMemory | None = None

# In-memory job store (moves to Supabase for multi-tenant)
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory
    memory = RekollectMemory(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "memory-engine-dev"),
        group_id=os.getenv("REKOLLECT_GROUP_ID", ""),
    )
    await memory.init()
    yield
    await memory.close()


app = FastAPI(
    title="Rekollect Memory Engine",
    version="0.3.0",
    description="Graph-based agent memory with hybrid search, temporal awareness, and importance scoring",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Models ─────────────────────────────────────────────────────

class RememberRequest(BaseModel):
    content: str = Field(..., description="Text to remember (any length — auto-chunks if needed)")
    source: str = Field(default="manual", description="Source label for provenance")

class RecallRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    format: str = Field(default="json", description="'json' (structured) or 'prompt' (markdown for LLM)")
    max_chars: int = Field(default=4000, ge=500, le=20000, description="Max chars when format=prompt")


# ─── Background Processing ─────────────────────────────────────

CHUNK_SIZE = 4000

async def _process_remember(job_id: str, content: str, source: str):
    """Background task: chunk content and ingest into graph."""
    job = jobs[job_id]
    try:
        job["status"] = "processing"

        # Auto-chunk if content is large
        if len(content) <= CHUNK_SIZE:
            chunks = [content]
        else:
            from rekollect.ingestion import chunk_messages
            # Wrap as a single "message" for the chunker
            msgs = [{"role": "user", "content": content}]
            chunks = chunk_messages(msgs, max_chars=CHUNK_SIZE)

        job["chunks"] = len(chunks)

        for i, chunk in enumerate(chunks):
            job["progress"] = f"{i + 1}/{len(chunks)}"
            await memory.remember(chunk, source)

        job["status"] = "done"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


# ─── Endpoints ──────────────────────────────────────────────────

@app.post("/v1/remember", status_code=202)
async def remember(req: RememberRequest):
    """Add memory. Auto-chunks large content. Returns immediately with a job ID.

    For short content (<4000 chars), processing is near-instant.
    For large content, poll GET /v1/remember/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "source": req.source,
        "content_length": len(req.content),
        "chunks": 1 if len(req.content) <= CHUNK_SIZE else None,
        "progress": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }

    asyncio.create_task(_process_remember(job_id, req.content, req.source))

    return {
        "job_id": job_id,
        "status": "pending",
        "content_length": len(req.content),
        "chunks": jobs[job_id]["chunks"],
    }


@app.get("/v1/remember/{job_id}")
async def remember_status(job_id: str):
    """Check the status of a remember job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.post("/v1/recall")
async def recall_post(req: RecallRequest):
    """Search memory with hybrid search (BM25 + vector + RRF).

    format=json: structured facts, entities, and episode citations.
    format=prompt: markdown block ready for LLM system prompt injection.
    """
    if req.format == "prompt":
        ctx = await memory.get_context(req.query, max_chars=req.max_chars)
        return {"context": ctx, "token_estimate": len(ctx) // 4}
    return await memory.recall(req.query, limit=req.limit)


@app.get("/v1/recall")
async def recall_get(
    query: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    format: str = Query("json", description="'json' or 'prompt'"),
    max_chars: int = Query(4000, ge=500, le=20000),
):
    if format == "prompt":
        ctx = await memory.get_context(query, max_chars=max_chars)
        return {"context": ctx, "token_estimate": len(ctx) // 4}
    return await memory.recall(query, limit=limit)


@app.get("/v1/stats")
async def stats():
    """Graph statistics: entity/fact/episode counts, top entities, core memories."""
    return await memory.stats()


@app.get("/v1/whats-new")
async def whats_new(since_hours: int = Query(24, ge=1, le=168)):
    """Facts added or changed in the last N hours."""
    return await memory.whats_new(since_hours)


@app.get("/v1/entities")
async def list_entities(query: str = Query(None), limit: int = Query(50, ge=1, le=200)):
    """List or search entities in the graph."""
    g = memory.graphiti
    async with g.driver.session() as s:
        if query:
            result = await s.run(
                "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($q) "
                "RETURN n.name AS name, n.entity_type AS entity_type, n.summary AS summary LIMIT $limit",
                {"q": query, "limit": limit},
            )
        else:
            result = await s.run(
                "MATCH (n:Entity) RETURN n.name AS name, n.entity_type AS entity_type, "
                "n.summary AS summary ORDER BY n.name LIMIT $limit",
                {"limit": limit},
            )
        return [r.data() async for r in result]


@app.get("/v1/timeline")
async def timeline(entity: str = Query(...), limit: int = Query(20, ge=1, le=100)):
    """Chronological facts about an entity."""
    g = memory.graphiti
    async with g.driver.session() as s:
        result = await s.run(
            "MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity) WHERE toLower(n.name) = toLower($name) "
            "RETURN r.fact AS fact, r.valid_at AS valid_from, r.invalid_at AS invalid_at, "
            "r.name AS relation, m.name AS related_entity ORDER BY r.valid_at DESC LIMIT $limit",
            {"name": entity, "limit": limit},
        )
        facts = [
            {**r.data(), "is_current": r.data().get("invalid_at") is None}
            async for r in result
        ]
    return {"entity": entity, "facts": facts}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rekollect", "version": "0.3.0"}
