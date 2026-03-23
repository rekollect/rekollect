"""Rekollect Memory Engine REST API — FastAPI wrapper around JarvisMemory."""
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from jarvis_memory import JarvisMemory

# ─── Globals ────────────────────────────────────────────────────

memory: JarvisMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory
    memory = JarvisMemory()
    await memory.init()
    yield
    await memory.close()


app = FastAPI(
    title="Rekollect Memory Engine",
    version="0.2.0",
    description="Graph-based agent memory with hybrid search, temporal awareness, and importance scoring",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ────────────────────────────────────

class RememberRequest(BaseModel):
    content: str = Field(..., description="Text to remember")
    source: str = Field(default="manual", description="Source identifier")


class RememberResponse(BaseModel):
    status: str
    episodes: int = 0


class IngestSessionRequest(BaseModel):
    session_path: str = Field(..., description="Path to OpenClaw .jsonl session file")


class IngestBatchRequest(BaseModel):
    directory: str = Field(..., description="Path to directory of .jsonl session files")
    since_hours: int | None = Field(None, description="Only ingest files modified in last N hours")


class IngestBatchResponse(BaseModel):
    total_files: int
    ingested: int
    episodes: int
    errors: list[dict] = []


class RecallRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)


class ContextRequest(BaseModel):
    query: str
    max_chars: int = Field(default=4000, ge=500, le=20000)


class ContextResponse(BaseModel):
    context: str
    token_estimate: int


class StatsResponse(BaseModel):
    entities: int
    facts: int
    episodes: int
    top_entities: list[dict] = []
    core_memories: list[dict] = []


class EntityInfo(BaseModel):
    name: str
    entity_type: str | None = None
    summary: str | None = None


# ─── Endpoints ──────────────────────────────────────────────────

@app.post("/v1/remember", response_model=RememberResponse)
async def remember(req: RememberRequest):
    """Add a memory manually."""
    await memory.remember(req.content, req.source)
    return RememberResponse(status="remembered", episodes=1)


@app.post("/v1/ingest/session", response_model=RememberResponse)
async def ingest_session(req: IngestSessionRequest):
    """Ingest an OpenClaw session log into the memory graph."""
    result = await memory.ingest_session(req.session_path)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return RememberResponse(status="ingested", episodes=result.get("episodes", 0))


@app.post("/v1/ingest/batch", response_model=IngestBatchResponse)
async def ingest_batch(req: IngestBatchRequest):
    """Ingest all .jsonl session files from a directory."""
    result = await memory.ingest_sessions_batch(req.directory, req.since_hours)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return IngestBatchResponse(**result)


@app.post("/v1/recall")
async def recall(req: RecallRequest):
    """Hybrid search the memory graph (BM25 + vector + RRF).
    
    Returns facts (edges), entities (nodes), and episode snippets (citations).
    Automatically updates importance metrics on recalled facts.
    """
    return await memory.recall(req.query, limit=req.limit)


@app.get("/v1/recall")
async def recall_get(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """GET version of recall for easy testing."""
    return await memory.recall(query, limit=limit)


@app.post("/v1/context", response_model=ContextResponse)
async def get_context(req: ContextRequest):
    """Assemble a token-aware context block for LLM prompts."""
    ctx = await memory.get_context(req.query, max_chars=req.max_chars)
    return ContextResponse(context=ctx, token_estimate=len(ctx) // 4)


@app.get("/v1/context")
async def get_context_get(
    query: str = Query(..., description="Context query"),
    max_chars: int = Query(4000, ge=500, le=20000),
):
    """GET version of context assembly."""
    ctx = await memory.get_context(query, max_chars=max_chars)
    return ContextResponse(context=ctx, token_estimate=len(ctx) // 4)


@app.get("/v1/stats", response_model=StatsResponse)
async def stats():
    """Get memory graph stats including top entities and core memories."""
    return await memory.stats()


@app.get("/v1/whats-new")
async def whats_new(since_hours: int = Query(24, ge=1, le=168)):
    """Get facts added or changed in the last N hours."""
    return await memory.whats_new(since_hours)


@app.get("/v1/entities", response_model=list[EntityInfo])
async def list_entities(
    query: str = Query(None, description="Filter by name"),
    limit: int = Query(50, ge=1, le=200),
):
    """List entities in the graph."""
    g = memory.graphiti
    async with g.driver.session() as s:
        if query:
            result = await s.run(
                "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($q) "
                "RETURN n.name AS name, n.entity_type AS entity_type, n.summary AS summary "
                "LIMIT $limit",
                {"q": query, "limit": limit},
            )
        else:
            result = await s.run(
                "MATCH (n:Entity) RETURN n.name AS name, n.entity_type AS entity_type, "
                "n.summary AS summary ORDER BY n.name LIMIT $limit",
                {"limit": limit},
            )
        records = [r.data() async for r in result]
    return [EntityInfo(**r) for r in records]


@app.get("/v1/timeline")
async def timeline(
    entity: str = Query(..., description="Entity name"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get chronological facts about an entity."""
    g = memory.graphiti
    async with g.driver.session() as s:
        result = await s.run(
            """
            MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity)
            WHERE toLower(n.name) = toLower($name)
            RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at,
                   r.name AS relation, m.name AS related_entity
            ORDER BY r.valid_at DESC
            LIMIT $limit
            """,
            {"name": entity, "limit": limit},
        )
        records = [r.data() async for r in result]

    return {
        "entity": entity,
        "facts": [
            {
                "fact": r["fact"],
                "related_to": r["related_entity"],
                "relation": r["relation"],
                "valid_from": str(r["valid_at"]) if r["valid_at"] else None,
                "invalid_at": str(r["invalid_at"]) if r["invalid_at"] else None,
                "is_current": r["invalid_at"] is None,
            }
            for r in records
        ],
    }


# ─── Health ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rekollect-memory-engine", "version": "0.2.0"}
