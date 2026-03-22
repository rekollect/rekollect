"""Memory Engine REST API — FastAPI wrapper around JarvisMemory."""
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
    title="Memory Engine",
    version="0.1.0",
    description="Graph-based agent memory with temporal awareness and citations",
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


class RecallResult(BaseModel):
    fact: str
    name: str | None = None
    valid_from: str | None = None
    invalid_at: str | None = None
    is_current: bool = True
    source: str | None = None
    target: str | None = None


class ContextResponse(BaseModel):
    context: str
    token_estimate: int


class EntityInfo(BaseModel):
    name: str
    entity_type: str | None = None
    summary: str | None = None


class StatsResponse(BaseModel):
    entities: int
    facts: int
    episodes: int


# ─── Endpoints ──────────────────────────────────────────────────

@app.post("/v1/remember", response_model=RememberResponse)
async def remember(req: RememberRequest):
    """Add a memory manually."""
    result = await memory.remember(req.content, req.source)
    return RememberResponse(status="remembered", episodes=1)


@app.post("/v1/ingest", response_model=RememberResponse)
async def ingest_session(req: IngestSessionRequest):
    """Ingest an OpenClaw session log into the memory graph."""
    result = await memory.ingest_session(req.session_path)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return RememberResponse(status="ingested", episodes=result.get("episodes", 0))


@app.get("/v1/recall", response_model=list[RecallResult])
async def recall(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """Search the memory graph."""
    results = await memory.recall(query, limit=limit)
    return [RecallResult(**r) for r in results]


@app.get("/v1/context")
async def get_context(
    query: str = Query(..., description="Context query"),
    max_chars: int = Query(4000, ge=500, le=20000),
):
    """Assemble a token-aware context block for LLM prompts."""
    ctx = await memory.get_context(query, max_chars=max_chars)
    # Rough token estimate: ~4 chars per token
    return ContextResponse(context=ctx, token_estimate=len(ctx) // 4)


@app.get("/v1/stats", response_model=StatsResponse)
async def stats():
    """Get memory graph stats."""
    g = memory.graphiti
    async with g.driver.session() as s:
        r1 = await s.run("MATCH (n:Entity) RETURN count(n) as c")
        r2 = await s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as c")
        r3 = await s.run("MATCH (e:Episodic) RETURN count(e) as c")
        return StatsResponse(
            entities=(await r1.single())["c"],
            facts=(await r2.single())["c"],
            episodes=(await r3.single())["c"],
        )


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
    return {"status": "ok", "service": "memory-engine"}
