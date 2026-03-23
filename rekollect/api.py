"""Rekollect REST API — FastAPI wrapper around RekollectMemory."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rekollect.memory import RekollectMemory

memory: RekollectMemory | None = None


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
    version="0.2.0",
    description="Graph-based agent memory with hybrid search, temporal awareness, and importance scoring",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Models ─────────────────────────────────────────────────────

class RememberRequest(BaseModel):
    content: str
    source: str = "manual"

class IngestSessionRequest(BaseModel):
    session_path: str

class IngestBatchRequest(BaseModel):
    directory: str
    since_hours: int | None = None

class RecallRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    format: str = Field(default="json", description="Response format: 'json' (structured) or 'prompt' (markdown for LLM injection)")
    max_chars: int = Field(default=4000, ge=500, le=20000, description="Max chars when format=prompt")


# ─── Endpoints ──────────────────────────────────────────────────

@app.post("/v1/remember")
async def remember(req: RememberRequest):
    await memory.remember(req.content, req.source)
    return {"status": "remembered", "episodes": 1}


@app.post("/v1/ingest/session")
async def ingest_session(req: IngestSessionRequest):
    result = await memory.ingest_session(req.session_path)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"status": "ingested", **result}


@app.post("/v1/ingest/batch")
async def ingest_batch(req: IngestBatchRequest):
    result = await memory.ingest_sessions_batch(req.directory, req.since_hours)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/v1/recall")
async def recall_post(req: RecallRequest):
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
    return await memory.stats()


@app.get("/v1/whats-new")
async def whats_new(since_hours: int = Query(24, ge=1, le=168)):
    return await memory.whats_new(since_hours)


@app.get("/v1/entities")
async def list_entities(query: str = Query(None), limit: int = Query(50, ge=1, le=200)):
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
    return {"status": "ok", "service": "rekollect", "version": "0.2.0"}
