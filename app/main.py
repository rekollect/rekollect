"""Rekollect -- FastAPI + FastMCP memory engine."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.db import get_pool, close_pool
from app.graph.engine import GraphEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.pool = await get_pool(settings.database_url)
    logger.info("Database connected")

    try:
        app.state.graph = GraphEngine(settings)
        await app.state.graph.init()
        logger.info("Graph engine initialized")
    except Exception as e:
        logger.warning(f"Graph unavailable: {e}")
        app.state.graph = None

    # Share state with MCP tools
    from app import mcp_tools
    mcp_tools.init(settings, app.state.pool, app.state.graph)

    yield

    if app.state.graph:
        await app.state.graph.close()
    await close_pool()


app = FastAPI(title="Rekollect", version="2.0.0", lifespan=lifespan)

# Register routers
from app.routers import add, process, recall, remember, memories, collections, keys

app.include_router(add.router, prefix="/v1", tags=["add"])
app.include_router(process.router, prefix="/v1", tags=["process"])
app.include_router(recall.router, prefix="/v1", tags=["recall"])
app.include_router(remember.router, prefix="/v1", tags=["remember"])
app.include_router(memories.router, prefix="/v1", tags=["memories"])
app.include_router(collections.router, prefix="/v1", tags=["collections"])
app.include_router(keys.router, prefix="/v1", tags=["keys"])

if os.environ.get("DEBUG") == "true":
    from app.routers import debug
    app.include_router(debug.router, prefix="/v1", tags=["debug"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "graph": app.state.graph is not None}


# MCP server
from fastmcp import FastMCP
from app import mcp_tools

mcp = FastMCP("Rekollect")
mcp_tools.register(mcp)
mcp_app = mcp.http_app(path="/mcp")
app.mount("/", mcp_app)
