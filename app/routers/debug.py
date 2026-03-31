"""DEBUG-ONLY endpoints for inspecting Neo4j graph state.

Only registered when DEBUG=true environment variable is set.
"""

import logging

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class EntityResult(BaseModel):
    name: str
    summary: str | None = None
    group_id: str | None = None


class FactResult(BaseModel):
    source: str
    target: str
    fact: str
    created_at: str | None = None


@router.get("/debug/entities", response_model=list[EntityResult])
async def debug_entities(
    request: Request,
    query: str = Query(..., description="Search entity names"),
    limit: int = Query(10, ge=1, le=100),
    user_id: str = Depends(get_current_user),
):
    """Query Neo4j for entities matching a name substring."""
    graph = request.app.state.graph
    if graph is None:
        return []

    result = await graph.graphiti.driver.execute_query(
        "MATCH (n:Entity) "
        "WHERE toLower(n.name) CONTAINS toLower($query) "
        "RETURN n.name AS name, n.summary AS summary, n.group_id AS group_id "
        "LIMIT $limit",
        params={"query": query, "limit": limit},
    )
    records = result.records if hasattr(result, "records") else []
    return [
        EntityResult(
            name=r.data()["name"],
            summary=r.data().get("summary"),
            group_id=r.data().get("group_id"),
        )
        for r in records
    ]


@router.get("/debug/facts", response_model=list[FactResult])
async def debug_facts(
    request: Request,
    query: str = Query(..., description="Search facts or entity names"),
    limit: int = Query(10, ge=1, le=100),
    user_id: str = Depends(get_current_user),
):
    """Query Neo4j for RELATES_TO edges matching a fact or entity name."""
    graph = request.app.state.graph
    if graph is None:
        return []

    result = await graph.graphiti.driver.execute_query(
        "MATCH (s)-[r:RELATES_TO]->(t) "
        "WHERE toLower(r.fact) CONTAINS toLower($query) "
        "   OR toLower(s.name) CONTAINS toLower($query) "
        "RETURN s.name AS source, t.name AS target, r.fact AS fact, "
        "       toString(r.created_at) AS created_at "
        "LIMIT $limit",
        params={"query": query, "limit": limit},
    )
    records = result.records if hasattr(result, "records") else []
    return [
        FactResult(
            source=r.data()["source"],
            target=r.data()["target"],
            fact=r.data()["fact"],
            created_at=r.data().get("created_at"),
        )
        for r in records
    ]
