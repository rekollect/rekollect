"""POST /v1/recall -- query pgvector + graph, merge results."""

import logging

from fastapi import APIRouter, Depends, Request

from app.auth import get_current_user
from app.models.schemas import RecallRequest, RecallResponse, RecallResult
from app.services.memory import search_memories

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recall", response_model=RecallResponse, operation_id="recall")
async def recall(
    req: RecallRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Search your personal memory. Use before starting any task, when
    answering questions about past decisions, or anytime prior context
    would help -- regardless of which app or agent originally saved it.
    """
    pool = request.app.state.pool
    results = await search_memories(
        pool,
        request.app.state.graph,
        request.app.state.settings,
        user_id=user_id,
        query=req.query,
        limit=req.limit,
        collection=req.collection,
    )
    return RecallResponse(results=[RecallResult(**r) for r in results])
