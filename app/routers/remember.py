"""POST /v1/remember -- add + process in one call."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import get_current_user
from app.models.schemas import AddRequest, RememberResponse
from app.services.memory import resolve_content, store_document, process_document

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/remember", response_model=RememberResponse, operation_id="remember")
async def remember(
    req: AddRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Save anything worth remembering -- for personal use or professional work.

    Personal: notes, reminders, ideas, places, people, dates, articles,
    quotes, anything you want to recall later.

    Professional: decisions made, bugs fixed, code patterns, architecture
    choices, research findings, meeting notes.

    Also call at the end of any session to save the full conversation as a
    complete record -- so it can be recalled later on any platform, through
    any AI agent or tool.

    Call proactively. Do not wait to be asked.
    """
    pool = request.app.state.pool
    settings = request.app.state.settings
    graph = request.app.state.graph

    content = resolve_content(req)

    try:
        # Step 1: Store document
        doc = await store_document(
            pool,
            user_id=user_id,
            content=content,
            doc_type=req.type,
            source=req.source,
            title=req.title,
            metadata=req.metadata,
            collection=req.collection,
        )

        doc_id = doc["id"]

        # Step 2: Process it
        result = await process_document(
            pool,
            graph,
            settings,
            doc_id,
            reprocess=False,
        )

        return RememberResponse(
            id=result.doc_id,
            status=result.status,
            chunks=result.chunks,
            entities=result.entities,
            document=result.document,
        )

    except Exception as e:
        logger.error(f"Remember operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
