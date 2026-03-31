"""POST /v1/process -- process a document by ID."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import get_current_user
from app.models.schemas import ProcessRequest, ProcessResponse
from app.services.memory import process_document
from app.storage import postgres

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/process", response_model=ProcessResponse, operation_id="process")
async def process(
    req: ProcessRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Process a document: chunk, embed, extract to graph, dedup.

    Works for both first-time processing and reprocessing.
    If chunks exist and reprocess=True, deletes them first.
    """
    pool = request.app.state.pool
    settings = request.app.state.settings
    graph = request.app.state.graph

    # Verify document exists
    doc = await postgres.get_document(pool, str(req.id))
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = await process_document(
            pool,
            graph,
            settings,
            str(req.id),
            reprocess=req.reprocess,
        )

        return ProcessResponse(
            id=result.doc_id,
            status=result.status,
            chunks=result.chunks,
            entities=result.entities,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Processing failed for {req.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
