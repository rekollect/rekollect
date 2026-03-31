"""POST /v1/add -- store content without processing."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import get_current_user
from app.models.schemas import AddRequest, AddResponse
from app.services.memory import resolve_content, store_document

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/add", response_model=AddResponse, operation_id="add")
async def add(
    req: AddRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Store content without processing. Use for large batches or when you
    want to control when processing happens. For most use cases, prefer
    /v1/remember which stores and processes in one call.
    """
    pool = request.app.state.pool
    content = resolve_content(req)

    try:
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

        return AddResponse(
            id=doc["id"],
            status=doc["processing_status"],
            chunks=0,
            entities=0,
            document=doc,
        )

    except Exception as e:
        logger.error(f"Failed to store document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
