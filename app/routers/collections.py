"""GET /v1/collections -- list user's collections with document counts."""

from fastapi import APIRouter, Depends, Request

from app.auth import get_current_user
from app.models.schemas import CollectionsResponse, CollectionSummary
from app.storage import postgres

router = APIRouter()


@router.get("/collections", response_model=CollectionsResponse, operation_id="list_collections")
async def list_collections(
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """List all collections for the current user with document counts."""
    pool = request.app.state.pool
    rows = await postgres.get_user_collections(pool, user_id)
    return CollectionsResponse(
        collections=[
            CollectionSummary(
                collection=r["collection"],
                document_count=r["document_count"],
            )
            for r in rows
        ]
    )
