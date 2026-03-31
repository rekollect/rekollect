"""API key management -- create, list, revoke."""

import secrets
import logging
from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import get_current_user
from app.models.schemas import ApiKeyCreate, ApiKeyCreated, ApiKeyResponse

logger = logging.getLogger(__name__)
router = APIRouter()

KEY_PREFIX = "rk_live_"


def _generate_key() -> str:
    """Generate a new API key: rk_live_ + 32 URL-safe random chars."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


@router.post("/keys", response_model=ApiKeyCreated)
async def create_key(
    req: ApiKeyCreate,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Create a new API key. The full key is returned once and never stored."""
    pool = request.app.state.pool

    raw_key = _generate_key()
    key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
    key_prefix = raw_key[:12]

    row = await pool.fetchrow(
        """
        INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
        VALUES ($1, $2, $3, $4)
        RETURNING id, created_at
        """,
        user_id,
        key_hash,
        key_prefix,
        req.name,
    )

    return ApiKeyCreated(
        id=row["id"],
        key=raw_key,
        key_prefix=key_prefix,
        name=req.name,
        created_at=row["created_at"],
    )


@router.get("/keys", response_model=list[ApiKeyResponse])
async def list_keys(
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """List all API keys for the current user. Never returns the full key."""
    pool = request.app.state.pool

    rows = await pool.fetch(
        """
        SELECT id, key_prefix, name, last_used_at, created_at
        FROM api_keys
        WHERE user_id = $1 AND revoked_at IS NULL
        ORDER BY created_at DESC
        """,
        user_id,
    )

    return [ApiKeyResponse(**dict(r)) for r in rows]


@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """Revoke an API key by setting revoked_at."""
    pool = request.app.state.pool

    # Verify ownership
    existing = await pool.fetchrow(
        "SELECT id FROM api_keys WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL",
        key_id,
        user_id,
    )

    if not existing:
        raise HTTPException(status_code=404, detail="Key not found")

    await pool.execute(
        "UPDATE api_keys SET revoked_at = $1 WHERE id = $2",
        datetime.now(timezone.utc),
        key_id,
    )

    return {"status": "revoked", "id": key_id}
