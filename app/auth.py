"""Simple API key authentication."""

import bcrypt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate API key and return user_id."""
    token = credentials.credentials
    pool = request.app.state.pool

    # Look up API key by checking bcrypt hash
    rows = await pool.fetch(
        "SELECT user_id, key_hash, revoked_at FROM api_keys WHERE revoked_at IS NULL"
    )
    matched_row = None
    for row in rows:
        if bcrypt.checkpw(token.encode(), row["key_hash"].encode()):
            matched_row = row
            break

    if not matched_row:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    if matched_row["revoked_at"]:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    # Update last_used_at (fire and forget)
    await pool.execute(
        "UPDATE api_keys SET last_used_at = now() WHERE key_hash = $1",
        matched_row["key_hash"],
    )

    return str(matched_row["user_id"])
