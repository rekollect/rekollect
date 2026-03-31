"""Database connection pool using asyncpg."""

import asyncpg

_pool: asyncpg.Pool | None = None


async def get_pool(database_url: str) -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(database_url)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
