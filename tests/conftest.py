import pytest
import pytest_asyncio
import httpx

BASE_URL = "http://localhost:8181"
API_KEY = "rk_dev_rekollect"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client() -> httpx.AsyncClient:
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=120.0,
    ) as c:
        yield c
