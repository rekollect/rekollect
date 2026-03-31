"""End-to-end tests for the Rekollect backend API.

Tests run in order and share state via module-level dict.
Requires: backend at localhost:8181, Postgres, Neo4j.
"""

import httpx
import pytest

from .conftest import BASE_URL

pytestmark = pytest.mark.asyncio(loop_scope="module")

# Shared state between tests
state: dict = {}


@pytest.mark.order(1)
async def test_health():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.order(2)
async def test_remember_text(client: httpx.AsyncClient):
    resp = await client.post(
        "/v1/remember",
        json={
            "content": "Rekollect test: Cooper Flagg scored 22 points against the Lakers on March 15th",
            "title": "Flagg Game Log",
            "type": "text",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("processed", "completed", "failed")
    assert "id" in data
    state["text_doc_id"] = data["id"]


@pytest.mark.order(3)
async def test_remember_chat_session(client: httpx.AsyncClient):
    conversation = [
        {"role": "user", "content": "What's the best architecture for a memory system?"},
        {"role": "assistant", "content": "A RAG pipeline with vector search and a knowledge graph works well."},
        {"role": "user", "content": "How should we handle deduplication?"},
        {"role": "assistant", "content": "Use entity resolution on the graph side and cosine similarity for chunks."},
    ]
    resp = await client.post(
        "/v1/remember",
        json={
            "type": "chat",
            "content_json": {"messages": conversation},
            "title": "Architecture Discussion",
            "source": "claude-code",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    state["chat_doc_id"] = data["id"]


@pytest.mark.order(4)
async def test_add_without_processing(client: httpx.AsyncClient):
    resp = await client.post(
        "/v1/add",
        json={
            "content": "Raw note: remember to update the Neo4j index dimensions",
            "type": "text",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    state["raw_doc_id"] = data["id"]
    doc = data.get("document") or {}
    status = doc.get("processing_status") or data.get("status", "")
    assert status != "completed"


@pytest.mark.order(5)
async def test_recall_by_content(client: httpx.AsyncClient):
    resp = await client.post(
        "/v1/recall",
        json={"query": "Cooper Flagg points", "limit": 5},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0
    matched = any(
        "flagg" in r.get("content", "").lower() or "points" in r.get("content", "").lower()
        for r in results
    )
    assert matched, "Expected at least one result mentioning Flagg or points"


@pytest.mark.order(6)
async def test_recall_chat_session(client: httpx.AsyncClient):
    resp = await client.post(
        "/v1/recall",
        json={"query": "architecture discussion"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) > 0


@pytest.mark.order(7)
async def test_list_memories(client: httpx.AsyncClient):
    resp = await client.get("/v1/memories", params={"limit": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) >= 3
    assert data["total"] >= 3


@pytest.mark.order(8)
async def test_list_memories_with_filter(client: httpx.AsyncClient):
    resp = await client.get("/v1/memories", params={"type": "chat"})
    assert resp.status_code == 200
    assert "memories" in resp.json()

    # Verify chat doc exists in unfiltered list with correct type in metadata
    resp2 = await client.get("/v1/memories", params={"limit": 50})
    assert resp2.status_code == 200
    memories = resp2.json()["memories"]
    assert any(
        m.get("type") == "chat" or m["id"] == state.get("chat_doc_id")
        for m in memories
    )


@pytest.mark.order(9)
async def test_get_memory_by_id(client: httpx.AsyncClient):
    doc_id = state["text_doc_id"]
    resp = await client.get(f"/v1/memories/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document"] is not None
    assert "chunks" in data


@pytest.mark.order(10)
async def test_create_api_key(client: httpx.AsyncClient):
    resp = await client.post("/v1/keys", json={"name": "E2E Test Key"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["key"].startswith("rk_live_")
    assert "key_prefix" in data
    assert data["name"] == "E2E Test Key"
    state["api_key_id"] = data["id"]
    state["api_key"] = data["key"]


@pytest.mark.order(11)
async def test_use_api_key_for_recall():
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {state['api_key']}"},
        timeout=30.0,
    ) as c:
        resp = await c.post("/v1/recall", json={"query": "test", "limit": 3})
    assert resp.status_code == 200
    assert "results" in resp.json()


@pytest.mark.order(12)
async def test_list_api_keys(client: httpx.AsyncClient):
    resp = await client.get("/v1/keys")
    assert resp.status_code == 200
    keys = resp.json()
    assert len(keys) >= 1
    for key in keys:
        assert "key_prefix" in key
        assert "key" not in key  # full key should not be exposed


@pytest.mark.order(13)
async def test_delete_memory(client: httpx.AsyncClient):
    doc_id = state["raw_doc_id"]
    resp = await client.delete(f"/v1/memories/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp2 = await client.get(f"/v1/memories/{doc_id}")
    assert resp2.status_code == 404


@pytest.mark.order(14)
async def test_revoke_api_key(client: httpx.AsyncClient):
    key_id = state["api_key_id"]
    resp = await client.delete(f"/v1/keys/{key_id}")
    assert resp.status_code == 200


@pytest.mark.order(15)
async def test_revoked_key_fails():
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {state['api_key']}"},
        timeout=10.0,
    ) as c:
        resp = await c.post("/v1/recall", json={"query": "test"})
    assert resp.status_code in (401, 403)


@pytest.mark.order(16)
async def test_bulk_delete_requires_confirm(client: httpx.AsyncClient):
    resp = await client.delete("/v1/memories")
    assert resp.status_code == 400
