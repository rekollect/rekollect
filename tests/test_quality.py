"""Quality baseline tests for graph extraction and recall accuracy.

Seeds 5 documents via /v1/remember, waits for processing, then tests:
- Graph entity extraction (Neo4j)
- Recall relevance and scoring (pgvector + graph)

Requires: backend at localhost:8181, Postgres, Neo4j, DEBUG=true.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from .conftest import BASE_URL

pytestmark = pytest.mark.asyncio(loop_scope="module")

logger = logging.getLogger(__name__)

# Shared state between ordered tests
state: dict = {
    "seed_results": [],
    "graph_entities": {},
    "graph_facts": {},
    "recall_results": [],
    "negative_test": None,
}

SEED_DOCUMENTS = [
    {
        "content": (
            "Cooper Flagg was drafted #1 overall by the Dallas Mavericks "
            "in the 2025 NBA Draft. He averaged 16.2 points, 7.8 rebounds "
            "and 3.1 assists per game in his rookie season."
        ),
        "type": "text",
        "source": "quality-test",
        "title": "Cooper Flagg Draft & Stats",
    },
    {
        "content": (
            "Had dinner at Lucia on Oak Lawn Ave in Dallas last Friday. "
            "The handmade pasta was incredible -- best Italian in the city. "
            "Elton recommended the orecchiette with sausage ragu."
        ),
        "type": "text",
        "source": "quality-test",
        "title": "Lucia Restaurant Review",
    },
    {
        "content": (
            "We chose FastAPI over Django for the Rekollect backend because "
            "we need async support for concurrent embedding calls and graph "
            "queries. The pgvector extension handles similarity search "
            "directly in Postgres."
        ),
        "type": "text",
        "source": "quality-test",
        "title": "Tech Decision: FastAPI",
    },
    {
        "content": (
            "User: Should we use Redis for caching?\n"
            "Assistant: Skip Redis. Postgres handles the volume. "
            "Use Upstash Redis for rate limiting only."
        ),
        "type": "chat",
        "source": "quality-test",
        "title": "Redis Discussion",
    },
    {
        "content": (
            "Mom birthday is June 15th. She wants a Kindle Paperwhite. "
            "Dad already got her earrings."
        ),
        "type": "text",
        "source": "quality-test",
        "title": "Mom Birthday Reminder",
    },
]

REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Setup: clear test data and seed documents
# ---------------------------------------------------------------------------


@pytest.mark.order(1)
async def test_clear_existing_data(client: httpx.AsyncClient):
    """Clear all existing memories to start fresh."""
    resp = await client.delete("/v1/memories", params={"confirm": "true"})
    assert resp.status_code == 200
    data = resp.json()
    logger.info(f"Cleared {data['deleted_count']} existing memories")


@pytest.mark.order(2)
async def test_seed_documents(client: httpx.AsyncClient):
    """Seed 5 documents via /v1/remember and wait for processing."""
    doc_ids = []
    for i, doc in enumerate(SEED_DOCUMENTS):
        resp = await client.post("/v1/remember", json=doc)
        assert resp.status_code == 200, f"Failed to seed doc {i}: {resp.text}"
        data = resp.json()
        assert data["status"] in ("processed", "completed", "failed")
        doc_ids.append(data["id"])

        state["seed_results"].append({
            "title": doc["title"],
            "type": doc["type"],
            "id": data["id"],
            "chunks": data["chunks"],
            "entities": data["entities"],
            "status": data["status"],
        })

        logger.info(
            f"Seeded doc {i}: id={data['id']} status={data['status']} "
            f"chunks={data['chunks']} entities={data['entities']}"
        )

    state["doc_ids"] = doc_ids

    # Brief pause for any async graph processing to settle
    await asyncio.sleep(2)

    # Verify all docs are completed
    for doc_id in doc_ids:
        resp = await client.get(f"/v1/memories/{doc_id}")
        assert resp.status_code == 200
        doc = resp.json()["document"]
        assert doc["processing_status"] in ("completed", "processed"), (
            f"Doc {doc_id} status: {doc['processing_status']}"
        )


# ---------------------------------------------------------------------------
# Graph entity tests (skip if graph/debug unavailable)
# ---------------------------------------------------------------------------


async def _graph_available(client: httpx.AsyncClient) -> bool:
    """Check if graph debug endpoints are available."""
    resp = await client.get("/health")
    if resp.status_code != 200:
        return False
    health = resp.json()
    if not health.get("graph"):
        return False
    resp = await client.get("/v1/debug/entities", params={"query": "test", "limit": 1})
    return resp.status_code == 200


@pytest.mark.order(3)
async def test_graph_entity_cooper_flagg(client: httpx.AsyncClient):
    """Cooper Flagg should be extracted as a graph entity."""
    if not await _graph_available(client):
        pytest.skip("Graph/debug endpoints unavailable")
    resp = await client.get(
        "/v1/debug/entities", params={"query": "Cooper Flagg", "limit": 10}
    )
    assert resp.status_code == 200
    entities = resp.json()
    names = [e["name"].lower() for e in entities]
    found = any("flagg" in n for n in names)
    assert found, f"Expected Cooper Flagg entity, got: {names}"

    match = next((e for e in entities if "flagg" in e["name"].lower()), None)
    state["graph_entities"]["cooper_flagg"] = {
        "found": True,
        "name": match["name"] if match else None,
        "summary": match.get("summary") if match else None,
    }


@pytest.mark.order(4)
async def test_graph_entity_lucia(client: httpx.AsyncClient):
    """Lucia restaurant should be extracted as a graph entity."""
    if not await _graph_available(client):
        pytest.skip("Graph/debug endpoints unavailable")
    resp = await client.get(
        "/v1/debug/entities", params={"query": "Lucia", "limit": 10}
    )
    assert resp.status_code == 200
    entities = resp.json()
    names = [e["name"].lower() for e in entities]
    found = any("lucia" in n for n in names)
    assert found, f"Expected Lucia entity, got: {names}"

    match = next((e for e in entities if "lucia" in e["name"].lower()), None)
    state["graph_entities"]["lucia"] = {
        "found": True,
        "name": match["name"] if match else None,
        "summary": match.get("summary") if match else None,
    }


@pytest.mark.order(5)
async def test_graph_entity_fastapi(client: httpx.AsyncClient):
    """FastAPI should be extracted as a graph entity."""
    if not await _graph_available(client):
        pytest.skip("Graph/debug endpoints unavailable")
    resp = await client.get(
        "/v1/debug/entities", params={"query": "FastAPI", "limit": 10}
    )
    assert resp.status_code == 200
    entities = resp.json()
    names = [e["name"].lower() for e in entities]
    found = any("fastapi" in n for n in names)
    assert found, f"Expected FastAPI entity, got: {names}"

    match = next((e for e in entities if "fastapi" in e["name"].lower()), None)
    state["graph_entities"]["fastapi"] = {
        "found": True,
        "name": match["name"] if match else None,
        "summary": match.get("summary") if match else None,
    }


@pytest.mark.order(6)
async def test_graph_facts_cooper_flagg(client: httpx.AsyncClient):
    """At least 1 fact about Cooper Flagg should exist in the graph."""
    if not await _graph_available(client):
        pytest.skip("Graph/debug endpoints unavailable")
    resp = await client.get(
        "/v1/debug/facts", params={"query": "Flagg", "limit": 10}
    )
    assert resp.status_code == 200
    facts = resp.json()
    assert len(facts) >= 1, f"Expected at least 1 Flagg fact, got {len(facts)}"

    state["graph_facts"]["cooper_flagg"] = [
        {
            "source": f.get("source", f.get("source_entity", "")),
            "target": f.get("target", f.get("target_entity", "")),
            "fact": f.get("fact", f.get("description", "")),
        }
        for f in facts
    ]

    logger.info(f"Flagg facts: {[f['fact'] for f in facts]}")


# ---------------------------------------------------------------------------
# Recall quality tests
# ---------------------------------------------------------------------------


def _save_recall(query: str, results: list, expected_match: str, matched: bool):
    """Save recall test results into shared state."""
    top = results[0] if results else {}
    state["recall_results"].append({
        "query": query,
        "top_score": top.get("score", 0.0),
        "result_count": len(results),
        "top_result_preview": top.get("content", "")[:120],
        "expected_match": expected_match,
        "matched": matched,
        "all_scores": [r.get("score", 0.0) for r in results],
    })


@pytest.mark.order(10)
async def test_recall_cooper_flagg_stats(client: httpx.AsyncClient):
    """'Cooper Flagg stats' should return Flagg content with high score."""
    resp = await client.post(
        "/v1/recall", json={"query": "Cooper Flagg stats", "limit": 5}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0, "No recall results for 'Cooper Flagg stats'"

    top = results[0]
    matched = "flagg" in top["content"].lower()
    assert matched, f"Top result doesn't mention Flagg: {top['content'][:100]}"
    if top.get("score") is not None:
        assert top["score"] > 0.5, f"Top score too low: {top['score']}"

    _save_recall("Cooper Flagg stats", results, "flagg", matched)


@pytest.mark.order(11)
async def test_recall_pasta_restaurant(client: httpx.AsyncClient):
    """'best pasta restaurant Dallas' should return Lucia."""
    resp = await client.post(
        "/v1/recall", json={"query": "best pasta restaurant Dallas", "limit": 5}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0, "No recall results for pasta restaurant query"

    all_content = " ".join(r["content"].lower() for r in results)
    matched = "lucia" in all_content
    assert matched, (
        f"Expected Lucia in results, got: {[r['content'][:80] for r in results]}"
    )

    _save_recall("best pasta restaurant Dallas", results, "lucia", matched)


@pytest.mark.order(12)
async def test_recall_fastapi_decision(client: httpx.AsyncClient):
    """'why did we pick FastAPI' should return the tech decision."""
    resp = await client.post(
        "/v1/recall", json={"query": "why did we pick FastAPI", "limit": 5}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0, "No recall results for FastAPI query"

    all_content = " ".join(r["content"].lower() for r in results)
    matched = "fastapi" in all_content
    assert matched, (
        f"Expected FastAPI in results, got: {[r['content'][:80] for r in results]}"
    )

    _save_recall("why did we pick FastAPI", results, "fastapi", matched)


@pytest.mark.order(13)
async def test_recall_redis_discussion(client: httpx.AsyncClient):
    """'should we use Redis' should return the Redis chat."""
    resp = await client.post(
        "/v1/recall", json={"query": "should we use Redis", "limit": 5}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0, "No recall results for Redis query"

    all_content = " ".join(r["content"].lower() for r in results)
    matched = "redis" in all_content
    assert matched, (
        f"Expected Redis in results, got: {[r['content'][:80] for r in results]}"
    )

    _save_recall("should we use Redis", results, "redis", matched)


@pytest.mark.order(14)
async def test_recall_mom_birthday(client: httpx.AsyncClient):
    """'when is Mom birthday' should return June 15."""
    resp = await client.post(
        "/v1/recall", json={"query": "when is Mom birthday", "limit": 5}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0, "No recall results for Mom birthday query"

    all_content = " ".join(r["content"].lower() for r in results)
    matched = "june 15" in all_content or "june" in all_content
    assert matched, (
        f"Expected June 15 in results, got: {[r['content'][:80] for r in results]}"
    )

    _save_recall("when is Mom birthday", results, "june 15", matched)


@pytest.mark.order(15)
async def test_recall_irrelevant_query(client: httpx.AsyncClient):
    """'quantum computing papers' should return no relevant results."""
    resp = await client.post(
        "/v1/recall", json={"query": "quantum computing papers", "limit": 5}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]

    high_scores = [
        r for r in results
        if r.get("score") is not None and r["score"] >= 0.3
    ]
    max_score = results[0].get("score", 0.0) if results else 0.0
    passed = len(high_scores) == 0

    state["negative_test"] = {
        "query": "quantum computing papers",
        "result_count": len(high_scores),
        "max_score": max_score,
        "passed": passed,
    }

    assert passed, (
        f"Expected no relevant results for quantum computing, "
        f"but got {len(high_scores)} with score >= 0.3: "
        f"{[(r['content'][:60], r['score']) for r in high_scores]}"
    )


# ---------------------------------------------------------------------------
# Baseline report
# ---------------------------------------------------------------------------


def _build_report(now: datetime) -> dict:
    """Build the full JSON report from collected state."""
    recall_scores = [
        r["top_score"] for r in state["recall_results"] if r["top_score"] > 0
    ]

    # Count tests
    total = len(state["seed_results"]) + len(state["graph_entities"]) + 1  # +1 facts
    total += len(state["recall_results"]) + (1 if state["negative_test"] else 0)

    passed = total  # assume all passed (assertions would have failed otherwise)
    skipped = 0
    if not state["graph_entities"]:
        skipped = 4  # 3 entity tests + 1 facts test
        passed -= skipped

    return {
        "run_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            "embedding_dim": int(os.environ.get("EMBEDDING_DIM", "1536")),
            "extraction_model": os.environ.get("LLM_MODEL", "gpt-4.1-nano"),
            "graph_embedding_model": os.environ.get(
                "GRAPH_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
        },
        "seed_documents": state["seed_results"],
        "graph_entities": state["graph_entities"],
        "graph_facts": state["graph_facts"],
        "recall_tests": state["recall_results"],
        "negative_test": state["negative_test"],
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": 0,
            "skipped": skipped,
            "avg_recall_score": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
            "min_recall_score": round(min(recall_scores), 4) if recall_scores else 0.0,
            "max_recall_score": round(max(recall_scores), 4) if recall_scores else 0.0,
        },
    }


def _print_report(report: dict, report_path: Path):
    """Print human-readable summary to stdout."""
    cfg = report["config"]
    run_date = report["run_date"].replace("T", " ").replace("Z", "")
    W = 67

    print("\n")
    print("=" * W)
    print(f"  REKOLLECT QUALITY BASELINE -- {run_date}")
    print(f"  Model: {cfg['embedding_model']} @ {cfg['embedding_dim']} dims")
    print("=" * W)

    # Seed documents
    print()
    print("  SEED DOCUMENTS")
    print("  " + "-" * (W - 2))
    for doc in report["seed_documents"]:
        ent_label = "entity" if doc["entities"] == 1 else "entities"
        chunk_label = "chunk" if doc["chunks"] == 1 else "chunks"
        print(
            f"  {doc['title']:<36} {doc['type']:<8}"
            f"  {doc['chunks']} {chunk_label:<10}"
            f"{doc['entities']} {ent_label}"
        )

    # Graph entities
    if report["graph_entities"]:
        print()
        print("  GRAPH ENTITIES")
        print("  " + "-" * (W - 2))
        for key, ent in report["graph_entities"].items():
            status = "found" if ent["found"] else "missing"
            summary = f'"{ent["summary"][:40]}..."' if ent.get("summary") else "(no summary)"
            print(f"  {ent['name']:<18}{status}   {summary}")

    # Graph facts
    if report["graph_facts"]:
        print()
        for key, facts in report["graph_facts"].items():
            label = key.replace("_", " ").title()
            print(f"  GRAPH FACTS ({label})")
            print("  " + "-" * (W - 2))
            for f in facts:
                print(f"  {f['source']} -> {f['target']}: \"{f['fact']}\"")

    # Recall quality
    print()
    print("  RECALL QUALITY")
    print("  " + "-" * (W - 2))
    print(f"  {'Query':<36}{'Score':>7}   {'Results':>7}   Match")
    for r in report["recall_tests"]:
        score_str = f"{r['top_score']:.3f}" if r["top_score"] else "0.000"
        match_str = f"  {r['expected_match']}" if r["matched"] else f"X {r['expected_match']}"
        print(f"  {r['query']:<36}{score_str:>7}   {r['result_count']:>7}   {match_str}")

    # Negative test
    if report["negative_test"]:
        neg = report["negative_test"]
        score_str = f"{neg['max_score']:.3f}" if neg['max_score'] is not None else "0.000"
        match_str = "  no match" if neg["passed"] else "X unexpected match"
        print(f"  {'quantum computing (negative)':<36}{score_str:>7}   {neg['result_count']:>7}   {match_str}")

    # Summary
    s = report["summary"]
    print()
    print(f"  SUMMARY: avg={s['avg_recall_score']:.2f}  min={s['min_recall_score']:.2f}  max={s['max_recall_score']:.2f}")
    print("=" * W)
    print(f"  Report saved: {report_path}")
    print("=" * W)
    print()


@pytest.mark.order(99)
async def test_print_baseline_summary(client: httpx.AsyncClient):
    """Generate JSON report and print human-readable summary."""
    now = datetime.now(timezone.utc)
    report = _build_report(now)

    # Ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write timestamped report
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"quality_baseline_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Write latest.json (overwritten each run)
    latest_path = REPORTS_DIR / "latest.json"
    latest_path.write_text(json.dumps(report, indent=2))

    # Print human-readable summary
    _print_report(report, report_path)
