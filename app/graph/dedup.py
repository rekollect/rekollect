"""Post-ingestion deduplication sweep.

Simplified port from ~/Develop/OpenClaw/rekollect/rekollect/dedup.py.
Uses OpenAI for both embeddings and LLM decisions.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MERGE_CONFIDENCE_THRESHOLD = 0.92
VECTOR_INDEX_NAME = "entity_dedup_embedding"


def _entity_embed_text(entity: dict) -> str:
    """Build embedding text for an entity (name + summary)."""
    name = entity.get("name", "unknown")
    summary = entity.get("summary")
    if summary and len(summary) > 10:
        return f"{name}: {summary[:200]}"
    return name


async def ensure_vector_index(driver, embedding_dim: int = 1536):
    """Create the dedup vector index if it doesn't exist."""
    async with driver.session() as s:
        try:
            await s.run(f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (n:Entity) ON (n.dedup_embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            logger.info("Dedup vector index ensured")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")


async def run_dedup_sweep(
    driver,
    embedder,
    dedup_model: str = "gpt-4.1",
    since_hours: float = 2.0,
) -> dict:
    """Run a dedup sweep on recently created entities.

    Args:
        driver: Neo4j async driver
        embedder: Object with create_batch(texts) -> list[list[float]]
        dedup_model: OpenAI model for merge decisions
        since_hours: How far back to scan
    """
    run_id = str(uuid.uuid4())[:12]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    before_entities, before_facts = await _get_counts(driver)

    # 1. Pull recent entities
    recent = await _get_recent_entities(driver, cutoff)
    if not recent:
        return {
            "run_id": run_id,
            "merges": 0,
            "skips": 0,
            "errors": 0,
            "before_entities": before_entities,
            "after_entities": before_entities,
        }

    logger.info(f"Dedup sweep: {len(recent)} entities since {cutoff.isoformat()}")

    # 2. Ensure dedup embeddings
    await _ensure_dedup_embeddings(driver, embedder, recent)

    # 3. Find candidate pairs via vector similarity
    merges, skips, errors = 0, 0, 0
    merged_uuids: set[str] = set()
    candidate_pairs = []

    for entity in recent:
        if entity["uuid"] in merged_uuids:
            continue
        candidates = await _find_similar_via_index(driver, entity, top_k=5)
        candidates = [
            c
            for c in candidates
            if c["uuid"] not in merged_uuids
            and c["uuid"] != entity["uuid"]
            and c.get("score", 0) < 0.99
        ]
        if candidates:
            candidate_pairs.append((entity, candidates))

    # 4. Batch LLM dedup decisions
    for batch_start in range(0, len(candidate_pairs), 10):
        batch = candidate_pairs[batch_start : batch_start + 10]
        decisions = await _batch_llm_dedup(client, batch, dedup_model)

        for (entity, candidates), decision in zip(batch, decisions):
            try:
                if (
                    decision["action"] == "merge"
                    and decision["confidence"] >= MERGE_CONFIDENCE_THRESHOLD
                ):
                    target = decision["merge_with"]
                    edges_moved = await _merge_entities(driver, entity, target)
                    merged_uuids.add(entity["uuid"])
                    merges += 1
                    logger.info(
                        f"  MERGE: '{entity['name']}' -> '{target['name']}' "
                        f"({decision['confidence']:.2f})"
                    )
                else:
                    skips += 1
            except Exception as e:
                errors += 1
                logger.warning(f"  ERROR on '{entity.get('name')}': {e}")

    after_entities, after_facts = await _get_counts(driver)
    return {
        "run_id": run_id,
        "merges": merges,
        "skips": skips,
        "errors": errors,
        "before_entities": before_entities,
        "after_entities": after_entities,
    }


# --- Helpers ---


async def _get_counts(driver) -> tuple[int, int]:
    async with driver.session() as s:
        r1 = await s.run("MATCH (n:Entity) RETURN count(n) as c")
        entities = (await r1.single())["c"]
        r2 = await s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as c")
        facts = (await r2.single())["c"]
    return entities, facts


async def _get_recent_entities(driver, cutoff: datetime) -> list[dict]:
    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (n:Entity)
            WHERE n.created_at >= $cutoff
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                   n.dedup_embedding IS NOT NULL AS has_dedup_embedding
            """,
            {"cutoff": cutoff},
        )
        return [r.data() async for r in result]


async def _ensure_dedup_embeddings(driver, embedder, entities: list[dict]):
    needs = [e for e in entities if not e.get("has_dedup_embedding")]
    if not needs:
        return

    texts = [_entity_embed_text(e) for e in needs]
    embeddings = await embedder.create_batch(texts)

    async with driver.session() as s:
        for entity, emb in zip(needs, embeddings):
            await s.run(
                "MATCH (n:Entity {uuid: $uuid}) SET n.dedup_embedding = $emb",
                {"uuid": entity["uuid"], "emb": emb},
            )
    logger.info(f"Embedded {len(needs)} entities for dedup")


async def _find_similar_via_index(
    driver, entity: dict, top_k: int = 5
) -> list[dict]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (n:Entity {uuid: $uuid}) RETURN n.dedup_embedding AS emb",
            {"uuid": entity["uuid"]},
        )
        rec = await r.single()
        emb = rec["emb"] if rec else None

    if not emb:
        return []

    async with driver.session() as s:
        try:
            result = await s.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $k, $embedding)
                YIELD node, score
                WHERE node.uuid <> $uuid
                RETURN node.uuid AS uuid, node.name AS name,
                       node.summary AS summary, score
                ORDER BY score DESC
                LIMIT $k
                """,
                {"k": top_k + 1, "embedding": list(emb), "uuid": entity["uuid"]},
            )
            return [r.data() async for r in result]
        except Exception:
            return []


async def _batch_llm_dedup(
    client: AsyncOpenAI,
    pairs: list[tuple],
    model: str,
) -> list[dict]:
    import asyncio

    async def _decide(entity, candidates):
        return await _llm_dedup_decision(client, entity, candidates, model)

    tasks = [_decide(entity, candidates) for entity, candidates in pairs]
    return await asyncio.gather(*tasks)


async def _llm_dedup_decision(
    client: AsyncOpenAI,
    entity: dict,
    candidates: list[dict],
    model: str,
) -> dict:
    candidate_lines = "\n".join(
        f'  {i+1}. "{c["name"]}" (summary: {(c.get("summary") or "none")[:100]}, '
        f'score: {c.get("score", "?")})'
        for i, c in enumerate(candidates)
    )

    prompt = f"""Determine if any candidate entities are the SAME real-world entity as the target.

TARGET: "{entity['name']}" -- {(entity.get('summary') or 'none')[:150]}

CANDIDATES:
{candidate_lines}

Rules:
- ONLY merge if they clearly refer to the SAME specific thing
- Different people are NEVER duplicates even if they play the same sport
- Abbreviation/alias of the same thing = duplicate
- When uncertain, do NOT merge

JSON response (no markdown):
{{"is_duplicate": true/false, "merge_with_index": <1-based or null>, "confidence": <0.0-1.0>, "reason": "<brief>"}}"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)

        if result.get("is_duplicate") and result.get("merge_with_index"):
            idx = result["merge_with_index"] - 1
            if 0 <= idx < len(candidates):
                return {
                    "action": "merge",
                    "merge_with": candidates[idx],
                    "confidence": result.get("confidence", 0.5),
                    "reason": result.get("reason", "LLM confirmed duplicate"),
                }
        return {
            "action": "skip",
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", "Not a duplicate"),
        }
    except Exception as e:
        return {"action": "skip", "confidence": 0.0, "reason": f"LLM error: {str(e)[:100]}"}


async def _merge_entities(driver, source: dict, target: dict) -> int:
    edges_moved = 0
    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (src:Entity {uuid: $src})-[r:RELATES_TO]->(dest)
            WHERE dest.uuid <> $tgt
            WITH r, dest, properties(r) AS props
            MERGE (tgt:Entity {uuid: $tgt})-[nr:RELATES_TO]->(dest)
            SET nr = props
            DELETE r
            RETURN count(nr) as moved
            """,
            {"src": source["uuid"], "tgt": target["uuid"]},
        )
        rec = await result.single()
        edges_moved += rec["moved"] if rec else 0

        result = await s.run(
            """
            MATCH (src)-[r:RELATES_TO]->(entity:Entity {uuid: $src})
            WHERE src.uuid <> $tgt
            WITH r, src, properties(r) AS props
            MERGE (src)-[nr:RELATES_TO]->(tgt:Entity {uuid: $tgt})
            SET nr = props
            DELETE r
            RETURN count(nr) as moved
            """,
            {"src": source["uuid"], "tgt": target["uuid"]},
        )
        rec = await result.single()
        edges_moved += rec["moved"] if rec else 0

        await s.run(
            "MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n",
            {"uuid": source["uuid"]},
        )

    return edges_moved
