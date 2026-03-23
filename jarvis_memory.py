"""Rekollect Memory Engine — wraps Graphiti for agent memory.

This is the bridge between OpenClaw's session system and Graphiti's
temporal knowledge graph. It handles:
1. Auto-ingestion of session transcripts
2. Context assembly for LLM prompts
3. Importance-weighted recall with hybrid search
4. Episode-level raw conversation search (citations)
5. "What changed?" queries
"""
import asyncio
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.embedder.ollama import OllamaEmbedder, OllamaEmbedderConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.utils.bulk_utils import RawEpisode


class JarvisMemory:
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "memory-engine-dev",
        group_id: str = "",
    ):
        embedder = OllamaEmbedder(OllamaEmbedderConfig(
            embedding_model="nomic-embed-text",
            base_url="http://localhost:11434",
            embedding_dim=768,
        ))
        self.graphiti = Graphiti(
            neo4j_uri, neo4j_user, neo4j_password,
            embedder=embedder,
        )
        self.group_id = group_id

    async def init(self):
        """Initialize graph indices."""
        await self.graphiti.build_indices_and_constraints()

    async def close(self):
        await self.graphiti.close()

    # ─── 1. INGEST ──────────────────────────────────────────────

    async def ingest_session(self, session_path: str) -> dict:
        """Ingest an OpenClaw session log (.jsonl) into the graph."""
        path = Path(session_path)
        if not path.exists():
            return {"error": f"Session file not found: {session_path}"}

        messages = []
        session_start = None

        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "message":
                continue

            msg = entry.get("message", {})
            role = msg.get("role")
            timestamp = entry.get("timestamp")

            if role in ("user", "assistant"):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if c.get("type") == "text"
                    )
                if content and len(content) > 20:
                    messages.append({"role": role, "content": content, "ts": timestamp})
                    if not session_start:
                        session_start = timestamp

        if not messages:
            return {"episodes": 0, "message": "No substantial messages found"}

        # Chunk messages into episodes (~4000 chars each for extraction quality)
        episodes = []
        current_chunk = []
        current_len = 0

        for msg in messages:
            line = f"{msg['role'].upper()}: {msg['content']}"
            if current_len + len(line) > 4000 and current_chunk:
                episodes.append("\n".join(current_chunk))
                current_chunk = [line]
                current_len = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line)

        if current_chunk:
            episodes.append("\n".join(current_chunk))

        ref_time = (
            datetime.fromisoformat(session_start.replace("Z", "+00:00"))
            if session_start
            else datetime.now(timezone.utc)
        )

        raw_episodes = [
            RawEpisode(
                name=f"Session {path.stem} part {i+1}",
                content=episode_text,
                source=EpisodeType.text,
                source_description=f"openclaw:session:{path.stem}",
                reference_time=ref_time + timedelta(minutes=i * 5),
                group_id=self.group_id,
            )
            for i, episode_text in enumerate(episodes)
        ]

        await self.graphiti.add_episode_bulk(raw_episodes)

        return {"episodes": len(episodes), "messages": len(messages)}

    async def ingest_sessions_batch(self, directory: str, since_hours: int | None = None) -> dict:
        """Ingest all .jsonl session files from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        files = sorted(dir_path.glob("*.jsonl"))
        if since_hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            files = [
                f for f in files
                if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) >= cutoff
            ]

        results = {"total_files": len(files), "ingested": 0, "episodes": 0, "errors": []}

        for f in files:
            try:
                result = await self.ingest_session(str(f))
                if "error" in result:
                    results["errors"].append({"file": f.name, "error": result["error"]})
                else:
                    results["ingested"] += 1
                    results["episodes"] += result.get("episodes", 0)
            except Exception as e:
                results["errors"].append({"file": f.name, "error": str(e)})

        return results

    async def remember(self, text: str, source: str = "manual") -> dict:
        """Manually add a memory."""
        await self.graphiti.add_episode(
            name=f"Manual memory - {datetime.now(timezone.utc).isoformat()[:16]}",
            episode_body=text,
            source=EpisodeType.text,
            source_description=source,
            reference_time=datetime.now(timezone.utc),
            group_id=self.group_id,
        )
        return {"status": "remembered"}

    # ─── 2. RECALL (Hybrid Search + Importance) ─────────────────

    async def recall(self, query: str, limit: int = 10) -> dict:
        """Search the memory graph with hybrid search (BM25 + vector + RRF).
        
        Returns facts (edges), entities (nodes), and episode snippets.
        Updates importance metrics on recalled edges.
        """
        config = COMBINED_HYBRID_SEARCH_RRF
        config.limit = limit

        results = await self.graphiti.search_(
            query=query,
            config=config,
            group_ids=[self.group_id],
        )

        # Format edges (facts)
        facts = []
        edge_uuids = []
        for edge in (results.edges or []):
            facts.append({
                "type": "fact",
                "uuid": edge.uuid,
                "fact": edge.fact,
                "name": edge.name,
                "valid_from": str(edge.valid_at) if edge.valid_at else None,
                "invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
                "is_current": edge.invalid_at is None,
                "source_node": edge.source_node_uuid,
                "target_node": edge.target_node_uuid,
            })
            edge_uuids.append(edge.uuid)

        # Format nodes (entities)
        entities = []
        for node in (results.nodes or []):
            entities.append({
                "type": "entity",
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
                "entity_type": node.labels[0] if node.labels else None,
            })

        # Format episodes (raw conversation citations)
        episodes = []
        for ep in (results.episodes or []):
            episodes.append({
                "type": "episode",
                "uuid": ep.uuid,
                "name": ep.name,
                "content": ep.content[:500] if ep.content else None,  # truncate
                "source": ep.source_description,
                "created_at": str(ep.created_at) if ep.created_at else None,
            })

        # Update importance metrics on recalled edges (fire-and-forget)
        if edge_uuids:
            asyncio.create_task(self._update_importance(edge_uuids, query))

        return {
            "facts": facts,
            "entities": entities,
            "episodes": episodes,
            "total": len(facts) + len(entities) + len(episodes),
        }

    async def _update_importance(self, edge_uuids: list[str], query: str):
        """Update importance metrics on recalled edges."""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        now = datetime.now(timezone.utc).isoformat()

        try:
            async with self.graphiti.driver.session() as session:
                for uuid in edge_uuids:
                    await session.run(
                        """
                        MATCH ()-[r:RELATES_TO {uuid: $uuid}]-()
                        SET r.recall_count = COALESCE(r.recall_count, 0) + 1,
                            r.last_recalled_at = $now,
                            r.query_hashes = CASE
                                WHEN r.query_hashes IS NULL THEN [$hash]
                                WHEN NOT $hash IN r.query_hashes THEN r.query_hashes + $hash
                                ELSE r.query_hashes
                            END
                        WITH r,
                             COALESCE(r.recall_count, 1) AS rc,
                             SIZE(COALESCE(r.query_hashes, [$hash])) AS uq,
                             COALESCE(r.base_importance, 50.0) AS base,
                             duration.between(
                                 COALESCE(datetime(r.last_recalled_at), datetime()),
                                 datetime()
                             ).days AS days_since
                        SET r.unique_queries = uq,
                            r.importance = CASE
                                WHEN base * (1.0 + log(1.0 + rc) / log(20.0))
                                     * (1.0 + (toFloat(uq) / toFloat(CASE WHEN rc > 0 THEN rc ELSE 1 END)) * 0.5)
                                     * exp(-0.01 * COALESCE(days_since, 0)) > 100.0
                                THEN 100.0
                                ELSE base * (1.0 + log(1.0 + rc) / log(20.0))
                                     * (1.0 + (toFloat(uq) / toFloat(CASE WHEN rc > 0 THEN rc ELSE 1 END)) * 0.5)
                                     * exp(-0.01 * COALESCE(days_since, 0))
                            END
                        """,
                        {"uuid": uuid, "now": now, "hash": query_hash},
                    )
        except Exception as e:
            # Don't let importance tracking failures break recall
            print(f"Warning: importance update failed: {e}")

    # ─── 3. CONTEXT ASSEMBLY ────────────────────────────────────

    async def get_context(self, query: str, max_chars: int = 4000) -> str:
        """Assemble a context block for LLM system prompt.
        
        Combines relevant facts + entities + episode citations into a compact
        markdown string, truncated to max_chars.
        """
        results = await self.recall(query, limit=20)

        # Also get behavioral rules / preferences
        prefs = await self.recall("preferences decisions rules lessons", limit=10)

        # Merge and dedupe facts
        seen_facts = set()
        all_facts = []
        for f in results["facts"] + prefs["facts"]:
            if f["fact"] not in seen_facts:
                seen_facts.add(f["fact"])
                all_facts.append(f)

        # Build context
        lines = ["## Memory Context\n"]

        # Core entities
        if results["entities"]:
            lines.append("### Relevant Entities")
            for e in results["entities"][:5]:
                summary = f" — {e['summary']}" if e.get("summary") else ""
                lines.append(f"- **{e['name']}** ({e.get('entity_type', '?')}){summary}")
            lines.append("")

        # Current facts
        current = [f for f in all_facts if f["is_current"]]
        if current:
            lines.append("### Known Facts")
            for f in current[:15]:
                lines.append(f"- {f['fact']}")
            lines.append("")

        # Historical facts (superseded)
        historical = [f for f in all_facts if not f["is_current"]]
        if historical:
            lines.append("### Historical (superseded)")
            for f in historical[:5]:
                date = f['invalid_at'][:10] if f.get('invalid_at') else '?'
                lines.append(f"- ~~{f['fact']}~~ (changed {date})")
            lines.append("")

        # Episode citations
        if results["episodes"]:
            lines.append("### Conversation Citations")
            for ep in results["episodes"][:3]:
                snippet = ep["content"][:200] + "..." if ep.get("content") else ""
                lines.append(f"- [{ep.get('source', 'unknown')}] {snippet}")
            lines.append("")

        context = "\n".join(lines)

        if len(context) > max_chars:
            context = context[:max_chars - 20] + "\n\n[...truncated]"

        return context

    # ─── 4. WHAT CHANGED? ──────────────────────────────────────

    async def whats_new(self, since_hours: int = 24) -> list[dict]:
        """Get facts added or changed in the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        results = await self.graphiti.search(
            f"facts and decisions from the last {since_hours} hours",
            num_results=20,
            group_ids=[self.group_id],
        )

        recent = []
        for r in results:
            if r.created_at and r.created_at >= cutoff:
                recent.append({"fact": r.fact, "created": str(r.created_at), "type": "new"})
            elif r.invalid_at and r.invalid_at >= cutoff:
                recent.append({"fact": r.fact, "invalidated": str(r.invalid_at), "type": "changed"})

        return recent

    # ─── 5. STATS ───────────────────────────────────────────────

    async def stats(self) -> dict:
        """Get memory graph stats."""
        async with self.graphiti.driver.session() as s:
            r1 = await s.run("MATCH (n:Entity) RETURN count(n) as c")
            r2 = await s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as c")
            r3 = await s.run("MATCH (e:Episodic) RETURN count(e) as c")

            # Top entities by connection count
            r4 = await s.run(
                "MATCH (n:Entity)-[r:RELATES_TO]-() "
                "RETURN n.name AS name, count(r) AS connections "
                "ORDER BY connections DESC LIMIT 10"
            )

            # Core memories (high importance)
            r5 = await s.run(
                "MATCH ()-[r:RELATES_TO]->() "
                "WHERE r.importance IS NOT NULL AND r.importance >= 75 "
                "RETURN r.fact AS fact, r.importance AS importance "
                "ORDER BY r.importance DESC LIMIT 10"
            )

            entities = (await r1.single())["c"]
            facts = (await r2.single())["c"]
            episodes = (await r3.single())["c"]
            top_entities = [r.data() async for r in r4]
            core_memories = [r.data() async for r in r5]

        return {
            "entities": entities,
            "facts": facts,
            "episodes": episodes,
            "top_entities": top_entities,
            "core_memories": core_memories,
        }


# ─── CLI TEST ───────────────────────────────────────────────────

async def test():
    mem = JarvisMemory()
    await mem.init()

    print("=== Stats ===")
    s = await mem.stats()
    print(f"  Entities: {s['entities']}, Facts: {s['facts']}, Episodes: {s['episodes']}")

    print("\n=== Hybrid Recall: 'DFS cheatsheet project' ===")
    results = await mem.recall("DFS cheatsheet project", limit=5)
    print(f"  Facts: {len(results['facts'])}, Entities: {len(results['entities'])}, Episodes: {len(results['episodes'])}")
    for f in results["facts"][:3]:
        marker = "✓" if f["is_current"] else "✗"
        print(f"  {marker} {f['fact']}")
    for e in results["entities"][:2]:
        print(f"  🔵 {e['name']} ({e.get('entity_type', '?')})")
    for ep in results["episodes"][:1]:
        print(f"  📝 {ep['content'][:100]}...")

    print("\n=== Context Assembly: 'Rekollect architecture' ===")
    ctx = await mem.get_context("Rekollect architecture decisions")
    print(ctx[:500])

    await mem.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(test())
