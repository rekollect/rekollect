"""RekollectMemory — the core memory engine.

Wraps Graphiti with hybrid search, importance scoring,
context assembly, and session ingestion.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from rekollect.embedders import OllamaEmbedder, OllamaEmbedderConfig
from rekollect.importance import IMPORTANCE_UPDATE_CYPHER, query_hash
from rekollect.ingestion import messages_to_episodes, parse_openclaw_session


class RekollectMemory:
    """Graph-based agent memory with hybrid search and importance scoring."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        group_id: str = "",
        embedder=None,
    ):
        if embedder is None:
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

    # ─── INGEST ─────────────────────────────────────────────────

    async def ingest_session(self, session_path: str) -> dict:
        """Ingest an OpenClaw session log (.jsonl) into the graph."""
        try:
            messages = parse_openclaw_session(session_path)
        except FileNotFoundError as e:
            return {"error": str(e)}

        if not messages:
            return {"episodes": 0, "messages": 0, "message": "No substantial messages found"}

        session_id = Path(session_path).stem
        episodes = messages_to_episodes(messages, session_id, self.group_id)
        await self.graphiti.add_episode_bulk(episodes)

        return {"episodes": len(episodes), "messages": len(messages)}

    async def ingest_sessions_batch(self, directory: str, since_hours: int | None = None) -> dict:
        """Ingest all .jsonl session files from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        files = sorted(dir_path.glob("*.jsonl"))
        # Filter out deleted, backup, and lock files
        files = [f for f in files if ".deleted." not in f.name and ".bak." not in f.name and ".lock" not in f.name]

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

    # ─── RECALL ─────────────────────────────────────────────────

    async def recall(self, query: str, limit: int = 10) -> dict:
        """Hybrid search (BM25 + vector + RRF) across facts, entities, and episodes.

        Returns structured results and updates importance metrics.
        """
        config = COMBINED_HYBRID_SEARCH_RRF
        config.limit = limit

        results = await self.graphiti.search_(
            query=query,
            config=config,
            group_ids=[self.group_id],
        )

        facts = [
            {
                "type": "fact",
                "uuid": edge.uuid,
                "fact": edge.fact,
                "name": edge.name,
                "valid_from": str(edge.valid_at) if edge.valid_at else None,
                "invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
                "is_current": edge.invalid_at is None,
                "source_node": edge.source_node_uuid,
                "target_node": edge.target_node_uuid,
            }
            for edge in (results.edges or [])
        ]

        entities = [
            {
                "type": "entity",
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
                "entity_type": node.labels[0] if node.labels else None,
            }
            for node in (results.nodes or [])
        ]

        episodes = [
            {
                "type": "episode",
                "uuid": ep.uuid,
                "name": ep.name,
                "content": ep.content[:500] if ep.content else None,
                "source": ep.source_description,
                "created_at": str(ep.created_at) if ep.created_at else None,
            }
            for ep in (results.episodes or [])
        ]

        # Update importance (fire-and-forget)
        edge_uuids = [f["uuid"] for f in facts]
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
        qhash = query_hash(query)
        now = datetime.now(timezone.utc).isoformat()
        try:
            async with self.graphiti.driver.session() as session:
                for uuid in edge_uuids:
                    await session.run(
                        IMPORTANCE_UPDATE_CYPHER,
                        {"uuid": uuid, "now": now, "hash": qhash},
                    )
        except Exception as e:
            pass  # Don't let importance tracking break recall

    # ─── CONTEXT ASSEMBLY ───────────────────────────────────────

    async def get_context(self, query: str, max_chars: int = 4000) -> str:
        """Assemble a markdown context block for LLM prompts."""
        results = await self.recall(query, limit=20)
        prefs = await self.recall("preferences decisions rules lessons", limit=10)

        # Dedupe facts
        seen = set()
        all_facts = []
        for f in results["facts"] + prefs["facts"]:
            if f["fact"] not in seen:
                seen.add(f["fact"])
                all_facts.append(f)

        lines = ["## Memory Context\n"]

        if results["entities"]:
            lines.append("### Relevant Entities")
            for e in results["entities"][:5]:
                summary = f" — {e['summary']}" if e.get("summary") else ""
                lines.append(f"- **{e['name']}** ({e.get('entity_type', '?')}){summary}")
            lines.append("")

        current = [f for f in all_facts if f["is_current"]]
        if current:
            lines.append("### Known Facts")
            for f in current[:15]:
                lines.append(f"- {f['fact']}")
            lines.append("")

        historical = [f for f in all_facts if not f["is_current"]]
        if historical:
            lines.append("### Historical (superseded)")
            for f in historical[:5]:
                date = f["invalid_at"][:10] if f.get("invalid_at") else "?"
                lines.append(f"- ~~{f['fact']}~~ (changed {date})")
            lines.append("")

        if results["episodes"]:
            lines.append("### Conversation Citations")
            for ep in results["episodes"][:3]:
                snippet = ep["content"][:200] + "..." if ep.get("content") else ""
                lines.append(f"- [{ep.get('source', 'unknown')}] {snippet}")
            lines.append("")

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[: max_chars - 20] + "\n\n[...truncated]"
        return context

    # ─── STATS ──────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Get memory graph stats."""
        async with self.graphiti.driver.session() as s:
            r1 = await s.run("MATCH (n:Entity) RETURN count(n) as c")
            r2 = await s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as c")
            r3 = await s.run("MATCH (e:Episodic) RETURN count(e) as c")
            r4 = await s.run(
                "MATCH (n:Entity)-[r:RELATES_TO]-() "
                "RETURN n.name AS name, count(r) AS connections "
                "ORDER BY connections DESC LIMIT 10"
            )
            r5 = await s.run(
                "MATCH ()-[r:RELATES_TO]->() "
                "WHERE r.importance IS NOT NULL AND r.importance >= 75 "
                "RETURN r.fact AS fact, r.importance AS importance "
                "ORDER BY r.importance DESC LIMIT 10"
            )
            return {
                "entities": (await r1.single())["c"],
                "facts": (await r2.single())["c"],
                "episodes": (await r3.single())["c"],
                "top_entities": [r.data() async for r in r4],
                "core_memories": [r.data() async for r in r5],
            }

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
