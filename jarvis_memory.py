"""Jarvis Memory Layer — wraps Graphiti for OpenClaw agent use.

This is the bridge between OpenClaw's session system and Graphiti's
temporal knowledge graph. It handles:
1. Auto-ingestion of session transcripts
2. Context assembly for session start
3. Behavioral rule injection
4. Importance-weighted recall
5. "What changed?" queries
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from graphiti_core import Graphiti
from graphiti_core.embedder.ollama import OllamaEmbedder, OllamaEmbedderConfig
from graphiti_core.nodes import EpisodeType


class JarvisMemory:
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "memory-engine-dev",
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

    async def init(self):
        """Initialize graph indices."""
        await self.graphiti.build_indices_and_constraints()

    async def close(self):
        await self.graphiti.close()

    # ─── 1. INGEST ──────────────────────────────────────────────

    async def ingest_session(self, session_path: str) -> dict:
        """Ingest an OpenClaw session log (.jsonl) into the graph.
        
        Extracts user + assistant messages, ignores tool calls.
        Returns count of episodes created.
        """
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

            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if c.get("type") == "text"
                    )
                if content and len(content) > 20:  # skip short system messages
                    messages.append({"role": "user", "content": content, "ts": timestamp})
                    if not session_start:
                        session_start = timestamp

            elif role == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if c.get("type") == "text"
                    )
                if content and len(content) > 20:
                    messages.append({"role": "assistant", "content": content, "ts": timestamp})

        if not messages:
            return {"episodes": 0, "message": "No substantial messages found"}

        # Chunk messages into episodes (~2000 chars each for extraction quality)
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

        # Ingest each chunk
        ref_time = datetime.fromisoformat(session_start.replace("Z", "+00:00")) if session_start else datetime.now(timezone.utc)

        for i, episode_text in enumerate(episodes):
            await self.graphiti.add_episode(
                name=f"Session {path.stem} part {i+1}",
                episode_body=episode_text,
                source=EpisodeType.text,
                source_description=f"openclaw:session:{path.stem}",
                reference_time=ref_time + timedelta(minutes=i * 5),
            )

        return {"episodes": len(episodes), "messages": len(messages)}

    async def remember(self, text: str, source: str = "manual") -> dict:
        """Manually add a memory."""
        await self.graphiti.add_episode(
            name=f"Manual memory - {datetime.now(timezone.utc).isoformat()[:16]}",
            episode_body=text,
            source=EpisodeType.text,
            source_description=source,
            reference_time=datetime.now(timezone.utc),
        )
        return {"status": "remembered"}

    # ─── 2. RECALL ──────────────────────────────────────────────

    async def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Search the memory graph. Returns facts with metadata."""
        results = await self.graphiti.search(query, num_results=limit)

        return [
            {
                "fact": r.fact,
                "name": r.name,
                "valid_from": str(r.valid_at) if r.valid_at else None,
                "invalid_at": str(r.invalid_at) if r.invalid_at else None,
                "is_current": r.invalid_at is None,
                "source": r.source_node_uuid,
                "target": r.target_node_uuid,
            }
            for r in results
        ]

    # ─── 3. CONTEXT ASSEMBLY ────────────────────────────────────

    async def get_context(self, query: str, max_chars: int = 4000) -> str:
        """Assemble a context block for LLM system prompt.
        
        Combines relevant facts + behavioral rules into a compact
        markdown string, truncated to max_chars.
        """
        # Get relevant facts
        facts = await self.recall(query, limit=20)

        # Get behavioral rules (always included)
        rules = await self.recall("behavioral rules preferences lessons learned", limit=10)

        # Deduplicate
        seen = set()
        unique_facts = []
        for f in facts + rules:
            if f["fact"] not in seen:
                seen.add(f["fact"])
                unique_facts.append(f)

        # Separate current from historical
        current = [f for f in unique_facts if f["is_current"]]
        historical = [f for f in unique_facts if not f["is_current"]]

        # Build context
        lines = ["## Memory Context\n"]

        if current:
            lines.append("### Current Facts")
            for f in current:
                lines.append(f"- {f['fact']}")
            lines.append("")

        if historical:
            lines.append("### Historical (superseded)")
            for f in historical[:5]:  # limit historical
                lines.append(f"- ~~{f['fact']}~~ (changed {f['invalid_at'][:10] if f['invalid_at'] else '?'})")
            lines.append("")

        context = "\n".join(lines)

        # Truncate to budget
        if len(context) > max_chars:
            context = context[:max_chars - 20] + "\n\n[...truncated]"

        return context

    # ─── 4. WHAT CHANGED? ──────────────────────────────────────

    async def whats_new(self, since_hours: int = 24) -> list[dict]:
        """Get facts added or changed in the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        # Search for recent activity
        results = await self.graphiti.search(
            f"facts and decisions from the last {since_hours} hours",
            num_results=20,
        )

        recent = []
        for r in results:
            if r.created_at and r.created_at >= cutoff:
                recent.append({
                    "fact": r.fact,
                    "created": str(r.created_at),
                    "type": "new",
                })
            elif r.invalid_at and r.invalid_at >= cutoff:
                recent.append({
                    "fact": r.fact,
                    "invalidated": str(r.invalid_at),
                    "type": "changed",
                })

        return recent


# ─── CLI TEST ───────────────────────────────────────────────────

async def test():
    mem = JarvisMemory()
    await mem.init()

    # Test recall
    print("=== Recall: 'graph database decision' ===")
    results = await mem.recall("graph database decision", limit=5)
    for r in results:
        marker = "✓" if r["is_current"] else "✗"
        print(f"  {marker} {r['fact']}")

    # Test context assembly
    print("\n=== Context for 'DFS project' ===")
    context = await mem.get_context("DFS cheatsheet project status")
    print(context)

    # Test manual remember
    print("\n=== Remember a new fact ===")
    result = await mem.remember(
        "Elton opened the OpenClaw workspace in Obsidian for direct file access. "
        "This gives him visibility into all files Jarvis creates or edits."
    )
    print(f"  {result}")

    # Verify it stuck
    print("\n=== Recall: 'Obsidian' ===")
    results = await mem.recall("Obsidian", limit=3)
    for r in results:
        print(f"  {r['fact']}")

    await mem.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(test())
