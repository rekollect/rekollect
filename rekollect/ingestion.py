"""Session ingestion for OpenClaw and other chat formats."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode


def parse_openclaw_session(session_path: str | Path) -> list[dict]:
    """Parse an OpenClaw .jsonl session into messages.

    Returns list of {role, content, ts} dicts.
    """
    path = Path(session_path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {session_path}")

    messages = []
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

    return messages


def chunk_messages(messages: list[dict], max_chars: int = 4000) -> list[str]:
    """Chunk messages into episode-sized text blocks."""
    episodes = []
    current_chunk = []
    current_len = 0

    for msg in messages:
        line = f"{msg['role'].upper()}: {msg['content']}"
        if current_len + len(line) > max_chars and current_chunk:
            episodes.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)

    if current_chunk:
        episodes.append("\n".join(current_chunk))

    return episodes


def messages_to_episodes(
    messages: list[dict],
    session_id: str,
    group_id: str = "",
) -> list[RawEpisode]:
    """Convert parsed messages into Graphiti RawEpisode objects."""
    chunks = chunk_messages(messages)

    # Use first message timestamp as reference
    first_ts = messages[0].get("ts") if messages else None
    ref_time = (
        datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        if first_ts
        else datetime.now(timezone.utc)
    )

    return [
        RawEpisode(
            name=f"Session {session_id} part {i + 1}",
            content=chunk,
            source=EpisodeType.text,
            source_description=f"openclaw:session:{session_id}",
            reference_time=ref_time + timedelta(minutes=i * 5),
            group_id=group_id,
        )
        for i, chunk in enumerate(chunks)
    ]
