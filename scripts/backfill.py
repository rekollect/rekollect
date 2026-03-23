"""Backfill OpenClaw session history into Rekollect.

Usage:
    uv run python backfill_sessions.py [--since-hours N] [--dry-run]
"""
import argparse
import asyncio
import sys
from pathlib import Path

from rekollect import RekollectMemory

SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"


async def backfill(since_hours: int | None = None, dry_run: bool = False):
    mem = RekollectMemory(neo4j_password="memory-engine-dev", group_id="")
    await mem.init()

    # Get current stats
    stats = await mem.stats()
    print(f"📊 Current graph: {stats['entities']} entities, {stats['facts']} facts, {stats['episodes']} episodes\n")

    if not SESSIONS_DIR.is_dir():
        print(f"❌ Sessions directory not found: {SESSIONS_DIR}")
        return

    # Find session files (skip deleted, lock files, and backups)
    files = sorted([
        f for f in SESSIONS_DIR.glob("*.jsonl")
        if ".deleted." not in f.name and ".bak." not in f.name and ".lock" not in f.name
    ])

    if since_hours:
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        files = [
            f for f in files
            if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) >= cutoff
        ]

    print(f"📁 Found {len(files)} session files in {SESSIONS_DIR}")
    
    if dry_run:
        for f in files:
            size = f.stat().st_size / 1024
            print(f"  📝 {f.name} ({size:.0f}KB)")
        print(f"\n🔍 Dry run — would ingest {len(files)} files")
        await mem.close()
        return

    total_episodes = 0
    total_messages = 0
    errors = []

    for i, f in enumerate(files):
        size = f.stat().st_size / 1024
        print(f"  [{i+1}/{len(files)}] {f.name} ({size:.0f}KB)...", end=" ", flush=True)
        try:
            result = await mem.ingest_session(str(f))
            if "error" in result:
                print(f"⚠️ {result['error']}")
                errors.append((f.name, result["error"]))
            else:
                eps = result.get("episodes", 0)
                msgs = result.get("messages", 0)
                total_episodes += eps
                total_messages += msgs
                print(f"✅ {eps} episodes, {msgs} messages")
        except Exception as e:
            print(f"❌ {e}")
            errors.append((f.name, str(e)))

    # Final stats
    stats_after = await mem.stats()
    print(f"\n📊 After backfill:")
    print(f"  Entities: {stats['entities']} → {stats_after['entities']} (+{stats_after['entities'] - stats['entities']})")
    print(f"  Facts: {stats['facts']} → {stats_after['facts']} (+{stats_after['facts'] - stats['facts']})")
    print(f"  Episodes: {stats['episodes']} → {stats_after['episodes']} (+{stats_after['episodes'] - stats['episodes']})")
    print(f"  Total messages processed: {total_messages}")
    
    if errors:
        print(f"\n⚠️ {len(errors)} errors:")
        for name, err in errors:
            print(f"  {name}: {err}")

    await mem.close()
    print("\n✅ Backfill complete!")


def main():
    parser = argparse.ArgumentParser(description="Backfill OpenClaw sessions into Rekollect")
    parser.add_argument("--since-hours", type=int, help="Only ingest files modified in last N hours")
    parser.add_argument("--dry-run", action="store_true", help="List files without ingesting")
    args = parser.parse_args()

    asyncio.run(backfill(since_hours=args.since_hours, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
