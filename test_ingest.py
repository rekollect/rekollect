"""Test: Ingest a conversation episode and query it back."""
import asyncio
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

async def main():
    # Connect to Neo4j
    g = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "memory-engine-dev",
    )

    # Initialize the graph (creates indexes etc)
    print("Initializing graph...")
    await g.build_indices_and_constraints()

    # Ingest a real conversation excerpt from today's session
    print("\nIngesting episode 1: DFS architecture decision...")
    await g.add_episode(
        name="DFS v3 Architecture Discussion",
        episode_body="""
        Elton and Jarvis discussed the DFS Cheatsheet v3 architecture.
        Elton decided to rebuild the iOS app in SwiftUI with a fully online API-driven approach.
        The backend uses FastAPI with Supabase Postgres.
        They decided to use purpose-built GET endpoints instead of the legacy flexible query engine.
        The optimizer was merged from the dfs-api Fly.io deployment into the cheatsheet-api.
        All 14 priority API items were completed and verified against live data.
        The iOS app scaffold was created at ~/Develop/290design/dfc-ios with 22 Swift files.
        """,
        source=EpisodeType.text,
        source_description="OpenClaw session - DFS v3 planning",
        reference_time=datetime(2026, 3, 21, 14, 0),
    )

    print("\nIngesting episode 2: Memory engine decision...")
    await g.add_episode(
        name="Memory Engine Architecture Decision",
        episode_body="""
        Elton wants to build a memory engine that gives AI agents human-like recall.
        They decided to fork Graphiti as the graph engine rather than building from scratch.
        The engine will use Neo4j for the best quality graph, with FalkorDB as a lightweight option.
        Elton specifically said he wants the memory to be unmatched, not just good enough.
        The memory engine will be dual-use: an OpenClaw skill for Jarvis and the Rekollect backend.
        Rekollect is Elton's memory assistant app with a chat-driven interface.
        The design includes importance decay, behavior extraction, and citation tracking.
        Elton prefers to test against live data before declaring things done.
        """,
        source=EpisodeType.text,
        source_description="OpenClaw session - Memory engine brainstorm",
        reference_time=datetime(2026, 3, 21, 21, 30),
    )

    # Query the graph
    print("\n--- Querying: 'What did Elton decide about the graph database?' ---")
    results = await g.search("What did Elton decide about the graph database?")
    for r in results:
        print(f"  [{r.score:.2f}] {r.fact}")

    print("\n--- Querying: 'What is Rekollect?' ---")
    results = await g.search("What is Rekollect?")
    for r in results:
        print(f"  [{r.score:.2f}] {r.fact}")

    print("\n--- Querying: 'What are Elton preferences?' ---")
    results = await g.search("What are Elton's preferences?")
    for r in results:
        print(f"  [{r.score:.2f}] {r.fact}")

    # Close connection
    await g.close()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
