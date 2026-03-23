"""Test Phase 0 — Hybrid search + importance scoring."""
import asyncio
from jarvis_memory import JarvisMemory


async def test_phase0():
    print("🧪 Phase 0 Test: Rekollect Memory Engine\n")
    
    mem = JarvisMemory(group_id="")  # Match existing data
    await mem.init()
    
    # 1. Stats
    print("=== 1. Graph Stats ===")
    stats = await mem.stats()
    print(f"  Entities: {stats['entities']}")
    print(f"  Facts: {stats['facts']}")
    print(f"  Episodes: {stats['episodes']}")
    print(f"  Top entities: {[e['name'] for e in stats['top_entities'][:3]]}")
    print()
    
    # 2. Hybrid recall (first call)
    print("=== 2. Hybrid Recall (1st call): 'DFS cheatsheet' ===")
    results1 = await mem.recall("DFS cheatsheet project status", limit=5)
    print(f"  Results: {results1['total']} total")
    print(f"    Facts: {len(results1['facts'])}")
    print(f"    Entities: {len(results1['entities'])}")
    print(f"    Episodes: {len(results1['episodes'])}")
    
    if results1['facts']:
        print(f"\n  Sample fact: {results1['facts'][0]['fact']}")
    if results1['entities']:
        print(f"  Sample entity: {results1['entities'][0]['name']} ({results1['entities'][0].get('entity_type', '?')})")
    if results1['episodes']:
        snippet = results1['episodes'][0]['content'][:100] if results1['episodes'][0].get('content') else ''
        print(f"  Sample episode: {snippet}...")
    print()
    
    # 3. Recall again (importance should increment)
    print("=== 3. Hybrid Recall (2nd call, same query) ===")
    results2 = await mem.recall("DFS cheatsheet project status", limit=5)
    print(f"  Results: {results2['total']} total")
    print(f"  (Importance metrics updated in background)")
    print()
    
    # 4. Context assembly
    print("=== 4. Context Assembly: 'Rekollect memory engine' ===")
    ctx = await mem.get_context("Rekollect memory engine architecture", max_chars=1000)
    print(f"  Context length: {len(ctx)} chars")
    print(f"  Preview:\n{ctx[:300]}...")
    print()
    
    # 5. What's new
    print("=== 5. What's New (last 48 hours) ===")
    recent = await mem.whats_new(since_hours=48)
    print(f"  Recent changes: {len(recent)}")
    if recent:
        print(f"  Sample: {recent[0]}")
    print()
    
    await mem.close()
    print("✅ Phase 0 Test Complete!\n")
    print("Next steps:")
    print("  - Add MCP server tools (ingest_session, recall_context)")
    print("  - Wire into OpenClaw config")
    print("  - Backfill historical sessions")


if __name__ == "__main__":
    asyncio.run(test_phase0())
