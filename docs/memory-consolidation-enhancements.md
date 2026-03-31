# Memory Consolidation Enhancements for Rekollect

Inspired by Claude Code's Auto Dream memory system. These enhancements would make Rekollect's memory layer smarter over time -- not just storing everything, but actively maintaining and improving stored knowledge.

Source: Claude Code Memory 2.0 / Auto Dream (Anthropic, March 2026)

---

## 1. Automated Memory Consolidation ("Dream Cycles")

### What Claude Code does
A background process reviews all stored memories and performs maintenance:
- Converts relative dates to absolute ("yesterday" -> "2026-03-28")
- Deletes contradicted facts (old decision replaced by new one)
- Removes stale entries (references to deleted files, old debugging notes)
- Merges duplicate/overlapping entries into single clean records

### How Rekollect should implement this
- **New endpoint: `POST /v1/consolidate`** -- triggers a consolidation cycle for a user
- **Background job**: runs automatically based on trigger conditions (see #2)
- **Process**:
  1. Fetch all documents for user, ordered by created_at
  2. Use LLM to identify contradictions, duplicates, and stale content across chunks
  3. Update or soft-delete outdated chunks (never hard-delete -- keep audit trail)
  4. Merge duplicate entities in the graph (Graphiti dedup already does some of this)
  5. Update document metadata with `last_consolidated_at` timestamp
- **Constraint**: consolidation only writes to memory data. Never touches user's external systems.

### Database changes
- Add `last_consolidated_at` timestamp to `documents` table
- Add `consolidation_log` table: user_id, run_at, documents_reviewed, chunks_pruned, entities_merged, duration_ms

---

## 2. Smart Trigger Conditions (Dual-Gate)

### What Claude Code does
Consolidation only runs when BOTH conditions are true:
- 24 hours since last consolidation
- 5+ sessions/ingestions since last consolidation

This prevents wasting compute on inactive users while ensuring active users get regular cleanup.

### How Rekollect should implement this
- Track per-user: `last_consolidation_at`, `ingestions_since_consolidation`
- Increment `ingestions_since_consolidation` on every `/v1/remember` or `/v1/add` call
- Check trigger conditions on each API call (cheap check) or via periodic cron
- Configurable thresholds per user/tier:
  - Free tier: 48h + 10 ingestions
  - Pro tier: 24h + 5 ingestions
  - Enterprise: 12h + 3 ingestions (or on-demand)

### Database changes
- Add to `profiles` or new `user_settings` table: `last_consolidation_at`, `ingestions_since_consolidation`, `consolidation_threshold_hours`, `consolidation_threshold_ingestions`

---

## 3. Targeted Signal Extraction (Not Full Reads)

### What Claude Code does
During consolidation, it doesn't read every session transcript end-to-end. It does targeted grep-style searches for:
- User corrections ("you're wrong", redirections)
- Explicit saves ("remember this")
- Recurring themes across sessions
- Important decisions (architecture, tools, workflows)

### How Rekollect should implement this
- During recall, boost results that match high-signal patterns:
  - Content containing decision language ("we decided", "the choice was", "going with")
  - Content with correction language ("actually", "no, it's", "that's wrong")
  - Content explicitly tagged as important (via metadata)
- During consolidation, prioritize reviewing documents with these signals
- Add an `importance_score` to chunks, computed during processing:
  - Decisions = high importance
  - Facts with dates = medium importance
  - Casual notes = low importance
  - Stale debugging notes = decay over time

### Database changes
- Add `importance_score` float to `chunks` table
- Add `signal_type` enum to chunks: 'decision', 'correction', 'fact', 'note', 'debug'

---

## 4. Memory Index with Size Limit

### What Claude Code does
MEMORY.md is kept under 200 lines -- it's an index pointing to topic files, not a dump of everything. This ensures the most important context loads fast at startup.

### How Rekollect should implement this
- **New concept: Memory Summary** -- a per-user summary document that's always returned first in recall results
- Generated during consolidation: a concise overview of the user's key knowledge areas
- Limited to ~2000 tokens (equivalent of 200 lines)
- Structure:
  ```
  ## Topics
  - Cooper Flagg: NBA stats, Mavericks draft pick, rookie season
  - Restaurants: Lucia (Dallas, Italian), [others]
  - Tech Stack: FastAPI, Supabase, pgvector, Neo4j
  - Personal: Mom's birthday June 15, [others]

  ## Recent Decisions
  - 2026-03-27: Chose pgvector over Pinecone for vector storage
  - 2026-03-28: OAuth only for production, email/password for local

  ## Active Projects
  - Rekollect: memory platform, MCP server, API keys
  - DFS Cheatsheet: v3 rebuild in progress
  ```
- This summary gets injected into MCP tool context so agents always have a high-level view of what's stored

### Database changes
- Add `memory_summary` text field to `profiles` table (or separate `user_summaries` table)
- Add `summary_updated_at` timestamp

---

## 5. Temporal Fact Resolution

### What Claude Code does
Converts relative dates to absolute during consolidation. "Yesterday we decided X" becomes "On 2026-03-15 we decided X."

### How Rekollect should implement this
- During `/v1/remember` processing, detect relative time references in content
- Resolve them to absolute dates based on the document's `created_at`
- Store both the original content and a `resolved_content` field
- During consolidation, re-resolve any remaining relative references
- Graphiti already handles temporal facts via `valid_from` on edges -- enhance this with resolved dates

### Processing changes
- Add date resolution step to the chunking/processing pipeline
- Use LLM or regex patterns: "yesterday", "last week", "this morning", "a few days ago"
- Store resolved dates in chunk metadata

---

## 6. Contradiction Detection

### What Claude Code does
When new information contradicts old information, the old entry is deleted during consolidation.

### How Rekollect should implement this
- During `/v1/remember`, after embedding + graph extraction:
  1. Search for existing chunks with high similarity (>0.85) to new content
  2. If found, use LLM to classify: duplicate, update, or contradiction
  3. If contradiction: mark old chunk as `superseded_by` pointing to new chunk
  4. During recall, filter out superseded chunks (or rank them lower)
- During consolidation: batch-process all superseded relationships and clean up

### Database changes
- Add `superseded_by` UUID (nullable FK to chunks) on `chunks` table
- Add `superseded_at` timestamp

---

## 7. Namespace-Scoped Consolidation

### What Rekollect needs (not in Claude Code)
Since we're adding namespaces (snippets, architecture, personal, etc.), consolidation should run per-namespace:
- Snippets namespace: merge duplicate code patterns, remove outdated versions
- Architecture namespace: resolve decision contradictions, update stale references
- Personal namespace: consolidate recurring facts about people/places

Each namespace can have different consolidation rules and thresholds.

---

## Implementation Priority

### Phase 1 (next sprint)
1. Namespace support on documents/recall (prerequisite for scoped consolidation)
2. `importance_score` on chunks (computed during processing)
3. `last_consolidated_at` tracking per user

### Phase 2 (following sprint)
4. Basic consolidation endpoint -- contradiction detection + duplicate merging
5. Smart trigger conditions (dual-gate)
6. Temporal fact resolution during processing

### Phase 3 (future)
7. Memory summary generation (index document per user)
8. Namespace-scoped consolidation rules
9. Full background consolidation job (cron-based)

---

## Key Principle
> "Store everything, consolidate intelligently, recall precisely."

The user shouldn't curate what goes in. Rekollect accepts everything and uses consolidation to maintain quality over time -- just like the brain uses sleep to strengthen important memories and discard noise.
