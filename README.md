# Rekollect

Graph-based agent memory with hybrid search, temporal facts, and importance scoring.

Built on [Graphiti](https://github.com/getzep/graphiti) + Neo4j.

## Why Rekollect?

| Feature | Rekollect | Mem0 | Graphiti | LightRAG |
|---------|-----------|------|----------|----------|
| Knowledge Graph | ✅ | ❌ | ✅ | ✅ |
| Temporal Facts | ✅ | ❌ | ✅ | ❌ |
| Hybrid Search (BM25 + Vector + RRF) | ✅ | ❌ | ❌ | Partial |
| Importance Scoring | ✅ | ❌ | ❌ | ❌ |
| Context Assembly | ✅ | ❌ | ❌ | ❌ |
| MCP Server | ✅ | ✅ | ❌ | ❌ |
| REST API | ✅ | ✅ | ❌ | ❌ |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/rekollect/rekollect.git
cd rekollect
uv sync

# 2. Start Neo4j
docker compose up -d

# 3. Set your OpenAI key (used for entity extraction)
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Run the API
make dev
```

API is now at `http://localhost:8100`.

## Usage

### Remember something
```bash
curl -X POST http://localhost:8100/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Use Railway for always-on hosting", "source": "decision"}'
```

### Recall with hybrid search
```bash
curl "http://localhost:8100/v1/recall?query=hosting+decision&limit=5"
```

Returns facts (graph edges), entities (nodes), and episode citations.

### Get LLM-ready context
```bash
curl "http://localhost:8100/v1/context?query=database+architecture&max_chars=4000"
```

### Ingest OpenClaw sessions
```bash
curl -X POST http://localhost:8100/v1/ingest/session \
  -H "Content-Type: application/json" \
  -d '{"session_path": "/path/to/session.jsonl"}'
```

### Stats
```bash
curl http://localhost:8100/v1/stats
```

## How it works

**Ingestion:** Text → Graphiti extracts entities, relationships, and temporal facts → stored in Neo4j with embeddings.

**Search:** Query → BM25 full-text + vector cosine similarity → Reciprocal Rank Fusion merges results → importance-weighted ranking.

**Importance:** Every memory starts at 50/100. Recalled frequently? Goes up. Across diverse queries? Goes up faster. Not recalled in weeks? Decays. Memories that cross 75+ become "core memories" — always included in context.

## Architecture

```
rekollect/
├── memory.py       # RekollectMemory — the main class
├── api.py          # FastAPI REST endpoints
├── embedders.py    # Ollama adapter (pluggable)
├── importance.py   # Scoring algorithm + Cypher
└── ingestion.py    # Session parsing + chunking
```

Dependencies: `graphiti-core` (graph engine), `fastapi` (API), `httpx` (Ollama client).

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/remember` | Add a memory |
| POST | `/v1/recall` | Hybrid search |
| GET | `/v1/recall?query=...` | Hybrid search (GET) |
| POST | `/v1/context` | LLM context assembly |
| GET | `/v1/context?query=...` | LLM context assembly (GET) |
| POST | `/v1/ingest/session` | Ingest OpenClaw session |
| POST | `/v1/ingest/batch` | Batch ingest directory |
| GET | `/v1/stats` | Graph statistics |
| GET | `/v1/whats-new` | Recent changes |
| GET | `/v1/entities` | List/search entities |
| GET | `/v1/timeline?entity=...` | Entity timeline |
| GET | `/health` | Health check |

## License

MIT
