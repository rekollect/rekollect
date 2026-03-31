# Rekollect

Memory infrastructure for AI agents. Store, search, and recall anything.

## Quick Start

```bash
# Clone and start
git clone https://github.com/rekollect/rekollect.git
cd rekollect
docker compose up -d
cp .env.example .env  # add your OpenAI key
uv sync
uv run uvicorn app.main:app --port 8181 --reload
```

Default API key: `rk_dev_rekollect`

## Test it

```bash
# Store something
curl -X POST http://localhost:8181/v1/remember \
  -H "Authorization: Bearer rk_dev_rekollect" \
  -H "Content-Type: application/json" \
  -d '{"content": "Cooper Flagg was drafted #1 by the Mavericks in 2025"}'

# Search for it
curl -X POST http://localhost:8181/v1/recall \
  -H "Authorization: Bearer rk_dev_rekollect" \
  -H "Content-Type: application/json" \
  -d '{"query": "Cooper Flagg draft"}'
```

## MCP Server

Connect Claude Desktop, Cursor, or any MCP client:

```json
{
  "mcpServers": {
    "rekollect": {
      "url": "http://localhost:8181/mcp",
      "headers": {
        "Authorization": "Bearer rk_dev_rekollect"
      }
    }
  }
}
```

## Connect to an MCP Client

Rekollect uses HTTP transport — every client uses the same URL and bearer token.

> **Production / remote use:** Replace `http://localhost:8181` with your server URL and `rk_dev_rekollect` with your actual API key.

<details>
<summary>Claude Code</summary>
<br>

```bash
claude mcp add rekollect \
  --transport http \
  --url http://localhost:8181/mcp \
  --header "Authorization: Bearer rk_dev_rekollect"
```

</details>

<details>
<summary>Cursor</summary>
<br>

Add to `.cursor/mcp.json` (project-scoped) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "rekollect": {
      "url": "http://localhost:8181/mcp",
      "headers": {
        "Authorization": "Bearer rk_dev_rekollect"
      }
    }
  }
}
```

</details>

<details>
<summary>Claude Desktop</summary>
<br>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rekollect": {
      "url": "http://localhost:8181/mcp",
      "headers": {
        "Authorization": "Bearer rk_dev_rekollect"
      }
    }
  }
}
```

</details>

<details>
<summary>VS Code / VS Code Insiders</summary>
<br>

Add to your VS Code settings JSON:

```json
{
  "mcp.servers": {
    "rekollect": {
      "url": "http://localhost:8181/mcp",
      "headers": {
        "Authorization": "Bearer rk_dev_rekollect"
      }
    }
  }
}
```

</details>

<details>
<summary>Codex CLI</summary>
<br>

```bash
codex mcp add rekollect \
  --transport http \
  --url http://localhost:8181/mcp \
  --header "Authorization: Bearer rk_dev_rekollect"
```

Or add to `~/.codex/config.toml`:

```toml
[mcp_servers.rekollect]
url = "http://localhost:8181/mcp"

[mcp_servers.rekollect.headers]
Authorization = "Bearer rk_dev_rekollect"
```

</details>

<details>
<summary>Windsurf</summary>
<br>

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "rekollect": {
      "url": "http://localhost:8181/mcp",
      "headers": {
        "Authorization": "Bearer rk_dev_rekollect"
      }
    }
  }
}
```

</details>

<details>
<summary>Kiro</summary>
<br>

Add to `.kiro/settings/mcp.json` (project-scoped) or `~/.kiro/settings/mcp.json` (global):

```json
{
  "mcpServers": {
    "rekollect": {
      "url": "http://localhost:8181/mcp",
      "headers": {
        "Authorization": "Bearer rk_dev_rekollect"
      }
    }
  }
}
```

</details>

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/remember | Store + process content |
| POST | /v1/recall | Hybrid search (vector + graph) |
| POST | /v1/add | Store without processing |
| POST | /v1/process | Process existing document |
| GET | /v1/memories | List memories |
| GET | /v1/memories/{id} | Get memory by ID |
| DELETE | /v1/memories/{id} | Delete memory |
| GET | /v1/collections | List collections |
| POST | /v1/keys | Create API key |
| GET | /v1/keys | List API keys |
| DELETE | /v1/keys/{id} | Revoke API key |
| GET | /health | Health check |

## Architecture

- **Postgres + pgvector** -- document storage + vector similarity search
- **Neo4j + Graphiti** -- knowledge graph extraction + fact search
- **FastAPI** -- async API server
- **FastMCP** -- MCP server for AI agent integration
- **OpenAI** -- embeddings + entity extraction

## Tests

```bash
uv run pytest tests/test_e2e.py -v              # E2E tests
DEBUG=true uv run pytest tests/test_quality.py -v -s  # Quality baseline
```

## License

MIT
