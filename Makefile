.PHONY: dev test ingest

dev:
	docker compose up -d
	@echo "Waiting for Neo4j..."
	@sleep 5
	uv run uvicorn rekollect.api:app --host 0.0.0.0 --port 8100 --reload

test:
	uv run pytest tests/ -v

ingest:
	uv run python scripts/backfill.py $(ARGS)

stats:
	@curl -s http://localhost:8100/v1/stats | python3 -m json.tool

recall:
	@curl -s "http://localhost:8100/v1/recall?query=$(Q)&limit=5" | python3 -m json.tool
