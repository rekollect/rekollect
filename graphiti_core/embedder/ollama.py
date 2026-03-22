"""Ollama embedder for Graphiti — runs embeddings locally via Ollama."""

from collections.abc import Iterable

import httpx
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_MODEL)
    base_url: str = Field(default=DEFAULT_BASE_URL)
    embedding_dim: int = Field(default=768, frozen=True)


class OllamaEmbedder(EmbedderClient):
    """Embedder that uses Ollama for local inference."""

    def __init__(self, config: OllamaEmbedderConfig | None = None):
        if config is None:
            config = OllamaEmbedderConfig()
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = " ".join(input_data)
        else:
            text = str(input_data)

        response = await self.client.post(
            f"{self.config.base_url}/api/embeddings",
            json={"model": self.config.embedding_model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Batch embed — Ollama doesn't have native batch, so we parallelize."""
        import asyncio
        tasks = [self.create(text) for text in input_data_list]
        return await asyncio.gather(*tasks)
