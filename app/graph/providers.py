"""Config-driven LLM and embedding factory.

Ported from jarvis_memory.py _get_llm_client() and rekollect embedders.
Supports: openai, anthropic, ollama, groq, openrouter, gemini.
"""

import os

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig


def get_llm_client(
    provider: str = "openai",
    model: str = "gpt-4.1-nano",
    small_model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Create an LLM client based on provider config."""
    small_model = small_model or model
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if provider == "ollama":
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
        return OpenAIGenericClient(
            config=LLMConfig(
                api_key="ollama",
                model=model,
                small_model=small_model,
                base_url=base_url,
            )
        )
    elif provider == "anthropic":
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        return AnthropicClient(
            config=LLMConfig(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                model=model,
                small_model=small_model,
            )
        )
    elif provider == "gemini":
        from graphiti_core.llm_client.gemini_client import GeminiClient

        return GeminiClient(
            config=LLMConfig(
                api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                model=model,
                small_model=small_model,
            )
        )
    elif provider == "groq":
        from graphiti_core.llm_client.groq_client import GroqClient

        return GroqClient(
            config=LLMConfig(
                api_key=api_key or os.getenv("GROQ_API_KEY"),
                model=model,
                small_model=small_model,
            )
        )
    elif provider == "openrouter":
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        return OpenAIGenericClient(
            config=LLMConfig(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                model=model,
                small_model=small_model,
                base_url="https://openrouter.ai/api/v1",
            )
        )
    else:  # openai (default)
        from graphiti_core.llm_client import OpenAIClient

        return OpenAIClient(
            config=LLMConfig(
                api_key=api_key,
                model=model,
                small_model=small_model,
            )
        )


def get_embedder(model: str = "text-embedding-3-small", dim: int = 1536):
    """Create an OpenAI embedder for Graphiti (graph + dedup)."""
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    return OpenAIEmbedder(
        OpenAIEmbedderConfig(
            embedding_model=model,
            embedding_dim=dim,
        )
    )
