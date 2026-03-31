"""Settings from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    port: int = 8181

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rekollect"

    # Neo4j
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rekollect-dev"
    neo4j_database: str = "neo4j"

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-nano"
    llm_small_model: str = "gpt-4.1-nano"
    llm_provider: str = "openai"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Graph embeddings
    graph_embedding_model: str = "text-embedding-3-small"
    graph_embedding_dim: int = 1536

    # Dedup
    dedup_model: str = "gpt-4.1"

    model_config = {"env_file": ".env", "extra": "ignore"}
