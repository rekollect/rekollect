"""Graphiti wrapper — init, episode ingestion, search, close.

Ported from JarvisMemory in jarvis_memory.py, simplified for the v2 API.
"""

import logging
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

from app.config import Settings
from app.graph.extraction import PERSONAL_MEMORY_INSTRUCTIONS
from app.graph.providers import get_embedder, get_llm_client

logger = logging.getLogger(__name__)


class GraphEngine:
    """Thin wrapper around Graphiti with config-driven providers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        llm_client = get_llm_client(
            provider=settings.llm_provider,
            model=settings.llm_model,
            small_model=settings.llm_small_model,
            api_key=settings.openai_api_key,
        )
        embedder = get_embedder(
            model=settings.graph_embedding_model,
            dim=settings.graph_embedding_dim,
        )
        driver = Neo4jDriver(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            database=settings.neo4j_database,
        )
        self.graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
        )

    async def init(self):
        """Validate embedding dimensions, then build graph indices and constraints."""
        # --- Validate embedding model output matches config ---
        test_embedding = await self.graphiti.embedder.create("dimension check")
        actual_dim = len(test_embedding)
        configured_dim = self.settings.graph_embedding_dim
        if actual_dim != configured_dim:
            raise RuntimeError(
                f"Embedding model produces {actual_dim} dimensions but "
                f"graph_embedding_dim is configured as {configured_dim}. "
                f"Update config or switch models."
            )
        logger.info(
            f"Embedding dimension validated: {actual_dim} matches config"
        )

        # --- Validate Neo4j index matches config (if it exists) ---
        result = await self.graphiti.driver.execute_query(
            "SHOW INDEXES YIELD name, options "
            "WHERE name = 'entity_dedup_embedding' "
            "RETURN options"
        )
        records = result.records if hasattr(result, "records") else []
        if records:
            options = records[0].data().get("options", {})
            index_dim = (
                options.get("indexConfig", {})
                .get("vector.dimensions")
            )
            if index_dim is not None and index_dim != configured_dim:
                raise RuntimeError(
                    f"Neo4j entity_dedup_embedding index has {index_dim} dimensions "
                    f"but config specifies {configured_dim}. Drop and recreate the "
                    f"index manually: DROP INDEX entity_dedup_embedding IF EXISTS; "
                    f"then restart."
                )
            logger.info(
                f"Neo4j entity_dedup_embedding index dimension validated: {index_dim}"
            )
        else:
            logger.info(
                "Neo4j entity_dedup_embedding index not found — "
                "Graphiti will create it"
            )

        await self.graphiti.build_indices_and_constraints()
        logger.info("Graphiti indices built")

    @staticmethod
    def make_group_id(user_id: str, collection: str = "default") -> str:
        # Graphiti group_id only allows alphanumeric, dashes, underscores
        return f"{user_id}__{collection}"

    @staticmethod
    def make_group_ids(user_id: str, collections: list[str]) -> list[str]:
        return [GraphEngine.make_group_id(user_id, c) for c in collections]

    async def add_episodes(
        self,
        chunks: list[str],
        source: str = "rekollect-api",
        user_id: str = "anonymous",
        collection: str = "default",
    ) -> int:
        """Ingest text chunks as episodes into the graph.

        Returns number of episodes submitted.
        """
        group_id = self.make_group_id(user_id, collection)
        now = datetime.now(timezone.utc)
        episodes = [
            RawEpisode(
                name=f"{source}:chunk:{i}",
                content=chunk,
                source=EpisodeType.text,
                source_description=source,
                reference_time=now,
            )
            for i, chunk in enumerate(chunks)
        ]

        if not episodes:
            return 0

        await self.graphiti.add_episode_bulk(
            bulk_episodes=episodes,
            group_id=group_id,
            custom_extraction_instructions=PERSONAL_MEMORY_INSTRUCTIONS,
        )
        logger.info(f"Ingested {len(episodes)} episodes from {source} (group={group_id})")
        return len(episodes)

    async def search(
        self, query: str, limit: int = 10, group_ids: list[str] | None = None,
    ) -> list[dict]:
        """Search the graph for facts matching query.

        Returns list of dicts with content, entity, valid_from, created_at.
        """
        results = await self.graphiti.search(
            query, num_results=limit, group_ids=group_ids,
        )
        facts = []
        for edge in results:
            facts.append(
                {
                    "content": edge.fact if hasattr(edge, "fact") else str(edge),
                    "entity": edge.source_node.name if hasattr(edge, "source_node") and edge.source_node else None,
                    "valid_from": str(edge.valid_at) if hasattr(edge, "valid_at") and edge.valid_at else None,
                    "created_at": str(edge.created_at) if hasattr(edge, "created_at") and edge.created_at else None,
                }
            )
        return facts

    async def close(self):
        """Close the Graphiti driver."""
        await self.graphiti.close()
        logger.info("Graphiti closed")
