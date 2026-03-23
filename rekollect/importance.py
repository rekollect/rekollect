"""Importance scoring for memory facts.

Every memory starts at 50/100. Importance rises with recall frequency
and query breadth, falls with time decay. Facts that cross the core
memory threshold (10+ recalls, 5+ unique queries) are always included
in context assembly.
"""

import hashlib
from datetime import datetime, timezone


CORE_MEMORY_THRESHOLD = 75.0  # importance score to be "always on"
DEFAULT_BASE_IMPORTANCE = 50.0


def compute_importance(
    base: float,
    recall_count: int,
    unique_queries: int,
    days_since_last_recall: float,
) -> float:
    """Compute importance score (0-100).

    Formula:
        importance = base × frequency_boost × breadth_boost × decay

    Where:
        frequency_boost = 1 + log(1 + recall_count) / log(20)
        breadth_boost = 1 + (unique_queries / max(recall_count, 1)) × 0.5
        decay = e^(-0.01 × days_since_last_recall)
    """
    import math

    frequency_boost = 1.0 + math.log(1.0 + recall_count) / math.log(20.0)
    breadth_boost = 1.0 + (unique_queries / max(recall_count, 1)) * 0.5
    decay = math.exp(-0.01 * days_since_last_recall)

    score = base * frequency_boost * breadth_boost * decay
    return min(max(score, 0.0), 100.0)


def query_hash(query: str) -> str:
    """Short hash for tracking unique queries."""
    return hashlib.md5(query.encode()).hexdigest()[:12]


# Cypher query for updating importance on recalled edges
IMPORTANCE_UPDATE_CYPHER = """
MATCH ()-[r:RELATES_TO {uuid: $uuid}]-()
SET r.recall_count = COALESCE(r.recall_count, 0) + 1,
    r.last_recalled_at = $now,
    r.query_hashes = CASE
        WHEN r.query_hashes IS NULL THEN [$hash]
        WHEN NOT $hash IN r.query_hashes THEN r.query_hashes + $hash
        ELSE r.query_hashes
    END
WITH r,
     COALESCE(r.recall_count, 1) AS rc,
     SIZE(COALESCE(r.query_hashes, [$hash])) AS uq,
     COALESCE(r.base_importance, 50.0) AS base,
     duration.between(
         COALESCE(datetime(r.last_recalled_at), datetime()),
         datetime()
     ).days AS days_since
SET r.unique_queries = uq,
    r.importance = CASE
        WHEN base * (1.0 + log(1.0 + rc) / log(20.0))
             * (1.0 + (toFloat(uq) / toFloat(CASE WHEN rc > 0 THEN rc ELSE 1 END)) * 0.5)
             * exp(-0.01 * COALESCE(days_since, 0)) > 100.0
        THEN 100.0
        ELSE base * (1.0 + log(1.0 + rc) / log(20.0))
             * (1.0 + (toFloat(uq) / toFloat(CASE WHEN rc > 0 THEN rc ELSE 1 END)) * 0.5)
             * exp(-0.01 * COALESCE(days_since, 0))
    END
"""
