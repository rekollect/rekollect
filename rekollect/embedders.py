"""Embedder re-exports from our Graphiti fork.

Our fork (rekollect/graphiti) adds the Ollama embedder to graphiti-core.
This module re-exports for convenience.
"""

from graphiti_core.embedder.ollama import OllamaEmbedder, OllamaEmbedderConfig

__all__ = ["OllamaEmbedder", "OllamaEmbedderConfig"]
