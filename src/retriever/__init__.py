"""Module: data_processing.

Vector store helpers for persisting and querying multimodal embeddings.
"""

from .chroma_store import ChromaStore
from .vector_builder import build_chroma_from_markdown

__all__ = ["ChromaStore", "build_chroma_from_markdown"]

