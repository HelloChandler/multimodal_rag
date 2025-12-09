"""Module: pipeline.

Pipeline construction utilities for indexing and online retrieval.
"""

from .build_index import main as build_index_main
from .rag_pipeline import RAGPipeline, create_pipeline

__all__ = ["build_index_main", "RAGPipeline", "create_pipeline"]

