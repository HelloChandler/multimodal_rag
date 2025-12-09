"""Module: data_processing.

FAISS-based persistence layer that stores and retrieves multimodal embeddings
used by both indexing and online inference stages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np


logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, index_path: Path, dimension: int):
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(".meta.json")
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[dict] = []

    def add(self, vectors: Sequence[Sequence[float]], metadatas: Sequence[dict]) -> None:
        if not vectors:
            return
        matrix = np.array(vectors, dtype="float32")
        if matrix.shape[1] != self.dimension:
            raise ValueError("Vector dimension mismatch")
        self.index.add(matrix)
        self.metadata.extend(metadatas)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        self.dimension = self.index.d
        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = []

    def search(self, vector: Sequence[float], top_k: int) -> List[Tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        query = np.array([vector], dtype="float32")
        scores, indices = self.index.search(query, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(score)))
        return results


__all__ = ["FaissStore"]

