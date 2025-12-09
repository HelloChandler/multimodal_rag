"""模块：data_processing测试.

确保ChromaDB的持久化和检索行为保持稳定。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.retriever.chroma_store import ChromaStore


def test_chroma_store_save_and_load(tmp_path: Path) -> None:
    index_path = tmp_path / "chroma.index"
    store = ChromaStore(index_path, dimension=3)
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    metas = [{"content": "a"}, {"content": "b"}]
    store.add(vectors, metas)
    store.save()

    other = ChromaStore(index_path, dimension=3)
    other.load()
    results = other.search([1.0, 0.0, 0.0], top_k=1)
    assert results
    assert results[0][0]["content"] == "a"

