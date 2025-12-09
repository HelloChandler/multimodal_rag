"""模块：data_processing测试.

验证自定义多模态嵌入适配器的健壮性。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.settings import ModelSettings, PathSettings, Settings
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings


class DummyArk:
    async def create_embedding(self, **_: str):
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    paths = PathSettings(
        base_dir=tmp_path,
        raw_data=tmp_path / "raw",
        processed_data=tmp_path / "processed",
        index_path=tmp_path / "index" / "faiss.index",
    )
    for path in (paths.raw_data, paths.processed_data, paths.index_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    models = ModelSettings(
        ark_api_key="key",
        ark_api_secret="secret",
        ark_region="cn",
        llm_model="llm",
        embed_model="embed",
    )
    return Settings(paths=paths, models=models)


def test_embed_documents(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr("src.embeddings.multimodal_embeddings.AsyncArk", lambda **_: DummyArk())
    embeddings = MultimodalEmbeddings(settings)
    docs = [{"type": "text", "content": "hello", "source": "s"}]
    vectors = embeddings.embed_documents(docs)
    assert len(vectors) == 1
    assert pytest.approx(sum(vectors[0])) == 0.6


def test_embed_handles_error(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    class ErrorArk:
        async def create_embedding(self, **_: str):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.embeddings.multimodal_embeddings.AsyncArk", lambda **_: ErrorArk())
    embeddings = MultimodalEmbeddings(settings)
    vectors = embeddings.embed_documents([{"type": "text", "content": "bad", "source": "s"}])
    assert len(vectors[0]) == embeddings._dimension  # type: ignore[attr-defined]
    assert all(value == 0.0 for value in vectors[0])

