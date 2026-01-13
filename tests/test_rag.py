"""模块：pipeline测试.

为编排好的RAG流水线进行轻量级契约测试。
"""

from __future__ import annotations

from pathlib import Path

from src.config.settings import ModelSettings, PathSettings, Settings
from src.pipelines.rag_pipeline import RAGPipeline


class StubEmbeddings:
    def embed_documents(self, docs):
        return [[float(i + 1) for i in range(3)] for _ in docs]

    def zero_vector(self):
        return [0.0, 0.0, 0.0]


class StubLLM:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt: str, images=None):
        self.last_prompt = prompt
        return "stub-answer"


class StubStore:
    def search(self, vector, top_k):
        return [({"content": "context text", "type": "text"}, 0.1)]


def make_settings(tmp_path: Path) -> Settings:
    paths = PathSettings(
        base_dir=tmp_path,
        raw_data=tmp_path / "raw",
        processed_data=tmp_path / "processed",
        index_path=tmp_path / "index" / "faiss.index",
    )
    for path in (paths.raw_data, paths.processed_data, paths.index_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    models = ModelSettings(
        ark_api_key="k",
        ark_api_secret="s",
        ark_region="cn",
        deepseek_api_key="",
        llm_model="llm",
        embed_model="emb",
    )
    return Settings(paths=paths, models=models)


def test_rag_pipeline_runs(tmp_path: Path) -> None:
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        embeddings=StubEmbeddings(),
        llm=StubLLM(),
        store=StubStore(),
    )
    answer = pipeline.run("what is this?")
    assert answer == "stub-answer"

