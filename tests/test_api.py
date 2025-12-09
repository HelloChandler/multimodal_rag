"""模块：pipeline测试.

验证FastAPI RAG端点的接线是否正确。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.app.api import app


class DummyPipeline:
    def run(self, query, image_base64=None, top_k=None):
        return f"echo:{query}"


def test_rag_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("src.app.api._get_pipeline", lambda: DummyPipeline())
    client = TestClient(app)
    resp = client.post("/rag", json={"query": "hello"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "echo:hello"

