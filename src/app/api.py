"""模块：pipeline.

FastAPI接口层，将多模态RAG工作流作为HTTP端点暴露，用于产品集成。
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.rag_pipeline import create_pipeline, RAGPipeline


class RAGRequest(BaseModel):
    query: str
    image_base64: str | None = None
    top_k: int | None = None


class RAGResponse(BaseModel):
    answer: str


app = FastAPI(title="Multimodal RAG")
pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global pipeline  # noqa: PLW0603
    if pipeline is None:
        pipeline = create_pipeline()
    return pipeline


@app.post("/rag", response_model=RAGResponse)
def run_rag(request: RAGRequest) -> RAGResponse:
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    pipeline = _get_pipeline()
    answer = pipeline.run(request.query, image_base64=request.image_base64, top_k=request.top_k)
    return RAGResponse(answer=answer)

