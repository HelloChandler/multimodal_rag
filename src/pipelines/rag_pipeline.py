"""模块：pipeline.

实现端到端的检索增强生成工作流：嵌入查询、获取相似上下文，
并使用多模态输入调用大语言模型。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.config.settings import Settings, get_settings
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings
from src.llm.doubao_llm import DoubaoLLM, LLMBase
from src.llm.deepseek_llm import DeepseekLLM
from src.model.model_manager import LLMManager, EmbeddingsManager
from src.retriever.chroma_store import ChromaStore


logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline:
    settings: Settings
    embeddings_manager: EmbeddingsManager
    llm_manager: LLMManager
    store: ChromaStore

    def run(self, query: str, image_base64: Optional[str] = None, top_k: Optional[int] = None) -> str:
        top_k = top_k or self.settings.default_top_k
        query_vector = self._embed_query(query, image_base64)
        matches = self.store.search(query_vector, top_k)
        context = self._format_context(matches)
        prompt = self._build_prompt(query, context)
        images = [image_base64] if image_base64 else self._collect_image_context(matches)
        return self.llm_manager.invoke((prompt, images))

    def _embed_query(self, query: str, image_base64: Optional[str]) -> List[float]:
        docs = [{"type": "text", "content": query, "source": "user-query"}]
        if image_base64:
            docs.append({"type": "image", "content": image_base64, "source": "user-image"})
        try:
            vectors = self.embeddings_manager.invoke(docs)
            if not vectors:
                # 获取第一个嵌入模型的零向量作为默认值
                return self.embeddings_manager.models[0].zero_vector()
            return _average_vectors(vectors)
        except Exception as e:
            logger.error("嵌入模型调用失败: %s", e)
            # 使用第一个嵌入模型的零向量作为回退
            return self.embeddings_manager.models[0].zero_vector()

    def _format_context(self, matches: Sequence) -> str:
        formatted = []
        for meta, score in matches:
            snippet = meta.get("content") or meta.get("alt")
            formatted.append(f"[score={score:.3f}] {snippet}")
        return "\n".join(formatted)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "你是严谨的多模态助手。"
            "请基于给定上下文回答问题，无法回答时须明确说明。\n"
            f"上下文:\n{context}\n"
            f"问题: {query}"
        )

    def _collect_image_context(self, matches: Sequence) -> List[str]:
        images = []
        for meta, _ in matches:
            if meta.get("type") == "image" and meta.get("image"):
                images.append(meta["image"])
        return images


def create_pipeline() -> RAGPipeline:
    settings = get_settings()
    
    # 创建嵌入模型列表
    embeddings_models = [MultimodalEmbeddings(settings)]
    embeddings_manager = EmbeddingsManager(embeddings_models, max_retries=1)
    
    # 创建LLM模型列表，支持多模型回退
    llm_models = [DoubaoLLM(settings), DeepseekLLM(settings)]
    llm_manager = LLMManager(llm_models, max_retries=1)
    
    # 获取嵌入维度（使用第一个模型的默认维度）
    embedding_dim = embeddings_models[0].dimension
    
    store = ChromaStore(settings.paths.index_path, embedding_dim)
    try:
        store.load()
    except Exception as e:
        logger.warning("Failed to load ChromaDB index: %s. Run build_index first.", str(e))
    
    return RAGPipeline(
        settings=settings, 
        embeddings_manager=embeddings_manager, 
        llm_manager=llm_manager, 
        store=store
    )


def _average_vectors(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    avg = [0.0] * length
    for vec in vectors:
        for i, value in enumerate(vec):
            avg[i] += value
    return [v / len(vectors) for v in avg]


__all__ = ["RAGPipeline", "create_pipeline"]

