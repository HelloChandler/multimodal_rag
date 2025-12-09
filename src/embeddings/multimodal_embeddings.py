"""模块：data_processing.

自定义LangChain嵌入接口，用于豆包多模态模型，为文本和图像生成向量表示。
本模块遵循[data_processing]规范进行嵌入生成。

功能特性：
- 多模态嵌入支持（文本和图像）
- 异步嵌入生成，提高性能
- 固定维度向量输出
- 带日志记录的错误处理
- 与LangChain Embeddings接口集成
- 通过配置文件可配置
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from langchain.embeddings.base import Embeddings
from volcenginesdkarkruntime import AsyncArk

from src.config.settings import Settings


logger = logging.getLogger(__name__)

# 默认嵌入维度（遵循data_processing规范）
DEFAULT_DIM = 1024


class MultimodalEmbeddings(Embeddings):
    """文本和图像的多模态嵌入生成器。
    
    该类实现了LangChain Embeddings接口，并遵循[data_processing]规范进行嵌入生成。
    
    属性：
        settings: 包含模型配置的应用程序设置
        _client: 用于嵌入生成的AsyncArk客户端
        _dimension: 当前嵌入维度（自动检测）
    """
    
    def __init__(self, settings: Settings):
        """初始化多模态嵌入生成器。
        
        参数：
            settings: 包含嵌入模型配置的应用程序设置
        """
        self.settings = settings
        self._client = AsyncArk(
            ak=settings.models.ark_api_key,
            sk=settings.models.ark_api_secret,
            region=settings.models.ark_region,
        )
        self._dimension = DEFAULT_DIM  # 默认维度（遵循data_processing规范）

    def embed_query(self, text: str) -> List[float]:
        """为单个文本查询生成嵌入向量。
        
        参数：
            text: 输入文本查询
            
        返回：
            嵌入向量，作为浮点数列表（遵循data_processing规范）
            
        遵循规则（data_processing规范）：
            - 固定维度向量输出
            - 错误时回退到零向量
        """
        payload = [{"type": "text", "content": text, "source": "query"}]
        vectors = self.embed_documents(payload)
        return vectors[0] if vectors else self._zero_vector()

    def embed_documents(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        """为多个文档（文本和/或图像）生成嵌入向量。
        
        参数：
            docs: 包含类型、内容和可选来源的文档负载列表
            
        返回：
            嵌入向量列表（遵循data_processing规范）
            
        遵循规则（data_processing规范）：
            - 所有嵌入使用固定维度向量
            - 处理空列表
            - 异步处理以提高性能
        """
        if not docs:
            return []
        return asyncio.run(self._embed_async(docs))

    async def _embed_async(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        """异步为多个文档生成嵌入向量。
        
        参数：
            docs: 文档负载列表
            
        返回：
            嵌入向量列表
            
        处理流程：
            1. 为每个文档创建嵌入任务
            2. 并发运行任务
            3. 处理错误并替换为零向量
            4. 根据实际嵌入输出更新维度
        """
        tasks = [self._embed_single(doc) for doc in docs]
        results = await asyncio.gather(*tasks)
        
        # 用零向量替换None结果（遵循data_processing规范）
        valid = [self._zero_vector() if vec is None else vec for vec in results]
        
        # 根据实际嵌入输出更新维度
        if valid and valid[0]:
            self._dimension = len(valid[0])
            logger.info("Auto-detected embedding dimension: %d", self._dimension)
        
        return valid

    async def _embed_single(self, doc: Dict[str, Any]) -> Optional[List[float]]:
        """为单个文档生成嵌入向量。
        
        参数：
            doc: 包含类型、内容和可选来源的文档负载
            
        返回：
            嵌入向量，失败时返回None
            
        处理流程：
            1. 准备嵌入API的负载
            2. 异步调用嵌入API
            3. 从响应中提取嵌入向量
            4. 带日志记录的错误处理
            
        遵循规则（data_processing规范）：
            - 多模态支持（文本和图像）
            - UTF-8编码兼容
            - 带日志记录的错误处理
        """
        payload = {
            "model": self.settings.models.embed_model,
            "input": [
                {"type": doc.get("type", "text"),  # 如果未指定类型，则默认为文本
                    "content": doc.get("content", ""),  # 如果未指定内容，则为空字符串
                }
            ],
        }
        
        try:
            # 使用AsyncArk客户端生成嵌入向量
            response = await self._client.create_embedding(**payload)
            
            # 从响应中提取嵌入向量
            data = _extract_embedding(response)
            if data is None:
                raise ValueError("Empty embedding response")
            
            return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding failed for %s: %s", doc.get("source", "unknown"), exc)
            return None

    def _zero_vector(self) -> List[float]:
        """创建当前维度的零向量。
        
        返回：
            当前维度的零向量（遵循data_processing规范）
        """
        return [0.0] * self._dimension

    def zero_vector(self) -> List[float]:
        """获取当前维度零向量的公共方法。
        
        返回：
            当前维度的零向量（遵循data_processing规范）
        """
        return self._zero_vector()

    @property
    def dimension(self) -> int:
        """获取当前嵌入维度。
        
        返回：
            嵌入维度（遵循data_processing规范）
        """
        return self._dimension


def _extract_embedding(response: Any) -> Optional[List[float]]:
    """从API响应中提取嵌入向量。
    
    参数：
        response: API响应对象或字典
        
    返回：
        嵌入向量，提取失败时返回None
        
    处理流程：
        1. 处理对象和字典两种响应类型
        2. 从响应中提取data字段
        3. 从第一个data项中提取嵌入向量
        4. 处理对象和字典两种data项类型
    """
    # 处理对象和字典两种响应类型
    data = getattr(response, "data", None) or response.get("data") if isinstance(response, dict) else None
    if not data:
        return None
    
    # 从第一个data项中提取嵌入向量
    first = data[0]
    if isinstance(first, dict):
        return first.get("embedding")
    return getattr(first, "embedding", None)


__all__ = ["MultimodalEmbeddings"]

