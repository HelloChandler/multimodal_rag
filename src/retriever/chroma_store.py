"""模块：data_processing.

基于ChromaDB的持久化层，用于存储和检索多模态嵌入向量，
同时支持索引构建和在线推理阶段使用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings


logger = logging.getLogger(__name__)


class ChromaStore:
    """基于ChromaDB的向量存储类，用于存储和检索多模态嵌入向量。
    
    属性：
        index_path: 索引文件路径
        dimension: 嵌入向量的维度
        collection_name: 集合名称
        client: ChromaDB客户端
        collection: ChromaDB集合对象
    """
    def __init__(self, index_path: Path, dimension: int):
        """初始化ChromaStore对象。
        
        参数：
            index_path: 索引文件路径
            dimension: 嵌入向量的维度
        """
        self.index_path = index_path
        self.dimension = dimension
        self.collection_name = "multimodal_embeddings"
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=str(index_path.parent),
            settings=ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(index_path.parent)
            )
        )
        
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"dimension": dimension}
        )

    def add(self, vectors: Sequence[Sequence[float]], metadatas: Sequence[dict]) -> None:
        """向ChromaDB添加向量和元数据。
        
        参数：
            vectors: 要添加的向量序列
            metadatas: 与向量对应的元数据序列
        """
        if not vectors:
            return
        
        # 为每个向量生成唯一ID
        ids = [f"vec_{i}" for i in range(self.collection.count(), self.collection.count() + len(vectors))]
        
        # 将向量添加到集合
        self.collection.add(
            embeddings=[list(vector) for vector in vectors],
            documents=[meta.get("content", "") for meta in metadatas],
            metadatas=list(metadatas),
            ids=ids
        )

    def save(self) -> None:
        """保存ChromaDB索引到磁盘。
        
        注意：ChromaDB会自动持久化变更，此方法确保变更立即保存。
        """
        # ChromaDB会自动持久化变更
        self.client.persist()
        logger.info("将ChromaDB索引保存到 %s", self.index_path.parent)

    def load(self) -> None:
        """从磁盘加载ChromaDB索引。
        
        注意：ChromaDB在客户端初始化时会自动加载集合，
        此方法主要用于更新维度信息。
        """
        # ChromaDB在客户端初始化时会自动加载集合
        # 从集合更新维度信息
        collection_info = self.client.get_collection(name=self.collection_name)
        self.dimension = collection_info.metadata.get("dimension", self.dimension)

    def search(self, vector: Sequence[float], top_k: int) -> List[Tuple[dict, float]]:
        """根据查询向量搜索最相似的向量。
        
        参数：
            vector: 查询向量
            top_k: 返回的最相似向量数量
            
        返回：
            元组列表，每个元组包含元数据和相似度分数
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        matches = []
        if results["metadatas"] and results["distances"]:
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                matches.append((metadata, float(distance)))
        
        return matches


__all__ = ["ChromaStore"]
