"""模块：pipeline.

命令行入口，用于使用多模态嵌入工作流将Markdown知识库转换为可搜索的ChromaDB索引。
"""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

from src.config.settings import get_settings
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings
from src.model.model_manager import EmbeddingsManager
from src.retriever.vector_builder import build_chroma_from_markdown


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从Markdown构建ChromaDB索引")
    parser.add_argument("--source", type=str, default=None, help="包含原始Markdown数据的目录")
    parser.add_argument("--chunk-size", type=int, default=None, help="每个分块的令牌数")
    parser.add_argument("--overlap", type=int, default=None, help="令牌重叠数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = args.overlap or settings.chunk_overlap
    source_dir = Path(args.source).resolve() if args.source else settings.paths.raw_data
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    # 使用EmbeddingsManager确保索引构建过程也能利用模型回退机制
    embeddings_models = [MultimodalEmbeddings(settings)]
    embeddings_manager = EmbeddingsManager(embeddings_models, max_retries=1)
    build_chroma_from_markdown(settings, embeddings_manager, chunk_size, overlap, source_dir=source_dir)
    logger.info("Index build complete")


if __name__ == "__main__":
    main()

