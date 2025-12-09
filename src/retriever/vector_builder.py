"""模块：data_processing.

将解析后的Markdown文档转换为嵌入向量负载，并持久化到ChromaDB中，
架起原始文档与可搜索向量存储之间的桥梁。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.config.settings import Settings
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings
from src.utils.file_utils import DocumentPayload, ImagePayload, parse_markdown_documents
from src.utils.text_utils import split_markdown

from .chroma_store import ChromaStore


logger = logging.getLogger(__name__)


def build_chroma_from_markdown(
    settings: Settings,
    embeddings: MultimodalEmbeddings,
    chunk_size: int,
    overlap: int,
    source_dir: Optional[Path] = None,
) -> None:
    origin_dir = source_dir or settings.paths.raw_data
    documents = parse_markdown_documents(origin_dir, settings.paths.raw_data)
    payloads = _to_embedding_payloads(documents, chunk_size, overlap, settings.paths.processed_data)
    vectors = embeddings.embed_documents(payloads)
    if not vectors:
        logger.warning("No vectors generated from documents")
        return
    dimension = len(vectors[0])
    store = ChromaStore(settings.paths.index_path, dimension)
    store.add(vectors, [payload["metadata"] for payload in payloads])
    store.save()
    logger.info("Saved %d vectors to %s", len(vectors), settings.paths.index_path.parent)


def _to_embedding_payloads(
    documents: List[DocumentPayload],
    chunk_size: int,
    overlap: int,
    processed_dir: Path,
) -> List[dict]:
    payloads: List[dict] = []
    for doc in documents:
        text_chunks = split_markdown(doc.content, chunk_size, overlap)
        for i, chunk in enumerate(text_chunks):
            _persist_chunk(processed_dir, doc.path.name, i, chunk)
            payloads.append(
                {
                    "type": "text",
                    "content": chunk,
                    "source": f"{doc.path.name}#chunk-{i}",
                    "metadata": {
                        "source": str(doc.path),
                        "chunk_id": i,
                        "type": "text",
                        "content": chunk,
                    },
                }
            )
        for img in doc.images:
            payloads.append(_image_payload(doc.path, img))
    return payloads


def _image_payload(doc_path: Path, image: ImagePayload) -> dict:
    return {
        "type": "image",
        "content": image.base64,
        "source": f"{doc_path.name}#{image.path.name}",
        "metadata": {
            "source": str(doc_path),
            "type": "image",
            "alt": image.alt,
            "image": image.base64,
        },
    }


def _persist_chunk(processed_dir: Path, doc_name: str, chunk_id: int, content: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    target = processed_dir / f"{doc_name}.chunk{chunk_id}.txt"
    target.write_text(content, encoding="utf-8")


__all__ = ["build_chroma_from_markdown"]

