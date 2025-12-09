"""模块：pipeline.

RAG系统的CLI入口，允许操作人员针对本地知识库提交临时的多模态查询。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipelines.rag_pipeline import create_pipeline
from src.utils.image_utils import image_to_base64


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for multimodal RAG")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--image", help="Optional image path")
    parser.add_argument("--top-k", type=int, default=None, help="Number of retrieved chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = create_pipeline()
    image_b64 = load_image_arg(args.image) if args.image else None
    answer = pipeline.run(args.query, image_base64=image_b64, top_k=args.top_k)
    print(answer)


def load_image_arg(image_path: str | None) -> str | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image %s not found, ignoring", image_path)
        return None
    return image_to_base64(path)


if __name__ == "__main__":
    main()

