"""模块：pipeline.

定义了LLM抽象类和Doubao多模态具体类，用于RAG流水线中基于事实的回答生成。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from volcenginesdkarkruntime import AsyncArk

from src.config.settings import Settings


logger = logging.getLogger(__name__)


class LLMBase(ABC):
    @abstractmethod
    def generate(self, prompt: str, images: Optional[List[str]] = None) -> str: ...


class DoubaoLLM(LLMBase):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = AsyncArk(
            ak=settings.models.ark_api_key,
            sk=settings.models.ark_api_secret,
            region=settings.models.ark_region,
        )

    def generate(self, prompt: str, images: Optional[List[str]] = None) -> str:
        return asyncio.run(self._generate_async(prompt, images or []))

    async def _generate_async(self, prompt: str, images: List[str]) -> str:
        contents: List[Any] = [{"type": "text", "text": prompt}]
        for img in images:
            contents.append({"type": "image", "image_base64": img})

        request = {
            "model": self.settings.models.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": contents,
                }
            ],
        }

        try:
            response = await self._client.create_chat_completion(**request)
            return _extract_text(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed: %s", exc)
            return "抱歉，当前无法生成回答，请稍后再试。"


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or response.get("choices") if isinstance(response, dict) else None
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", {})
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", [])
    if isinstance(content, list):
        texts = [item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(filter(None, texts))
    return content or ""


__all__ = ["LLMBase", "DoubaoLLM"]

