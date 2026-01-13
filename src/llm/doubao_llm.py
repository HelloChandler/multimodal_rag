"""模块：pipeline.

定义了LLM抽象类和Doubao多模态具体类，用于RAG流水线中基于事实的回答生成。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
 
from volcenginesdkarkruntime import AsyncArk

from src.config.settings import Settings
from src.model.unified_interface import UnifiedLLM

# Define LLMBase as an alias for UnifiedLLM for backward compatibility
LLMBase = UnifiedLLM


logger = logging.getLogger(__name__)


class DoubaoLLM(UnifiedLLM):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = AsyncArk(
            ak=settings.models.ark_api_key,
            sk=settings.models.ark_api_secret,
            region=settings.models.ark_region,
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.settings.models.llm_model,
            "type": "llm",
            "provider": "doubao",
            "version": "v1"
        }

    def health_check(self) -> bool:
        try:
            # 简单的健康检查，尝试生成一个短响应
            response = self.generate("Hello, are you there?", max_tokens=5)
            return response and "抱歉" not in response
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        return asyncio.run(self.generate_async(prompt, images or [], **kwargs))

    async def generate_async(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        contents: List[Any] = [{"type": "text", "text": prompt}]
        for img in images or []:
            contents.append({"type": "image", "image_base64": img})

        request = {
            "model": self.settings.models.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": contents,
                }
            ],
            **kwargs
        }

        try:
            response = await self._client.create_chat_completion(**request)
            return _extract_text(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed: %s", exc)
            raise

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return asyncio.run(self.chat_async(messages, **kwargs))

    async def chat_async(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        try:
            request = {
                "model": self.settings.models.llm_model,
                "messages": messages,
                **kwargs
            }
            response = await self._client.create_chat_completion(**request)
            return {
                "message": response.choices[0].message,
                "model": response.model,
                "usage": response.usage
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM chat failed: %s", exc)
            raise


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

