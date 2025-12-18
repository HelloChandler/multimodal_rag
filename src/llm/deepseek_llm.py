"""模块：pipeline.

定义了DeepSeek V3.2多模态模型的实现，用于RAG流水线中基于事实的回答生成。
"""

from __future__ import annotations

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional

import aiohttp

from src.config.settings import Settings
from src.model.unified_interface import UnifiedLLM
from src.llm.doubao_llm import _extract_text


logger = logging.getLogger(__name__)


class DeepseekLLM(UnifiedLLM):
    """DeepSeek V3.2多模态大语言模型的实现。"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._api_key = settings.models.deepseek_api_key if hasattr(settings.models, 'deepseek_api_key') else None
        self._api_url = "https://api.deepseek.com/chat/completions"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "deepseek-vl-1.5-chat",
            "type": "llm",
            "provider": "deepseek",
            "version": "v3.2"
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
        """使用DeepSeek V3.2模型生成文本。
        
        参数：
            prompt: 输入提示文本
            images: 可选的图片base64字符串列表
            **kwargs: 额外的模型参数
            
        返回：
            生成的文本
        """
        return asyncio.run(self.generate_async(prompt, images or [], **kwargs))
    
    async def generate_async(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """异步使用DeepSeek V3.2模型生成文本。"""
        if not self._api_key:
            logger.error("DeepSeek API密钥未配置")
            raise ValueError("DeepSeek API密钥未配置")
        
        # 构建请求内容
        contents: List[Any] = ["<image>", prompt]
        for img in images or []:
            contents.append(f"<image>{img}</image>")
        
        request = {
            "model": "deepseek-vl-1.5-chat",  # DeepSeek V3.2多模态模型
            "messages": [
                {
                    "role": "user",
                    "content": contents,
                }
            ],
            "stream": False,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._api_url, headers=headers, json=request) as response:
                    if response.status != 200:
                        logger.warning(f"DeepSeek API请求失败，状态码: {response.status}")
                        raise Exception(f"DeepSeek API请求失败，状态码: {response.status}")
                    
                    response_data = await response.json()
                    return _extract_text(response_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek LLM generation failed: %s", exc)
            raise
    
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """使用DeepSeek V3.2模型进行多轮对话。
        
        参数：
            messages: 对话历史，每个消息包含role和content
            **kwargs: 额外的模型参数
            
        返回：
            包含生成的消息和其他信息的字典
        """
        return asyncio.run(self.chat_async(messages, **kwargs))
    
    async def chat_async(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """异步使用DeepSeek V3.2模型进行多轮对话。"""
        if not self._api_key:
            logger.error("DeepSeek API密钥未配置")
            raise ValueError("DeepSeek API密钥未配置")
        
        request = {
            "model": "deepseek-vl-1.5-chat",  # DeepSeek V3.2多模态模型
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._api_url, headers=headers, json=request) as response:
                    if response.status != 200:
                        logger.warning(f"DeepSeek API请求失败，状态码: {response.status}")
                        raise Exception(f"DeepSeek API请求失败，状态码: {response.status}")
                    
                    response_data = await response.json()
                    return {
                        "message": response_data.get("choices", [{}])[0].get("message"),
                        "model": response_data.get("model"),
                        "usage": response_data.get("usage")
                    }
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek LLM chat failed: %s", exc)
            raise


__all__ = ["DeepseekLLM"]