"""模块：model_integration_framework

模型管理器模块，负责统一调用模型并实现回退策略。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from src.config.config_manager import get_model_integration_config
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings
from src.llm.doubao_llm import LLMBase
from src.model.model_factory import get_model_factory
from src.model.unified_interface import (
    ModelRequest,
    ModelResponse,
    UnifiedEmbeddings,
    UnifiedLLM,
    UnifiedModel,
)


logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器，负责统一调用模型并实现回退策略。"""

    def __init__(self):
        self._config = get_model_integration_config()
        self._factory = get_model_factory()

    def generate_completion(
        self, prompt: str, images: Optional[List[str]] = None, **kwargs
    ) -> ModelResponse:
        """使用LLM生成文本完成。

        参数：
            prompt: 输入提示文本
            images: 可选的图片base64字符串列表
            **kwargs: 额外的模型参数

        返回：
            包含生成结果的统一响应对象
        """
        request = ModelRequest(
            prompt=prompt,
            images=images or [],
            parameters=kwargs
        )
        return self._invoke_llm_model(request)

    async def generate_completion_async(
        self, prompt: str, images: Optional[List[str]] = None, **kwargs
    ) -> ModelResponse:
        """异步使用LLM生成文本完成。

        参数：
            prompt: 输入提示文本
            images: 可选的图片base64字符串列表
            **kwargs: 额外的模型参数

        返回：
            包含生成结果的统一响应对象
        """
        request = ModelRequest(
            prompt=prompt,
            images=images or [],
            parameters=kwargs
        )
        return await self._invoke_llm_model_async(request)

    def chat_completion(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> ModelResponse:
        """使用LLM进行多轮对话。

        参数：
            messages: 对话历史，每个消息包含role和content
            **kwargs: 额外的模型参数

        返回：
            包含生成结果的统一响应对象
        """
        request = ModelRequest(
            messages=messages,
            parameters=kwargs
        )
        return self._invoke_llm_model(request, chat_mode=True)

    async def chat_completion_async(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> ModelResponse:
        """异步使用LLM进行多轮对话。

        参数：
            messages: 对话历史，每个消息包含role和content
            **kwargs: 额外的模型参数

        返回：
            包含生成结果的统一响应对象
        """
        request = ModelRequest(
            messages=messages,
            parameters=kwargs
        )
        return await self._invoke_llm_model_async(request, chat_mode=True)

    def embed_query(
        self, text: str, **kwargs
    ) -> ModelResponse:
        """为单个文本查询生成嵌入向量。

        参数：
            text: 输入文本查询
            **kwargs: 额外的模型参数

        返回：
            包含嵌入向量的统一响应对象
        """
        request = ModelRequest(
            documents=text,
            parameters=kwargs
        )
        return self._invoke_embeddings_model(request, query_mode=True)

    async def embed_query_async(
        self, text: str, **kwargs
    ) -> ModelResponse:
        """异步为单个文本查询生成嵌入向量。

        参数：
            text: 输入文本查询
            **kwargs: 额外的模型参数

        返回：
            包含嵌入向量的统一响应对象
        """
        request = ModelRequest(
            documents=text,
            parameters=kwargs
        )
        return await self._invoke_embeddings_model_async(request, query_mode=True)

    def embed_documents(
        self, documents: Union[str, List[str], List[Dict[str, Any]]], **kwargs
    ) -> ModelResponse:
        """为多个文档生成嵌入向量。

        参数：
            documents: 文档列表或单个文档
            **kwargs: 额外的模型参数

        返回：
            包含嵌入向量的统一响应对象
        """
        request = ModelRequest(
            documents=documents,
            parameters=kwargs
        )
        return self._invoke_embeddings_model(request)

    async def embed_documents_async(
        self, documents: Union[str, List[str], List[Dict[str, Any]]], **kwargs
    ) -> ModelResponse:
        """异步为多个文档生成嵌入向量。

        参数：
            documents: 文档列表或单个文档
            **kwargs: 额外的模型参数

        返回：
            包含嵌入向量的统一响应对象
        """
        request = ModelRequest(
            documents=documents,
            parameters=kwargs
        )
        return await self._invoke_embeddings_model_async(request)

    def invoke_llm_model(
        self, request: ModelRequest, chat_mode: bool = False
    ) -> ModelResponse:
        """同步调用LLM模型并实现回退策略。

        参数：
            request: 模型请求对象
            chat_mode: 是否使用聊天模式

        返回：
            包含生成结果的统一响应对象
        """
        return asyncio.run(self._invoke_llm_model_async(request, chat_mode))
    
    def _invoke_llm_model(
        self, request: ModelRequest, chat_mode: bool = False
    ) -> ModelResponse:
        """同步调用LLM模型并实现回退策略（内部方法）。

        参数：
            request: 模型请求对象
            chat_mode: 是否使用聊天模式

        返回：
            包含生成结果的统一响应对象
        """
        return asyncio.run(self._invoke_llm_model_async(request, chat_mode))

    async def _invoke_llm_model_async(
        self, request: ModelRequest, chat_mode: bool = False
    ) -> ModelResponse:
        """异步调用LLM模型并实现回退策略。

        参数：
            request: 模型请求对象
            chat_mode: 是否使用聊天模式

        返回：
            包含生成结果的统一响应对象
        """
        fallback_config = self._config.fallback.get("llm")
        if not fallback_config:
            logger.error("未配置LLM回退策略")
            return ModelResponse(
                content=None,
                model_name="",
                model_type="llm",
                is_success=False,
                error_message="未配置LLM回退策略"
            )

        model_order = fallback_config.order
        error_messages: List[str] = []

        for model_name in model_order:
            try:
                logger.info(f"尝试调用LLM模型: {model_name}")
                start_time = datetime.now()

                llm_model = self._factory.get_llm_model(model_name)
                
                if chat_mode:
                    content = await llm_model.chat_async(
                        request.get_messages(),
                        **request.get_parameters()
                    )
                else:
                    content = await llm_model.generate_async(
                        request.get_prompt(),
                        request.get_images(),
                        **request.get_parameters()
                    )

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info(f"成功调用LLM模型: {model_name}, 耗时: {duration:.2f}秒")

                return ModelResponse(
                    content=content,
                    model_name=model_name,
                    model_type="llm",
                    is_success=True,
                    metadata={"duration": duration}
                )

            except Exception as e:
                error_msg = f"调用LLM模型 {model_name} 失败: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)

                # 检查是否需要重试
                if fallback_config.retry_on_failure:
                    for retry in range(fallback_config.max_retries_per_model):
                        try:
                            logger.info(f"重试调用LLM模型: {model_name} (第{retry+1}次)")
                            start_time = datetime.now()

                            llm_model = self._factory.get_llm_model(model_name)
                            
                            if chat_mode:
                                content = await llm_model.chat_async(
                                    request.get_messages(),
                                    **request.get_parameters()
                                )
                            else:
                                content = await llm_model.generate_async(
                                    request.get_prompt(),
                                    request.get_images(),
                                    **request.get_parameters()
                                )

                            end_time = datetime.now()
                            duration = (end_time - start_time).total_seconds()

                            logger.info(f"重试成功调用LLM模型: {model_name}, 耗时: {duration:.2f}秒")

                            return ModelResponse(
                                content=content,
                                model_name=model_name,
                                model_type="llm",
                                is_success=True,
                                metadata={"duration": duration, "retry_count": retry+1}
                            )
                        except Exception as retry_e:
                            retry_error_msg = f"重试调用LLM模型 {model_name} 失败 (第{retry+1}次): {str(retry_e)}"
                            logger.warning(retry_error_msg)
                            error_messages.append(retry_error_msg)

        # 所有模型都调用失败
        logger.error("所有LLM模型调用失败")
        return ModelResponse(
            content=None,
            model_name="",
            model_type="llm",
            is_success=False,
            error_message="所有LLM模型调用失败: " + "; ".join(error_messages)
        )

    def _invoke_embeddings_model(
        self, request: ModelRequest, query_mode: bool = False
    ) -> ModelResponse:
        """同步调用嵌入模型并实现回退策略。

        参数：
            request: 模型请求对象
            query_mode: 是否为查询模式

        返回：
            包含嵌入向量的统一响应对象
        """
        return asyncio.run(self._invoke_embeddings_model_async(request, query_mode))

    async def _invoke_embeddings_model_async(
        self, request: ModelRequest, query_mode: bool = False
    ) -> ModelResponse:
        """异步调用嵌入模型并实现回退策略。

        参数：
            request: 模型请求对象
            query_mode: 是否为查询模式

        返回：
            包含嵌入向量的统一响应对象
        """
        fallback_config = self._config.fallback.get("embedding")
        if not fallback_config:
            logger.error("未配置嵌入模型回退策略")
            return ModelResponse(
                content=None,
                model_name="",
                model_type="embedding",
                is_success=False,
                error_message="未配置嵌入模型回退策略"
            )

        model_order = fallback_config.order
        error_messages: List[str] = []

        for model_name in model_order:
            try:
                logger.info(f"尝试调用嵌入模型: {model_name}")
                start_time = datetime.now()

                emb_model = self._factory.get_embeddings_model(model_name)
                
                if query_mode:
                    content = await emb_model.embed_query_async(
                        request.get_documents(),
                        **request.get_parameters()
                    )
                else:
                    content = await emb_model.embed_documents_async(
                        request.get_documents(),
                        **request.get_parameters()
                    )

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info(f"成功调用嵌入模型: {model_name}, 耗时: {duration:.2f}秒")

                return ModelResponse(
                    content=content,
                    model_name=model_name,
                    model_type="embedding",
                    is_success=True,
                    metadata={"duration": duration}
                )

            except Exception as e:
                error_msg = f"调用嵌入模型 {model_name} 失败: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)

                # 检查是否需要重试
                if fallback_config.retry_on_failure:
                    for retry in range(fallback_config.max_retries_per_model):
                        try:
                            logger.info(f"重试调用嵌入模型: {model_name} (第{retry+1}次)")
                            start_time = datetime.now()

                            emb_model = self._factory.get_embeddings_model(model_name)
                            
                            if query_mode:
                                content = await emb_model.embed_query_async(
                                    request.get_documents(),
                                    **request.get_parameters()
                                )
                            else:
                                content = await emb_model.embed_documents_async(
                                    request.get_documents(),
                                    **request.get_parameters()
                                )

                            end_time = datetime.now()
                            duration = (end_time - start_time).total_seconds()

                            logger.info(f"重试成功调用嵌入模型: {model_name}, 耗时: {duration:.2f}秒")

                            return ModelResponse(
                                content=content,
                                model_name=model_name,
                                model_type="embedding",
                                is_success=True,
                                metadata={"duration": duration, "retry_count": retry+1}
                            )
                        except Exception as retry_e:
                            retry_error_msg = f"重试调用嵌入模型 {model_name} 失败 (第{retry+1}次): {str(retry_e)}"
                            logger.warning(retry_error_msg)
                            error_messages.append(retry_error_msg)

        # 所有模型都调用失败
        logger.error("所有嵌入模型调用失败")
        return ModelResponse(
            content=None,
            model_name="",
            model_type="embedding",
            is_success=False,
            error_message="所有嵌入模型调用失败: " + "; ".join(error_messages)
        )

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息。

        参数：
            model_name: 模型名称

        返回：
            模型信息字典，如果模型不存在则返回None
        """
        try:
            # 尝试获取LLM模型
            llm_model = self._factory.get_llm_model(model_name)
            return llm_model.get_model_info()
        except Exception:
            pass

        try:
            # 尝试获取嵌入模型
            emb_model = self._factory.get_embeddings_model(model_name)
            return emb_model.get_model_info()
        except Exception:
            pass

        logger.error(f"未找到模型: {model_name}")
        return None

    def health_check(self, model_name: str) -> bool:
        """检查模型是否健康可用。

        参数：
            model_name: 模型名称

        返回：
            如果模型健康可用则返回True，否则返回False
        """
        try:
            # 尝试获取LLM模型
            llm_model = self._factory.get_llm_model(model_name)
            return llm_model.health_check()
        except Exception:
            pass

        try:
            # 尝试获取嵌入模型
            emb_model = self._factory.get_embeddings_model(model_name)
            return emb_model.health_check()
        except Exception:
            pass

        logger.error(f"未找到模型: {model_name}")
        return False


# 创建全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例。

    返回：
        全局模型管理器实例
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def init_model_manager() -> None:
    """初始化全局模型管理器。"""
    global _model_manager
    _model_manager = ModelManager()


def reload_model_manager() -> None:
    """重新加载全局模型管理器。"""
    global _model_manager
    _model_manager = None
    init_model_manager()

class LLMManager:
    """LLM模型管理器，负责管理多个LLM模型并实现回退策略。
    
    参数：
        models: LLM模型列表
        max_retries: 每个模型的最大重试次数
    """
    
    def __init__(self, models: List[LLMBase], max_retries: int = 1):
        self.models = models
        self.max_retries = max_retries
    
    def invoke(self, prompt_payload: tuple[str, List[str]]):
        """调用LLM模型生成文本。
        
        参数：
            prompt_payload: 包含提示文本和图片列表的元组
            
        返回：
            生成的文本结果
            
        抛出：
            Exception: 所有模型调用失败时
        """
        prompt, images = prompt_payload
        
        for model in self.models:
            for retry in range(self.max_retries + 1):
                try:
                    if images:
                        return model.generate(prompt, images)
                    else:
                        return model.generate(prompt)
                except Exception as e:
                    logger.warning(f"模型 {model.__class__.__name__} 调用失败 (第{retry+1}次): {e}")
        
        raise Exception("所有LLM模型调用失败")


class EmbeddingsManager:
    """嵌入模型管理器，负责管理多个嵌入模型并实现回退策略。
    
    参数：
        models: 嵌入模型列表
        max_retries: 每个模型的最大重试次数
    """
    
    def __init__(self, models: List[MultimodalEmbeddings], max_retries: int = 1):
        self.models = models
        self.max_retries = max_retries
    
    def invoke(self, documents: List[Dict[str, Any]]):
        """调用嵌入模型生成向量。
        
        参数：
            documents: 文档列表，每个文档包含类型、内容和来源
            
        返回：
            生成的向量列表
            
        抛出：
            Exception: 所有模型调用失败时
        """
        for model in self.models:
            for retry in range(self.max_retries + 1):
                try:
                    return model.get_embeddings(documents)
                except Exception as e:
                    logger.warning(f"模型 {model.__class__.__name__} 调用失败 (第{retry+1}次): {e}")
        
        raise Exception("所有嵌入模型调用失败")


__all__ = [
    "ModelManager",
    "LLMManager",
    "EmbeddingsManager",
    "get_model_manager",
    "init_model_manager",
    "reload_model_manager"
]