"""模块：model_integration_framework

OpenAI接口适配层，用于处理不同模型间的参数差异和响应格式差异。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI

from src.config.config_manager import ModelConfig
from src.model.unified_interface import UnifiedLLM, UnifiedEmbeddings, ModelResponse

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """OpenAI接口适配层，用于将统一接口映射到OpenAI兼容接口。"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = self._init_client()
        self.async_client = self._init_async_client()
    
    def _init_client(self) -> openai.OpenAI:
        """初始化OpenAI客户端。"""
        client_kwargs = {
            "api_key": self.model_config.api_key,
        }
        
        if self.model_config.base_url:
            client_kwargs["base_url"] = self.model_config.base_url
        
        return openai.OpenAI(**client_kwargs)
    
    def _init_async_client(self) -> AsyncOpenAI:
        """初始化异步OpenAI客户端。"""
        client_kwargs = {
            "api_key": self.model_config.api_key,
        }
        
        if self.model_config.base_url:
            client_kwargs["base_url"] = self.model_config.base_url
        
        return AsyncOpenAI(**client_kwargs)
    
    def map_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """将统一参数映射到模型特定参数。"""
        mapped_params = {}
        
        # 如果有自定义参数映射，则使用映射关系
        if self.model_config.parameter_mapping:
            for unified_param, model_param in self.model_config.parameter_mapping.items():
                if unified_param in params:
                    mapped_params[model_param] = params[unified_param]
        else:
            # 默认情况下，直接使用统一参数名称
            mapped_params = params.copy()
        
        # 添加模型名称到参数中
        if "model" not in mapped_params:
            mapped_params["model"] = self.model_config.name
        
        # 添加默认参数（如果配置中有的话）
        if hasattr(self.model_config, 'parameters') and self.model_config.parameters:
            for param_name, param_value in self.model_config.parameters.items():
                if param_name not in mapped_params:
                    mapped_params[param_name] = param_value
        
        return mapped_params
    
    def parse_response(self, response: Dict[str, Any], model_type: str) -> ModelResponse:
        """将模型响应转换为统一格式。"""
        try:
            if self.model_config.response_format:
                # 使用响应格式映射解析响应
                content = self._extract_value(response, self.model_config.response_format.get("content", "choices[0].message.content"))
                model_name = self._extract_value(response, self.model_config.response_format.get("model", "model"))
                
                # 提取额外的元数据（如果配置了的话）
                metadata = {"original_response": response}
                for key, path in self.model_config.response_format.items():
                    if key not in ["content", "model"]:
                        value = self._extract_value(response, path)
                        if value is not None:
                            metadata[key] = value
            else:
                # 默认解析方式
                if model_type == "llm":
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    # 提取额外的元数据
                    metadata = {
                        "original_response": response,
                        "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                        "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                        "total_tokens": response.get("usage", {}).get("total_tokens")
                    }
                elif model_type == "embedding":
                    content = response.get("data", [{}])[0].get("embedding", [])
                    # 提取额外的元数据
                    metadata = {
                        "original_response": response,
                        "model": response.get("model"),
                        "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                        "total_tokens": response.get("usage", {}).get("total_tokens")
                    }
                else:
                    content = response
                    metadata = {"original_response": response}
                
                model_name = response.get("model", self.model_config.name)
            
            return ModelResponse(
                content=content,
                model_name=model_name,
                model_type=model_type,
                is_success=True,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"解析模型响应失败: {e}")
            return ModelResponse(
                content=None,
                model_name=self.model_config.name,
                model_type=model_type,
                is_success=False,
                error_message=str(e),
                metadata={"original_response": response}
            )
    
    def _extract_value(self, data: Any, path: str) -> Any:
        """从嵌套字典中提取指定路径的值，支持复杂的JSONPath表达式。"""
        import re
        
        # 移除首尾空格
        path = path.strip()
        if not path:
            return data
        
        # 处理根路径
        if path == "$" or path == "$":
            return data
        
        # 移除根路径前缀
        if path.startswith("$.") or path.startswith("$"):
            path = path[2:] if path.startswith("$.") else path[1:]
        
        # 分割路径为段
        segments = re.split(r"(?<!\\)\.", path)  # 用未转义的点分割
        result = data
        
        for segment in segments:
            # 处理转义的点
            segment = segment.replace("\\.", ".")
            
            # 处理数组索引
            if "[" in segment and "]" in segment:
                # 提取数组键和索引部分
                arr_key = segment.split("[")[0]
                indices_part = segment[segment.find("["):]
                
                # 处理普通键部分
                if arr_key:
                    if isinstance(result, dict) and arr_key in result:
                        result = result[arr_key]
                    else:
                        return None
                
                # 处理数组索引部分
                if isinstance(result, list):
                    # 提取所有索引表达式
                    indices = re.findall(r"\[(\d+)\]", indices_part)
                    for idx_str in indices:
                        try:
                            idx = int(idx_str)
                            if idx < len(result):
                                result = result[idx]
                            else:
                                return None
                        except (ValueError, IndexError):
                            return None
                else:
                    return None
            else:
                # 处理普通键
                if isinstance(result, dict) and segment in result:
                    result = result[segment]
                else:
                    return None
        
        return result


class OpenAICompatibleLLM(UnifiedLLM):
    """OpenAI兼容的LLM实现。"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.adapter = OpenAIAdapter(model_config)
        self.default_params = model_config.parameters.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_config.name,
            "type": self.model_config.type,
            "interface_type": self.model_config.interface_type,
            "base_url": self.model_config.base_url
        }
    
    def health_check(self) -> bool:
        try:
            # 发送一个简单的测试请求来检查模型是否可用
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            params = {
                "messages": test_messages,
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = self.adapter.client.chat.completions.create(**params)
            
            # 验证响应格式是否符合OpenAI规范
            if not hasattr(response, 'choices') or response.choices is None:
                return False
            
            if len(response.choices) == 0:
                return False
            
            if not hasattr(response.choices[0], 'message') or response.choices[0].message is None:
                return False
            
            if not hasattr(response.choices[0].message, 'content'):
                return False
            
            # 验证内容是否为非空字符串
            content = response.choices[0].message.content
            if not isinstance(content, str) or len(content.strip()) == 0:
                return False
            
            # 验证基本的元数据字段
            if hasattr(response, 'model'):
                if not isinstance(response.model, str) or len(response.model.strip()) == 0:
                    return False
            
            if hasattr(response, 'created'):
                if not isinstance(response.created, int):
                    return False
            
            # 验证finish_reason字段
            if hasattr(response.choices[0], 'finish_reason'):
                finish_reason = response.choices[0].finish_reason
                if finish_reason not in [None, 'stop', 'length', 'content_filter', 'tool_calls', 'function_call']:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"模型健康检查失败: {e}")
            return False
    
    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 处理图片输入
        if images:
            content = [
                {"type": "text", "text": prompt}
            ]
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                })
            messages = [{
                "role": "user",
                "content": content
            }]
        
        params = self.default_params.copy()
        params.update({"messages": messages})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = self.adapter.client.chat.completions.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "llm")
        
        if model_response.is_successful():
            return model_response.get_content()
        else:
            raise Exception(model_response.get_error_message())
    
    async def generate_async(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 处理图片输入
        if images:
            content = [
                {"type": "text", "text": prompt}
            ]
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                })
            messages = [{
                "role": "user",
                "content": content
            }]
        
        params = self.default_params.copy()
        params.update({"messages": messages})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = await self.adapter.async_client.chat.completions.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "llm")
        
        if model_response.is_successful():
            return model_response.get_content()
        else:
            raise Exception(model_response.get_error_message())
    
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update({"messages": messages})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = self.adapter.client.chat.completions.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "llm")
        
        if model_response.is_successful():
            return {
                "content": model_response.get_content(),
                "model": model_response.model_name,
                "metadata": model_response.get_metadata()
            }
        else:
            raise Exception(model_response.get_error_message())
    
    async def chat_async(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update({"messages": messages})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = await self.adapter.async_client.chat.completions.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "llm")
        
        if model_response.is_successful():
            return {
                "content": model_response.get_content(),
                "model": model_response.model_name,
                "metadata": model_response.get_metadata()
            }
        else:
            raise Exception(model_response.get_error_message())


class OpenAICompatibleEmbeddings(UnifiedEmbeddings):
    """OpenAI兼容的Embeddings实现。"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.adapter = OpenAIAdapter(model_config)
        self.default_params = model_config.parameters.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_config.name,
            "type": self.model_config.type,
            "interface_type": self.model_config.interface_type,
            "base_url": self.model_config.base_url
        }
    
    def health_check(self) -> bool:
        try:
            # 发送一个简单的测试请求来检查模型是否可用
            test_text = "Hello, world!"
            
            params = {
                "input": test_text,
            }
            
            response = self.adapter.client.embeddings.create(**params)
            
            # 验证响应格式是否符合OpenAI规范
            if not hasattr(response, 'data') or response.data is None:
                return False
            
            if len(response.data) == 0:
                return False
            
            if not hasattr(response.data[0], 'embedding'):
                return False
            
            # 验证嵌入向量是否为非空列表
            embedding = response.data[0].embedding
            if not isinstance(embedding, list) or len(embedding) == 0:
                return False
            
            # 验证嵌入向量元素是否为浮点数
            if not all(isinstance(val, (int, float)) for val in embedding):
                return False
            
            return True
        except Exception as e:
            logger.error(f"模型健康检查失败: {e}")
            return False
    
    def embed_query(self, text: str, **kwargs) -> List[float]:
        params = self.default_params.copy()
        params.update({"input": text})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = self.adapter.client.embeddings.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "embedding")
        
        if model_response.is_successful():
            return model_response.get_content()
        else:
            raise Exception(model_response.get_error_message())
    
    async def embed_query_async(self, text: str, **kwargs) -> List[float]:
        params = self.default_params.copy()
        params.update({"input": text})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = await self.adapter.async_client.embeddings.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "embedding")
        
        if model_response.is_successful():
            return model_response.get_content()
        else:
            raise Exception(model_response.get_error_message())
    
    def embed_documents(self, documents: Union[str, List[str], List[Dict[str, Any]]], **kwargs) -> List[List[float]]:
        # 确保documents是列表
        if isinstance(documents, str):
            documents = [documents]
        elif isinstance(documents, dict):
            documents = [documents]
        
        # 提取文本内容
        texts = []
        for doc in documents:
            if isinstance(doc, dict):
                texts.append(doc.get("text", ""))
            else:
                texts.append(doc)
        
        params = self.default_params.copy()
        params.update({"input": texts})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = self.adapter.client.embeddings.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "embedding")
        
        if model_response.is_successful():
            return [item["embedding"] for item in response.data]
        else:
            raise Exception(model_response.get_error_message())
    
    async def embed_documents_async(self, documents: Union[str, List[str], List[Dict[str, Any]]], **kwargs) -> List[List[float]]:
        # 确保documents是列表
        if isinstance(documents, str):
            documents = [documents]
        elif isinstance(documents, dict):
            documents = [documents]
        
        # 提取文本内容
        texts = []
        for doc in documents:
            if isinstance(doc, dict):
                texts.append(doc.get("text", ""))
            else:
                texts.append(doc)
        
        params = self.default_params.copy()
        params.update({"input": texts})
        params.update(kwargs)
        
        # 映射参数
        mapped_params = self.adapter.map_parameters(params)
        
        # 调用API
        response = await self.adapter.async_client.embeddings.create(**mapped_params)
        
        # 解析响应
        model_response = self.adapter.parse_response(response.model_dump(), "embedding")
        
        if model_response.is_successful():
            return [item["embedding"] for item in response.data]
        else:
            raise Exception(model_response.get_error_message())
    
    def get_dimension(self) -> int:
        # 发送一个简单的请求来获取嵌入维度
        try:
            response = self.embed_query("test")
            return len(response)
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {e}")
            return 0
