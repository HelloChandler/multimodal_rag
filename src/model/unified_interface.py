"""模块：model_integration_framework

统一模型接口层，定义了所有AI模型必须实现的通用接口。
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Union


logger = logging.getLogger(__name__)


class UnifiedModel(abc.ABC):
    """所有AI模型的统一基类接口。
    
    这个类定义了所有AI模型必须实现的核心方法，确保不同模型之间的一致性。
    """
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型的基本信息。
        
        返回：
            包含模型名称、类型、版本等信息的字典
        """
        ...
    
    @abc.abstractmethod
    def health_check(self) -> bool:
        """检查模型是否健康可用。
        
        返回：
            如果模型健康可用则返回True，否则返回False
        """
        ...
    
    def close(self) -> None:
        """关闭模型资源。
        
        子类可以重写此方法来释放资源。
        """
        pass


class UnifiedLLM(UnifiedModel, abc.ABC):
    """统一的LLM（大语言模型）接口。"""
    
    @abc.abstractmethod
    def generate(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """生成文本响应。
        
        参数：
            prompt: 输入提示文本
            images: 可选的图片base64字符串列表
            kwargs: 其他可选参数
            
        返回：
            生成的文本响应
        """
        ...
    
    @abc.abstractmethod
    async def generate_async(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """异步生成文本响应。
        
        参数：
            prompt: 输入提示文本
            images: 可选的图片base64字符串列表
            kwargs: 其他可选参数
            
        返回：
            生成的文本响应
        """
        ...
    
    @abc.abstractmethod
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """进行多轮对话。
        
        参数：
            messages: 对话历史，每个消息包含role和content
            kwargs: 其他可选参数
            
        返回：
            包含生成的消息和其他信息的字典
        """
        ...
    
    @abc.abstractmethod
    async def chat_async(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """异步进行多轮对话。
        
        参数：
            messages: 对话历史，每个消息包含role和content
            kwargs: 其他可选参数
            
        返回：
            包含生成的消息和其他信息的字典
        """
        ...


class UnifiedEmbeddings(UnifiedModel, abc.ABC):
    """统一的Embeddings（嵌入）接口。"""
    
    @abc.abstractmethod
    def embed_query(self, text: str, **kwargs) -> List[float]:
        """为单个文本查询生成嵌入向量。
        
        参数：
            text: 输入文本查询
            kwargs: 其他可选参数
            
        返回：
            嵌入向量
        """
        ...
    
    @abc.abstractmethod
    async def embed_query_async(self, text: str, **kwargs) -> List[float]:
        """异步为单个文本查询生成嵌入向量。
        
        参数：
            text: 输入文本查询
            kwargs: 其他可选参数
            
        返回：
            嵌入向量
        """
        ...
    
    @abc.abstractmethod
    def embed_documents(self, documents: Union[str, List[str], List[Dict[str, Any]]], **kwargs) -> List[List[float]]:
        """为多个文档生成嵌入向量。
        
        参数：
            documents: 文档列表或单个文档
            kwargs: 其他可选参数
            
        返回：
            嵌入向量列表
        """
        ...
    
    @abc.abstractmethod
    async def embed_documents_async(self, documents: Union[str, List[str], List[Dict[str, Any]]], **kwargs) -> List[List[float]]:
        """异步为多个文档生成嵌入向量。
        
        参数：
            documents: 文档列表或单个文档
            kwargs: 其他可选参数
            
        返回：
            嵌入向量列表
        """
        ...
    
    @abc.abstractmethod
    def get_dimension(self) -> int:
        """获取嵌入向量的维度。
        
        返回：
            嵌入向量的维度
        """
        ...


class ModelResponse:
    """统一的模型响应包装器。
    
    这个类用于包装所有模型的响应，提供一致的访问接口。
    """
    
    def __init__(self, content: Any, model_name: str, model_type: str, 
                 is_success: bool = True, error_message: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """初始化模型响应。
        
        参数：
            content: 模型响应的内容
            model_name: 生成响应的模型名称
            model_type: 模型类型（llm或embedding）
            is_success: 响应是否成功
            error_message: 错误消息（如果失败）
            metadata: 额外的元数据
        """
        self.content = content
        self.model_name = model_name
        self.model_type = model_type
        self.is_success = is_success
        self.error_message = error_message
        self.metadata = metadata or {}
    
    def get_content(self) -> Any:
        """获取响应内容。
        
        返回：
            响应内容
        """
        return self.content
    
    def get_model_info(self) -> Dict[str, str]:
        """获取生成响应的模型信息。
        
        返回：
            包含模型名称和类型的字典
        """
        return {
            "name": self.model_name,
            "type": self.model_type
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取响应的元数据。
        
        返回：
            元数据字典
        """
        return self.metadata
    
    def is_successful(self) -> bool:
        """检查响应是否成功。
        
        返回：
            如果响应成功则返回True，否则返回False
        """
        return self.is_success
    
    def get_error_message(self) -> Optional[str]:
        """获取错误消息（如果失败）。
        
        返回：
            错误消息，如果响应成功则返回None
        """
        return self.error_message


class ModelRequest:
    """统一的模型请求包装器。
    
    这个类用于包装所有模型的请求，提供一致的接口。
    """
    
    def __init__(self, prompt: Optional[str] = None, 
                 images: Optional[List[str]] = None,
                 documents: Optional[Union[str, List[str], List[Dict[str, Any]]]] = None,
                 messages: Optional[List[Dict[str, Any]]] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        """初始化模型请求。
        
        参数：
            prompt: 输入提示文本（用于LLM）
            images: 图片base64字符串列表（用于多模态LLM）
            documents: 文档列表（用于Embeddings）
            messages: 对话历史（用于聊天型LLM）
            parameters: 额外的参数
        """
        self.prompt = prompt
        self.images = images or []
        self.documents = documents
        self.messages = messages or []
        self.parameters = parameters or {}
    
    def get_prompt(self) -> Optional[str]:
        """获取输入提示文本。
        
        返回：
            输入提示文本
        """
        return self.prompt
    
    def get_images(self) -> List[str]:
        """获取图片列表。
        
        返回：
            图片base64字符串列表
        """
        return self.images
    
    def get_documents(self) -> Optional[Union[str, List[str], List[Dict[str, Any]]]]:
        """获取文档列表。
        
        返回：
            文档列表或单个文档
        """
        return self.documents
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """获取对话历史。
        
        返回：
            对话历史列表
        """
        return self.messages
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取额外的参数。
        
        返回：
            额外参数的字典
        """
        return self.parameters
    
    def set_parameter(self, key: str, value: Any) -> None:
        """设置额外参数。
        
        参数：
            key: 参数名称
            value: 参数值
        """
        self.parameters[key] = value


__all__ = [
    "UnifiedModel",
    "UnifiedLLM",
    "UnifiedEmbeddings",
    "ModelResponse",
    "ModelRequest"
]
