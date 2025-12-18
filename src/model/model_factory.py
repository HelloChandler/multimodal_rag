"""模块：model_integration_framework

模型工厂模块，负责根据配置创建和管理各种AI模型实例。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from src.config.config_manager import ModelConfig, get_model_integration_config
from src.llm.deepseek_llm import DeepseekLLM
from src.llm.doubao_llm import DoubaoLLM, LLMBase
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings
from src.model.openai_adapter import OpenAICompatibleLLM, OpenAICompatibleEmbeddings
from src.model.unified_interface import UnifiedLLM, UnifiedEmbeddings

# 定义泛型类型变量
T_LLM = TypeVar('T_LLM', bound=LLMBase)
T_Embeddings = TypeVar('T_Embeddings', bound=MultimodalEmbeddings)


logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类，负责创建和管理模型实例。"""
    
    # 模型类型注册表
    _llm_registry: Dict[str, Type[LLMBase]] = {}
    _embeddings_registry: Dict[str, Type[MultimodalEmbeddings]] = {}
    
    # 接口类型注册表
    _interface_type_registry: Dict[str, Dict[str, Optional[Type]]] = {}
    _interface_type_registry['openai'] = {'llm': OpenAICompatibleLLM, 'embeddings': OpenAICompatibleEmbeddings}
    _interface_type_registry['doubao'] = {'llm': DoubaoLLM, 'embeddings': MultimodalEmbeddings}
    _interface_type_registry['deepseek'] = {'llm': DeepseekLLM, 'embeddings': None}
    
    def __init__(self):
        self._llm_instances: Dict[str, LLMBase] = {}
        self._embeddings_instances: Dict[str, MultimodalEmbeddings] = {}
        self._config = get_model_integration_config()
    
    def init_all_models(self) -> None:
        """初始化所有配置的模型。"""
        logger.info("开始初始化所有配置的模型...")
        
        # 初始化LLM模型
        for model_name, model_config in self._config.models.items():
            if model_config.type == 'llm':
                try:
                    model_instance = self.get_llm_model(model_name)
                    
                    # 执行模型类型自动检测
                    if model_config.interface_type == 'openai':
                        from src.model.unified_interface import UnifiedLLM
                        if isinstance(model_instance, UnifiedLLM) and hasattr(model_instance, 'health_check'):
                            if not model_instance.health_check():
                                logger.warning(f"警告: 模型 {model_name} 配置为OpenAI接口，但健康检查失败，可能不兼容OpenAI规范")
                                logger.warning(f"建议: 请检查模型配置或考虑通过继承UnifiedLLM类实现自定义接入逻辑")
                    
                    logger.info(f"成功初始化LLM模型: {model_name}")
                except Exception as e:
                    logger.error(f"初始化LLM模型失败: {model_name}, 错误: {e}")
                    logger.error(f"建议: 请检查模型配置或考虑通过继承UnifiedLLM类实现自定义接入逻辑")
            elif model_config.type == 'embedding':
                try:
                    model_instance = self.get_embeddings_model(model_name)
                    
                    # 执行模型类型自动检测
                    if model_config.interface_type == 'openai':
                        from src.model.unified_interface import UnifiedEmbeddings
                        if isinstance(model_instance, UnifiedEmbeddings) and hasattr(model_instance, 'health_check'):
                            if not model_instance.health_check():
                                logger.warning(f"警告: 嵌入模型 {model_name} 配置为OpenAI接口，但健康检查失败，可能不兼容OpenAI规范")
                                logger.warning(f"建议: 请检查模型配置或考虑通过继承UnifiedEmbeddings类实现自定义接入逻辑")
                    
                    logger.info(f"成功初始化Embedding模型: {model_name}")
                except Exception as e:
                    logger.error(f"初始化Embedding模型失败: {model_name}, 错误: {e}")
                    logger.error(f"建议: 请检查模型配置或考虑通过继承UnifiedEmbeddings类实现自定义接入逻辑")
    
    def get_llm_model(self, model_name: str) -> LLMBase:
        """获取LLM模型实例。
        
        参数：
            model_name: 模型名称
            
        返回：
            LLM模型实例
            
        抛出：
            ValueError: 模型不存在时
            RuntimeError: 模型创建失败时
        """
        if model_name in self._llm_instances:
            return self._llm_instances[model_name]
        
        # 获取模型配置
        model_config = self._config.models.get(model_name)
        if not model_config:
            raise ValueError(f"模型 {model_name} 未在配置中定义")
        
        if model_config.type != 'llm':
            raise ValueError(f"模型 {model_name} 不是LLM类型")
        
        # 创建模型实例
        try:
            llm_instance = self._create_llm_model(model_name, model_config)
            self._llm_instances[model_name] = llm_instance
            return llm_instance
        except Exception as e:
            raise RuntimeError(f"创建LLM模型实例失败: {model_name}, 错误: {e}") from e
    
    def get_embeddings_model(self, model_name: str) -> MultimodalEmbeddings:
        """获取Embeddings模型实例。
        
        参数：
            model_name: 模型名称
            
        返回：
            Embeddings模型实例
            
        抛出：
            ValueError: 模型不存在时
            RuntimeError: 模型创建失败时
        """
        if model_name in self._embeddings_instances:
            return self._embeddings_instances[model_name]
        
        # 获取模型配置
        model_config = self._config.models.get(model_name)
        if not model_config:
            raise ValueError(f"模型 {model_name} 未在配置中定义")
        
        if model_config.type != 'embedding':
            raise ValueError(f"模型 {model_name} 不是Embedding类型")
        
        # 创建模型实例
        try:
            embeddings_instance = self._create_embeddings_model(model_name, model_config)
            self._embeddings_instances[model_name] = embeddings_instance
            return embeddings_instance
        except Exception as e:
            raise RuntimeError(f"创建Embeddings模型实例失败: {model_name}, 错误: {e}") from e
    
    def get_all_llm_models(self) -> Dict[str, LLMBase]:
        """获取所有LLM模型实例。
        
        返回：
            所有LLM模型实例的字典
        """
        # 确保所有LLM模型都已初始化
        for model_name, model_config in self._config.models.items():
            if model_config.type == 'llm' and model_name not in self._llm_instances:
                try:
                    self.get_llm_model(model_name)
                except Exception as e:
                    logger.error(f"获取LLM模型失败: {model_name}, 错误: {e}")
        
        return self._llm_instances.copy()
    
    def get_all_embeddings_models(self) -> Dict[str, MultimodalEmbeddings]:
        """获取所有Embeddings模型实例。
        
        返回：
            所有Embeddings模型实例的字典
        """
        # 确保所有Embedding模型都已初始化
        for model_name, model_config in self._config.models.items():
            if model_config.type == 'embedding' and model_name not in self._embeddings_instances:
                try:
                    self.get_embeddings_model(model_name)
                except Exception as e:
                    logger.error(f"获取Embedding模型失败: {model_name}, 错误: {e}")
        
        return self._embeddings_instances.copy()
    
    def reload_models(self) -> None:
        """重新加载所有模型实例。"""
        logger.info("开始重新加载所有模型...")
        
        # 清空现有实例
        self._llm_instances.clear()
        self._embeddings_instances.clear()
        
        # 重新初始化所有模型
        self.init_all_models()
    
    def _create_llm_model(self, model_name: str, model_config: ModelConfig) -> LLMBase:
        """创建LLM模型实例。
        
        参数：
            model_name: 模型名称
            model_config: 模型配置
            
        返回：
            LLM模型实例
            
        抛出：
            ValueError: 不支持的模型类型
        """
        # 首先检查注册表中是否有对应的模型类型
        if model_config.name in self._llm_registry:
            return self._llm_registry[model_config.name](model_config)
        
        # 检查接口类型注册表
        interface_type = model_config.interface_type
        if interface_type in self._interface_type_registry:
            interface_registry = self._interface_type_registry[interface_type]
            if interface_registry['llm']:
                if interface_type == 'doubao' or interface_type == 'deepseek':
                    # 对于doubao和deepseek，需要使用特殊的创建方法
                    return getattr(self, f"_create_{interface_type}_llm")(model_config)
                else:
                    return interface_registry['llm'](model_config)
            else:
                logger.warning(f"接口类型 {interface_type} 没有注册LLM类")
        else:
            logger.warning(f"未找到接口类型 {interface_type} 的注册")
        
        # 尝试使用OpenAI适配器作为默认选项
        try:
            from src.model.openai_adapter import OpenAICompatibleLLM
            model = OpenAICompatibleLLM(model_config)
            # 进行健康检查，验证是否兼容OpenAI接口
            if model.health_check():
                return model
            else:
                logger.warning(f"模型 {model_name} 声明为OpenAI接口，但健康检查失败，可能需要自定义实现")
                raise ValueError(f"模型 {model_name} 不兼容OpenAI接口规范")
        except Exception as e:
            logger.error(f"使用OpenAI适配器创建模型 {model_name} 失败: {e}")
            raise ValueError(f"不支持的LLM模型: {model_name} ({model_config.name})，且不兼容OpenAI接口规范") from e
    
    def _create_embeddings_model(self, model_name: str, model_config: ModelConfig) -> MultimodalEmbeddings:
        """创建Embeddings模型实例。
        
        参数：
            model_name: 模型名称
            model_config: 模型配置
            
        返回：
            Embeddings模型实例
            
        抛出：
            ValueError: 不支持的模型类型
        """
        # 首先检查注册表中是否有对应的模型类型
        if model_config.name in self._embeddings_registry:
            return self._embeddings_registry[model_config.name](model_config)
        
        # 检查接口类型注册表
        interface_type = model_config.interface_type
        if interface_type in self._interface_type_registry:
            interface_registry = self._interface_type_registry[interface_type]
            if interface_registry['embeddings']:
                if interface_type == 'doubao':
                    # 对于doubao，需要使用特殊的创建方法
                    return getattr(self, f"_create_{interface_type}_embeddings")(model_config)
                else:
                    return interface_registry['embeddings'](model_config)
            else:
                logger.warning(f"接口类型 {interface_type} 没有注册Embeddings类")
        else:
            logger.warning(f"未找到接口类型 {interface_type} 的注册")
        
        # 尝试使用OpenAI适配器作为默认选项
        try:
            from src.model.openai_adapter import OpenAICompatibleEmbeddings
            model = OpenAICompatibleEmbeddings(model_config)
            # 进行健康检查，验证是否兼容OpenAI接口
            if model.health_check():
                return model
            else:
                logger.warning(f"Embedding模型 {model_name} 声明为OpenAI接口，但健康检查失败，可能需要自定义实现")
                raise ValueError(f"Embedding模型 {model_name} 不兼容OpenAI接口规范")
        except Exception as e:
            logger.error(f"使用OpenAI适配器创建Embedding模型 {model_name} 失败: {e}")
            raise ValueError(f"不支持的Embedding模型: {model_name} ({model_config.name})，且不兼容OpenAI接口规范") from e
    
    def _create_doubao_llm(self, model_config: ModelConfig) -> DoubaoLLM:
        """创建Doubao LLM实例。
        
        参数：
            model_config: 模型配置
            
        返回：
            DoubaoLLM实例
        """
        # 从配置创建DoubaoLLM需要的设置
        from src.config.settings import Settings, ModelSettings
        
        model_settings = ModelSettings(
            ark_api_key=model_config.api_key or '',
            ark_api_secret=model_config.api_secret or '',
            ark_region=model_config.region or 'cn-beijing',
            deepseek_api_key='',  # 不需要DeepSeek API密钥
            llm_model=model_config.name,
            embed_model=''
        )
        
        settings = Settings(
            paths=object(),  # 只需要模型配置
            models=model_settings
        )
        
        return DoubaoLLM(settings)
    
    def _create_deepseek_llm(self, model_config: ModelConfig) -> DeepseekLLM:
        """创建Deepseek LLM实例。
        
        参数：
            model_config: 模型配置
            
        返回：
            DeepseekLLM实例
        """
        # 从配置创建DeepseekLLM需要的设置
        from src.config.settings import Settings, ModelSettings
        
        model_settings = ModelSettings(
            ark_api_key='',  # 不需要Ark API密钥
            ark_api_secret='',
            ark_region='',
            deepseek_api_key=model_config.api_key or '',
            llm_model='',
            embed_model=''
        )
        
        settings = Settings(
            paths=object(),  # 只需要模型配置
            models=model_settings
        )
        
        return DeepseekLLM(settings)
    
    def _create_doubao_embeddings(self, model_config: ModelConfig) -> MultimodalEmbeddings:
        """创建Doubao Embeddings实例。
        
        参数：
            model_config: 模型配置
            
        返回：
            MultimodalEmbeddings实例
        """
        # 从配置创建MultimodalEmbeddings需要的设置
        from src.config.settings import Settings, ModelSettings
        
        model_settings = ModelSettings(
            ark_api_key=model_config.api_key or '',
            ark_api_secret=model_config.api_secret or '',
            ark_region=model_config.region or 'cn-beijing',
            deepseek_api_key='',  # 不需要DeepSeek API密钥
            llm_model='',
            embed_model=model_config.name
        )
        
        settings = Settings(
            paths=object(),  # 只需要模型配置
            models=model_settings
        )
        
        return MultimodalEmbeddings(settings)
    
    @classmethod
    def register_llm(cls, name: str, model_class: Type[LLMBase]) -> None:
        """注册LLM模型类型。
        
        参数：
            name: 模型名称
            model_class: 模型类
        """
        cls._llm_registry[name] = model_class
        logger.info(f"已注册LLM模型类型: {name}")
    
    @classmethod
    def register_embeddings(cls, name: str, model_class: Type[MultimodalEmbeddings]) -> None:
        """注册Embeddings模型类型。
        
        参数：
            name: 模型名称
            model_class: 模型类
        """
        cls._embeddings_registry[name] = model_class
        logger.info(f"已注册Embedding模型类型: {name}")
    
    @classmethod
    def register_interface_type(cls, interface_type: str, llm_class: Optional[Type[LLMBase]] = None, embeddings_class: Optional[Type[MultimodalEmbeddings]] = None) -> None:
        """注册新的接口类型支持。
        
        参数：
            interface_type: 接口类型名称（如：anthropic、google等）
            llm_class: LLM模型类（可选）
            embeddings_class: Embeddings模型类（可选）
        """
        # 如果接口类型已存在，更新它
        if interface_type in cls._interface_type_registry:
            current_registry = cls._interface_type_registry[interface_type]
            if llm_class:
                current_registry['llm'] = llm_class
            if embeddings_class:
                current_registry['embeddings'] = embeddings_class
        else:
            # 创建新的接口类型注册
            cls._interface_type_registry[interface_type] = {
                'llm': llm_class,
                'embeddings': embeddings_class
            }
        
        logger.info(f"已注册接口类型支持: {interface_type}")
        if llm_class:
            logger.info(f"  - LLM模型类: {llm_class.__name__}")
        if embeddings_class:
            logger.info(f"  - Embeddings模型类: {embeddings_class.__name__}")
    
    @classmethod
    def unregister_llm(cls, name: str) -> None:
        """注销LLM模型类型。
        
        参数：
            name: 模型名称
        """
        if name in cls._llm_registry:
            del cls._llm_registry[name]
            logger.info(f"已注销LLM模型类型: {name}")
    
    @classmethod
    def unregister_embeddings(cls, name: str) -> None:
        """注销Embeddings模型类型。
        
        参数：
            name: 模型名称
        """
        if name in cls._embeddings_registry:
            del cls._embeddings_registry[name]
            logger.info(f"已注销Embedding模型类型: {name}")


# 创建全局模型工厂实例
_model_factory: Optional[ModelFactory] = None


def get_model_factory() -> ModelFactory:
    """获取全局模型工厂实例。
    
    返回：
        全局模型工厂实例
    """
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory


def init_model_factory() -> None:
    """初始化全局模型工厂并创建所有配置的模型实例。"""
    factory = get_model_factory()
    factory.init_all_models()


def reload_model_factory() -> None:
    """重新加载全局模型工厂的配置和模型实例。"""
    global _model_factory
    _model_factory = None
    init_model_factory()


__all__ = [
    "ModelFactory",
    "get_model_factory",
    "init_model_factory",
    "reload_model_factory"
]
