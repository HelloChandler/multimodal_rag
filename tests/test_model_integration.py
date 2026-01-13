"""模块：model_integration_framework测试.

验证模型集成框架的核心功能，包括初始化、回退策略和错误处理。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config.config_manager import ModelIntegrationConfig, ModelConfig, FallbackConfig, CacheConfig, GeneralConfig, LoggingConfig
from src.model.model_factory import ModelFactory
from src.model.model_manager import ModelManager
from src.model.unified_interface import ModelRequest, ModelResponse
from src.llm.doubao_llm import DoubaoLLM
from src.llm.deepseek_llm import DeepseekLLM


@pytest.fixture
def mock_model_config() -> ModelIntegrationConfig:
    """创建模拟的模型配置。"""
    # 创建LLM模型配置
    models = {
        "doubao": ModelConfig(
            type="llm",
            name="doubao-seed-1-6-vision-250815",
            api_key="test-api-key",
            api_secret="test-api-secret",
            base_url="https://test.ark.cn/api/v3",
            region="cn-beijing",
            parameters={"temperature": 0.7, "max_tokens": 2000}
        ),
        "deepseek": ModelConfig(
            type="llm",
            name="deepseek-vl-1.5-chat",
            api_key="test-deepseek-key",
            base_url="https://api.deepseek.com",
            parameters={"temperature": 0.7, "max_tokens": 2000}
        )
    }
    
    # 创建回退策略配置
    fallback = {
        "llm": FallbackConfig(
            order=["doubao", "deepseek"],
            strategy="sequential",
            retry_on_failure=True,
            max_retries_per_model=1
        )
    }
    
    # 创建缓存配置
    caching = CacheConfig(
        enabled=True,
        type="memory",
        ttl=3600,
        max_size=1000
    )
    
    # 创建通用配置
    general = GeneralConfig(
        default_timeout=30,
        max_retries=2,
        enable_async=True
    )
    
    # 创建日志配置
    logging = LoggingConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return ModelIntegrationConfig(
        general=general,
        models=models,
        fallback=fallback,
        caching=caching,
        logging=logging
    )


@pytest.fixture
def mock_doubao_llm() -> MagicMock:
    """创建模拟的DoubaoLLM实例。"""
    mock_llm = MagicMock(spec=DoubaoLLM)
    mock_llm.get_model_info.return_value = {"name": "doubao-seed-1-6-vision-250815", "type": "llm"}
    mock_llm.health_check.return_value = True
    mock_llm.generate.return_value = "Doubao响应"
    mock_llm.generate_async.return_value = "Doubao异步响应"
    return mock_llm


@pytest.fixture
def mock_deepseek_llm() -> MagicMock:
    """创建模拟的DeepseekLLM实例。"""
    mock_llm = MagicMock(spec=DeepseekLLM)
    mock_llm.get_model_info.return_value = {"name": "deepseek-vl-1.5-chat", "type": "llm"}
    mock_llm.health_check.return_value = True
    mock_llm.generate.return_value = "Deepseek响应"
    mock_llm.generate_async.return_value = "Deepseek异步响应"
    return mock_llm


class TestModelFactory:
    """测试模型工厂类。"""
    
    def test_init_all_models(self, mock_model_config: ModelIntegrationConfig, 
                            mock_doubao_llm: MagicMock, mock_deepseek_llm: MagicMock):
        """测试初始化所有模型。"""
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            with patch.object(ModelFactory, "_create_deepseek_llm", return_value=mock_deepseek_llm):
                factory = ModelFactory()
                
                # 替换工厂的配置
                factory._config = mock_model_config
                
                # 初始化所有模型
                factory.init_all_models()
                
                # 验证模型实例已创建
                assert len(factory._llm_instances) == 2
                assert "doubao" in factory._llm_instances
                assert "deepseek" in factory._llm_instances
    
    def test_get_llm_model(self, mock_model_config: ModelIntegrationConfig, mock_doubao_llm: MagicMock):
        """测试获取LLM模型实例。"""
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            factory = ModelFactory()
            factory._config = mock_model_config
            
            # 获取模型实例
            llm_model = factory.get_llm_model("doubao")
            
            # 验证模型实例
            assert llm_model is mock_doubao_llm
            assert factory._llm_instances["doubao"] is llm_model
    
    def test_get_nonexistent_model(self, mock_model_config: ModelIntegrationConfig):
        """测试获取不存在的模型。"""
        factory = ModelFactory()
        factory._config = mock_model_config
        
        # 验证获取不存在的模型会抛出ValueError
        with pytest.raises(ValueError) as excinfo:
            factory.get_llm_model("nonexistent")
        
        assert "模型 nonexistent 未在配置中定义" in str(excinfo.value)


class TestModelManager:
    """测试模型管理器类。"""
    
    def test_initialize(self, mock_model_config: ModelIntegrationConfig, 
                        mock_doubao_llm: MagicMock, mock_deepseek_llm: MagicMock):
        """测试模型管理器初始化。"""
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            with patch.object(ModelFactory, "_create_deepseek_llm", return_value=mock_deepseek_llm):
                with patch.object(ModelFactory, "_config", mock_model_config):
                    manager = ModelManager()
                    
                    # 验证管理器已初始化
                    assert manager is not None
    
    def test_invoke_llm_model(self, mock_model_config: ModelIntegrationConfig, 
                            mock_doubao_llm: MagicMock, mock_deepseek_llm: MagicMock):
        """测试调用LLM模型。"""
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            with patch.object(ModelFactory, "_create_deepseek_llm", return_value=mock_deepseek_llm):
                manager = ModelManager()
                manager._config = mock_model_config
                
                # 创建请求
                request = ModelRequest(prompt="Hello, world!", parameters={"temperature": 0.7})
                
                # 调用模型
                response = manager.invoke_llm_model(request)
                
                # 验证响应
                assert response.is_success
                assert response.content == "Doubao响应"
                assert response.model_name == "doubao"
                assert response.model_type == "llm"
    
    def test_fallback_strategy(self, mock_model_config: ModelIntegrationConfig, 
                            mock_doubao_llm: MagicMock, mock_deepseek_llm: MagicMock):
        """测试回退策略。"""
        # 模拟doubao模型失败
        mock_doubao_llm.generate.side_effect = Exception("Doubao API failure")
        
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            with patch.object(ModelFactory, "_create_deepseek_llm", return_value=mock_deepseek_llm):
                manager = ModelManager()
                manager._config = mock_model_config
                
                # 创建请求
                request = ModelRequest(prompt="Hello, world!", parameters={"temperature": 0.7})
                
                # 调用模型
                response = manager.invoke_llm_model(request)
                
                # 验证回退策略生效
                assert response.is_success
                assert response.content == "Deepseek响应"
                assert response.model_name == "deepseek"
                assert response.model_type == "llm"
    
    def test_all_models_failure(self, mock_model_config: ModelIntegrationConfig, 
                            mock_doubao_llm: MagicMock, mock_deepseek_llm: MagicMock):
        """测试所有模型都失败的情况。"""
        # 模拟所有模型失败
        mock_doubao_llm.generate.side_effect = Exception("Doubao API failure")
        mock_deepseek_llm.generate.side_effect = Exception("Deepseek API failure")
        
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            with patch.object(ModelFactory, "_create_deepseek_llm", return_value=mock_deepseek_llm):
                manager = ModelManager()
                manager._config = mock_model_config
                
                # 创建请求
                request = ModelRequest(prompt="Hello, world!", parameters={"temperature": 0.7})
                
                # 调用模型
                response = manager.invoke_llm_model(request)
                
                # 验证响应
                assert not response.is_success
                assert "所有LLM模型调用失败" in response.error_message
                assert response.content is None
    
    @pytest.mark.asyncio
    async def test_async_invoke_llm_model(self, mock_model_config: ModelIntegrationConfig, 
                                        mock_doubao_llm: MagicMock, mock_deepseek_llm: MagicMock):
        """测试异步调用LLM模型。"""
        with patch.object(ModelFactory, "_create_doubao_llm", return_value=mock_doubao_llm):
            with patch.object(ModelFactory, "_create_deepseek_llm", return_value=mock_deepseek_llm):
                manager = ModelManager()
                manager._config = mock_model_config
                
                # 创建请求
                request = ModelRequest(prompt="Hello, world!", parameters={"temperature": 0.7})
                
                # 调用模型
                response = await manager.invoke_llm_model_async(request)
                
                # 验证响应
                assert response.is_success
                assert response.content == "Doubao异步响应"
                assert response.model_name == "doubao"
                assert response.model_type == "llm"
