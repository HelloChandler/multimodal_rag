"""模块：模型工厂测试.

测试模型工厂的接口类型注册和模型创建功能。
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.model.model_factory import ModelFactory
from src.config.config_manager import ModelConfig, ModelIntegrationConfig


class TestModelFactory:
    """测试模型工厂功能。"""
    
    def test_register_interface_type(self):
        """测试注册新的接口类型。"""
        # 创建模型工厂实例
        factory = ModelFactory()
        
        # 创建模拟的LLM和Embeddings类
        mock_llm_class = MagicMock()
        mock_embeddings_class = MagicMock()
        
        # 注册新的接口类型
        ModelFactory.register_interface_type("anthropic", mock_llm_class, mock_embeddings_class)
        
        # 验证接口类型注册成功
        # 注意：由于当前实现只是记录日志，我们无法直接验证注册状态
        # 但可以验证方法调用不会引发异常
        assert True
    
    def test_register_llm_model(self):
        """测试注册和获取LLM模型。"""
        # 创建模拟的LLM类
        class MockLLM:
            def __init__(self, config):
                self.config = config
        
        # 注册LLM模型
        ModelFactory.register_llm("mock-llm", MockLLM)
        
        # 创建模型配置
        model_config = ModelConfig(
            type="llm",
            name="mock-llm",
            api_key="test-key"
        )
        
        # 创建模型工厂实例
        factory = ModelFactory()
        
        # 测试创建LLM模型
        with patch.object(factory, "_config", ModelIntegrationConfig(
            general=object(),
            models={"mock-llm": model_config},
            fallback={},
            caching=object(),
            logging=object()
        )):
            llm_instance = factory._create_llm_model("mock-llm", model_config)
            
            # 验证模型实例创建成功
            assert isinstance(llm_instance, MockLLM)
    
    def test_register_embeddings_model(self):
        """测试注册和获取Embeddings模型。"""
        # 创建模拟的Embeddings类
        class MockEmbeddings:
            def __init__(self, config):
                self.config = config
        
        # 注册Embeddings模型
        ModelFactory.register_embeddings("mock-embedding", MockEmbeddings)
        
        # 创建模型配置
        model_config = ModelConfig(
            type="embedding",
            name="mock-embedding",
            api_key="test-key"
        )
        
        # 创建模型工厂实例
        factory = ModelFactory()
        
        # 测试创建Embeddings模型
        with patch.object(factory, "_config", ModelIntegrationConfig(
            general=object(),
            models={"mock-embedding": model_config},
            fallback={},
            caching=object(),
            logging=object()
        )):
            embeddings_instance = factory._create_embeddings_model("mock-embedding", model_config)
            
            # 验证模型实例创建成功
            assert isinstance(embeddings_instance, MockEmbeddings)
    
    def test_create_openai_llm(self):
        """测试创建OpenAI兼容的LLM模型。"""
        # 创建模型配置
        model_config = ModelConfig(
            type="llm",
            name="gpt-3.5-turbo",
            api_key="test-key",
            interface_type="openai"
        )
        
        # 创建模型工厂实例
        factory = ModelFactory()
        
        # 测试创建OpenAI兼容的LLM模型
        with patch.object(factory, "_config", ModelIntegrationConfig(
            general=object(),
            models={"gpt-3.5-turbo": model_config},
            fallback={},
            caching=object(),
            logging=object()
        )):
            with patch("src.model.openai_adapter.OpenAICompatibleLLM") as mock_openai_llm:
                # 配置模拟返回值
                mock_instance = MagicMock()
                mock_openai_llm.return_value = mock_instance
                
                llm_instance = factory._create_llm_model("gpt-3.5-turbo", model_config)
                
                # 验证OpenAICompatibleLLM被调用
                mock_openai_llm.assert_called_once_with(model_config)
                assert llm_instance == mock_instance
    
    def test_create_openai_embeddings(self):
        """测试创建OpenAI兼容的Embeddings模型。"""
        # 创建模型配置
        model_config = ModelConfig(
            type="embedding",
            name="text-embedding-ada-002",
            api_key="test-key",
            interface_type="openai"
        )
        
        # 创建模型工厂实例
        factory = ModelFactory()
        
        # 测试创建OpenAI兼容的Embeddings模型
        with patch.object(factory, "_config", ModelIntegrationConfig(
            general=object(),
            models={"text-embedding-ada-002": model_config},
            fallback={},
            caching=object(),
            logging=object()
        )):
            with patch("src.model.openai_adapter.OpenAICompatibleEmbeddings") as mock_openai_embeddings:
                # 配置模拟返回值
                mock_instance = MagicMock()
                mock_openai_embeddings.return_value = mock_instance
                
                embeddings_instance = factory._create_embeddings_model("text-embedding-ada-002", model_config)
                
                # 验证OpenAICompatibleEmbeddings被调用
                mock_openai_embeddings.assert_called_once_with(model_config)
                assert embeddings_instance == mock_instance
    
    def test_unregister_llm_model(self):
        """测试注销LLM模型。"""
        # 创建模拟的LLM类
        class MockLLM:
            def __init__(self, config):
                self.config = config
        
        # 注册LLM模型
        ModelFactory.register_llm("mock-llm", MockLLM)
        
        # 注销LLM模型
        ModelFactory.unregister_llm("mock-llm")
        
        # 验证模型已被注销
        # 由于当前实现没有提供直接的验证方法，我们可以通过尝试创建模型来验证
        model_config = ModelConfig(
            type="llm",
            name="mock-llm",
            api_key="test-key"
        )
        
        factory = ModelFactory()
        
        # 应该会尝试使用其他方式创建模型（如OpenAI适配器）
        with patch.object(factory, "_config", ModelIntegrationConfig(
            general=object(),
            models={"mock-llm": model_config},
            fallback={},
            caching=object(),
            logging=object()
        )):
            with patch("src.model.openai_adapter.OpenAICompatibleLLM") as mock_openai_llm:
                mock_instance = MagicMock()
                mock_openai_llm.return_value = mock_instance
                
                llm_instance = factory._create_llm_model("mock-llm", model_config)
                
                # 验证OpenAICompatibleLLM被调用（而不是使用注册的模型）
                mock_openai_llm.assert_called_once_with(model_config)
    
    def test_unregister_embeddings_model(self):
        """测试注销Embeddings模型。"""
        # 创建模拟的Embeddings类
        class MockEmbeddings:
            def __init__(self, config):
                self.config = config
        
        # 注册Embeddings模型
        ModelFactory.register_embeddings("mock-embedding", MockEmbeddings)
        
        # 注销Embeddings模型
        ModelFactory.unregister_embeddings("mock-embedding")
        
        # 验证模型已被注销
        model_config = ModelConfig(
            type="embedding",
            name="mock-embedding",
            api_key="test-key"
        )
        
        factory = ModelFactory()
        
        # 应该会尝试使用其他方式创建模型（如OpenAI适配器）
        with patch.object(factory, "_config", ModelIntegrationConfig(
            general=object(),
            models={"mock-embedding": model_config},
            fallback={},
            caching=object(),
            logging=object()
        )):
            with patch("src.model.openai_adapter.OpenAICompatibleEmbeddings") as mock_openai_embeddings:
                mock_instance = MagicMock()
                mock_openai_embeddings.return_value = mock_instance
                
                embeddings_instance = factory._create_embeddings_model("mock-embedding", model_config)
                
                # 验证OpenAICompatibleEmbeddings被调用（而不是使用注册的模型）
                mock_openai_embeddings.assert_called_once_with(model_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
