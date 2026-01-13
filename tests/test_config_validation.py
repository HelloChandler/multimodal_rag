"""模块：配置验证和JSONPath解析测试.

验证配置验证功能和JSONPath解析机制的正确性。
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from src.config.config_manager import ConfigManager, ConfigError, ModelIntegrationConfig, ModelConfig
from src.model.openai_adapter import OpenAIAdapter


class TestConfigValidation:
    """测试配置验证功能。"""
    
    def test_validate_parameter_mapping(self):
        """测试参数映射验证。"""
        # 创建包含有效参数映射的配置文件
        valid_config = {
            "general": {
                "default_timeout": 30,
                "max_retries": 2,
                "enable_async": True
            },
            "models": {
                "test-model": {
                    "type": "llm",
                    "name": "test-model",
                    "api_key": "test-key",
                    "parameter_mapping": {
                        "temperature": "temp",
                        "max_tokens": "max_length"
                    }
                }
            },
            "fallback": {
                "llm": {
                    "order": ["test-model"],
                    "strategy": "sequential",
                    "retry_on_failure": True,
                    "max_retries_per_model": 1
                }
            },
            "caching": {
                "enabled": False,
                "type": "memory",
                "ttl": 3600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 测试有效参数映射
        manager = ConfigManager("configs/models.yaml")
        with patch.object(manager, "_build_config_object"):
            manager._validate_config(valid_config)  # 不应该抛出异常
    
    def test_validate_invalid_parameter_mapping(self):
        """测试无效的参数映射。"""
        # 创建包含无效参数映射的配置文件
        invalid_config = {
            "general": {
                "default_timeout": 30,
                "max_retries": 2,
                "enable_async": True
            },
            "models": {
                "test-model": {
                    "type": "llm",
                    "name": "test-model",
                    "api_key": "test-key",
                    "parameter_mapping": {
                        "temperature": 0.7,  # 无效：值应该是字符串
                        "max_tokens": "max_length"
                    }
                }
            },
            "fallback": {
                "llm": {
                    "order": ["test-model"],
                    "strategy": "sequential",
                    "retry_on_failure": True,
                    "max_retries_per_model": 1
                }
            },
            "caching": {
                "enabled": False,
                "type": "memory",
                "ttl": 3600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 测试无效参数映射
        manager = ConfigManager("configs/models.yaml")
        with pytest.raises(ConfigError) as excinfo:
            manager._validate_config(invalid_config)
        
        assert "参数映射键值必须是字符串" in str(excinfo.value)
    
    def test_validate_jsonpath_valid(self):
        """测试有效的JSONPath表达式。"""
        # 创建包含有效JSONPath的配置文件
        valid_config = {
            "general": {
                "default_timeout": 30,
                "max_retries": 2,
                "enable_async": True
            },
            "models": {
                "test-model": {
                    "type": "llm",
                    "name": "test-model",
                    "api_key": "test-key",
                    "response_format": {
                        "content": "choices[0].message.content",
                        "model": "model",
                        "usage": "usage.total_tokens"
                    }
                }
            },
            "fallback": {
                "llm": {
                    "order": ["test-model"],
                    "strategy": "sequential",
                    "retry_on_failure": True,
                    "max_retries_per_model": 1
                }
            },
            "caching": {
                "enabled": False,
                "type": "memory",
                "ttl": 3600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 测试有效JSONPath
        manager = ConfigManager("configs/models.yaml")
        with patch.object(manager, "_build_config_object"):
            manager._validate_config(valid_config)  # 不应该抛出异常
    
    def test_validate_jsonpath_invalid(self):
        """测试无效的JSONPath表达式。"""
        # 创建包含无效JSONPath的配置文件
        invalid_config = {
            "general": {
                "default_timeout": 30,
                "max_retries": 2,
                "enable_async": True
            },
            "models": {
                "test-model": {
                    "type": "llm",
                    "name": "test-model",
                    "api_key": "test-key",
                    "response_format": {
                        "content": "choices[0].message.content",
                        "model": "model",
                        "invalid": "$.choices[0].message@content"  # 包含非法字符
                    }
                }
            },
            "fallback": {
                "llm": {
                    "order": ["test-model"],
                    "strategy": "sequential",
                    "retry_on_failure": True,
                    "max_retries_per_model": 1
                }
            },
            "caching": {
                "enabled": False,
                "type": "memory",
                "ttl": 3600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 测试无效JSONPath
        manager = ConfigManager("configs/models.yaml")
        with pytest.raises(ConfigError) as excinfo:
            manager._validate_config(invalid_config)
        
        assert "包含非法字符" in str(excinfo.value)
    
    def test_validate_jsonpath_bracket_mismatch(self):
        """测试JSONPath括号不匹配的情况。"""
        # 创建包含括号不匹配的JSONPath的配置文件
        invalid_config = {
            "general": {
                "default_timeout": 30,
                "max_retries": 2,
                "enable_async": True
            },
            "models": {
                "test-model": {
                    "type": "llm",
                    "name": "test-model",
                    "api_key": "test-key",
                    "response_format": {
                        "content": "choices[0].message.content",
                        "model": "model",
                        "invalid": "$.choices[0].message[content"  # 括号不匹配
                    }
                }
            },
            "fallback": {
                "llm": {
                    "order": ["test-model"],
                    "strategy": "sequential",
                    "retry_on_failure": True,
                    "max_retries_per_model": 1
                }
            },
            "caching": {
                "enabled": False,
                "type": "memory",
                "ttl": 3600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 测试括号不匹配的JSONPath
        manager = ConfigManager("configs/models.yaml")
        with pytest.raises(ConfigError) as excinfo:
            manager._validate_config(invalid_config)
        
        assert "括号不匹配" in str(excinfo.value)
    
    def test_unknown_unified_parameters(self):
        """测试未知的统一参数。"""
        # 创建包含未知统一参数的配置文件
        invalid_config = {
            "general": {
                "default_timeout": 30,
                "max_retries": 2,
                "enable_async": True
            },
            "models": {
                "test-model": {
                    "type": "llm",
                    "name": "test-model",
                    "api_key": "test-key",
                    "parameter_mapping": {
                        "unknown_parameter": "unknown_param",  # 未知统一参数
                        "temperature": "temp"
                    }
                }
            },
            "fallback": {
                "llm": {
                    "order": ["test-model"],
                    "strategy": "sequential",
                    "retry_on_failure": True,
                    "max_retries_per_model": 1
                }
            },
            "caching": {
                "enabled": False,
                "type": "memory",
                "ttl": 3600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 测试未知统一参数
        manager = ConfigManager("configs/models.yaml")
        with patch.object(manager, "_build_config_object"):
            with pytest.raises(ConfigError) as excinfo:
                manager._validate_config(invalid_config)
        
        assert "未知的统一参数" in str(excinfo.value)


class TestJSONPathExtraction:
    """测试JSONPath解析机制。"""
    
    def test_extract_value_complex_jsonpath(self):
        """测试复杂JSONPath表达式的解析。"""
        test_data = {
            "choices": [
                {
                    "message": {"content": "test content"},
                    "metadata": {"usage": {"total_tokens": 15}}
                }
            ],
            "model": "test-model",
            "created": 1234567890
        }
        
        adapter = OpenAIAdapter(ModelConfig(type="llm", name="test-model"))
        
        # 测试简单路径
        assert adapter._extract_value(test_data, "model") == "test-model"
        
        # 测试嵌套路径
        assert adapter._extract_value(test_data, "choices[0].message.content") == "test content"
        
        # 测试多级嵌套路径
        assert adapter._extract_value(test_data, "choices[0].metadata.usage.total_tokens") == 15
    
    def test_extract_value_root_path(self):
        """测试根路径$的解析。"""
        test_data = {
            "choices": [
                {"message": {"content": "test content"}}
            ],
            "model": "test-model"
        }
        
        adapter = OpenAIAdapter(ModelConfig(type="llm", name="test-model"))
        
        # 测试$路径
        assert adapter._extract_value(test_data, "$") == test_data
        
        # 测试$.model路径
        assert adapter._extract_value(test_data, "$.model") == "test-model"
        
        # 测试$.choices[0].message.content路径
        assert adapter._extract_value(test_data, "$.choices[0].message.content") == "test content"
    
    def test_extract_value_escaped_dots(self):
        """测试转义点的解析。"""
        test_data = {
            "model.name": "test-model-123",
            "choices": [
                {"message.content": "test content with escaped dot"}
            ]
        }
        
        adapter = OpenAIAdapter(ModelConfig(type="llm", name="test-model"))
        
        # 测试转义点的解析
        assert adapter._extract_value(test_data, "model\.name") == "test-model-123"
        assert adapter._extract_value(test_data, "choices[0].message\.content") == "test content with escaped dot"
    
    def test_extract_value_multi_level_arrays(self):
        """测试多级数组索引的解析。"""
        test_data = {
            "choices": [
                [
                    {"message": {"content": "inner content"}}
                ]
            ]
        }
        
        adapter = OpenAIAdapter(ModelConfig(type="llm", name="test-model"))
        
        # 测试多级数组索引
        assert adapter._extract_value(test_data, "choices[0][0].message.content") == "inner content"
    
    def test_extract_value_nonexistent_path(self):
        """测试不存在路径的解析。"""
        test_data = {
            "choices": [
                {"message": {"content": "test content"}}
            ],
            "model": "test-model"
        }
        
        adapter = OpenAIAdapter(ModelConfig(type="llm", name="test-model"))
        
        # 测试不存在的路径
        assert adapter._extract_value(test_data, "nonexistent.path") is None
        assert adapter._extract_value(test_data, "choices[1].message.content") is None
        assert adapter._extract_value(test_data, "choices[0].nonexistent.content") is None