"""模块：model_integration_framework

配置管理模块，负责加载、解析和验证模型集成框架的配置文件。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """单个模型的配置信息。"""
    type: str  # 模型类型：llm或embedding
    name: str  # 模型名称
    api_key: Optional[str] = None  # API密钥
    api_secret: Optional[str] = None  # API密钥对
    base_url: Optional[str] = None  # API基础URL
    region: Optional[str] = None  # 服务区域
    interface_type: str = "openai"  # 接口类型：openai（默认）、anthropic、google等
    parameter_mapping: Dict[str, str] = field(default_factory=dict)  # 参数映射关系
    response_format: Dict[str, str] = field(default_factory=dict)  # 响应格式映射
    parameters: Dict[str, Any] = field(default_factory=dict)  # 可选参数


@dataclass(frozen=True)
class FallbackConfig:
    """回退策略配置。"""
    order: List[str]  # 模型调用顺序
    strategy: str  # 回退策略
    retry_on_failure: bool  # 失败时是否重试
    max_retries_per_model: int  # 每个模型的最大重试次数


@dataclass(frozen=True)
class CacheConfig:
    """缓存配置。"""
    enabled: bool  # 是否启用缓存
    type: str  # 缓存类型
    ttl: int  # 缓存过期时间（秒）
    max_size: int  # 最大缓存条目数
    redis: Dict[str, Any] = field(default_factory=dict)  # Redis配置


@dataclass(frozen=True)
class GeneralConfig:
    """通用配置。"""
    default_timeout: int  # 默认超时时间（秒）
    max_retries: int  # 默认最大重试次数
    enable_async: bool  # 是否启用异步调用


@dataclass(frozen=True)
class LoggingConfig:
    """日志配置。"""
    level: str  # 日志级别
    format: str  # 日志格式
    file: Optional[str] = None  # 日志文件路径
    rotate: bool = False  # 是否启用日志轮转
    max_bytes: int = 10485760  # 单个日志文件最大大小（10MB）
    backup_count: int = 5  # 保留的日志文件数量

@dataclass(frozen=True)
class ModelIntegrationConfig:
    """模型集成框架的完整配置。"""
    general: GeneralConfig  # 通用配置
    models: Dict[str, ModelConfig]  # 模型配置字典
    fallback: Dict[str, FallbackConfig]  # 回退策略配置
    caching: CacheConfig  # 缓存配置
    logging: LoggingConfig  # 日志配置


class ConfigError(Exception):
    """配置相关错误的异常类。"""
    pass


class ConfigManager:
    """配置管理器，负责加载和解析配置文件。"""
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^:}]+)(?::([^}]*))?\}')
    
    def __init__(self, config_path: Union[str, Path]):
        """初始化配置管理器。
        
        参数：
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Optional[ModelIntegrationConfig] = None
        
    def load(self) -> ModelIntegrationConfig:
        """加载并解析配置文件。
        
        返回：
            解析后的配置对象
            
        抛出：
            ConfigError: 配置文件加载或解析失败时
        """
        if not self.config_path.exists():
            raise ConfigError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            if not raw_config:
                raise ConfigError(f"配置文件为空: {self.config_path}")
            
            # 解析环境变量
            parsed_config = self._resolve_env_vars(raw_config)
            
            # 验证配置
            self._validate_config(parsed_config)
            
            # 构建配置对象
            self._config = self._build_config_object(parsed_config)
            
            return self._config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"配置文件解析错误: {e}")
        except Exception as e:
            raise ConfigError(f"配置文件处理错误: {e}")
    
    def reload(self) -> ModelIntegrationConfig:
        """重新加载配置文件。
        
        返回：
            重新解析后的配置对象
        """
        self._config = None
        return self.load()
    
    @property
    def config(self) -> ModelIntegrationConfig:
        """获取当前配置。
        
        返回：
            当前配置对象
            
        抛出：
            ConfigError: 配置未加载时
        """
        if not self._config:
            raise ConfigError("配置未加载，请先调用load()方法")
        return self._config
    
    def _resolve_env_vars(self, config: Any) -> Any:
        """解析配置中的环境变量引用。
        
        参数：
            config: 配置对象
            
        返回：
            解析后的配置对象
        """
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config
    
    def _substitute_env_vars(self, value: str) -> str:
        """替换字符串中的环境变量引用。
        
        参数：
            value: 包含环境变量引用的字符串
            
        返回：
            替换后的字符串
        """
        def replace_match(match: re.Match) -> str:
            env_var = match.group(1)
            default_value = match.group(2)
            return os.getenv(env_var, default_value) if default_value is not None else os.getenv(env_var, env_var)
        
        return self.ENV_VAR_PATTERN.sub(replace_match, value)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置的完整性和正确性。
        
        参数：
            config: 解析后的配置字典
            
        抛出：
            ConfigError: 配置不完整或不正确时
        """
        # 验证必需的配置部分
        required_sections = ['general', 'models', 'fallback', 'caching', 'logging']
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"配置缺少必需的部分: {section}")
        
        # 验证模型配置
        models = config['models']
        if not isinstance(models, dict) or len(models) == 0:
            raise ConfigError("模型配置不能为空")
        
        for model_name, model_config in models.items():
            if not isinstance(model_config, dict):
                raise ConfigError(f"模型配置格式错误: {model_name}")
            
            # 验证模型配置的必需字段
            required_model_fields = ['type', 'name']
            for field in required_model_fields:
                if field not in model_config:
                    raise ConfigError(f"模型 {model_name} 缺少必需字段: {field}")
            
            # 验证模型类型
            model_type = model_config['type']
            if model_type not in ['llm', 'embedding']:
                raise ConfigError(f"模型 {model_name} 的类型无效: {model_type}")
            
            # 验证接口类型
            interface_type = model_config.get('interface_type', 'openai')
            supported_interface_types = ['openai', 'doubao', 'deepseek']
            if interface_type not in supported_interface_types:
                # 警告但不阻止加载，允许使用自定义接口类型
                import logging
                logging.warning(f"模型 {model_name} 使用了未验证的接口类型: {interface_type}")
            
            # 验证参数映射格式
            parameter_mapping = model_config.get('parameter_mapping', {})
            if not isinstance(parameter_mapping, dict):
                raise ConfigError(f"模型 {model_name} 的参数映射格式错误")
            
            # 验证参数映射的键是否为字符串
            for key, value in parameter_mapping.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ConfigError(f"模型 {model_name} 的参数映射键值必须是字符串: {key} -> {value}")
            
            # 检查参数映射中的键是否为系统支持的统一参数名称
            supported_unified_params = [
                'temperature', 'top_p', 'max_tokens', 'frequency_penalty', 
                'presence_penalty', 'stop', 'seed', 'n', 'stream', 'model'
            ]
            for key in parameter_mapping.keys():
                if key not in supported_unified_params:
                    import logging
                    logging.warning(f"模型 {model_name} 的参数映射中包含未识别的统一参数: {key}")
                    logging.warning(f"支持的统一参数: {', '.join(supported_unified_params)}")
            
            # 验证响应格式映射格式
            response_format = model_config.get('response_format', {})
            if not isinstance(response_format, dict):
                raise ConfigError(f"模型 {model_name} 的响应格式映射格式错误")
            
            # 验证响应格式中的路径是否为有效的JSONPath格式
            for key, path in response_format.items():
                if not isinstance(path, str):
                    raise ConfigError(f"模型 {model_name} 的响应格式路径必须是字符串: {key}")
                
                # 更严格地验证JSONPath格式
                self._validate_jsonpath(path, model_name, key)
            
            # 验证API密钥和基础URL的一致性
            api_key = model_config.get('api_key')
            base_url = model_config.get('base_url')
            if base_url and not api_key:
                import logging
                logging.warning(f"模型 {model_name} 配置了基础URL但没有提供API密钥")
            
            # 验证可选参数格式
            parameters = model_config.get('parameters', {})
            if not isinstance(parameters, dict):
                raise ConfigError(f"模型 {model_name} 的可选参数格式错误")
            
            # 验证可选参数的值类型是否合法
            for param_name, param_value in parameters.items():
                if param_value is None:
                    raise ConfigError(f"模型 {model_name} 的可选参数 {param_name} 不能为None")
        
        # 验证回退策略配置
        fallback = config['fallback']
        if not isinstance(fallback, dict):
            raise ConfigError("回退策略配置格式错误")
        
        for model_type, fallback_config in fallback.items():
            if not isinstance(fallback_config, dict):
                raise ConfigError(f"回退策略配置格式错误: {model_type}")
            
            # 验证回退策略的必需字段
            required_fallback_fields = ['order', 'strategy', 'retry_on_failure', 'max_retries_per_model']
            for field in required_fallback_fields:
                if field not in fallback_config:
                    raise ConfigError(f"回退策略 {model_type} 缺少必需字段: {field}")
            
            # 验证模型顺序中的模型是否存在
            for model_name in fallback_config['order']:
                if model_name not in models:
                    raise ConfigError(f"回退策略 {model_type} 引用了不存在的模型: {model_name}")
            
            # 验证回退策略类型
            strategy = fallback_config['strategy']
            if strategy not in ['sequential', 'random']:
                raise ConfigError(f"回退策略 {model_type} 的策略类型无效: {strategy}")
            
            # 验证重试次数
            max_retries = fallback_config['max_retries_per_model']
            if not isinstance(max_retries, int) or max_retries < 0:
                raise ConfigError(f"回退策略 {model_type} 的最大重试次数必须是非负整数: {max_retries}")
                
    def _validate_jsonpath(self, path: str, model_name: str, key: str) -> None:
        """验证JSONPath表达式的有效性。
        
        参数：
            path: JSONPath表达式
            model_name: 模型名称
            key: 响应格式键名
            
        抛出：
            ConfigError: JSONPath表达式无效时
        """
        # 基本JSONPath格式验证
        if not path:
            raise ConfigError(f"模型 {model_name} 的响应格式路径不能为空: {key}")
        
        # 移除首尾空格
        path = path.strip()
        
        # 验证根路径格式
        if path.startswith("$") and len(path) > 1 and not path.startswith("$.") and not path.startswith("$["):
            raise ConfigError(f"模型 {model_name} 的响应格式路径根前缀格式错误: {key} -> {path}")
        
        # 检查是否包含非法字符（允许字母、数字、$、.、_、[、]）
        import re
        if not re.match(r'^[a-zA-Z0-9$._\[\]]+$', path):
            raise ConfigError(f"模型 {model_name} 的响应格式路径包含非法字符: {key} -> {path}")
        
        # 验证括号匹配
        if path.count('[') != path.count(']'):
            raise ConfigError(f"模型 {model_name} 的响应格式路径括号不匹配: {key} -> {path}")
        
        # 验证索引格式
        indices = re.findall(r'\[([^\]]+)\]', path)
        for index in indices:
            # 只允许数字索引
            if not index.isdigit():
                raise ConfigError(f"模型 {model_name} 的响应格式路径索引必须是数字: {key} -> {path}")
        
        # 验证路径结构（确保至少有一个有效键或索引）
        if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*|\[\d+\]', path):
            raise ConfigError(f"模型 {model_name} 的响应格式路径结构无效: {key} -> {path}")
        
        # 检查转义字符的使用
        if '\\' in path:
            escaped_chars = re.findall(r'\\.', path)
            for esc_char in escaped_chars:
                if esc_char not in ['\\.', '\\$', '\\[', '\\]']:
                    raise ConfigError(f"模型 {model_name} 的响应格式路径包含无效的转义字符: {key} -> {path}")
    
    def _build_config_object(self, config: Dict[str, Any]) -> ModelIntegrationConfig:
        """构建类型安全的配置对象。
        
        参数：
            config: 解析后的配置字典
            
        返回：
            类型安全的配置对象
        """
        # 构建通用配置
        general_config = GeneralConfig(
            default_timeout=config['general'].get('default_timeout', 30),
            max_retries=config['general'].get('max_retries', 2),
            enable_async=config['general'].get('enable_async', True)
        )
        
        # 构建模型配置
        models_config = {}
        for model_name, model_config in config['models'].items():
            models_config[model_name] = ModelConfig(
                type=model_config['type'],
                name=model_config['name'],
                api_key=model_config.get('api_key'),
                api_secret=model_config.get('api_secret'),
                base_url=model_config.get('base_url'),
                region=model_config.get('region'),
                interface_type=model_config.get('interface_type', 'openai'),
                parameter_mapping=model_config.get('parameter_mapping', {}),
                response_format=model_config.get('response_format', {}),
                parameters=model_config.get('parameters', {})
            )
        
        # 构建回退策略配置
        fallback_config = {}
        for model_type, fb_config in config['fallback'].items():
            fallback_config[model_type] = FallbackConfig(
                order=fb_config['order'],
                strategy=fb_config['strategy'],
                retry_on_failure=fb_config['retry_on_failure'],
                max_retries_per_model=fb_config['max_retries_per_model']
            )
        
        # 构建缓存配置
        cache_config = CacheConfig(
            enabled=config['caching'].get('enabled', False),
            type=config['caching'].get('type', 'memory'),
            ttl=config['caching'].get('ttl', 3600),
            max_size=config['caching'].get('max_size', 1000),
            redis=config['caching'].get('redis', {})
        )
        
        # 构建日志配置
        logging_config = LoggingConfig(
            level=config['logging'].get('level', 'INFO'),
            format=config['logging'].get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            file=config['logging'].get('file'),
            rotate=config['logging'].get('rotate', False),
            max_bytes=config['logging'].get('max_bytes', 10485760),
            backup_count=config['logging'].get('backup_count', 5)
        )
        
        # 构建完整配置对象
        return ModelIntegrationConfig(
            general=general_config,
            models=models_config,
            fallback=fallback_config,
            caching=cache_config,
            logging=logging_config
        )


@lru_cache(maxsize=1)
def get_config_manager(config_path: Union[str, Path] = "configs/models.yaml") -> ConfigManager:
    """获取配置管理器实例。
    
    参数：
        config_path: 配置文件路径
        
    返回：
        配置管理器实例
    """
    return ConfigManager(config_path)


@lru_cache(maxsize=1)
def get_model_integration_config(config_path: Union[str, Path] = "configs/models.yaml") -> ModelIntegrationConfig:
    """获取模型集成框架配置。
    
    参数：
        config_path: 配置文件路径
        
    返回：
        模型集成框架配置
    """
    manager = get_config_manager(config_path)
    return manager.load()


__all__ = [
    "ModelConfig",
    "FallbackConfig",
    "CacheConfig",
    "GeneralConfig",
    "LoggingConfig",
    "ModelIntegrationConfig",
    "ConfigError",
    "ConfigManager",
    "get_config_manager",
    "get_model_integration_config"
]
