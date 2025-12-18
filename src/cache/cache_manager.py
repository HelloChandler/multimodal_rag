"""模块：model_integration_framework

缓存管理模块，负责模型响应的缓存和检索。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, Optional, Union

from src.config.config_manager import CacheConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheItem:
    """缓存项"""
    value: Any
    created_at: float
    ttl: int


class CacheManager:
    """缓存管理器，负责模型响应的缓存和检索。"""
    
    def __init__(self, config: CacheConfig):
        """初始化缓存管理器。
        
        参数：
            config: 缓存配置
        """
        self.config = config
        self._cache: Dict[str, CacheItem] = {}
        self._redis_client = None
        
        # 初始化缓存
        if config.enabled:
            if config.type == 'redis':
                self._init_redis()
            elif config.type != 'memory':
                logger.warning(f"不支持的缓存类型: {config.type}，将使用内存缓存")
            logger.info(f"缓存已启用，类型: {config.type}")
        else:
            logger.info("缓存已禁用")
    
    def _init_redis(self) -> None:
        """初始化Redis客户端。"""
        try:
            import redis
            self._redis_client = redis.Redis(**self.config.redis)
            # 测试连接
            self._redis_client.ping()
            logger.info("Redis缓存客户端初始化成功")
        except ImportError:
            logger.error("Redis模块未安装，请安装redis包")
            raise ImportError("Redis模块未安装，请安装redis包")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise RuntimeError(f"Redis连接失败: {e}")
    
    def _get_cache_key(self, model_name: str, model_type: str, request_data: Dict[str, Any]) -> str:
        """生成缓存键。
        
        参数：
            model_name: 模型名称
            model_type: 模型类型
            request_data: 请求数据
            
        返回：
            缓存键
        """
        import hashlib
        import json
        
        # 将请求数据转换为JSON字符串并排序，确保相同内容生成相同哈希
        sorted_request = json.dumps(request_data, sort_keys=True, ensure_ascii=False)
        request_hash = hashlib.md5(sorted_request.encode('utf-8')).hexdigest()
        
        return f"{model_type}:{model_name}:{request_hash}"
    
    def get(self, model_name: str, model_type: str, request_data: Dict[str, Any]) -> Optional[Any]:
        """从缓存中获取响应。
        
        参数：
            model_name: 模型名称
            model_type: 模型类型
            request_data: 请求数据
            
        返回：
            缓存的响应内容，如果没有缓存则返回None
        """
        if not self.config.enabled:
            return None
        
        cache_key = self._get_cache_key(model_name, model_type, request_data)
        
        try:
            if self._redis_client:
                # 使用Redis缓存
                value = self._redis_client.get(cache_key)
                if value:
                    return self._deserialize(value)
            else:
                # 使用内存缓存
                if cache_key in self._cache:
                    item = self._cache[cache_key]
                    # 检查是否过期
                    if time.time() - item.created_at < item.ttl:
                        return item.value
                    else:
                        # 移除过期缓存
                        del self._cache[cache_key]
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        
        return None
    
    async def get_async(self, model_name: str, model_type: str, request_data: Dict[str, Any]) -> Optional[Any]:
        """异步从缓存中获取响应。
        
        参数：
            model_name: 模型名称
            model_type: 模型类型
            request_data: 请求数据
            
        返回：
            缓存的响应内容，如果没有缓存则返回None
        """
        # 对于内存缓存，同步和异步操作相同
        if not self._redis_client:
            return self.get(model_name, model_type, request_data)
        
        # 对于Redis，使用异步客户端
        try:
            import aioredis
            
            cache_key = self._get_cache_key(model_name, model_type, request_data)
            async with aioredis.Redis(**self.config.redis) as redis:
                value = await redis.get(cache_key)
                if value:
                    return self._deserialize(value)
        except ImportError:
            logger.error("aioredis模块未安装，请安装aioredis包")
        except Exception as e:
            logger.error(f"异步获取缓存失败: {e}")
        
        return None
    
    def set(self, model_name: str, model_type: str, request_data: Dict[str, Any], response_data: Any) -> bool:
        """将响应缓存起来。
        
        参数：
            model_name: 模型名称
            model_type: 模型类型
            request_data: 请求数据
            response_data: 响应数据
            
        返回：
            如果缓存成功则返回True，否则返回False
        """
        if not self.config.enabled:
            return False
        
        cache_key = self._get_cache_key(model_name, model_type, request_data)
        
        try:
            if self._redis_client:
                # 使用Redis缓存
                serialized_data = self._serialize(response_data)
                self._redis_client.setex(cache_key, self.config.ttl, serialized_data)
            else:
                # 使用内存缓存
                # 检查缓存大小
                if len(self._cache) >= self.config.max_size:
                    # 移除最旧的缓存项
                    oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
                    del self._cache[oldest_key]
                
                # 添加新缓存项
                self._cache[cache_key] = CacheItem(
                    value=response_data,
                    created_at=time.time(),
                    ttl=self.config.ttl
                )
            return True
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    async def set_async(self, model_name: str, model_type: str, request_data: Dict[str, Any], response_data: Any) -> bool:
        """异步将响应缓存起来。
        
        参数：
            model_name: 模型名称
            model_type: 模型类型
            request_data: 请求数据
            response_data: 响应数据
            
        返回：
            如果缓存成功则返回True，否则返回False
        """
        # 对于内存缓存，同步和异步操作相同
        if not self._redis_client:
            return self.set(model_name, model_type, request_data, response_data)
        
        # 对于Redis，使用异步客户端
        try:
            import aioredis
            
            cache_key = self._get_cache_key(model_name, model_type, request_data)
            serialized_data = self._serialize(response_data)
            
            async with aioredis.Redis(**self.config.redis) as redis:
                await redis.setex(cache_key, self.config.ttl, serialized_data)
            return True
        except ImportError:
            logger.error("aioredis模块未安装，请安装aioredis包")
        except Exception as e:
            logger.error(f"异步设置缓存失败: {e}")
        
        return False
    
    def clear(self, model_name: Optional[str] = None, model_type: Optional[str] = None) -> bool:
        """清除缓存。
        
        参数：
            model_name: 可选，指定要清除的模型名称
            model_type: 可选，指定要清除的模型类型
            
        返回：
            如果清除成功则返回True，否则返回False
        """
        if not self.config.enabled:
            return False
        
        try:
            if self._redis_client:
                # 使用Redis缓存
                if model_name and model_type:
                    pattern = f"{model_type}:{model_name}:*"
                elif model_type:
                    pattern = f"{model_type}:*"
                else:
                    pattern = "*"
                
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            else:
                # 使用内存缓存
                if model_name and model_type:
                    prefix = f"{model_type}:{model_name}:"
                    keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
                elif model_type:
                    prefix = f"{model_type}:"
                    keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
                else:
                    keys_to_delete = list(self._cache.keys())
                
                for key in keys_to_delete:
                    del self._cache[key]
            
            return True
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
            return False
    
    async def clear_async(self, model_name: Optional[str] = None, model_type: Optional[str] = None) -> bool:
        """异步清除缓存。
        
        参数：
            model_name: 可选，指定要清除的模型名称
            model_type: 可选，指定要清除的模型类型
            
        返回：
            如果清除成功则返回True，否则返回False
        """
        # 对于内存缓存，同步和异步操作相同
        if not self._redis_client:
            return self.clear(model_name, model_type)
        
        # 对于Redis，使用异步客户端
        try:
            import aioredis
            
            if model_name and model_type:
                pattern = f"{model_type}:{model_name}:*"
            elif model_type:
                pattern = f"{model_type}:*"
            else:
                pattern = "*"
            
            async with aioredis.Redis(**self.config.redis) as redis:
                keys = await redis.keys(pattern)
                if keys:
                    await redis.delete(*keys)
            
            return True
        except ImportError:
            logger.error("aioredis模块未安装，请安装aioredis包")
        except Exception as e:
            logger.error(f"异步清除缓存失败: {e}")
        
        return False
    
    def _serialize(self, data: Any) -> Union[str, bytes]:
        """序列化数据。
        
        参数：
            data: 要序列化的数据
            
        返回：
            序列化后的数据
        """
        import json
        return json.dumps(data, ensure_ascii=False)
    
    def _deserialize(self, data: Union[str, bytes]) -> Any:
        """反序列化数据。
        
        参数：
            data: 要反序列化的数据
            
        返回：
            反序列化后的数据
        """
        import json
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    
    def cache_response(self, model_name: str, model_type: str):
        """响应缓存装饰器。
        
        参数：
            model_name: 模型名称
            model_type: 模型类型
            
        返回：
            装饰器函数
        """
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                # 异步函数装饰器
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    if not self.config.enabled:
                        return await func(*args, **kwargs)
                    
                    # 提取请求数据
                    request_data = kwargs.copy()
                    if 'self' in kwargs:
                        del request_data['self']
                    
                    # 尝试从缓存获取
                    cached_response = await self.get_async(model_name, model_type, request_data)
                    if cached_response is not None:
                        logger.debug(f"从缓存获取响应: {model_name}")
                        return cached_response
                    
                    # 调用原始函数
                    response = await func(*args, **kwargs)
                    
                    # 缓存响应
                    await self.set_async(model_name, model_type, request_data, response)
                    logger.debug(f"缓存响应: {model_name}")
                    
                    return response
                return async_wrapper
            else:
                # 同步函数装饰器
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    if not self.config.enabled:
                        return func(*args, **kwargs)
                    
                    # 提取请求数据
                    request_data = kwargs.copy()
                    if 'self' in kwargs:
                        del request_data['self']
                    
                    # 尝试从缓存获取
                    cached_response = self.get(model_name, model_type, request_data)
                    if cached_response is not None:
                        logger.debug(f"从缓存获取响应: {model_name}")
                        return cached_response
                    
                    # 调用原始函数
                    response = func(*args, **kwargs)
                    
                    # 缓存响应
                    self.set(model_name, model_type, request_data, response)
                    logger.debug(f"缓存响应: {model_name}")
                    
                    return response
                return sync_wrapper
        return decorator
    
    def close(self) -> None:
        """关闭缓存资源。"""
        if self._redis_client:
            try:
                self._redis_client.close()
                logger.info("Redis缓存客户端已关闭")
            except Exception as e:
                logger.error(f"关闭Redis缓存客户端失败: {e}")


# 全局缓存管理器实例
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例。
    
    返回：
        全局缓存管理器实例
    """
    global _cache_manager
    if _cache_manager is None:
        from src.config.config_manager import get_model_integration_config
        config = get_model_integration_config()
        _cache_manager = CacheManager(config.caching)
    return _cache_manager


def init_cache_manager() -> None:
    """初始化全局缓存管理器。"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = get_cache_manager()


def reload_cache_manager() -> None:
    """重新加载全局缓存管理器。"""
    global _cache_manager
    from src.config.config_manager import get_model_integration_config
    config = get_model_integration_config()
    _cache_manager = CacheManager(config.caching)


__all__ = [
    "CacheManager",
    "get_cache_manager",
    "init_cache_manager",
    "reload_cache_manager"
]