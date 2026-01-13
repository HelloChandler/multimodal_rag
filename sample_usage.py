#!/usr/bin/env python3
"""示例：模型集成框架使用演示

展示如何使用模型集成框架进行模型调用、回退策略和异步操作。
"""

import asyncio
import logging

from src.model.model_manager import ModelManager
from src.model.unified_interface import ModelRequest

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def demo_sync_model_invoke():
    """演示同步模型调用。"""
    logger.info("\n=== 演示同步模型调用 ===")
    
    # 创建模型管理器
    manager = ModelManager()
    
    # 调用模型
    response = manager.generate_completion(
        prompt="请解释什么是人工智能？",
        temperature=0.7,
        max_tokens=500
    )
    
    # 打印结果
    if response.is_success:
        logger.info(f"成功调用模型: {response.model_name}")
        logger.info(f"响应内容: {response.content[:200]}...")
        if "duration" in response.metadata:
            logger.info(f"调用耗时: {response.metadata['duration']:.2f}秒")
    else:
        logger.error(f"模型调用失败: {response.error_message}")
    
    return response

async def demo_async_model_invoke():
    """演示异步模型调用。"""
    logger.info("\n=== 演示异步模型调用 ===")
    
    # 创建模型管理器
    manager = ModelManager()
    
    # 异步调用模型
    response = await manager.generate_completion_async(
        prompt="请解释什么是机器学习？",
        temperature=0.7,
        max_tokens=500
    )
    
    # 打印结果
    if response.is_success:
        logger.info(f"成功调用模型: {response.model_name}")
        logger.info(f"响应内容: {response.content[:200]}...")
        if "duration" in response.metadata:
            logger.info(f"调用耗时: {response.metadata['duration']:.2f}秒")
    else:
        logger.error(f"模型调用失败: {response.error_message}")
    
    return response

def demo_health_check():
    """演示模型健康检查。"""
    logger.info("\n=== 演示模型健康检查 ===")
    
    # 创建模型管理器
    manager = ModelManager()
    
    # 从配置中获取模型列表
    from src.config.config_manager import get_model_integration_config
    config = get_model_integration_config()
    
    # 获取所有模型名称
    all_model_names = list(config.models.keys())
    
    for model_name in all_model_names:
        try:
            is_healthy = manager.health_check(model_name)
            model_info = manager.get_model_info(model_name)
            status = "健康" if is_healthy else "不健康"
            logger.info(f"模型 {model_name} ({model_info['name']}): {status}")
        except Exception as e:
            logger.error(f"模型 {model_name} 健康检查失败: {e}")

def demo_config_reload():
    """演示配置重新加载。"""
    logger.info("\n=== 演示配置重新加载 ===")
    
    # 重新加载模型管理器
    from src.model.model_manager import reload_model_manager
    reload_model_manager()
    logger.info("配置重新加载成功")

if __name__ == "__main__":
    logger.info("=== 模型集成框架演示开始 ===")
    
    try:
        # 演示健康检查
        demo_health_check()
        
        # 演示同步模型调用
        demo_sync_model_invoke()
        
        # 演示异步模型调用
        asyncio.run(demo_async_model_invoke())
        
        # 演示配置重新加载
        demo_config_reload()
        
        logger.info("=== 模型集成框架演示结束 ===")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}", exc_info=True)
