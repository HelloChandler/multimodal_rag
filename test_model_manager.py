#!/usr/bin/env python3
"""
测试脚本：验证模型调用机制的功能

该脚本用于测试 ModelManager、LLMManager 和 EmbeddingsManager 的基本功能，
包括正常调用和失败回退机制。
"""

import logging
import sys
from typing import List, Dict, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
sys.path.insert(0, '.')

from src.config.settings import get_settings
from src.model.model_manager import LLMManager, EmbeddingsManager
from src.llm.doubao_llm import DoubaoLLM
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings


def test_embeddings_manager():
    """测试 EmbeddingsManager 功能"""
    logger.info("测试 EmbeddingsManager...")
    settings = get_settings()
    
    # 创建嵌入模型
    embeddings_models = [MultimodalEmbeddings(settings)]
    embeddings_manager = EmbeddingsManager(embeddings_models, max_retries=1)
    
    # 测试文本嵌入
    text_payload = [{"type": "text", "content": "这是一个测试文本", "source": "test"}]
    try:
        vectors = embeddings_manager.invoke(text_payload)
        logger.info(f"文本嵌入成功，向量维度: {len(vectors[0]) if vectors else '无'}")
        return True
    except Exception as e:
        logger.error(f"文本嵌入失败: {e}")
        return False


def test_llm_manager():
    """测试 LLMManager 功能"""
    logger.info("测试 LLMManager...")
    settings = get_settings()
    
    # 创建 LLM 模型
    llm_models = [DoubaoLLM(settings)]
    llm_manager = LLMManager(llm_models, max_retries=1)
    
    # 测试简单生成
    prompt = "你好，请问今天天气怎么样？"
    try:
        result = llm_manager.invoke((prompt, []))
        logger.info(f"LLM 生成成功，结果: {result[:100]}...")
        return True
    except Exception as e:
        logger.error(f"LLM 生成失败: {e}")
        return False


if __name__ == "__main__":
    logger.info("开始测试模型调用机制...")
    
    success_count = 0
    total_tests = 0
    
    # 测试嵌入管理器
    total_tests += 1
    if test_embeddings_manager():
        success_count += 1
    
    # 测试 LLM 管理器
    total_tests += 1
    if test_llm_manager():
        success_count += 1
    
    logger.info(f"测试完成: {success_count}/{total_tests} 个测试通过")
    
    if success_count == total_tests:
        logger.info("所有测试通过！模型调用机制工作正常。")
        sys.exit(0)
    else:
        logger.error(f"{total_tests - success_count} 个测试失败！")
        sys.exit(1)
