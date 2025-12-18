#!/usr/bin/env python3
"""
测试脚本：验证DeepseekLLM模型的功能

该脚本用于测试DeepseekLLM模型的基本功能，包括文本生成和多模态功能。
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
from src.llm.deepseek_llm import DeepseekLLM
from src.embeddings.multimodal_embeddings import MultimodalEmbeddings


def test_deepseek_llm_direct():
    """直接测试DeepseekLLM模型功能"""
    logger.info("直接测试DeepseekLLM...")
    settings = get_settings()
    
    # 创建DeepseekLLM模型
    deepseek_llm = DeepseekLLM(settings)
    
    # 测试简单文本生成
    prompt = "请简要介绍一下DeepSeek V3.2模型的特点"
    try:
        result = deepseek_llm.generate(prompt)
        logger.info(f"DeepseekLLM文本生成成功，结果: {result[:100]}...")
        return True
    except Exception as e:
        logger.error(f"DeepseekLLM文本生成失败: {e}")
        return False


def test_deepseek_in_manager():
    """测试DeepseekLLM在LLMManager中的功能"""
    logger.info("测试DeepseekLLM在LLMManager中...")
    settings = get_settings()
    
    # 创建包含Deepseek的模型列表
    llm_models = [DoubaoLLM(settings), DeepseekLLM(settings)]
    llm_manager = LLMManager(llm_models, max_retries=1)
    
    # 测试生成功能
    prompt = "请解释什么是多模态RAG系统"
    try:
        result = llm_manager.invoke((prompt, []))
        logger.info(f"LLMManager生成成功，结果: {result[:100]}...")
        return True
    except Exception as e:
        logger.error(f"LLMManager生成失败: {e}")
        return False


if __name__ == "__main__":
    logger.info("开始测试DeepseekLLM模型...")
    
    success_count = 0
    total_tests = 0
    
    # 测试直接调用
    total_tests += 1
    if test_deepseek_llm_direct():
        success_count += 1
    
    # 测试在管理器中调用
    total_tests += 1
    if test_deepseek_in_manager():
        success_count += 1
    
    logger.info(f"测试完成: {success_count}/{total_tests} 个测试通过")
    
    if success_count == total_tests:
        logger.info("所有测试通过！DeepseekLLM模型工作正常。")
        sys.exit(0)
    else:
        logger.error(f"{total_tests - success_count} 个测试失败！")
        sys.exit(1)
