"""模块：pipeline.

集中式设置和路径配置加载器，确保每个流水线组件与环境变量(.env)定义保持一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)


@dataclass(frozen=True)
class PathSettings:
    """路径配置类，定义项目中的各种路径。"""
    base_dir: Path = BASE_DIR  # 项目根目录
    raw_data: Path = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")  # 原始数据目录
    processed_data: Path = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")  # 处理后数据目录
    index_path: Path = BASE_DIR / os.getenv("INDEX_PATH", "data/index/faiss.index")  # 索引文件路径


@dataclass(frozen=True)
class ModelSettings:
    """模型配置类，定义模型相关的API密钥和参数。"""
    ark_api_key: str  # Ark API密钥
    ark_api_secret: str  # Ark API密钥对
    ark_region: str  # Ark服务区域
    deepseek_api_key: str  # DeepSeek API密钥
    llm_model: str  # 大语言模型名称
    embed_model: str  # 嵌入模型名称


@dataclass(frozen=True)
class Settings:
    """全局配置类，包含路径、模型和其他流水线配置。"""
    paths: PathSettings  # 路径配置
    models: ModelSettings  # 模型配置
    default_top_k: int = 4  # 默认检索返回的文档数量
    chunk_size: int = 500  # 文本分块大小
    chunk_overlap: int = 100  # 文本分块重叠大小


def _env(key: str, default: Optional[str] = None) -> str:
    """从环境变量中获取值，如果不存在则使用默认值或抛出异常。
    
    参数：
        key: 环境变量名称
        default: 默认值（可选）
        
    返回：
        环境变量的值
        
    异常：
        RuntimeError: 如果环境变量不存在且没有提供默认值
    """
    value = os.getenv(key, default)
    if value is None:
        raise RuntimeError(f"缺少必需的环境变量: {key}")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置实例，使用LRU缓存确保只加载一次。
    
    返回：
        Settings实例，包含所有配置信息
    """
    paths = PathSettings()
    models = ModelSettings(
        ark_api_key=_env("ARK_API_KEY", ""),
        ark_api_secret=_env("ARK_API_SECRET", ""),
        ark_region=_env("ARK_REGION", "cn-beijing"),
        deepseek_api_key=_env("DEEPSEEK_API_KEY", ""),
        llm_model=_env("DOUBAO_LLM_MODEL", "doubao-seed-1-6-vision-250815"),
        embed_model=_env("DOUBAO_EMBED_MODEL", "doubao-embedding-vision-250615"),
    )
    # 创建所需的目录
    for target in (paths.raw_data, paths.processed_data, paths.index_path.parent):
        target.mkdir(parents=True, exist_ok=True)
    return Settings(paths=paths, models=models)


__all__ = ["Settings", "get_settings"]

