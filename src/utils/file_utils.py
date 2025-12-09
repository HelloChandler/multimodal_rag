"""模块：data_processing.

负责遍历Markdown/Text源文件，提取内联图像，并为下游嵌入步骤准备结构化负载。
本模块遵循[data_processing]规范进行多模态数据准备。

功能特性：
- 文件系统遍历，支持markdown和text文件
- Markdown文档解析与图像提取
- 为嵌入流水线创建结构化负载
- 强制执行UTF-8编码
- 带日志记录的错误处理
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .image_utils import image_to_base64, save_image_copy


logger = logging.getLogger(__name__)

# 匹配markdown内联图像的正则表达式
INLINE_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)")

# 文档处理支持的文件类型
MARKDOWN_SUFFIXES = (".md", ".markdown")
TEXT_SUFFIXES = (".txt",)

# 所有文本操作的字符编码（遵循data_processing规范）
DEFAULT_ENCODING = "utf-8"


@dataclass
class DocumentPayload:
    """处理后文档的结构化负载。
    
    属性：
        path: 原始文件路径
        content: 文本内容（UTF-8编码）
        images: 提取并处理后的图像列表
    """
    path: Path
    content: str
    images: List["ImagePayload"]


@dataclass
class ImagePayload:
    """处理后图像的结构化负载。
    
    属性：
        alt: Markdown中的替代文本
        path: 复制后的图像文件路径
        base64: 图像的Base64编码字符串
    """
    alt: str
    path: Path
    base64: str


def iter_source_files(source_dir: Path) -> Iterable[Path]:
    """遍历目录中所有支持的源文件。
    
    参数：
        source_dir: 搜索源文件的目录
        
    生成：
        找到的所有markdown和text文件的Path对象
        
    遵循规则（data_processing规范）：
        - UTF-8编码支持
        - 按文件类型过滤
        - 排序遍历以确保一致性
    """
    supported_suffixes = (*MARKDOWN_SUFFIXES, *TEXT_SUFFIXES)
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            yield path


def parse_markdown_documents(source_dir: Path, raw_dir: Path) -> List[DocumentPayload]:
    """解析markdown文档并提取内联图像。
    
    参数：
        source_dir: 包含markdown源文件的目录
        raw_dir: 存储复制的原始图像的目录
        
    返回：
        包含处理后的文档和图像的DocumentPayload对象列表
        
    处理流程：
        1. 遍历所有支持的源文件
        2. 使用UTF-8编码读取文本内容
        3. 提取并处理内联图像
        4. 为下游处理创建结构化负载
        
    遵循规则（data_processing规范）：
        - 强制执行UTF-8编码
        - 带日志记录的错误处理
        - 结构化数据输出
        - 图像处理集成
    """
    documents: List[DocumentPayload] = []
    
    for file_path in iter_source_files(source_dir):
        try:
            # 使用UTF-8编码读取文本（遵循data_processing规范）
            text = file_path.read_text(encoding=DEFAULT_ENCODING)
        except UnicodeDecodeError as exc:
            logger.warning("UTF-8 decoding failed for %s: %s", file_path, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", file_path, exc)
            continue

        # 从文档中提取并处理图像
        images = _extract_images(file_path, raw_dir, text)
        
        # 创建文档负载
        documents.append(DocumentPayload(path=file_path, content=text, images=images))
    
    return documents


def _extract_images(file_path: Path, raw_dir: Path, text: str) -> List[ImagePayload]:
    """从markdown文本中提取并处理内联图像。
    
    参数：
        file_path: Markdown文件的路径
        raw_dir: 存储复制的原始图像的目录
        text: Markdown文本内容
        
    返回：
        成功处理的图像的ImagePayload对象列表
        
    处理流程：
        1. 查找markdown中的所有内联图像引用
        2. 解析相对于文档的图像路径
        3. 将图像复制到raw目录
        4. 将图像转换为base64编码字符串
        5. 创建图像负载
        
    遵循规则（data_processing规范）：
        - 图像尺寸归一化
        - 为嵌入流水线进行Base64编码
        - 带日志记录的错误处理
    """
    images: List[ImagePayload] = []
    
    for match in INLINE_IMAGE_PATTERN.finditer(text):
        rel_path = match.group("path").strip()
        alt = match.group("alt").strip()
        
        # 解析相对于文档的图像路径
        img_path = (file_path.parent / rel_path).resolve()
        
        # 将图像复制到raw目录进行处理
        copied = save_image_copy(img_path, raw_dir)
        if not copied:
            logger.warning("Failed to copy image %s", img_path)
            continue
        
        # 将图像转换为base64（包括尺寸归一化）
        b64 = image_to_base64(copied)
        if not b64:
            logger.warning("Failed to convert image %s to base64", copied)
            continue
        
        # 创建图像负载
        images.append(ImagePayload(alt=alt, path=copied, base64=b64))
    
    return images


def read_text_file(path: Path, encoding: str = DEFAULT_ENCODING) -> Optional[str]:
    """读取文本文件并进行错误处理。
    
    参数：
        path: 文本文件的路径
        encoding: 使用的字符编码
        
    返回：
        成功时返回文本内容，否则返回None
        
    遵循规则（data_processing规范）：
        - 默认使用UTF-8编码
        - 带日志记录的错误处理
        - 一致的文本读取接口
    """
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        logger.warning("%s decoding failed for %s: %s", encoding, path, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None


__all__ = [
    "DocumentPayload",
    "ImagePayload",
    "parse_markdown_documents",
    "iter_source_files",
    "read_text_file"
]

