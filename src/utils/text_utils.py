"""模块：data_processing.

特定于Markdown的清理和智能分块辅助函数，用于在文本进入嵌入流水线之前对其进行标准化处理。

本模块实现了遵循data_processing规范的文本处理函数：
- 文本经过归一化和清理以确保一致性
- 函数为纯函数，不修改输入参数
- 全程使用UTF-8编码
- 支持Markdown特定功能，如标题和代码块
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional


HEADING_PATTERN = re.compile(r"^(#+\s.*)", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[\x00-\x1F\x7F]")  # 控制字符


def clean_markdown(text: str) -> str:
    """根据data_processing规范清理并归一化Markdown文本。
    
    参数：
        text: 要清理的输入Markdown文本
        
    返回：
        具有标准化格式的清理后文本
        
    遵循规则：
        - 移除回车符
        - 将多个空白字符替换为单个空格
        - 移除开头/结尾的空白字符
        - 确保UTF-8兼容性
    """
    text = text.replace("\r", "")  # 移除Windows风格的行结尾
    text = SPECIAL_CHAR_PATTERN.sub("", text)  # 移除控制字符
    text = WHITESPACE_PATTERN.sub(" ", text)  # 归一化空白字符
    return text.strip()


def extract_code_blocks(text: str) -> List[str]:
    """从Markdown文本中提取代码块。
    
    参数：
        text: 包含代码块的输入Markdown文本
        
    返回：
        提取的代码块列表
    """
    return CODE_BLOCK_PATTERN.findall(text)


def split_markdown(text: str, chunk_size: int, overlap: int) -> List[str]:
    """将Markdown文本分割成具有指定大小和重叠的块。
    
    参数：
        text: 要分割的输入Markdown文本
        chunk_size: 每个块的最大令牌数
        overlap: 连续块之间的重叠令牌数
        
    返回：
        遵循Markdown结构的文本块列表
        
    处理流程：
        1. 清理输入文本
        2. 按标题分割成章节
        3. 创建具有指定重叠的滑动窗口块
    """
    cleaned = clean_markdown(text)
    if not cleaned:
        return []

    sections = _split_by_heading(cleaned)
    return list(_window_sections(sections, chunk_size, overlap))


def _split_by_heading(text: str) -> List[str]:
    """根据Markdown标题将文本分割成章节。
    
    参数：
        text: 清理后的Markdown文本
        
    返回：
        文本章节列表，每个章节以标题开头
    """
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [text]  # 如果没有标题，则返回整个文本

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        # 当前章节的结束是下一个标题的开始或文本末尾
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())
    return sections


def _window_sections(sections: Iterable[str], chunk_size: int, overlap: int) -> Iterable[str]:
    """从文本章节创建滑动窗口块。
    
    参数：
        sections: 文本章节的可迭代对象
        chunk_size: 每个块的最大令牌数
        overlap: 连续块之间的重叠令牌数
        
    返回：
        具有指定重叠的文本块的可迭代对象
    """
    # 将章节展平为单个令牌列表
    tokens: List[str] = []
    for section in sections:
        tokens.extend(section.split(" "))

    if not tokens:
        return []

    # 计算步长以实现所需的重叠
    step = max(1, chunk_size - overlap)
    chunks = []
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if not chunk_tokens:
            continue
        chunks.append(" ".join(chunk_tokens).strip())
    return chunks


__all__ = ["clean_markdown", "split_markdown", "extract_code_blocks"]

