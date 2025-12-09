"""模块：data_processing.

用于规范化图像资产的辅助函数（调整大小、归一化、base64编码、复制）
使数据摄入流水线能够统一处理图像。本模块遵循[data_processing]规范进行图像处理。

功能特性：
- 图像加载和格式转换
- 可配置最大边缘的尺寸归一化
- 像素值归一化（0-1和Z-score）
- 用于嵌入流水线的Base64编码
- 批量处理支持
- 带日志记录的错误处理
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)

# 遵循data_processing规范的默认配置
MAX_EDGE = 1024  # 图像归一化的默认最大边缘尺寸
DEFAULT_QUALITY = 85  # 默认JPEG质量
RGB_CHANNELS = 3  # RGB图像的通道数


def _load_image(path: Path) -> Optional[Image.Image]:
    """从文件系统加载图像并转换为RGB格式。
    
    参数：
        path: 图像文件的路径
        
    返回：
        加载并转换后的图像，加载失败则返回None
    """
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")  # 确保RGB格式一致性
    except Exception as exc:  # noqa: BLE001 - 根据规范记录并忽略异常
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


def normalize_image_size(image: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """在保持宽高比的同时归一化图像尺寸。
    
    参数：
        image: 输入图像
        max_edge: 宽度或高度的最大允许尺寸
        
    返回：
        调整后的图像，最大维度 <= max_edge
    """
    w, h = image.size
    if max(w, h) <= max_edge:
        return image  # 无需调整大小
    
    # 计算缩放因子以保持宽高比
    scale = max_edge / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    
    # 使用高质量重采样
    return image.resize(new_size, Image.Resampling.LANCZOS)


def normalize_image_pixels(image: Image.Image, method: str = "0-1") -> np.ndarray:
    """根据指定方法归一化图像像素值。
    
    参数：
        image: 输入图像（PIL Image）
        method: 归一化方法："0-1"（最小-最大缩放）或 "z-score"
        
    返回：
        归一化后的图像，为float32值的numpy数组
        
    遵循规则（data_processing规范）：
        - 0-1归一化：将像素值从0-255缩放到0.0-1.0
        - Z-score归一化：标准化为每个通道的平均值=0，标准差=1
    """
    # 将图像转换为float32类型的numpy数组
    img_array = np.array(image, dtype=np.float32)
    
    if method == "0-1":
        # Min-max scaling to [0, 1] range
        return img_array / 255.0
    elif method == "z-score":
        # Standardize each channel independently
        if len(img_array.shape) == 3:
            # For RGB images: (H, W, C) -> normalize per channel
            mean = img_array.mean(axis=(0, 1), keepdims=True)
            std = img_array.std(axis=(0, 1), keepdims=True) + 1e-8  # Avoid division by zero
            return (img_array - mean) / std
        else:
            # For grayscale: (H, W) -> normalize globally
            mean = img_array.mean()
            std = img_array.std() + 1e-8
            return (img_array - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use '0-1' or 'z-score'")


def image_to_base64(path: Path, max_edge: int = MAX_EDGE, quality: int = DEFAULT_QUALITY) -> Optional[str]:
    """将图像转换为base64编码字符串，并进行尺寸归一化。
    
    参数：
        path: 图像文件的路径
        max_edge: 宽度或高度的最大允许尺寸
        quality: JPEG压缩质量（1-100）
        
    返回：
        图像的Base64编码字符串，处理失败则返回None
        
    处理流程：
        1. 从路径加载图像
        2. 归一化尺寸
        3. 转换为JPEG格式
        4. 编码为base64字符串
    """
    image = _load_image(path)
    if image is None:
        return None

    # 归一化图像尺寸
    image = normalize_image_size(image, max_edge)

    # 转换为base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    base64_bytes = base64.b64encode(buffer.getvalue())
    return base64_bytes.decode("utf-8")


def save_image_copy(src: Path, dst_dir: Path) -> Optional[Path]:
    """将图像的副本保存到目标目录。
    
    参数：
        src: 源图像路径
        dst_dir: 目标目录路径
        
    返回：
        保存的图像副本路径，保存失败则返回None
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    try:
        # 如果源文件和目标文件相同，则避免复制
        if src.resolve() != target.resolve():
            target.write_bytes(src.read_bytes())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to copy image %s -> %s: %s", src, target, exc)
        return None
    return target


def process_batch_images(
    paths: List[Path], 
    max_edge: int = MAX_EDGE,
    normalize_pixels: bool = False,
    pixel_method: str = "0-1"
) -> List[Tuple[Path, Optional[Union[Image.Image, np.ndarray]]]]:
    """批量处理多个图像。
    
    参数：
        paths: 要处理的图像路径列表
        max_edge: 宽度或高度的最大允许尺寸
        normalize_pixels: 是否归一化像素值
        pixel_method: 如果normalize_pixels为True，则为像素归一化方法
        
    返回：
        包含(image_path, processed_image)的元组列表。
        根据normalize_pixels标志，processed_image可以是PIL Image或numpy数组。
        失败的图像返回None。
        
    遵循规则（data_processing规范）：
        - 批量处理支持
        - 一致的尺寸归一化
        - 可选的像素值归一化
        - 带日志记录的错误处理
    """
    results = []
    for path in paths:
        image = _load_image(path)
        if image is None:
            results.append((path, None))
            continue
        
        # 归一化图像尺寸
        image = normalize_image_size(image, max_edge)
        
        # 可选地归一化像素值
        if normalize_pixels:
            try:
                processed = normalize_image_pixels(image, pixel_method)
                results.append((path, processed))
            except ValueError as exc:
                logger.warning("Failed to normalize pixels for %s: %s", path, exc)
                results.append((path, None))
        else:
            results.append((path, image))
    
    return results


__all__ = [
    "image_to_base64",
    "save_image_copy",
    "normalize_image_size",
    "normalize_image_pixels",
    "process_batch_images"
]

