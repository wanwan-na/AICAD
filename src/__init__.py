"""
AI_CAD - AI生成Mesh到参数化CAD转换工具

模块结构:
- preprocessor: 网格预处理（清理、修复、简化）
- retopology: 重拓扑处理（三角面→四边形）
- utils: 工具函数
"""

__version__ = "0.1.0"
__author__ = "AI_CAD Team"

from .preprocessor import MeshPreprocessor
from .retopology import RetopologyProcessor

__all__ = [
    "MeshPreprocessor",
    "RetopologyProcessor",
]
