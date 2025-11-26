"""
网格文件读写工具

支持格式: OBJ, STL, PLY, GLB/GLTF
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import pymeshlab
except ImportError:
    pymeshlab = None


class MeshIO:
    """网格文件输入输出处理器"""

    SUPPORTED_FORMATS = {
        'input': ['.obj', '.stl', '.ply', '.glb', '.gltf', '.off'],
        'output': ['.obj', '.stl', '.ply', '.off']
    }

    @classmethod
    def load_mesh(cls, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载网格文件

        Args:
            file_path: 网格文件路径

        Returns:
            vertices: 顶点数组 (N, 3)
            faces: 面索引数组 (M, 3) 或 (M, 4)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in cls.SUPPORTED_FORMATS['input']:
            raise ValueError(f"不支持的输入格式: {suffix}")

        if trimesh is None:
            raise ImportError("请安装 trimesh: pip install trimesh")

        mesh = trimesh.load(str(file_path), force='mesh')

        return np.array(mesh.vertices), np.array(mesh.faces)

    @classmethod
    def save_mesh(cls, file_path: str,
                  vertices: np.ndarray,
                  faces: np.ndarray,
                  vertex_normals: Optional[np.ndarray] = None) -> str:
        """
        保存网格文件

        Args:
            file_path: 输出文件路径
            vertices: 顶点数组
            faces: 面索引数组
            vertex_normals: 顶点法线（可选）

        Returns:
            保存的文件路径
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix not in cls.SUPPORTED_FORMATS['output']:
            raise ValueError(f"不支持的输出格式: {suffix}")

        # 确保输出目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if trimesh is None:
            raise ImportError("请安装 trimesh: pip install trimesh")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if vertex_normals is not None:
            mesh.vertex_normals = vertex_normals

        mesh.export(str(file_path))

        return str(file_path)

    @classmethod
    def get_mesh_info(cls, file_path: str) -> Dict[str, Any]:
        """
        获取网格文件基本信息

        Args:
            file_path: 网格文件路径

        Returns:
            包含网格信息的字典
        """
        vertices, faces = cls.load_mesh(file_path)

        # 计算包围盒
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_diagonal = np.linalg.norm(bbox_size)

        # 分析面类型
        face_sizes = np.array([len(f) for f in faces]) if faces.ndim == 1 else np.full(len(faces), faces.shape[1])
        tri_count = np.sum(face_sizes == 3)
        quad_count = np.sum(face_sizes == 4)

        return {
            'file_path': str(file_path),
            'vertex_count': len(vertices),
            'face_count': len(faces),
            'triangle_count': int(tri_count),
            'quad_count': int(quad_count),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'bbox_size': bbox_size.tolist(),
            'bbox_diagonal': float(bbox_diagonal),
            'center': ((bbox_min + bbox_max) / 2).tolist()
        }

    @classmethod
    def convert_format(cls, input_path: str, output_path: str) -> str:
        """
        转换网格格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        vertices, faces = cls.load_mesh(input_path)
        return cls.save_mesh(output_path, vertices, faces)
