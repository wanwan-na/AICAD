"""
四边形转换模块 V2 - 高效版

目标: 达到 80%+ 四边形比例

改进策略:
1. 更激进的三角形配对（放宽质量阈值）
2. 多轮迭代配对
3. 对奇数三角形区域进行特殊处理
4. 边界三角形保留策略优化
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class QuadConversionConfigV2:
    """四边形转换配置 V2"""
    # 第一轮配对 - 宽松条件
    first_pass_min_quality: float = 0.05
    first_pass_max_angle: float = np.pi * 0.7  # 126度

    # 第二轮配对 - 更宽松
    second_pass_min_quality: float = 0.01
    second_pass_max_angle: float = np.pi * 0.85  # 153度

    # 最大对角线比例
    max_diagonal_ratio: float = 5.0

    # 是否强制处理所有可配对三角形
    aggressive_mode: bool = True


class QuadConverterV2:
    """
    高效四边形转换器 V2

    使用多轮配对策略，目标达到80%+四边形比例
    """

    def __init__(self, config: Optional[QuadConversionConfigV2] = None):
        self.config = config or QuadConversionConfigV2()

    def convert(self, vertices: np.ndarray,
                faces: np.ndarray) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
        """
        转换三角面网格为四边形网格

        Args:
            vertices: 顶点数组 (N, 3)
            faces: 三角面数组 (M, 3)

        Returns:
            vertices: 顶点数组
            quad_faces: 四边形面列表
            tri_faces: 剩余三角形面列表
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)

        all_quads = []
        remaining_faces = faces.copy()

        # 第一轮：高质量配对
        quads1, remaining_faces = self._pair_pass(
            vertices, remaining_faces,
            min_quality=self.config.first_pass_min_quality,
            max_angle=self.config.first_pass_max_angle
        )
        all_quads.extend(quads1)

        # 第二轮：宽松配对
        if len(remaining_faces) > 0:
            quads2, remaining_faces = self._pair_pass(
                vertices, remaining_faces,
                min_quality=self.config.second_pass_min_quality,
                max_angle=self.config.second_pass_max_angle
            )
            all_quads.extend(quads2)

        # 第三轮：激进模式 - 几乎配对所有可能的三角形
        if self.config.aggressive_mode and len(remaining_faces) > 0:
            quads3, remaining_faces = self._aggressive_pair_pass(vertices, remaining_faces)
            all_quads.extend(quads3)

        # 转换剩余三角形为列表格式
        remaining_tris = [list(f) for f in remaining_faces]

        return vertices, all_quads, remaining_tris

    def _pair_pass(self, vertices: np.ndarray,
                   faces: np.ndarray,
                   min_quality: float,
                   max_angle: float) -> Tuple[List[List[int]], np.ndarray]:
        """
        执行一轮配对

        Returns:
            quads: 生成的四边形列表
            remaining: 剩余的三角形数组
        """
        if len(faces) == 0:
            return [], faces

        # 构建边-面邻接关系
        edge_to_faces = self._build_edge_adjacency(faces)

        # 找到所有可配对的三角形对，按质量排序
        pairs = []
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) != 2:
                continue

            f1_idx, f2_idx = face_indices
            quality = self._evaluate_pair(
                vertices, faces[f1_idx], faces[f2_idx], edge,
                min_quality, max_angle
            )

            if quality > 0:
                # 使用负质量作为优先级（堆是最小堆）
                heapq.heappush(pairs, (-quality, f1_idx, f2_idx, edge))

        # 贪心配对
        used = set()
        quads = []

        while pairs:
            neg_quality, f1_idx, f2_idx, edge = heapq.heappop(pairs)

            if f1_idx in used or f2_idx in used:
                continue

            used.add(f1_idx)
            used.add(f2_idx)

            # 创建四边形
            quad = self._make_quad(faces[f1_idx], faces[f2_idx], edge)
            if quad:
                quads.append(quad)

        # 收集未使用的三角形
        remaining = np.array([faces[i] for i in range(len(faces)) if i not in used])

        return quads, remaining

    def _aggressive_pair_pass(self, vertices: np.ndarray,
                              faces: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        激进配对 - 尽可能配对所有三角形

        放宽所有质量限制，只要能形成有效四边形就配对
        """
        if len(faces) == 0:
            return [], faces

        edge_to_faces = self._build_edge_adjacency(faces)

        pairs = []
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) != 2:
                continue

            f1_idx, f2_idx = face_indices

            # 只检查是否能形成有效四边形（不检查质量）
            quad = self._make_quad(faces[f1_idx], faces[f2_idx], edge)
            if quad and self._is_valid_quad(vertices, quad):
                pairs.append((f1_idx, f2_idx, edge))

        # 随机打乱以获得更均匀的分布
        np.random.shuffle(pairs)

        used = set()
        quads = []

        for f1_idx, f2_idx, edge in pairs:
            if f1_idx in used or f2_idx in used:
                continue

            used.add(f1_idx)
            used.add(f2_idx)

            quad = self._make_quad(faces[f1_idx], faces[f2_idx], edge)
            if quad:
                quads.append(quad)

        remaining = np.array([faces[i] for i in range(len(faces)) if i not in used])

        return quads, remaining

    def _build_edge_adjacency(self, faces: np.ndarray) -> Dict[tuple, List[int]]:
        """构建边-面邻接"""
        edge_to_faces = defaultdict(list)

        for idx, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_to_faces[edge].append(idx)

        return edge_to_faces

    def _evaluate_pair(self, vertices: np.ndarray,
                       f1: np.ndarray, f2: np.ndarray,
                       edge: tuple,
                       min_quality: float,
                       max_angle: float) -> float:
        """评估配对质量，返回0表示不可配对"""
        quad = self._make_quad(f1, f2, edge)
        if quad is None:
            return 0

        v = vertices[quad]

        # 检查凸性
        if not self._is_convex(v):
            return 0

        # 计算角度
        angles = self._compute_angles(v)
        max_dev = max(abs(a - np.pi / 2) for a in angles)
        if max_dev > max_angle:
            return 0

        # 检查对角线比例
        d1 = np.linalg.norm(v[2] - v[0])
        d2 = np.linalg.norm(v[3] - v[1])
        ratio = max(d1, d2) / (min(d1, d2) + 1e-10)
        if ratio > self.config.max_diagonal_ratio:
            return 0

        # 计算质量分数
        angle_score = 1.0 - max_dev / max_angle
        ratio_score = 1.0 - (ratio - 1) / (self.config.max_diagonal_ratio - 1)

        quality = 0.6 * angle_score + 0.4 * ratio_score

        return quality if quality >= min_quality else 0

    def _make_quad(self, f1: np.ndarray, f2: np.ndarray, edge: tuple) -> Optional[List[int]]:
        """从两个三角形创建四边形"""
        e0, e1 = edge

        # 找非共享顶点
        other1 = [v for v in f1 if v not in edge]
        other2 = [v for v in f2 if v not in edge]

        if len(other1) != 1 or len(other2) != 1:
            return None

        other1 = other1[0]
        other2 = other2[0]

        # 确定顺序
        f1_list = list(f1)
        idx = f1_list.index(other1)
        next_v = f1_list[(idx + 1) % 3]

        if next_v == e0:
            return [other1, e0, other2, e1]
        else:
            return [other1, e1, other2, e0]

    def _is_valid_quad(self, vertices: np.ndarray, quad: List[int]) -> bool:
        """检查四边形是否有效"""
        v = vertices[quad]

        # 检查面积不为0
        area = 0
        for i in range(4):
            j = (i + 1) % 4
            area += v[i][0] * v[j][1] - v[j][0] * v[i][1]

        return abs(area) > 1e-10

    def _is_convex(self, v: np.ndarray) -> bool:
        """
        检查是否为凸四边形（放宽检查）

        允许轻微的凹陷，因为在曲面上完美凸四边形很难实现
        """
        cross_products = []
        for i in range(4):
            v0 = v[i]
            v1 = v[(i + 1) % 4]
            v2 = v[(i + 2) % 4]

            e1 = v1 - v0
            e2 = v2 - v1
            cross = np.cross(e1, e2)
            cross_products.append(cross)

        # 计算法线一致性（允许一定偏差）
        if len(cross_products) < 4:
            return True

        # 检查法线方向是否大致一致
        normals = [c / (np.linalg.norm(c) + 1e-10) for c in cross_products]

        # 计算相邻法线的点积，允许一定的负值（轻微凹陷）
        min_dot = 1.0
        for i in range(4):
            dot = np.dot(normals[i], normals[(i + 1) % 4])
            min_dot = min(min_dot, dot)

        # 允许法线偏差达到120度（cos(120°) = -0.5）
        return min_dot > -0.5

    def _compute_angles(self, v: np.ndarray) -> List[float]:
        """计算四边形四个角"""
        angles = []
        for i in range(4):
            p0 = v[(i - 1) % 4]
            p1 = v[i]
            p2 = v[(i + 1) % 4]

            e1 = p0 - p1
            e2 = p2 - p1

            cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
            angles.append(np.arccos(np.clip(cos_a, -1, 1)))

        return angles


def convert_to_quad_mesh_v2(vertices: np.ndarray,
                            faces: np.ndarray,
                            aggressive: bool = True) -> Dict[str, Any]:
    """
    高效四边形转换（V2版本）

    Args:
        vertices: 顶点数组
        faces: 三角面数组
        aggressive: 是否使用激进模式

    Returns:
        转换结果字典
    """
    config = QuadConversionConfigV2(aggressive_mode=aggressive)
    converter = QuadConverterV2(config)

    verts, quads, tris = converter.convert(vertices, faces)

    total = len(quads) + len(tris)
    quad_ratio = len(quads) / total if total > 0 else 0

    return {
        'vertices': verts,
        'quad_faces': quads,
        'tri_faces': tris,
        'quad_count': len(quads),
        'tri_count': len(tris),
        'quad_ratio': quad_ratio,
        'original_tri_count': len(faces)
    }
