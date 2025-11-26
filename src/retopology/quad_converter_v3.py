"""
四边形转换模块 V3 - 边界处理增强版

目标: 达到 80%+ 四边形比例

V3改进:
1. 继承V2的多轮配对策略
2. 新增: 边界三角形链处理
3. 新增: 三角形扇形区域转四边形
4. 新增: 孤立三角形顶点合并

核心思想: 对于无法直接配对的边界三角形，通过几何分析找到可以组合的模式
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class QuadConversionConfigV3:
    """四边形转换配置 V3"""
    # 继承V2的配对参数
    first_pass_min_quality: float = 0.05
    first_pass_max_angle: float = np.pi * 0.75  # 135度
    second_pass_min_quality: float = 0.01
    second_pass_max_angle: float = np.pi * 0.9  # 162度
    max_diagonal_ratio: float = 8.0
    aggressive_mode: bool = True

    # V3新参数
    enable_boundary_chains: bool = True  # 启用边界链处理
    enable_fan_conversion: bool = True   # 启用扇形区域转换
    min_chain_length: int = 2            # 最小边界链长度
    max_vertex_valence: int = 8          # 顶点最大价数


class QuadConverterV3:
    """
    高效四边形转换器 V3 - 边界处理增强版

    在V2基础上增加对边界三角形的特殊处理，目标达到80%+四边形比例
    """

    def __init__(self, config: Optional[QuadConversionConfigV3] = None):
        self.config = config or QuadConversionConfigV3()

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

        # 第三轮：激进配对
        if self.config.aggressive_mode and len(remaining_faces) > 0:
            quads3, remaining_faces = self._aggressive_pair_pass(vertices, remaining_faces)
            all_quads.extend(quads3)

        # 第四轮（V3新增）：边界链处理
        if self.config.enable_boundary_chains and len(remaining_faces) > 0:
            quads4, remaining_faces = self._boundary_chain_pass(vertices, remaining_faces)
            all_quads.extend(quads4)

        # 第五轮（V3新增）：扇形区域处理
        if self.config.enable_fan_conversion and len(remaining_faces) > 0:
            quads5, remaining_faces = self._fan_conversion_pass(vertices, remaining_faces)
            all_quads.extend(quads5)

        # 转换剩余三角形为列表格式
        remaining_tris = [list(f) for f in remaining_faces]

        return vertices, all_quads, remaining_tris

    def _pair_pass(self, vertices: np.ndarray,
                   faces: np.ndarray,
                   min_quality: float,
                   max_angle: float) -> Tuple[List[List[int]], np.ndarray]:
        """执行一轮配对"""
        if len(faces) == 0:
            return [], faces

        edge_to_faces = self._build_edge_adjacency(faces)

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
                heapq.heappush(pairs, (-quality, f1_idx, f2_idx, edge))

        used = set()
        quads = []

        while pairs:
            neg_quality, f1_idx, f2_idx, edge = heapq.heappop(pairs)

            if f1_idx in used or f2_idx in used:
                continue

            used.add(f1_idx)
            used.add(f2_idx)

            quad = self._make_quad(faces[f1_idx], faces[f2_idx], edge)
            if quad:
                quads.append(quad)

        remaining = np.array([faces[i] for i in range(len(faces)) if i not in used])

        return quads, remaining

    def _aggressive_pair_pass(self, vertices: np.ndarray,
                              faces: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """激进配对"""
        if len(faces) == 0:
            return [], faces

        edge_to_faces = self._build_edge_adjacency(faces)

        pairs = []
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) != 2:
                continue

            f1_idx, f2_idx = face_indices
            quad = self._make_quad(faces[f1_idx], faces[f2_idx], edge)
            if quad and self._is_valid_quad(vertices, quad):
                pairs.append((f1_idx, f2_idx, edge))

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

    def _boundary_chain_pass(self, vertices: np.ndarray,
                             faces: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        边界链处理

        找到相邻的边界三角形链，尝试将偶数个三角形配对
        """
        if len(faces) < 2:
            return [], faces

        # 构建边-面邻接
        edge_to_faces = self._build_edge_adjacency(faces)

        # 找到所有边界边（只属于一个面的边）
        boundary_edges = {edge: flist[0] for edge, flist in edge_to_faces.items() if len(flist) == 1}

        # 构建面的边界边邻接
        face_boundary_edges = defaultdict(list)
        for edge, face_idx in boundary_edges.items():
            face_boundary_edges[face_idx].append(edge)

        # 找到共享顶点的边界三角形对
        vertex_to_boundary_faces = defaultdict(set)
        for face_idx, edges in face_boundary_edges.items():
            for edge in edges:
                for v in edge:
                    vertex_to_boundary_faces[v].add(face_idx)

        # 尝试配对共享边界顶点但不共享内部边的三角形
        used = set()
        quads = []

        for vertex, face_set in vertex_to_boundary_faces.items():
            face_list = [f for f in face_set if f not in used]
            if len(face_list) < 2:
                continue

            # 对于共享顶点的多个边界三角形，检查是否可以形成四边形
            for i in range(len(face_list)):
                if face_list[i] in used:
                    continue
                for j in range(i + 1, len(face_list)):
                    if face_list[j] in used:
                        continue

                    f1_idx, f2_idx = face_list[i], face_list[j]
                    f1, f2 = faces[f1_idx], faces[f2_idx]

                    # 检查是否共享边（如果共享边应该已经在前面配对了）
                    shared = set(f1) & set(f2)
                    if len(shared) == 2:  # 共享边
                        continue
                    if len(shared) != 1:  # 必须共享一个顶点
                        continue

                    shared_v = list(shared)[0]

                    # 尝试创建四边形（需要找到连接两个三角形的方式）
                    quad = self._try_make_quad_from_shared_vertex(
                        vertices, f1, f2, shared_v
                    )

                    if quad and self._is_valid_quad(vertices, quad):
                        quads.append(quad)
                        used.add(f1_idx)
                        used.add(f2_idx)
                        break

        remaining = np.array([faces[i] for i in range(len(faces)) if i not in used])

        return quads, remaining

    def _fan_conversion_pass(self, vertices: np.ndarray,
                             faces: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        扇形区域处理

        找到以某个顶点为中心的三角形扇形，将其转换为四边形
        """
        if len(faces) < 3:
            return [], faces

        # 构建顶点-面邻接
        vertex_to_faces = defaultdict(list)
        for idx, face in enumerate(faces):
            for v in face:
                vertex_to_faces[v].append(idx)

        # 找到高价顶点（连接多个三角形）
        used = set()
        quads = []

        # 按价数排序，优先处理高价顶点
        sorted_vertices = sorted(vertex_to_faces.keys(),
                                key=lambda v: len(vertex_to_faces[v]),
                                reverse=True)

        for center_v in sorted_vertices:
            face_indices = [f for f in vertex_to_faces[center_v] if f not in used]

            if len(face_indices) < 4:
                continue
            if len(face_indices) > self.config.max_vertex_valence:
                continue

            # 尝试将4个相邻三角形转换为2个四边形
            result = self._try_fan_to_quads(vertices, faces, center_v, face_indices)

            if result:
                new_quads, converted_faces = result
                quads.extend(new_quads)
                used.update(converted_faces)

        remaining = np.array([faces[i] for i in range(len(faces)) if i not in used])

        return quads, remaining

    def _try_fan_to_quads(self, vertices: np.ndarray, faces: np.ndarray,
                         center_v: int, face_indices: List[int]) -> Optional[Tuple[List[List[int]], Set[int]]]:
        """
        尝试将扇形区域的三角形转换为四边形

        策略：找到4个连续的三角形，转换为2个四边形
        """
        if len(face_indices) < 4:
            return None

        # 获取所有围绕center_v的三角形
        fan_faces = [faces[i] for i in face_indices]

        # 获取每个三角形的另外两个顶点
        other_vertices = []
        for face in fan_faces:
            others = [v for v in face if v != center_v]
            if len(others) == 2:
                other_vertices.append(others)

        if len(other_vertices) < 4:
            return None

        # 尝试找到4个连续三角形配对
        # 简化处理：只取前4个
        if len(other_vertices) >= 4:
            # 创建两个四边形
            v1, v2 = other_vertices[0]
            v3, v4 = other_vertices[1]
            v5, v6 = other_vertices[2]
            v7, v8 = other_vertices[3]

            # 检查共享边
            if v2 == v3:
                quad1 = [v1, center_v, v4, v2]
                if self._is_valid_quad(vertices, quad1):
                    return [quad1], {face_indices[0], face_indices[1]}

        return None

    def _try_make_quad_from_shared_vertex(self, vertices: np.ndarray,
                                          f1: np.ndarray, f2: np.ndarray,
                                          shared_v: int) -> Optional[List[int]]:
        """
        从共享单个顶点的两个三角形创建四边形

        这需要找到一个合理的顶点顺序
        """
        # 获取非共享顶点
        f1_others = [v for v in f1 if v != shared_v]
        f2_others = [v for v in f2 if v != shared_v]

        if len(f1_others) != 2 or len(f2_others) != 2:
            return None

        # 尝试不同的顶点排列组合
        for a, b in [(0, 1), (1, 0)]:
            for c, d in [(0, 1), (1, 0)]:
                # 四边形顶点顺序
                quad = [f1_others[a], shared_v, f2_others[c], f2_others[d]]

                # 检查是否所有顶点都不同
                if len(set(quad)) != 4:
                    continue

                if self._is_valid_quad(vertices, quad):
                    return quad

        return None

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
        """评估配对质量"""
        quad = self._make_quad(f1, f2, edge)
        if quad is None:
            return 0

        v = vertices[quad]

        if not self._is_convex(v):
            return 0

        angles = self._compute_angles(v)
        max_dev = max(abs(a - np.pi / 2) for a in angles)
        if max_dev > max_angle:
            return 0

        d1 = np.linalg.norm(v[2] - v[0])
        d2 = np.linalg.norm(v[3] - v[1])
        ratio = max(d1, d2) / (min(d1, d2) + 1e-10)
        if ratio > self.config.max_diagonal_ratio:
            return 0

        angle_score = 1.0 - max_dev / max_angle
        ratio_score = 1.0 - (ratio - 1) / (self.config.max_diagonal_ratio - 1)

        quality = 0.6 * angle_score + 0.4 * ratio_score

        return quality if quality >= min_quality else 0

    def _make_quad(self, f1: np.ndarray, f2: np.ndarray, edge: tuple) -> Optional[List[int]]:
        """从两个三角形创建四边形"""
        e0, e1 = edge

        other1 = [v for v in f1 if v not in edge]
        other2 = [v for v in f2 if v not in edge]

        if len(other1) != 1 or len(other2) != 1:
            return None

        other1 = other1[0]
        other2 = other2[0]

        f1_list = list(f1)
        idx = f1_list.index(other1)
        next_v = f1_list[(idx + 1) % 3]

        if next_v == e0:
            return [other1, e0, other2, e1]
        else:
            return [other1, e1, other2, e0]

    def _is_valid_quad(self, vertices: np.ndarray, quad: List[int]) -> bool:
        """检查四边形是否有效"""
        if len(set(quad)) != 4:
            return False

        v = vertices[quad]

        # 检查面积不为0
        area = 0
        for i in range(4):
            j = (i + 1) % 4
            area += v[i][0] * v[j][1] - v[j][0] * v[i][1]

        if abs(area) < 1e-10:
            return False

        # 检查凸性（放宽）
        return self._is_convex(v)

    def _is_convex(self, v: np.ndarray) -> bool:
        """检查是否为凸四边形（放宽检查）"""
        cross_products = []
        for i in range(4):
            v0 = v[i]
            v1 = v[(i + 1) % 4]
            v2 = v[(i + 2) % 4]

            e1 = v1 - v0
            e2 = v2 - v1
            cross = np.cross(e1, e2)
            cross_products.append(cross)

        if len(cross_products) < 4:
            return True

        normals = [c / (np.linalg.norm(c) + 1e-10) for c in cross_products]

        min_dot = 1.0
        for i in range(4):
            dot = np.dot(normals[i], normals[(i + 1) % 4])
            min_dot = min(min_dot, dot)

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


def convert_to_quad_mesh_v3(vertices: np.ndarray,
                            faces: np.ndarray,
                            aggressive: bool = True) -> Dict[str, Any]:
    """
    高效四边形转换（V3版本 - 边界增强）

    Args:
        vertices: 顶点数组
        faces: 三角面数组
        aggressive: 是否使用激进模式

    Returns:
        转换结果字典
    """
    config = QuadConversionConfigV3(
        aggressive_mode=aggressive,
        enable_boundary_chains=True,
        enable_fan_conversion=True
    )
    converter = QuadConverterV3(config)

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
