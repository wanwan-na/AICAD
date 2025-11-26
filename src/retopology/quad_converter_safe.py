"""
安全四边形转换模块

保证转换过程不会引入非流形边

核心策略：
1. 只配对共享边的相邻三角形
2. 配对前验证不会产生非流形边
3. 保守的质量阈值
4. 跳过任何可能引入问题的配对
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class SafeQuadConfig:
    """安全四边形转换配置"""
    # 质量阈值
    min_quality: float = 0.1
    max_angle_deviation: float = np.pi * 0.6  # 108度 (更保守)
    max_diagonal_ratio: float = 5.0

    # 是否启用多轮配对
    multi_pass: bool = True

    # 是否验证非流形
    check_non_manifold: bool = True


class SafeQuadConverter:
    """
    安全四边形转换器

    保证不会引入非流形边
    """

    def __init__(self, config: Optional[SafeQuadConfig] = None):
        self.config = config or SafeQuadConfig()

    def convert(self, vertices: np.ndarray,
                faces: np.ndarray) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
        """
        安全地转换三角面网格为四边形网格

        保证：
        1. 不引入非流形边
        2. 不改变边界
        3. 保持拓扑正确

        Args:
            vertices: 顶点数组 (N, 3)
            faces: 三角面数组 (M, 3)

        Returns:
            vertices: 顶点数组（不变）
            quad_faces: 四边形面列表
            tri_faces: 剩余三角形面列表
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)

        # 构建边-面邻接
        edge_to_faces = self._build_edge_adjacency(faces)

        # 找到所有可配对的三角形对
        candidate_pairs = self._find_candidate_pairs(vertices, faces, edge_to_faces)

        # 贪心配对，优先高质量对
        quad_faces, used_faces = self._greedy_pairing(
            vertices, faces, candidate_pairs, edge_to_faces
        )

        # 收集剩余三角形
        tri_faces = [list(faces[i]) for i in range(len(faces)) if i not in used_faces]

        # 如果启用多轮，尝试再次配对剩余的
        if self.config.multi_pass and len(tri_faces) > 1:
            remaining = np.array(tri_faces)
            edge_to_faces2 = self._build_edge_adjacency(remaining)
            pairs2 = self._find_candidate_pairs(vertices, remaining, edge_to_faces2)

            if pairs2:
                quads2, used2 = self._greedy_pairing(
                    vertices, remaining, pairs2, edge_to_faces2
                )
                quad_faces.extend(quads2)
                tri_faces = [list(remaining[i]) for i in range(len(remaining)) if i not in used2]

        return vertices, quad_faces, tri_faces

    def _build_edge_adjacency(self, faces: np.ndarray) -> Dict[tuple, List[int]]:
        """构建边-面邻接"""
        edge_to_faces = defaultdict(list)
        for idx, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_to_faces[edge].append(idx)
        return edge_to_faces

    def _find_candidate_pairs(self, vertices: np.ndarray,
                              faces: np.ndarray,
                              edge_to_faces: Dict) -> List[Tuple[int, int, tuple, float]]:
        """
        找到所有候选配对

        只返回：
        1. 共享恰好一条边的两个三角形
        2. 配对后形成有效凸四边形
        3. 满足质量要求
        """
        pairs = []

        for edge, face_indices in edge_to_faces.items():
            # 只处理恰好有两个面共享的边（内部边）
            if len(face_indices) != 2:
                continue

            f1_idx, f2_idx = face_indices
            f1, f2 = faces[f1_idx], faces[f2_idx]

            # 创建四边形
            quad = self._make_quad(f1, f2, edge)
            if quad is None:
                continue

            # 检查四边形有效性
            if not self._is_valid_quad(vertices, quad):
                continue

            # 计算质量
            quality = self._compute_quad_quality(vertices, quad)
            if quality < self.config.min_quality:
                continue

            pairs.append((f1_idx, f2_idx, edge, quality))

        # 按质量排序
        pairs.sort(key=lambda x: -x[3])

        return pairs

    def _greedy_pairing(self, vertices: np.ndarray,
                        faces: np.ndarray,
                        pairs: List[Tuple[int, int, tuple, float]],
                        edge_to_faces: Dict) -> Tuple[List[List[int]], Set[int]]:
        """
        贪心配对

        每次选择最高质量的可用对，同时检查不会引入非流形边

        关键：当两个三角形配对成四边形时：
        - 共享边被"消费"掉（不再是边界）
        - 四边形的4条边中，2条来自三角形1，2条来自三角形2
        - 这4条边各自已经被其相邻三角形使用1次
        - 配对不会增加边的使用次数（只是改变面的形状）
        """
        used_faces = set()
        quad_faces = []

        # 跟踪哪些边已经被"锁定"（其相邻三角形已被配对）
        locked_edges = set()

        for f1_idx, f2_idx, shared_edge, quality in pairs:
            # 检查面是否已被使用
            if f1_idx in used_faces or f2_idx in used_faces:
                continue

            # 创建四边形
            f1, f2 = faces[f1_idx], faces[f2_idx]
            quad = self._make_quad(f1, f2, shared_edge)

            if quad is None:
                continue

            # 检查非流形：确保不会与已配对的面产生冲突
            if self.config.check_non_manifold:
                # 获取两个三角形的所有边（除了共享边）
                f1_edges = set()
                for i in range(3):
                    e = tuple(sorted([f1[i], f1[(i + 1) % 3]]))
                    if e != shared_edge:
                        f1_edges.add(e)

                f2_edges = set()
                for i in range(3):
                    e = tuple(sorted([f2[i], f2[(i + 1) % 3]]))
                    if e != shared_edge:
                        f2_edges.add(e)

                # 检查这些边是否与已锁定的边冲突
                # 边冲突：如果一条边的两个相邻三角形都被不同的四边形使用
                will_cause_conflict = False
                for e in f1_edges | f2_edges:
                    if e in locked_edges:
                        # 这条边的另一个相邻三角形已被配对
                        # 检查是否会产生非流形
                        adjacent_faces = edge_to_faces.get(e, [])
                        if len(adjacent_faces) == 2:
                            # 内部边：两个面共享
                            # 如果另一个面已被配对到其他四边形，可能产生问题
                            other_face = [f for f in adjacent_faces if f != f1_idx and f != f2_idx]
                            if other_face and other_face[0] in used_faces:
                                will_cause_conflict = True
                                break

                if will_cause_conflict:
                    continue

                # 锁定这些边
                for e in f1_edges | f2_edges:
                    locked_edges.add(e)

            # 标记为已使用
            used_faces.add(f1_idx)
            used_faces.add(f2_idx)
            quad_faces.append(quad)

        return quad_faces, used_faces

    def _make_quad(self, f1: np.ndarray, f2: np.ndarray, edge: tuple) -> Optional[List[int]]:
        """从两个三角形创建四边形"""
        e0, e1 = edge

        # 找到非共享顶点
        other1 = [v for v in f1 if v not in edge]
        other2 = [v for v in f2 if v not in edge]

        if len(other1) != 1 or len(other2) != 1:
            return None

        other1 = other1[0]
        other2 = other2[0]

        # 确定顶点顺序
        f1_list = list(f1)
        idx = f1_list.index(other1)
        next_v = f1_list[(idx + 1) % 3]

        if next_v == e0:
            return [other1, e0, other2, e1]
        else:
            return [other1, e1, other2, e0]

    def _is_valid_quad(self, vertices: np.ndarray, quad: List[int]) -> bool:
        """检查四边形是否有效"""
        # 检查顶点是否唯一
        if len(set(quad)) != 4:
            return False

        v = vertices[quad]

        # 检查是否为凸四边形
        if not self._is_convex(v):
            return False

        # 检查面积是否为正
        area = self._compute_area(v)
        if area < 1e-10:
            return False

        return True

    def _is_convex(self, v: np.ndarray) -> bool:
        """检查是否为凸四边形"""
        # 计算每个顶点处的叉积方向
        cross_signs = []
        for i in range(4):
            v0 = v[i]
            v1 = v[(i + 1) % 4]
            v2 = v[(i + 2) % 4]

            e1 = v1 - v0
            e2 = v2 - v1
            cross = np.cross(e1, e2)

            # 取主方向的符号
            norm = np.linalg.norm(cross)
            if norm > 1e-10:
                # 使用z分量或最大分量
                max_idx = np.argmax(np.abs(cross))
                cross_signs.append(np.sign(cross[max_idx]))

        if len(cross_signs) < 4:
            return True  # 退化情况，允许

        # 所有符号相同则为凸
        return all(s == cross_signs[0] for s in cross_signs)

    def _compute_area(self, v: np.ndarray) -> float:
        """计算四边形面积"""
        # 分成两个三角形
        area1 = 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
        area2 = 0.5 * np.linalg.norm(np.cross(v[2] - v[0], v[3] - v[0]))
        return area1 + area2

    def _compute_quad_quality(self, vertices: np.ndarray, quad: List[int]) -> float:
        """计算四边形质量"""
        v = vertices[quad]

        # 角度评分：接近90度
        angles = self._compute_angles(v)
        target = np.pi / 2
        angle_devs = [abs(a - target) for a in angles]
        max_dev = max(angle_devs)

        if max_dev > self.config.max_angle_deviation:
            return 0

        angle_score = 1.0 - (np.mean(angle_devs) / target)

        # 对角线比例评分
        d1 = np.linalg.norm(v[2] - v[0])
        d2 = np.linalg.norm(v[3] - v[1])
        ratio = max(d1, d2) / (min(d1, d2) + 1e-10)

        if ratio > self.config.max_diagonal_ratio:
            return 0

        ratio_score = 1.0 - (ratio - 1) / (self.config.max_diagonal_ratio - 1)

        # 边长均匀性
        edges = [
            np.linalg.norm(v[(i + 1) % 4] - v[i])
            for i in range(4)
        ]
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        edge_score = 1.0 - min(edge_std / (edge_mean + 1e-10), 1.0)

        # 综合
        return 0.4 * angle_score + 0.3 * ratio_score + 0.3 * edge_score

    def _compute_angles(self, v: np.ndarray) -> List[float]:
        """计算四个角"""
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


def convert_to_quad_safe(vertices: np.ndarray,
                         faces: np.ndarray) -> Dict[str, Any]:
    """
    安全四边形转换便捷函数

    Args:
        vertices: 顶点数组
        faces: 三角面数组

    Returns:
        转换结果字典
    """
    converter = SafeQuadConverter()
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
    }
