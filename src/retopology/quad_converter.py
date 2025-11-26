"""
四边形转换模块

将三角面网格转换为四边形网格的核心算法

方法:
1. 三角形配对法 (Triangle Pairing) - 将相邻三角形配对成四边形
2. 基于边的方法 - 选择最佳边进行合并

输出格式: 混合网格 (四边形 + 少量三角形)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class QuadConversionConfig:
    """四边形转换配置"""
    # 最小四边形质量阈值 (0-1, 越高要求越严格)
    min_quad_quality: float = 0.3

    # 最大对角线比例 (四边形两条对角线的比值上限)
    max_diagonal_ratio: float = 3.0

    # 最大角度偏差 (与90度的最大偏差，弧度)
    max_angle_deviation: float = np.pi / 3  # 60度

    # 是否保留边界三角形
    preserve_boundary: bool = True


class QuadConverter:
    """
    四边形转换器

    使用三角形配对算法将三角面网格转换为四边形主导的网格

    使用示例:
        converter = QuadConverter()
        quad_verts, quad_faces, tri_faces = converter.convert(vertices, faces)
    """

    def __init__(self, config: Optional[QuadConversionConfig] = None):
        self.config = config or QuadConversionConfig()

    def convert(self, vertices: np.ndarray,
                faces: np.ndarray) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
        """
        转换三角面网格为四边形网格

        Args:
            vertices: 顶点数组 (N, 3)
            faces: 三角面数组 (M, 3)

        Returns:
            vertices: 顶点数组 (不变)
            quad_faces: 四边形面列表 [[v0,v1,v2,v3], ...]
            tri_faces: 剩余三角形面列表 [[v0,v1,v2], ...]
        """
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)

        # 构建边-面邻接关系
        edge_to_faces = self._build_edge_face_adjacency(faces)

        # 找到可配对的三角形对
        pairs = self._find_triangle_pairs(vertices, faces, edge_to_faces)

        # 执行配对，生成四边形
        quad_faces, remaining_tris = self._pair_triangles(vertices, faces, pairs)

        return vertices, quad_faces, remaining_tris

    def _build_edge_face_adjacency(self, faces: np.ndarray) -> Dict[tuple, List[int]]:
        """构建边到面的邻接关系"""
        edge_to_faces = defaultdict(list)

        for face_idx, face in enumerate(faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v0, v1]))
                edge_to_faces[edge].append(face_idx)

        return edge_to_faces

    def _find_triangle_pairs(self, vertices: np.ndarray,
                             faces: np.ndarray,
                             edge_to_faces: Dict) -> List[Tuple[int, int, tuple]]:
        """
        找到可以配对成四边形的三角形对

        返回: [(face_idx1, face_idx2, shared_edge), ...]
        """
        pairs = []

        for edge, face_indices in edge_to_faces.items():
            # 只处理恰好有两个面共享的边（非边界边）
            if len(face_indices) != 2:
                continue

            f1_idx, f2_idx = face_indices
            f1, f2 = faces[f1_idx], faces[f2_idx]

            # 评估配对质量
            quality = self._evaluate_pair_quality(vertices, f1, f2, edge)

            if quality >= self.config.min_quad_quality:
                pairs.append((f1_idx, f2_idx, edge, quality))

        # 按质量排序（优先配对质量高的）
        pairs.sort(key=lambda x: x[3], reverse=True)

        return [(p[0], p[1], p[2]) for p in pairs]

    def _evaluate_pair_quality(self, vertices: np.ndarray,
                               face1: np.ndarray,
                               face2: np.ndarray,
                               shared_edge: tuple) -> float:
        """
        评估两个三角形配对成四边形的质量

        质量指标:
        1. 四边形的角度接近90度
        2. 边长均匀
        3. 不会产生凹四边形
        """
        # 找到四边形的四个顶点（按顺序）
        quad_verts = self._get_quad_vertices(face1, face2, shared_edge)
        if quad_verts is None:
            return 0.0

        v = vertices[quad_verts]

        # 检查是否为凸四边形
        if not self._is_convex_quad(v):
            return 0.0

        # 计算边长
        edges = [
            np.linalg.norm(v[1] - v[0]),
            np.linalg.norm(v[2] - v[1]),
            np.linalg.norm(v[3] - v[2]),
            np.linalg.norm(v[0] - v[3])
        ]

        # 边长均匀性得分
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        edge_score = 1.0 - min(edge_std / (edge_mean + 1e-8), 1.0)

        # 角度得分（接近90度）
        angles = self._compute_quad_angles(v)
        target = np.pi / 2
        angle_deviations = [abs(a - target) for a in angles]
        max_deviation = max(angle_deviations)

        if max_deviation > self.config.max_angle_deviation:
            return 0.0

        angle_score = 1.0 - (np.mean(angle_deviations) / (np.pi / 2))

        # 对角线比例检查
        d1 = np.linalg.norm(v[2] - v[0])
        d2 = np.linalg.norm(v[3] - v[1])
        diag_ratio = max(d1, d2) / (min(d1, d2) + 1e-8)

        if diag_ratio > self.config.max_diagonal_ratio:
            return 0.0

        diag_score = 1.0 - (diag_ratio - 1.0) / (self.config.max_diagonal_ratio - 1.0)

        # 综合得分
        return 0.4 * edge_score + 0.4 * angle_score + 0.2 * diag_score

    def _get_quad_vertices(self, face1: np.ndarray,
                           face2: np.ndarray,
                           shared_edge: tuple) -> Optional[List[int]]:
        """
        获取四边形的四个顶点（按逆时针顺序）
        """
        e0, e1 = shared_edge

        # 找到非共享顶点
        other1 = [v for v in face1 if v not in shared_edge][0]
        other2 = [v for v in face2 if v not in shared_edge][0]

        # 确定顺序：从face1的非共享顶点开始，绕一圈
        # face1: other1 -> e0 -> e1 或 other1 -> e1 -> e0

        # 找到face1中other1后面的顶点
        f1_list = list(face1)
        idx = f1_list.index(other1)
        next_v = f1_list[(idx + 1) % 3]

        if next_v == e0:
            # 顺序: other1 -> e0 -> e1 -> other2
            return [other1, e0, other2, e1]
        else:
            # 顺序: other1 -> e1 -> e0 -> other2
            return [other1, e1, other2, e0]

    def _is_convex_quad(self, v: np.ndarray) -> bool:
        """检查四边形是否为凸多边形"""
        # 计算每个顶点处的叉积符号
        signs = []
        for i in range(4):
            v0 = v[i]
            v1 = v[(i + 1) % 4]
            v2 = v[(i + 2) % 4]

            e1 = v1 - v0
            e2 = v2 - v1

            cross = np.cross(e1, e2)
            # 取z分量的符号（假设大致在xy平面）或使用点积与法线
            sign = np.dot(cross, cross)  # 简化：检查是否非零
            if np.linalg.norm(cross) > 1e-10:
                signs.append(np.sign(cross[2]) if abs(cross[2]) > 1e-10 else
                           np.sign(cross[1]) if abs(cross[1]) > 1e-10 else
                           np.sign(cross[0]))

        # 所有叉积符号相同则为凸
        if len(signs) < 4:
            return True  # 退化情况
        return all(s == signs[0] for s in signs) or all(s == 0 for s in signs)

    def _compute_quad_angles(self, v: np.ndarray) -> List[float]:
        """计算四边形四个角的角度"""
        angles = []
        for i in range(4):
            v0 = v[(i - 1) % 4]
            v1 = v[i]
            v2 = v[(i + 1) % 4]

            e1 = v0 - v1
            e2 = v2 - v1

            cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        return angles

    def _pair_triangles(self, vertices: np.ndarray,
                        faces: np.ndarray,
                        pairs: List[Tuple[int, int, tuple]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        执行三角形配对，生成四边形

        使用贪心算法，每个三角形只能配对一次
        """
        used_faces: Set[int] = set()
        quad_faces = []

        for f1_idx, f2_idx, shared_edge in pairs:
            # 检查这两个面是否已被使用
            if f1_idx in used_faces or f2_idx in used_faces:
                continue

            # 标记为已使用
            used_faces.add(f1_idx)
            used_faces.add(f2_idx)

            # 创建四边形
            f1, f2 = faces[f1_idx], faces[f2_idx]
            quad_verts = self._get_quad_vertices(f1, f2, shared_edge)

            if quad_verts:
                quad_faces.append(quad_verts)

        # 收集未配对的三角形
        remaining_tris = []
        for i, face in enumerate(faces):
            if i not in used_faces:
                remaining_tris.append(list(face))

        return quad_faces, remaining_tris


def convert_to_quad_mesh(vertices: np.ndarray,
                         faces: np.ndarray,
                         config: Optional[QuadConversionConfig] = None) -> Dict[str, Any]:
    """
    便捷函数：将三角面网格转换为四边形网格

    Args:
        vertices: 顶点数组
        faces: 三角面数组
        config: 转换配置

    Returns:
        {
            'vertices': 顶点数组,
            'quad_faces': 四边形面列表,
            'tri_faces': 剩余三角形面列表,
            'quad_count': 四边形数量,
            'tri_count': 三角形数量,
            'quad_ratio': 四边形面积比例
        }
    """
    converter = QuadConverter(config)
    verts, quads, tris = converter.convert(vertices, faces)

    total_faces = len(quads) + len(tris)
    quad_ratio = len(quads) / total_faces if total_faces > 0 else 0

    return {
        'vertices': verts,
        'quad_faces': quads,
        'tri_faces': tris,
        'quad_count': len(quads),
        'tri_count': len(tris),
        'quad_ratio': quad_ratio
    }


def save_mixed_mesh_obj(filepath: str,
                        vertices: np.ndarray,
                        quad_faces: List[List[int]],
                        tri_faces: List[List[int]]):
    """
    保存混合网格（四边形+三角形）为OBJ格式

    OBJ格式天然支持混合面类型
    """
    with open(filepath, 'w') as f:
        f.write("# Mixed quad/tri mesh generated by AI_CAD\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Quads: {len(quad_faces)}\n")
        f.write(f"# Triangles: {len(tri_faces)}\n\n")

        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # 写入四边形面（OBJ索引从1开始）
        f.write("# Quad faces\n")
        for face in quad_faces:
            indices = " ".join(str(i + 1) for i in face)
            f.write(f"f {indices}\n")

        # 写入三角形面
        f.write("\n# Triangle faces\n")
        for face in tri_faces:
            indices = " ".join(str(i + 1) for i in face)
            f.write(f"f {indices}\n")


def load_mixed_mesh_obj(filepath: str) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
    """
    加载混合网格OBJ文件

    Returns:
        vertices, quad_faces, tri_faces
    """
    vertices = []
    quad_faces = []
    tri_faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()[1:4]
                vertices.append([float(x) for x in parts])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # 处理可能的 v/vt/vn 格式
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                if len(indices) == 4:
                    quad_faces.append(indices)
                elif len(indices) == 3:
                    tri_faces.append(indices)

    return np.array(vertices), quad_faces, tri_faces
