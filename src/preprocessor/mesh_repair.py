"""
网格修复模块

专门用于修复AI生成的3D模型中的孔洞和非流形问题
针对混元3D等AI模型的输出进行优化

支持方法:
1. 基于边界环的孔洞填充（改进版：支持复杂边界）
2. 非流形边修复
3. 重复面/顶点清理
4. 边界一致性修复
5. 泊松重建（需要Open3D）
6. 体素化重建（需要trimesh voxelization）
"""

import numpy as np
from typing import Tuple, Optional, List, Set, Dict
from collections import defaultdict
from dataclasses import dataclass
import heapq


@dataclass
class RepairConfig:
    """修复配置"""
    # 孔洞填充
    fill_holes: bool = True
    max_hole_vertices: int = 200  # 最大填充的孔洞顶点数

    # 高级孔洞填充（针对复杂孔洞）
    advanced_hole_filling: bool = True
    use_ear_clipping: bool = True  # 使用耳切法填充（更好的三角剖分）

    # 非流形修复
    repair_non_manifold: bool = True
    split_non_manifold_vertices: bool = True
    remove_non_manifold_edges: bool = True

    # 清理选项
    remove_duplicate_faces: bool = True
    remove_degenerate_faces: bool = True  # 移除退化面（面积为0）
    merge_close_vertices: bool = True
    merge_threshold: float = 1e-6

    # 重建方法
    use_voxel_repair: bool = False
    voxel_pitch: float = 0.01  # 体素大小

    # 是否保留原始几何
    preserve_original: bool = True


def fill_boundary_holes(vertices: np.ndarray,
                        faces: np.ndarray,
                        max_hole_size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    填充边界孔洞

    通过识别边界环并用扇形三角形填充

    Args:
        vertices: 顶点数组 (N, 3)
        faces: 面数组 (M, 3)
        max_hole_size: 最大填充的孔洞顶点数

    Returns:
        new_vertices, new_faces
    """
    # 找到所有边界边
    edge_counts = defaultdict(int)
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_counts[edge] += 1

    boundary_edges = [e for e, c in edge_counts.items() if c == 1]

    if not boundary_edges:
        return vertices, faces

    # 构建边界顶点邻接
    vertex_neighbors = defaultdict(list)
    for e in boundary_edges:
        vertex_neighbors[e[0]].append(e[1])
        vertex_neighbors[e[1]].append(e[0])

    # 找到所有边界环
    boundary_vertices = set(v for e in boundary_edges for v in e)
    visited = set()
    holes = []

    for start in boundary_vertices:
        if start in visited:
            continue

        # 追踪这个边界环
        ring = [start]
        visited.add(start)
        current = start

        while True:
            neighbors = [n for n in vertex_neighbors[current] if n not in visited]
            if not neighbors:
                # 检查是否回到起点形成闭环
                if start in vertex_neighbors[current]:
                    break
                else:
                    break

            next_v = neighbors[0]
            ring.append(next_v)
            visited.add(next_v)
            current = next_v

        if len(ring) >= 3:
            holes.append(ring)

    # 填充孔洞
    new_vertices = list(vertices)
    new_faces = list(faces)

    for hole in holes:
        if len(hole) > max_hole_size:
            continue  # 跳过太大的孔洞

        if len(hole) < 3:
            continue

        # 计算孔洞中心
        hole_verts = vertices[hole]
        center = hole_verts.mean(axis=0)

        # 添加中心顶点
        center_idx = len(new_vertices)
        new_vertices.append(center)

        # 用扇形三角形填充
        for i in range(len(hole)):
            v1 = hole[i]
            v2 = hole[(i + 1) % len(hole)]
            new_faces.append([v1, v2, center_idx])

    return np.array(new_vertices), np.array(new_faces)


def remove_duplicate_faces(faces: np.ndarray) -> np.ndarray:
    """移除重复面"""
    # 标准化每个面（排序顶点索引）
    normalized = np.sort(faces, axis=1)
    # 使用集合去重
    unique_faces = []
    seen = set()
    for i, face in enumerate(normalized):
        key = tuple(face)
        if key not in seen:
            seen.add(key)
            unique_faces.append(faces[i])
    return np.array(unique_faces) if unique_faces else np.array([]).reshape(0, 3)


def remove_degenerate_faces(vertices: np.ndarray,
                            faces: np.ndarray,
                            area_threshold: float = 1e-10) -> np.ndarray:
    """移除退化面（面积为0或接近0的面）"""
    valid_faces = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # 计算面积（叉积的一半）
        e1 = v1 - v0
        e2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        if area > area_threshold:
            valid_faces.append(face)
    return np.array(valid_faces) if valid_faces else np.array([]).reshape(0, 3)


def merge_close_vertices(vertices: np.ndarray,
                         faces: np.ndarray,
                         threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """合并距离很近的顶点"""
    n_verts = len(vertices)
    if n_verts == 0:
        return vertices, faces

    # 构建顶点映射
    vertex_map = np.arange(n_verts)

    # 使用简单的空间哈希来加速
    # 对于小规模网格，直接暴力检查
    if n_verts < 10000:
        for i in range(n_verts):
            if vertex_map[i] != i:  # 已经被合并
                continue
            for j in range(i + 1, n_verts):
                if vertex_map[j] != j:
                    continue
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if dist < threshold:
                    vertex_map[j] = i
    else:
        # 对于大规模网格，使用KD树
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(vertices)
            pairs = tree.query_pairs(threshold)
            for i, j in pairs:
                if vertex_map[j] > vertex_map[i]:
                    vertex_map[j] = vertex_map[i]
                else:
                    vertex_map[i] = vertex_map[j]
        except ImportError:
            pass  # 没有scipy，跳过

    # 创建新的顶点数组
    unique_indices = np.unique(vertex_map)
    new_index_map = {old: new for new, old in enumerate(unique_indices)}
    new_vertices = vertices[unique_indices]

    # 更新面索引
    new_faces = []
    for face in faces:
        new_face = [new_index_map[vertex_map[v]] for v in face]
        # 检查面是否退化（三个顶点相同）
        if len(set(new_face)) == 3:
            new_faces.append(new_face)

    return new_vertices, np.array(new_faces) if new_faces else np.array([]).reshape(0, 3)


def repair_non_manifold_edges(vertices: np.ndarray,
                              faces: np.ndarray) -> np.ndarray:
    """
    修复非流形边（被超过2个面共享的边）

    策略：移除共享非流形边的多余面
    """
    edge_to_faces = defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted([face[j], face[(j + 1) % 3]]))
            edge_to_faces[edge].append(i)

    # 找到非流形边
    non_manifold_edges = {e: flist for e, flist in edge_to_faces.items() if len(flist) > 2}

    if not non_manifold_edges:
        return faces

    # 标记要移除的面
    faces_to_remove = set()
    for edge, face_indices in non_manifold_edges.items():
        # 保留前2个面，移除其他
        for fi in face_indices[2:]:
            faces_to_remove.add(fi)

    # 返回过滤后的面
    return np.array([faces[i] for i in range(len(faces)) if i not in faces_to_remove])


def fill_hole_ear_clipping(vertices: np.ndarray,
                           hole_indices: List[int]) -> List[List[int]]:
    """
    使用耳切法填充孔洞

    这比简单的扇形填充产生更好的三角剖分
    """
    if len(hole_indices) < 3:
        return []

    if len(hole_indices) == 3:
        return [list(hole_indices)]

    # 获取孔洞顶点
    hole_verts = vertices[hole_indices]

    # 计算孔洞平面的法向量
    center = hole_verts.mean(axis=0)
    centered = hole_verts - center
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]

    # 将顶点投影到2D（用于耳切法）
    u = vh[0]
    v = vh[1]
    points_2d = np.column_stack([
        np.dot(centered, u),
        np.dot(centered, v)
    ])

    # 简化版耳切法
    triangles = []
    remaining = list(range(len(hole_indices)))

    max_iterations = len(hole_indices) * 2
    iteration = 0

    while len(remaining) > 3 and iteration < max_iterations:
        iteration += 1
        found_ear = False

        for i in range(len(remaining)):
            prev_i = (i - 1) % len(remaining)
            next_i = (i + 1) % len(remaining)

            p0 = points_2d[remaining[prev_i]]
            p1 = points_2d[remaining[i]]
            p2 = points_2d[remaining[next_i]]

            # 检查是否为凸顶点（耳朵）
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

            if cross > 0:  # 凸顶点
                # 检查三角形内部是否有其他点
                is_ear = True
                for j in range(len(remaining)):
                    if j in [prev_i, i, next_i]:
                        continue
                    pt = points_2d[remaining[j]]
                    if point_in_triangle_2d(pt, p0, p1, p2):
                        is_ear = False
                        break

                if is_ear:
                    # 添加三角形
                    triangles.append([
                        hole_indices[remaining[prev_i]],
                        hole_indices[remaining[i]],
                        hole_indices[remaining[next_i]]
                    ])
                    remaining.pop(i)
                    found_ear = True
                    break

        if not found_ear:
            # 如果找不到耳朵，可能是非简单多边形，使用扇形填充剩余部分
            break

    # 处理剩余的顶点
    if len(remaining) == 3:
        triangles.append([
            hole_indices[remaining[0]],
            hole_indices[remaining[1]],
            hole_indices[remaining[2]]
        ])
    elif len(remaining) > 3:
        # 使用扇形填充
        center_idx = remaining[0]
        for i in range(1, len(remaining) - 1):
            triangles.append([
                hole_indices[center_idx],
                hole_indices[remaining[i]],
                hole_indices[remaining[i + 1]]
            ])

    return triangles


def point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """检查2D点是否在三角形内部"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def fill_boundary_holes_advanced(vertices: np.ndarray,
                                 faces: np.ndarray,
                                 max_hole_size: int = 200,
                                 use_ear_clipping: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    高级孔洞填充

    改进点：
    1. 正确追踪多个独立的边界环
    2. 使用耳切法进行更好的三角剖分
    3. 处理复杂的边界情况
    """
    # 找到所有边界边
    edge_counts = defaultdict(int)
    edge_to_face = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_counts[edge] += 1
            edge_to_face[edge].append(fi)

    boundary_edges = [(e[0], e[1]) for e, c in edge_counts.items() if c == 1]

    if not boundary_edges:
        return vertices, faces

    # 构建有向边界图（保持方向一致性）
    # 对于边界边，需要确定正确的方向
    directed_edges = defaultdict(list)
    for e in boundary_edges:
        # 找到这条边属于哪个面
        fi = edge_to_face[tuple(sorted(e))][0]
        face = faces[fi]
        # 在面中找到这条边的方向
        for i in range(3):
            if set([face[i], face[(i + 1) % 3]]) == set(e):
                # 边界边应该与面的方向相反
                v1, v2 = face[(i + 1) % 3], face[i]
                directed_edges[v1].append(v2)
                break

    # 追踪所有边界环
    boundary_vertices = set(v for e in boundary_edges for v in e)
    visited_vertices = set()
    holes = []

    for start in boundary_vertices:
        if start in visited_vertices:
            continue

        ring = [start]
        visited_vertices.add(start)
        current = start

        max_steps = len(boundary_vertices) + 1
        steps = 0

        while steps < max_steps:
            steps += 1
            next_verts = directed_edges.get(current, [])
            next_v = None

            for v in next_verts:
                if v == start and len(ring) >= 3:
                    # 回到起点，环完成
                    next_v = None
                    break
                if v not in visited_vertices:
                    next_v = v
                    break

            if next_v is None:
                break

            ring.append(next_v)
            visited_vertices.add(next_v)
            current = next_v

        if len(ring) >= 3:
            holes.append(ring)

    # 填充每个孔洞
    new_faces = list(faces)

    for hole in holes:
        if len(hole) > max_hole_size:
            continue

        if use_ear_clipping and len(hole) > 3:
            tris = fill_hole_ear_clipping(vertices, hole)
            new_faces.extend(tris)
        else:
            # 简单扇形填充
            for i in range(1, len(hole) - 1):
                new_faces.append([hole[0], hole[i], hole[i + 1]])

    return vertices, np.array(new_faces)


def repair_mesh(vertices: np.ndarray,
                faces: np.ndarray,
                config: Optional[RepairConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    修复网格（增强版）

    针对混元3D等AI生成模型的问题进行优化

    Args:
        vertices: 顶点数组
        faces: 面数组
        config: 修复配置

    Returns:
        修复后的 vertices, faces
    """
    cfg = config or RepairConfig()

    new_verts = vertices.copy()
    new_faces = faces.copy()

    # 1. 合并相近顶点
    if cfg.merge_close_vertices:
        new_verts, new_faces = merge_close_vertices(
            new_verts, new_faces, cfg.merge_threshold
        )

    # 2. 移除重复面
    if cfg.remove_duplicate_faces and len(new_faces) > 0:
        new_faces = remove_duplicate_faces(new_faces)

    # 3. 移除退化面
    if cfg.remove_degenerate_faces and len(new_faces) > 0:
        new_faces = remove_degenerate_faces(new_verts, new_faces)

    # 4. 修复非流形边
    if cfg.repair_non_manifold and cfg.remove_non_manifold_edges and len(new_faces) > 0:
        new_faces = repair_non_manifold_edges(new_verts, new_faces)

    # 5. 填充孔洞
    if cfg.fill_holes and len(new_faces) > 0:
        if cfg.advanced_hole_filling:
            new_verts, new_faces = fill_boundary_holes_advanced(
                new_verts, new_faces,
                max_hole_size=cfg.max_hole_vertices,
                use_ear_clipping=cfg.use_ear_clipping
            )
        else:
            new_verts, new_faces = fill_boundary_holes(
                new_verts, new_faces,
                max_hole_size=cfg.max_hole_vertices
            )

    return new_verts, new_faces


def repair_mesh_aggressive(vertices: np.ndarray,
                           faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    激进模式修复网格

    对混元3D等AI生成模型使用最强力的修复策略
    """
    config = RepairConfig(
        fill_holes=True,
        max_hole_vertices=500,  # 更大的孔洞也尝试填充
        advanced_hole_filling=True,
        use_ear_clipping=True,
        repair_non_manifold=True,
        split_non_manifold_vertices=True,
        remove_non_manifold_edges=True,
        remove_duplicate_faces=True,
        remove_degenerate_faces=True,
        merge_close_vertices=True,
        merge_threshold=1e-5,  # 更宽松的合并阈值
    )
    return repair_mesh(vertices, faces, config)


def analyze_mesh_topology(vertices: np.ndarray,
                          faces: np.ndarray) -> dict:
    """
    分析网格拓扑

    Returns:
        拓扑分析报告
    """
    # 边统计
    edge_counts = defaultdict(int)
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_counts[edge] += 1

    boundary_edges = sum(1 for c in edge_counts.values() if c == 1)
    non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)

    # 找边界环
    boundary_edge_list = [e for e, c in edge_counts.items() if c == 1]
    vertex_neighbors = defaultdict(set)
    for e in boundary_edge_list:
        vertex_neighbors[e[0]].add(e[1])
        vertex_neighbors[e[1]].add(e[0])

    boundary_vertices = set(vertex_neighbors.keys())
    visited = set()
    hole_count = 0
    hole_sizes = []

    for start in boundary_vertices:
        if start in visited:
            continue

        # BFS
        queue = [start]
        component = []
        while queue:
            v = queue.pop(0)
            if v in visited:
                continue
            visited.add(v)
            component.append(v)
            for n in vertex_neighbors[v]:
                if n not in visited:
                    queue.append(n)

        hole_count += 1
        hole_sizes.append(len(component))

    # 欧拉特征
    V = len(vertices)
    E = len(edge_counts)
    F = len(faces)
    euler = V - E + F

    return {
        'vertex_count': V,
        'edge_count': E,
        'face_count': F,
        'euler_characteristic': euler,
        'boundary_edge_count': boundary_edges,
        'non_manifold_edge_count': non_manifold_edges,
        'hole_count': hole_count,
        'hole_sizes': sorted(hole_sizes, reverse=True)[:10],
        'is_watertight': boundary_edges == 0,
        'is_manifold': non_manifold_edges == 0
    }
