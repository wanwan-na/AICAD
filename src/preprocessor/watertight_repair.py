"""
水密网格修复模块

专门用于将混元3D等AI生成的有孔洞模型转换为水密网格

核心策略：
1. 迭代填充所有孔洞直到网格水密
2. 使用多种填充策略组合
3. 最后验证并清理
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class WatertightConfig:
    """水密修复配置"""
    max_iterations: int = 10  # 最大迭代次数
    max_hole_size: int = 1000  # 最大填充的孔洞大小（顶点数）
    use_center_fan: bool = True  # 使用中心扇形填充大孔洞
    use_ear_clipping: bool = True  # 使用耳切法填充小孔洞
    ear_clip_threshold: int = 50  # 小于此顶点数使用耳切法
    verbose: bool = True


def find_boundary_loops(faces: np.ndarray) -> List[List[int]]:
    """
    找到所有边界环（孔洞）

    返回: 边界环列表，每个环是顶点索引的有序列表

    改进：正确处理多个孔洞共享顶点的情况
    """
    # 统计每条边被使用的次数
    edge_count = defaultdict(int)
    edge_to_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] += 1
            edge_to_faces[edge].append((fi, v1, v2))

    # 边界边是只被使用一次的边
    boundary_edges_set = set()
    directed_boundary_edges = []
    for edge, count in edge_count.items():
        if count == 1:
            boundary_edges_set.add(edge)
            # 获取正确的方向
            fi, v1, v2 = edge_to_faces[edge][0]
            # 边界边方向应该与面的方向相反
            directed_boundary_edges.append((v2, v1))

    if not directed_boundary_edges:
        return []

    # 构建有向邻接图 - 允许一个顶点有多个出边
    next_vertices = defaultdict(list)
    for v1, v2 in directed_boundary_edges:
        next_vertices[v1].append(v2)

    # 追踪所有环 - 使用边而不是顶点来跟踪访问
    visited_edges = set()
    loops = []

    for start in next_vertices:
        for first_next in next_vertices[start]:
            # 检查这条边是否已被访问
            start_edge = tuple(sorted([start, first_next]))
            if start_edge in visited_edges:
                continue

            loop = [start]
            visited_edges.add(start_edge)
            current = first_next
            loop.append(current)

            max_steps = len(directed_boundary_edges) + 1
            steps = 0

            while steps < max_steps:
                steps += 1

                # 找下一个未访问的边
                found_next = False
                for next_v in next_vertices.get(current, []):
                    edge = tuple(sorted([current, next_v]))
                    if edge in visited_edges:
                        continue

                    if next_v == start and len(loop) >= 3:
                        # 完成一个闭环
                        visited_edges.add(edge)
                        found_next = True
                        break

                    loop.append(next_v)
                    visited_edges.add(edge)
                    current = next_v
                    found_next = True
                    break

                if not found_next:
                    break

                if current == start:
                    break

            # 只保存闭环
            if len(loop) >= 3 and (loop[-1] == start or
                tuple(sorted([loop[-1], start])) in boundary_edges_set):
                loops.append(loop)

    return loops


def fill_hole_fan(vertices: np.ndarray, hole: List[int]) -> Tuple[np.ndarray, List[List[int]]]:
    """
    使用扇形填充孔洞（添加中心点）

    适合大孔洞，产生规则的三角形
    """
    # 计算孔洞中心
    hole_verts = vertices[hole]
    center = hole_verts.mean(axis=0)

    # 添加中心顶点
    new_vertices = np.vstack([vertices, center.reshape(1, 3)])
    center_idx = len(vertices)

    # 创建扇形三角形
    new_faces = []
    for i in range(len(hole)):
        v1 = hole[i]
        v2 = hole[(i + 1) % len(hole)]
        new_faces.append([v1, v2, center_idx])

    return new_vertices, new_faces


def fill_hole_ear_clipping(vertices: np.ndarray, hole: List[int]) -> List[List[int]]:
    """
    使用耳切法填充孔洞

    适合小孔洞，产生更好的三角剖分
    """
    if len(hole) < 3:
        return []
    if len(hole) == 3:
        return [[hole[0], hole[1], hole[2]]]

    # 计算孔洞平面
    hole_verts = vertices[hole]
    center = hole_verts.mean(axis=0)
    centered = hole_verts - center

    # SVD找平面
    try:
        _, _, vh = np.linalg.svd(centered)
        u_axis = vh[0]
        v_axis = vh[1]
    except:
        # SVD失败，使用简单投影
        u_axis = np.array([1, 0, 0])
        v_axis = np.array([0, 1, 0])

    # 投影到2D
    points_2d = np.column_stack([
        np.dot(centered, u_axis),
        np.dot(centered, v_axis)
    ])

    # 耳切法
    triangles = []
    remaining = list(range(len(hole)))
    max_iter = len(hole) * 3

    for _ in range(max_iter):
        if len(remaining) < 3:
            break
        if len(remaining) == 3:
            triangles.append([hole[remaining[0]], hole[remaining[1]], hole[remaining[2]]])
            break

        found = False
        for i in range(len(remaining)):
            prev_i = (i - 1) % len(remaining)
            next_i = (i + 1) % len(remaining)

            p0 = points_2d[remaining[prev_i]]
            p1 = points_2d[remaining[i]]
            p2 = points_2d[remaining[next_i]]

            # 检查是否为凸顶点
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

            if cross > 1e-10:  # 凸顶点
                # 检查是否有其他点在三角形内
                is_ear = True
                for j in range(len(remaining)):
                    if j in [prev_i, i, next_i]:
                        continue
                    pt = points_2d[remaining[j]]
                    if point_in_triangle(pt, p0, p1, p2):
                        is_ear = False
                        break

                if is_ear:
                    triangles.append([
                        hole[remaining[prev_i]],
                        hole[remaining[i]],
                        hole[remaining[next_i]]
                    ])
                    remaining.pop(i)
                    found = True
                    break

        if not found:
            # 找不到耳朵，使用扇形填充剩余
            if len(remaining) >= 3:
                for i in range(1, len(remaining) - 1):
                    triangles.append([
                        hole[remaining[0]],
                        hole[remaining[i]],
                        hole[remaining[i + 1]]
                    ])
            break

    return triangles


def point_in_triangle(p, a, b, c):
    """检查点是否在三角形内"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def make_watertight(vertices: np.ndarray,
                    faces: np.ndarray,
                    config: Optional[WatertightConfig] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    将网格转换为水密网格

    通过迭代填充所有孔洞

    Args:
        vertices: 顶点数组
        faces: 面数组
        config: 配置

    Returns:
        new_vertices, new_faces, stats
    """
    cfg = config or WatertightConfig()

    vertices = np.array(vertices, dtype=np.float64)
    faces = [list(f) for f in faces]

    # 构建已存在面的集合，用于去重
    existing_faces = set()
    for f in faces:
        key = tuple(sorted(f))
        existing_faces.add(key)

    stats = {
        'initial_holes': 0,
        'holes_filled': 0,
        'vertices_added': 0,
        'faces_added': 0,
        'duplicates_skipped': 0,
        'iterations': 0,
    }

    for iteration in range(cfg.max_iterations):
        stats['iterations'] = iteration + 1

        # 找到所有孔洞
        holes = find_boundary_loops(np.array(faces))

        if iteration == 0:
            stats['initial_holes'] = len(holes)

        if not holes:
            if cfg.verbose:
                print(f"  迭代 {iteration + 1}: 网格已水密")
            break

        if cfg.verbose:
            print(f"  迭代 {iteration + 1}: 发现 {len(holes)} 个孔洞")

        # 按大小排序，先填小的
        holes.sort(key=len)

        # 填充孔洞
        filled = 0
        for hole in holes:
            if len(hole) > cfg.max_hole_size:
                continue

            if cfg.use_ear_clipping and len(hole) <= cfg.ear_clip_threshold:
                # 小孔洞用耳切法
                new_faces_list = fill_hole_ear_clipping(vertices, hole)
            else:
                # 大孔洞用扇形填充
                vertices, new_faces_list = fill_hole_fan(vertices, hole)
                stats['vertices_added'] += 1

            # 添加新面，但跳过已存在的
            for new_face in new_faces_list:
                key = tuple(sorted(new_face))
                if key not in existing_faces:
                    faces.append(new_face)
                    existing_faces.add(key)
                    stats['faces_added'] += 1
                else:
                    stats['duplicates_skipped'] += 1

            filled += 1

        stats['holes_filled'] += filled

        if filled == 0:
            if cfg.verbose:
                print(f"  无法填充更多孔洞")
            break

    return vertices, np.array(faces), stats


def verify_watertight(faces: np.ndarray) -> Tuple[bool, Dict]:
    """
    验证网格是否水密

    Returns:
        is_watertight, stats
    """
    edge_count = defaultdict(int)
    for face in faces:
        n = len(face)
        for i in range(n):
            edge = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_count[edge] += 1

    boundary_edges = sum(1 for c in edge_count.values() if c == 1)
    non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)

    return boundary_edges == 0, {
        'boundary_edges': boundary_edges,
        'non_manifold_edges': non_manifold_edges,
        'total_edges': len(edge_count),
    }


def repair_post_quad_conversion(vertices: np.ndarray,
                                quad_faces: List[List[int]],
                                tri_faces: List[List[int]]) -> Tuple[List[List[int]], List[List[int]], Dict]:
    """
    四边形转换后的修复

    修复转换过程中可能引入的非流形边问题

    Returns:
        repaired_quads, repaired_tris, stats
    """
    # 合并所有面进行分析
    all_faces = quad_faces + tri_faces

    # 检查非流形边
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(all_faces):
        n = len(face)
        for i in range(n):
            edge = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_to_faces[edge].append(fi)

    # 找到非流形边
    non_manifold_edges = {e: flist for e, flist in edge_to_faces.items() if len(flist) > 2}

    stats = {
        'non_manifold_edges_found': len(non_manifold_edges),
        'faces_removed': 0,
    }

    if not non_manifold_edges:
        return quad_faces, tri_faces, stats

    # 标记要移除的面
    faces_to_remove = set()
    for edge, face_indices in non_manifold_edges.items():
        # 保留前两个面，移除其他
        for fi in face_indices[2:]:
            faces_to_remove.add(fi)

    stats['faces_removed'] = len(faces_to_remove)

    # 过滤面
    n_quads = len(quad_faces)
    new_quads = []
    new_tris = []

    for fi, face in enumerate(all_faces):
        if fi in faces_to_remove:
            continue
        if fi < n_quads:
            new_quads.append(face)
        else:
            new_tris.append(face)

    return new_quads, new_tris, stats
