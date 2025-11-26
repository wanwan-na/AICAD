"""
诊断非流形边问题

找出为什么四边形转换会产生非流形边
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import trimesh
from src.preprocessor.watertight_repair import make_watertight, WatertightConfig
from src.preprocessor.mesh_repair import analyze_mesh_topology


def check_manifold(faces, name="mesh"):
    """检查非流形边"""
    edge_count = defaultdict(int)
    edge_to_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            e = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_count[e] += 1
            edge_to_faces[e].append(fi)

    non_manifold = {e: c for e, c in edge_count.items() if c > 2}
    boundary = {e: c for e, c in edge_count.items() if c == 1}

    print(f"\n{name}:")
    print(f"  总边数: {len(edge_count)}")
    print(f"  非流形边: {len(non_manifold)}")
    print(f"  边界边: {len(boundary)}")

    if non_manifold:
        print(f"\n  非流形边示例 (边, 使用次数):")
        for i, (e, c) in enumerate(list(non_manifold.items())[:5]):
            faces_using = edge_to_faces[e]
            print(f"    {e}: 被 {c} 个面使用: {faces_using[:10]}")

    return non_manifold, boundary


def simple_quad_convert(faces):
    """
    简单的四边形转换（仅用于诊断）

    只做最基本的配对，不做任何花哨的操作
    """
    faces = np.array(faces)

    # 构建边-面邻接
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_to_faces[e].append(fi)

    # 找可配对的三角形对（共享恰好一条边）
    pairs = []
    for edge, flist in edge_to_faces.items():
        if len(flist) == 2:
            pairs.append((flist[0], flist[1], edge))

    print(f"\n发现 {len(pairs)} 对可配对三角形")

    # 贪心配对
    used = set()
    quads = []

    for f1_idx, f2_idx, shared_edge in pairs:
        if f1_idx in used or f2_idx in used:
            continue

        f1 = faces[f1_idx]
        f2 = faces[f2_idx]

        # 创建四边形
        e0, e1 = shared_edge
        other1 = [v for v in f1 if v not in shared_edge][0]
        other2 = [v for v in f2 if v not in shared_edge][0]

        # 确定顺序
        f1_list = list(f1)
        idx = f1_list.index(other1)
        next_v = f1_list[(idx + 1) % 3]

        if next_v == e0:
            quad = [other1, e0, other2, e1]
        else:
            quad = [other1, e1, other2, e0]

        used.add(f1_idx)
        used.add(f2_idx)
        quads.append(quad)

    # 收集剩余三角形
    tris = [list(faces[i]) for i in range(len(faces)) if i not in used]

    print(f"生成 {len(quads)} 个四边形，{len(tris)} 个三角形")

    return quads, tris


def main():
    # 加载
    mesh = trimesh.load("Hy21_Glass_Mesh.glb", force='mesh')
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    print("=" * 60)
    print("非流形边诊断")
    print("=" * 60)

    # 检查原始网格
    check_manifold(faces, "原始网格")

    # 水密化
    print("\n" + "-" * 40)
    print("水密化...")
    print("-" * 40)

    config = WatertightConfig(verbose=False)
    vertices, faces, stats = make_watertight(vertices, faces, config)

    print(f"填充孔洞: {stats['holes_filled']}")
    print(f"添加面: {stats['faces_added']}")

    # 检查水密化后
    check_manifold(faces, "水密化后")

    # 四边形转换
    print("\n" + "-" * 40)
    print("四边形转换...")
    print("-" * 40)

    quads, tris = simple_quad_convert(faces)

    # 合并所有面
    all_faces = quads + tris

    # 检查转换后
    non_manifold, boundary = check_manifold(all_faces, "四边形转换后")

    # 如果有非流形，分析原因
    if non_manifold:
        print("\n" + "-" * 40)
        print("分析非流形原因...")
        print("-" * 40)

        # 检查这些非流形边在原始三角形中的使用情况
        original_edge_to_faces = defaultdict(list)
        for fi, face in enumerate(faces):
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                original_edge_to_faces[e].append(fi)

        print("\n检查非流形边在原始三角形中的使用:")
        for i, (e, c) in enumerate(list(non_manifold.items())[:10]):
            original_count = len(original_edge_to_faces.get(e, []))
            print(f"  边 {e}: 转换后被{c}面使用, 原始被{original_count}面使用")


if __name__ == "__main__":
    main()
