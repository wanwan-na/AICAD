"""
诊断水密化过程中产生的非流形边
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import trimesh


def check_duplicate_faces(faces):
    """检查重复面"""
    face_set = set()
    duplicates = []
    for i, face in enumerate(faces):
        key = tuple(sorted(face))
        if key in face_set:
            duplicates.append((i, key))
        face_set.add(key)
    return duplicates


def check_non_manifold_detailed(faces):
    """详细检查非流形边"""
    edge_to_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            e = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_to_faces[e].append(fi)

    # 找非流形边
    non_manifold = {}
    for e, flist in edge_to_faces.items():
        if len(flist) > 2:
            non_manifold[e] = flist

    return non_manifold, edge_to_faces


def main():
    # 加载
    mesh = trimesh.load("Hy21_Glass_Mesh.glb", force='mesh')
    vertices = np.array(mesh.vertices)
    original_faces = np.array(mesh.faces)

    print("=" * 60)
    print("水密化过程诊断")
    print("=" * 60)

    # 检查原始网格
    print(f"\n原始网格:")
    print(f"  顶点: {len(vertices)}")
    print(f"  面: {len(original_faces)}")

    dups = check_duplicate_faces(original_faces)
    print(f"  重复面: {len(dups)}")

    non_manifold, _ = check_non_manifold_detailed(original_faces)
    print(f"  非流形边: {len(non_manifold)}")

    # 手动执行水密化的每一步
    from src.preprocessor.watertight_repair import (
        find_boundary_loops,
        fill_hole_fan,
        fill_hole_ear_clipping
    )

    faces = list(original_faces)
    current_vertices = vertices.copy()

    # 找边界环
    holes = find_boundary_loops(np.array(faces))
    print(f"\n发现 {len(holes)} 个孔洞")

    # 显示孔洞信息
    hole_sizes = sorted([len(h) for h in holes], reverse=True)
    print(f"孔洞大小分布: {hole_sizes[:20]}")

    # 检查孔洞是否有重叠（共享顶点）
    all_hole_vertices = []
    for h in holes:
        all_hole_vertices.extend(h)

    unique_hole_verts = set(all_hole_vertices)
    print(f"\n孔洞总顶点: {len(all_hole_vertices)}")
    print(f"唯一顶点: {len(unique_hole_verts)}")
    print(f"重复顶点: {len(all_hole_vertices) - len(unique_hole_verts)}")

    if len(all_hole_vertices) != len(unique_hole_verts):
        # 找出哪些顶点被多个孔洞共享
        vertex_to_holes = defaultdict(list)
        for hi, h in enumerate(holes):
            for v in h:
                vertex_to_holes[v].append(hi)

        shared_verts = {v: hlist for v, hlist in vertex_to_holes.items() if len(hlist) > 1}
        print(f"\n共享顶点数: {len(shared_verts)}")
        print(f"共享顶点示例:")
        for v, hlist in list(shared_verts.items())[:5]:
            print(f"  顶点 {v}: 被孔洞 {hlist} 共享")

    # 填充孔洞并检查每步
    print(f"\n" + "-" * 40)
    print("填充孔洞...")
    print("-" * 40)

    new_faces = []
    added_face_count = 0

    for hi, hole in enumerate(holes):
        if len(hole) > 500:
            print(f"  孔洞 {hi}: {len(hole)} 顶点 - 跳过(太大)")
            continue

        if len(hole) <= 30:
            # 耳切法
            tris = fill_hole_ear_clipping(current_vertices, hole)
            new_faces.extend(tris)
            added_face_count += len(tris)
        else:
            # 扇形填充
            current_vertices, tris = fill_hole_fan(current_vertices, hole)
            new_faces.extend(tris)
            added_face_count += len(tris)

    print(f"  添加了 {added_face_count} 个面")

    # 合并面
    all_faces = list(faces) + new_faces

    # 检查填充后
    print(f"\n填充后:")
    print(f"  总面数: {len(all_faces)}")

    dups = check_duplicate_faces(all_faces)
    print(f"  重复面: {len(dups)}")
    if dups:
        print(f"  重复面示例: {dups[:5]}")

    non_manifold, edge_to_faces = check_non_manifold_detailed(all_faces)
    print(f"  非流形边: {len(non_manifold)}")

    if non_manifold:
        print(f"\n  非流形边详情:")
        for i, (e, flist) in enumerate(list(non_manifold.items())[:5]):
            print(f"    边 {e}: 被面 {flist} 使用")
            # 显示这些面
            for fi in flist:
                if fi < len(original_faces):
                    print(f"      面 {fi} (原始): {all_faces[fi]}")
                else:
                    print(f"      面 {fi} (新增): {all_faces[fi]}")


if __name__ == "__main__":
    main()
