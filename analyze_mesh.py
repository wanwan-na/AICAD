"""
分析网格文件的拓扑问题
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from collections import defaultdict

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("警告: trimesh未安装")


def analyze_mesh_topology_detailed(vertices, faces):
    """详细分析网格拓扑"""
    # 边统计
    edge_counts = defaultdict(int)
    edge_to_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            edge = tuple(sorted([v1, v2]))
            edge_counts[edge] += 1
            edge_to_faces[edge].append(fi)

    # 分类边
    boundary_edges = []  # 只被1个面使用
    manifold_edges = []  # 被2个面使用
    non_manifold_edges = []  # 被3个以上面使用

    for edge, count in edge_counts.items():
        if count == 1:
            boundary_edges.append(edge)
        elif count == 2:
            manifold_edges.append(edge)
        else:
            non_manifold_edges.append(edge)

    # 找边界环（孔洞）
    if boundary_edges:
        vertex_neighbors = defaultdict(list)
        for e in boundary_edges:
            vertex_neighbors[e[0]].append(e[1])
            vertex_neighbors[e[1]].append(e[0])

        boundary_vertices = set(vertex_neighbors.keys())
        visited = set()
        holes = []

        for start in boundary_vertices:
            if start in visited:
                continue

            ring = [start]
            visited.add(start)
            current = start

            while True:
                neighbors = [n for n in vertex_neighbors[current] if n not in visited]
                if not neighbors:
                    break
                next_v = neighbors[0]
                ring.append(next_v)
                visited.add(next_v)
                current = next_v

            if len(ring) >= 3:
                holes.append(ring)
    else:
        holes = []

    # 检查重复面
    face_set = set()
    duplicate_faces = 0
    for face in faces:
        key = tuple(sorted(face))
        if key in face_set:
            duplicate_faces += 1
        face_set.add(key)

    # 检查退化面
    degenerate_faces = 0
    for face in faces:
        if len(set(face)) < 3:
            degenerate_faces += 1
            continue
        # 检查面积
        if len(face) == 3:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            if area < 1e-10:
                degenerate_faces += 1

    # 顶点价数分析
    vertex_valence = defaultdict(int)
    for face in faces:
        for v in face:
            vertex_valence[v] += 1

    valences = list(vertex_valence.values())

    # 检查孤立顶点
    used_vertices = set()
    for face in faces:
        for v in face:
            used_vertices.add(v)
    isolated_vertices = len(vertices) - len(used_vertices)

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
        'expected_genus': (2 - euler) // 2 if euler <= 2 else 'invalid',

        'boundary_edge_count': len(boundary_edges),
        'manifold_edge_count': len(manifold_edges),
        'non_manifold_edge_count': len(non_manifold_edges),

        'hole_count': len(holes),
        'hole_sizes': sorted([len(h) for h in holes], reverse=True)[:20],

        'duplicate_face_count': duplicate_faces,
        'degenerate_face_count': degenerate_faces,
        'isolated_vertex_count': isolated_vertices,

        'is_watertight': len(boundary_edges) == 0,
        'is_manifold': len(non_manifold_edges) == 0,

        'valence_min': min(valences) if valences else 0,
        'valence_max': max(valences) if valences else 0,
        'valence_avg': np.mean(valences) if valences else 0,

        'non_manifold_edges_sample': non_manifold_edges[:10],
        'boundary_edges_sample': boundary_edges[:10],
    }


def analyze_quad_mesh(vertices, faces):
    """分析四边形网格"""
    tri_count = 0
    quad_count = 0
    other_count = 0

    for face in faces:
        if len(face) == 3:
            tri_count += 1
        elif len(face) == 4:
            quad_count += 1
        else:
            other_count += 1

    total = tri_count + quad_count + other_count
    quad_ratio = quad_count / total if total > 0 else 0

    return {
        'tri_count': tri_count,
        'quad_count': quad_count,
        'other_count': other_count,
        'quad_ratio': quad_ratio,
    }


def load_obj_with_quads(filepath):
    """加载OBJ文件，保留四边形"""
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    # 处理 v/vt/vn 格式
                    idx = int(p.split('/')[0]) - 1  # OBJ索引从1开始
                    face.append(idx)
                faces.append(face)

    return np.array(vertices), faces


def main():
    if not HAS_TRIMESH:
        print("需要安装trimesh: pip install trimesh")
        return

    print("=" * 70)
    print("网格拓扑分析报告")
    print("=" * 70)

    # 分析原始GLB文件
    glb_path = "Hy21_Glass_Mesh.glb"
    print(f"\n{'='*70}")
    print(f"1. 原始文件: {glb_path}")
    print("=" * 70)

    try:
        mesh = trimesh.load(glb_path, force='mesh')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        print(f"\n基本信息:")
        print(f"  顶点数: {len(vertices)}")
        print(f"  面数: {len(faces)}")
        print(f"  包围盒: {mesh.bounds}")

        topo = analyze_mesh_topology_detailed(vertices, faces)

        print(f"\n拓扑分析:")
        print(f"  欧拉特征: {topo['euler_characteristic']}")
        print(f"  预期亏格: {topo['expected_genus']}")
        print(f"  水密(无边界): {'是' if topo['is_watertight'] else '否'}")
        print(f"  流形: {'是' if topo['is_manifold'] else '否'}")

        print(f"\n边分类:")
        print(f"  边界边(孔洞边): {topo['boundary_edge_count']}")
        print(f"  流形边(正常): {topo['manifold_edge_count']}")
        print(f"  非流形边(问题): {topo['non_manifold_edge_count']}")

        print(f"\n孔洞信息:")
        print(f"  孔洞数量: {topo['hole_count']}")
        if topo['hole_sizes']:
            print(f"  孔洞大小(顶点数): {topo['hole_sizes']}")

        print(f"\n问题面:")
        print(f"  重复面: {topo['duplicate_face_count']}")
        print(f"  退化面: {topo['degenerate_face_count']}")
        print(f"  孤立顶点: {topo['isolated_vertex_count']}")

        print(f"\n顶点价数:")
        print(f"  最小: {topo['valence_min']}")
        print(f"  最大: {topo['valence_max']}")
        print(f"  平均: {topo['valence_avg']:.2f}")

        if topo['non_manifold_edges_sample']:
            print(f"\n非流形边示例: {topo['non_manifold_edges_sample']}")

    except Exception as e:
        print(f"  加载失败: {e}")

    # 分析转换后的四边形文件
    quad_path = "Hy21_Glass_Mesh_quad2.obj"
    print(f"\n{'='*70}")
    print(f"2. 四边形文件: {quad_path}")
    print("=" * 70)

    try:
        vertices, faces = load_obj_with_quads(quad_path)

        print(f"\n基本信息:")
        print(f"  顶点数: {len(vertices)}")
        print(f"  面数: {len(faces)}")

        quad_info = analyze_quad_mesh(vertices, faces)
        print(f"\n面类型统计:")
        print(f"  三角形: {quad_info['tri_count']}")
        print(f"  四边形: {quad_info['quad_count']}")
        print(f"  其他: {quad_info['other_count']}")
        print(f"  四边形比例: {quad_info['quad_ratio']*100:.1f}%")

        # 将四边形转为三角形进行拓扑分析
        tri_faces = []
        for face in faces:
            if len(face) == 3:
                tri_faces.append(face)
            elif len(face) == 4:
                tri_faces.append([face[0], face[1], face[2]])
                tri_faces.append([face[0], face[2], face[3]])
            else:
                # 扇形三角化
                for i in range(1, len(face) - 1):
                    tri_faces.append([face[0], face[i], face[i+1]])

        topo = analyze_mesh_topology_detailed(vertices, tri_faces)

        print(f"\n拓扑分析(三角化后):")
        print(f"  欧拉特征: {topo['euler_characteristic']}")
        print(f"  水密(无边界): {'是' if topo['is_watertight'] else '否'}")
        print(f"  流形: {'是' if topo['is_manifold'] else '否'}")

        print(f"\n边分类:")
        print(f"  边界边(孔洞边): {topo['boundary_edge_count']}")
        print(f"  流形边(正常): {topo['manifold_edge_count']}")
        print(f"  非流形边(问题): {topo['non_manifold_edge_count']}")

        print(f"\n孔洞信息:")
        print(f"  孔洞数量: {topo['hole_count']}")
        if topo['hole_sizes']:
            print(f"  孔洞大小(顶点数): {topo['hole_sizes']}")

    except Exception as e:
        print(f"  加载失败: {e}")
        import traceback
        traceback.print_exc()

    # 分析直接转换的文件
    direct_path = "Hy21_Glass_direct_quad.obj"
    print(f"\n{'='*70}")
    print(f"3. 直接转换文件: {direct_path}")
    print("=" * 70)

    try:
        vertices, faces = load_obj_with_quads(direct_path)

        print(f"\n基本信息:")
        print(f"  顶点数: {len(vertices)}")
        print(f"  面数: {len(faces)}")

        quad_info = analyze_quad_mesh(vertices, faces)
        print(f"\n面类型统计:")
        print(f"  三角形: {quad_info['tri_count']}")
        print(f"  四边形: {quad_info['quad_count']}")
        print(f"  四边形比例: {quad_info['quad_ratio']*100:.1f}%")

        # 拓扑分析
        tri_faces = []
        for face in faces:
            if len(face) == 3:
                tri_faces.append(face)
            elif len(face) == 4:
                tri_faces.append([face[0], face[1], face[2]])
                tri_faces.append([face[0], face[2], face[3]])

        topo = analyze_mesh_topology_detailed(vertices, tri_faces)

        print(f"\n拓扑分析:")
        print(f"  边界边(孔洞边): {topo['boundary_edge_count']}")
        print(f"  孔洞数量: {topo['hole_count']}")
        if topo['hole_sizes']:
            print(f"  孔洞大小: {topo['hole_sizes']}")

    except Exception as e:
        print(f"  加载失败: {e}")

    print(f"\n{'='*70}")
    print("分析结论")
    print("=" * 70)


if __name__ == "__main__":
    main()
