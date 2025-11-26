"""
混元3D到四边形网格的完整转换流程

专门针对混元3D (Hunyuan3D) 生成的模型优化：
1. 先将模型修复为水密网格
2. 使用保守的重网格化
3. 转换为四边形
4. 后处理修复

使用方法:
    python hunyuan_to_quad.py input.glb output_quad.obj
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import trimesh
except ImportError:
    print("错误: 需要安装 trimesh: pip install trimesh")
    sys.exit(1)

try:
    import pymeshlab
except ImportError:
    print("错误: 需要安装 pymeshlab: pip install pymeshlab")
    sys.exit(1)

from src.preprocessor.watertight_repair import (
    make_watertight,
    verify_watertight,
    repair_post_quad_conversion,
    WatertightConfig
)
from src.preprocessor.mesh_repair import analyze_mesh_topology
from src.retopology.quad_converter_safe import SafeQuadConverter, SafeQuadConfig


def load_mesh(filepath: str):
    """加载网格文件"""
    mesh = trimesh.load(str(filepath), force='mesh')
    return np.array(mesh.vertices), np.array(mesh.faces)


def save_obj(filepath: str, vertices: np.ndarray,
             quad_faces: list, tri_faces: list):
    """保存为OBJ文件"""
    with open(filepath, 'w') as f:
        f.write("# Hunyuan3D to Quad conversion result\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Quads: {len(quad_faces)}\n")
        f.write(f"# Triangles: {len(tri_faces)}\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")
        for face in quad_faces:
            indices = " ".join(str(i + 1) for i in face)
            f.write(f"f {indices}\n")

        for face in tri_faces:
            indices = " ".join(str(i + 1) for i in face)
            f.write(f"f {indices}\n")


def watertight_pymeshlab(vertices: np.ndarray, faces: np.ndarray,
                         verbose: bool = True) -> tuple:
    """
    使用PyMeshLab进行水密化修复

    PyMeshLab的孔洞填充更成熟，不会产生非流形边
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_in = f.name
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_out = f.name

    try:
        # 保存
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(temp_in)

        # PyMeshLab处理
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_in)

        # 1. 移除重复面和顶点
        try:
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()
        except:
            pass

        # 2. 修复非流形
        try:
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        except:
            pass

        # 3. 填充孔洞 - 使用多种策略
        max_hole_size = 500  # 最大孔洞边数

        # 尝试多次填充
        for attempt in range(5):
            try:
                ms.meshing_close_holes(maxholesize=max_hole_size)
            except Exception as e:
                if verbose:
                    print(f"  孔洞填充尝试 {attempt + 1} 失败: {e}")
                break

        # 4. 再次清理
        try:
            ms.meshing_remove_duplicate_faces()
            ms.meshing_repair_non_manifold_edges(method=0)
        except:
            pass

        # 5. 重新计算法线
        try:
            ms.compute_normal_per_vertex()
            ms.compute_normal_per_face()
            ms.meshing_re_orient_faces_coherentely()
        except:
            pass

        # 保存
        ms.save_current_mesh(temp_out)

        # 重新加载
        result = trimesh.load(temp_out, force='mesh')
        return np.array(result.vertices), np.array(result.faces)

    finally:
        try:
            os.unlink(temp_in)
            os.unlink(temp_out)
        except:
            pass


def remesh_conservative(vertices: np.ndarray, faces: np.ndarray,
                        target_edge_length: float = None,
                        iterations: int = 5) -> tuple:
    """
    保守的重网格化，尽量保持拓扑

    不使用可能破坏拓扑的激进操作
    """
    import tempfile
    import os

    # 计算目标边长
    if target_edge_length is None:
        # 基于包围盒估算
        bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        target_edge_length = bbox_diag / 50  # 适中的密度

    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_in = f.name
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_out = f.name

    try:
        # 保存
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(temp_in)

        # PyMeshLab处理
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_in)

        # 保守的等参线重网格化
        # 注意：不同版本的pymeshlab API不同
        try:
            # 新版API
            ms.meshing_isotropic_explicit_remeshing(
                iterations=iterations,
                targetlen=pymeshlab.PercentageValue(1.5),  # 相对于包围盒对角线的百分比
                checksurfdist=True,
                splitflag=True,
                collapseflag=True,
                swapflag=True,
                smoothflag=True,
                reprojectflag=True
            )
        except Exception as e:
            print(f"  警告: 等参线重网格化方法1失败: {e}")
            # 尝试旧版API或简化参数
            try:
                ms.meshing_isotropic_explicit_remeshing(
                    iterations=iterations,
                    targetlen=pymeshlab.PercentageValue(1.5)
                )
            except Exception as e2:
                print(f"  警告: 等参线重网格化方法2失败: {e2}")
                # 最后尝试：使用绝对值
                try:
                    ms.meshing_isotropic_explicit_remeshing(
                        iterations=iterations
                    )
                except:
                    print("  警告: 所有重网格化方法都失败，跳过")

        # 保存
        ms.save_current_mesh(temp_out)

        # 重新加载
        result = trimesh.load(temp_out, force='mesh')
        return np.array(result.vertices), np.array(result.faces)

    finally:
        try:
            os.unlink(temp_in)
            os.unlink(temp_out)
        except:
            pass


def hunyuan_to_quad(input_path: str,
                   output_path: str = None,
                   skip_watertight: bool = False,
                   skip_remesh: bool = False,
                   verbose: bool = True) -> dict:
    """
    混元3D到四边形的完整转换

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        skip_watertight: 跳过水密化（如果模型已经是水密的）
        skip_remesh: 跳过重网格化
        verbose: 详细输出

    Returns:
        处理结果字典
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_quad.obj"
    output_path = Path(output_path)

    results = {}

    if verbose:
        print("=" * 60)
        print("混元3D → 四边形网格 转换")
        print("=" * 60)
        print(f"\n输入: {input_path}")

    # 1. 加载网格
    if verbose:
        print("\n" + "-" * 40)
        print("阶段 1: 加载网格")
        print("-" * 40)

    vertices, faces = load_mesh(str(input_path))

    if verbose:
        print(f"  顶点数: {len(vertices)}")
        print(f"  面数: {len(faces)}")

    # 分析初始拓扑
    initial_topo = analyze_mesh_topology(vertices, faces)
    results['initial'] = initial_topo

    if verbose:
        print(f"  孔洞数: {initial_topo['hole_count']}")
        print(f"  边界边: {initial_topo['boundary_edge_count']}")
        print(f"  非流形边: {initial_topo['non_manifold_edge_count']}")
        print(f"  水密: {'是' if initial_topo['is_watertight'] else '否'}")

    # 2. 水密化 - 使用PyMeshLab
    if not skip_watertight and not initial_topo['is_watertight']:
        if verbose:
            print("\n" + "-" * 40)
            print("阶段 2: 水密化修复 (PyMeshLab)")
            print("-" * 40)

        vertices, faces = watertight_pymeshlab(vertices, faces, verbose)
        results['watertight'] = {'method': 'pymeshlab'}

        # 验证
        is_wt, wt_verify = verify_watertight(faces)

        if verbose:
            print(f"\n  水密验证: {'通过' if is_wt else '未通过'}")
            print(f"  边界边: {wt_verify['boundary_edges']}")
            print(f"  非流形边: {wt_verify['non_manifold_edges']}")
    else:
        if verbose:
            print("\n  跳过水密化")

    # 3. 重网格化
    if not skip_remesh:
        if verbose:
            print("\n" + "-" * 40)
            print("阶段 3: 重网格化")
            print("-" * 40)

        original_faces = len(faces)
        vertices, faces = remesh_conservative(vertices, faces, iterations=10)

        if verbose:
            print(f"  原面数: {original_faces}")
            print(f"  新面数: {len(faces)}")

        # 检查重网格化后的拓扑
        remesh_topo = analyze_mesh_topology(vertices, faces)
        results['after_remesh'] = remesh_topo

        if verbose:
            print(f"  孔洞数: {remesh_topo['hole_count']}")
            print(f"  边界边: {remesh_topo['boundary_edge_count']}")

        # 如果重网格化引入了新孔洞，再次修复
        if remesh_topo['hole_count'] > 0:
            if verbose:
                print(f"\n  重网格化引入了 {remesh_topo['hole_count']} 个孔洞，尝试修复...")

            vertices, faces, _ = make_watertight(
                vertices, faces,
                WatertightConfig(max_iterations=3, verbose=False)
            )
    else:
        if verbose:
            print("\n  跳过重网格化")

    # 4. 四边形转换
    if verbose:
        print("\n" + "-" * 40)
        print("阶段 4: 四边形转换 (安全模式)")
        print("-" * 40)

    # 使用安全转换器，保证不引入非流形边
    quad_config = SafeQuadConfig(
        min_quality=0.05,  # 较低阈值以获得更多四边形
        max_angle_deviation=np.pi * 0.7,  # 126度
        max_diagonal_ratio=6.0,
        multi_pass=True,
        check_non_manifold=True  # 关键：检查非流形
    )

    converter = SafeQuadConverter(quad_config)
    vertices, quad_faces, tri_faces = converter.convert(vertices, faces)

    total_faces = len(quad_faces) + len(tri_faces)
    quad_ratio = len(quad_faces) / total_faces if total_faces > 0 else 0

    if verbose:
        print(f"  四边形: {len(quad_faces)}")
        print(f"  三角形: {len(tri_faces)}")
        print(f"  四边形比例: {quad_ratio * 100:.1f}%")

    # 5. 后处理 - 只检查，不修复（修复可能引入新问题）
    if verbose:
        print("\n" + "-" * 40)
        print("阶段 5: 拓扑验证")
        print("-" * 40)

    # 检查非流形边（不移除）
    all_faces_check = quad_faces + tri_faces
    edge_count = defaultdict(int)
    for face in all_faces_check:
        n = len(face)
        for i in range(n):
            e = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_count[e] += 1

    non_manifold_count = sum(1 for c in edge_count.values() if c > 2)
    boundary_count = sum(1 for c in edge_count.values() if c == 1)

    results['post_check'] = {
        'non_manifold_edges': non_manifold_count,
        'boundary_edges': boundary_count,
    }

    if verbose:
        print(f"  非流形边: {non_manifold_count}")
        print(f"  边界边: {boundary_count}")

    # 如果有非流形边，尝试保守修复
    if non_manifold_count > 0:
        if verbose:
            print(f"\n  警告: 发现 {non_manifold_count} 条非流形边")
            print(f"  尝试移除问题面...")

        # 找到并移除非流形边相关的面
        non_manifold_edges = {e for e, c in edge_count.items() if c > 2}
        faces_to_keep_quad = []
        faces_to_keep_tri = []

        # 统计每个面涉及多少非流形边
        for face in quad_faces:
            nm_count = 0
            for i in range(4):
                e = tuple(sorted([face[i], face[(i + 1) % 4]]))
                if e in non_manifold_edges:
                    nm_count += 1
            if nm_count == 0:
                faces_to_keep_quad.append(face)

        for face in tri_faces:
            nm_count = 0
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if e in non_manifold_edges:
                    nm_count += 1
            if nm_count == 0:
                faces_to_keep_tri.append(face)

        removed = len(quad_faces) + len(tri_faces) - len(faces_to_keep_quad) - len(faces_to_keep_tri)
        if verbose:
            print(f"  移除了 {removed} 个问题面")

        quad_faces = faces_to_keep_quad
        tri_faces = faces_to_keep_tri

    # 最终拓扑检查
    all_faces = quad_faces + tri_faces
    # 转为三角形检查
    tri_for_check = []
    for face in all_faces:
        if len(face) == 3:
            tri_for_check.append(face)
        elif len(face) == 4:
            tri_for_check.append([face[0], face[1], face[2]])
            tri_for_check.append([face[0], face[2], face[3]])

    final_topo = analyze_mesh_topology(vertices, np.array(tri_for_check))
    results['final'] = final_topo

    if verbose:
        print(f"\n最终拓扑:")
        print(f"  孔洞数: {final_topo['hole_count']}")
        print(f"  边界边: {final_topo['boundary_edge_count']}")
        print(f"  非流形边: {final_topo['non_manifold_edge_count']}")

    # 6. 保存结果
    save_obj(str(output_path), vertices, quad_faces, tri_faces)

    if verbose:
        print(f"\n输出: {output_path}")
        print("\n" + "=" * 60)
        print("转换完成!")
        print("=" * 60)

    results['output_path'] = str(output_path)
    results['quad_count'] = len(quad_faces)
    results['tri_count'] = len(tri_faces)
    results['quad_ratio'] = quad_ratio

    return results


def main():
    parser = argparse.ArgumentParser(
        description='混元3D到四边形网格转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python hunyuan_to_quad.py model.glb
  python hunyuan_to_quad.py model.glb output.obj
  python hunyuan_to_quad.py model.glb --skip-remesh  # 跳过重网格化
        """
    )

    parser.add_argument('input', help='输入网格文件')
    parser.add_argument('output', nargs='?', help='输出文件路径')
    parser.add_argument('--skip-watertight', action='store_true',
                        help='跳过水密化')
    parser.add_argument('--skip-remesh', action='store_true',
                        help='跳过重网格化')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='安静模式')

    args = parser.parse_args()

    try:
        result = hunyuan_to_quad(
            args.input,
            args.output,
            skip_watertight=args.skip_watertight,
            skip_remesh=args.skip_remesh,
            verbose=not args.quiet
        )

        # 评估结果
        if result['final']['hole_count'] == 0:
            print("\n结果: 水密网格 (无孔洞)")
            sys.exit(0)
        else:
            print(f"\n结果: 仍有 {result['final']['hole_count']} 个孔洞")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
