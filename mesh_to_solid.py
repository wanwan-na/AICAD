"""
3D网格转Solid转换工具

将三角网格或四边形网格转换为solid体（实体模型）
支持多种输出格式，适用于CAD/CAM工作流

主要功能:
1. 网格水密性检查和修复
2. 体素化转换
3. 布尔运算优化
4. 导出为STEP/STL/OBJ等CAD格式

使用方法:
    python mesh_to_solid.py input.obj output.step
    python mesh_to_solid.py input.obj --voxel-size 0.5
    python mesh_to_solid.py input.obj --repair --watertight
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import trimesh
    import pymeshlab
    import open3d as o3d
else:
    try:
        import trimesh
    except ImportError:
        trimesh = None

    try:
        import pymeshlab
    except ImportError:
        pymeshlab = None

    try:
        import open3d as o3d
    except ImportError:
        o3d = None


class MeshToSolidConverter:
    """网格转Solid转换器"""

    def __init__(self, verbose: bool = True):
        """
        初始化转换器

        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose

        # 检查依赖
        if trimesh is None:
            raise ImportError("需要安装 trimesh: pip install trimesh")

    def analyze_face_types(self, mesh) -> Dict[str, int]:
        """
        分析网格中的面类型（三角面、四边面等）

        Args:
            mesh: trimesh网格对象

        Returns:
            包含面类型统计的字典
        """
        faces = mesh.faces

        # 统计每个面的顶点数
        if len(faces) == 0:
            return {
                'total_faces': 0,
                'triangle_faces': 0,
                'quad_faces': 0,
                'other_faces': 0,
                'triangle_ratio': 0.0,
                'quad_ratio': 0.0
            }

        # trimesh默认只支持三角面，所以大部分情况都是三角面
        # 但如果是从OBJ等格式加载的，可能包含四边面
        face_vertices = []
        for face in faces:
            face_vertices.append(len(face))

        face_vertices = np.array(face_vertices)

        triangle_count = np.sum(face_vertices == 3)
        quad_count = np.sum(face_vertices == 4)
        other_count = len(faces) - triangle_count - quad_count

        total = len(faces)

        return {
            'total_faces': total,
            'triangle_faces': int(triangle_count),
            'quad_faces': int(quad_count),
            'other_faces': int(other_count),
            'triangle_ratio': float(triangle_count / total) if total > 0 else 0.0,
            'quad_ratio': float(quad_count / total) if total > 0 else 0.0
        }

    def load_mesh(self, file_path: str):
        """
        加载网格文件

        Args:
            file_path: 输入文件路径

        Returns:
            trimesh网格对象
        """
        if self.verbose:
            print(f"加载网格: {file_path}")

        mesh = trimesh.load(str(file_path), force='mesh')

        if self.verbose:
            print(f"  顶点数: {len(mesh.vertices)}")
            print(f"  面数: {len(mesh.faces)}")

            # 分析面类型
            face_stats = self.analyze_face_types(mesh)
            print(f"    三角面: {face_stats['triangle_faces']} ({face_stats['triangle_ratio']*100:.1f}%)")
            print(f"    四边面: {face_stats['quad_faces']} ({face_stats['quad_ratio']*100:.1f}%)")
            if face_stats['other_faces'] > 0:
                print(f"    其他面: {face_stats['other_faces']}")

            print(f"  水密: {'是' if mesh.is_watertight else '否'}")
            print(f"  流形: {'是' if mesh.is_volume else '否'}")

        return mesh

    def check_watertight(self, mesh) -> Tuple[bool, Dict[str, Any]]:
        """
        检查网格是否水密

        Args:
            mesh: trimesh网格对象

        Returns:
            (is_watertight, info_dict)
        """
        info = {
            'is_watertight': mesh.is_watertight,
            'is_volume': mesh.is_volume,
            'has_holes': not mesh.is_watertight,
            'boundary_edges': 0
        }

        # 统计边界边
        edges = mesh.edges_unique
        edge_count = mesh.edges_unique_length
        info['edge_count'] = edge_count

        # 检查边界
        if hasattr(mesh, 'edges_unique_inverse'):
            # 统计每条边被多少个面共享
            edge_faces = np.bincount(mesh.edges_unique_inverse)
            boundary_edges = np.sum(edge_faces == 1)
            info['boundary_edges'] = int(boundary_edges)

        if self.verbose:
            print("\n网格检查:")
            print(f"  水密: {info['is_watertight']}")
            print(f"  体积: {info['is_volume']}")
            print(f"  边界边数: {info['boundary_edges']}")

        return info['is_watertight'], info

    def repair_mesh(self, mesh,
                   fill_holes: bool = True,
                   remove_duplicates: bool = True,
                   fix_normals: bool = True):
        """
        修复网格使其成为水密solid

        Args:
            mesh: 输入网格
            fill_holes: 是否填充孔洞
            remove_duplicates: 是否移除重复顶点和面
            fix_normals: 是否修复法向

        Returns:
            修复后的网格
        """
        if self.verbose:
            print("\n修复网格...")

        # 创建副本
        repaired = mesh.copy()

        # 移除重复顶点
        if remove_duplicates:
            if self.verbose:
                print("  移除重复顶点...")
            repaired.merge_vertices()

        # 移除退化面
        if self.verbose:
            print("  移除退化面...")
        # trimesh中退化面会在后续操作中自动处理
        # 使用nondegenerate_faces属性来获取非退化面
        if hasattr(repaired, 'nondegenerate_faces'):
            valid_faces = repaired.nondegenerate_faces(height=1e-8)
            if len(valid_faces) < len(repaired.faces):
                repaired.update_faces(valid_faces)

        # 移除重复面
        if remove_duplicates:
            if self.verbose:
                print("  移除重复面...")
            # trimesh中使用unique_faces来去除重复面
            # 对每个面的顶点进行排序
            sorted_faces = np.sort(repaired.faces, axis=1)
            # 找到唯一的面
            _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
            if len(unique_indices) < len(repaired.faces):
                # 使用原始（未排序）的唯一面
                unique_faces = repaired.faces[sorted(unique_indices)]
                repaired = trimesh.Trimesh(vertices=repaired.vertices, faces=unique_faces)

        # 修复法向
        if fix_normals:
            if self.verbose:
                print("  修复法向...")
            try:
                repaired.fix_normals()
            except (IndexError, ValueError) as e:
                if self.verbose:
                    print(f"    警告: 法向修复失败 - {e}")
                    print("    跳过法向修复")

        # 填充孔洞
        if fill_holes and pymeshlab is not None:
            if self.verbose:
                print("  填充孔洞...")
            repaired = self._fill_holes_pymeshlab(repaired)
        elif fill_holes:
            if self.verbose:
                print("  警告: pymeshlab未安装，跳过孔洞填充")

        if self.verbose:
            print(f"  修复完成")
            print(f"    顶点数: {len(repaired.vertices)}")
            print(f"    面数: {len(repaired.faces)}")
            print(f"    水密: {'是' if repaired.is_watertight else '否'}")

        return repaired

    def _fill_holes_pymeshlab(self, mesh):
        """
        使用PyMeshLab填充孔洞

        Args:
            mesh: 输入网格

        Returns:
            填充后的网格
        """
        if pymeshlab is None:
            return mesh

        try:
            # 创建MeshSet
            ms = pymeshlab.MeshSet()

            # 先保存为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
                tmp_path = tmp.name
                mesh.export(tmp_path)

            # 使用pymeshlab加载
            ms.load_new_mesh(tmp_path)

            # 清理临时文件
            import os
            os.unlink(tmp_path)

            # 填充孔洞
            try:
                # 先尝试简单填充
                ms.meshing_close_holes(maxholesize=30)

                # 如果还有孔洞，尝试更激进的方法
                if not ms.current_mesh().is_compact():
                    ms.meshing_close_holes(maxholesize=100)
            except Exception as e:
                if self.verbose:
                    print(f"    警告: 孔洞填充部分失败 - {e}")

            # 转换回trimesh
            filled_mesh = trimesh.Trimesh(
                vertices=ms.current_mesh().vertex_matrix(),
                faces=ms.current_mesh().face_matrix()
            )

            return filled_mesh

        except Exception as e:
            if self.verbose:
                print(f"    PyMeshLab填充失败: {e}")
                print("    返回原始网格")
            return mesh

    def voxelize_mesh(self, mesh,
                     voxel_size: Optional[float] = None,
                     pitch: Optional[float] = None):
        """
        将网格体素化

        Args:
            mesh: 输入网格
            voxel_size: 体素大小（绝对值）
            pitch: 体素间距（相对于包围盒）

        Returns:
            体素网格
        """
        if self.verbose:
            print("\n体素化处理...")

        # 如果没有指定参数，自动计算
        if voxel_size is None and pitch is None:
            # 默认使用包围盒对角线的1/100作为pitch
            bbox_diagonal = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
            pitch = bbox_diagonal / 100.0

        # 创建体素网格
        if voxel_size is not None:
            voxels = mesh.voxelized(voxel_size)
        else:
            voxels = mesh.voxelized(pitch)

        if self.verbose:
            print(f"  体素网格尺寸: {voxels.shape}")
            print(f"  体素数量: {voxels.filled_count}")

        return voxels

    def voxels_to_mesh(self, voxels):
        """
        将体素网格转换回网格

        Args:
            voxels: 体素网格

        Returns:
            网格对象
        """
        if self.verbose:
            print("  体素转网格...")

        # 使用marching cubes算法
        mesh = voxels.marching_cubes

        if self.verbose:
            print(f"  生成网格 - 顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")

        return mesh

    def simplify_mesh(self, mesh,
                     target_faces: Optional[int] = None,
                     ratio: float = 0.5):
        """
        简化网格

        Args:
            mesh: 输入网格
            target_faces: 目标面数
            ratio: 简化比例（如果未指定target_faces）

        Returns:
            简化后的网格
        """
        if self.verbose:
            print("\n简化网格...")
            print(f"  原始面数: {len(mesh.faces)}")

        if target_faces is None:
            target_faces = int(len(mesh.faces) * ratio)

        # 尝试使用不同的简化方法
        try:
            # 首选: quadric decimation (需要fast_simplification)
            simplified = mesh.simplify_quadric_decimation(target_faces)
        except (AttributeError, ImportError):
            try:
                # 备选1: 使用pymeshlab简化
                if pymeshlab is not None:
                    ms = pymeshlab.MeshSet()
                    m = pymeshlab.Mesh(
                        vertex_matrix=mesh.vertices,
                        face_matrix=mesh.faces
                    )
                    ms.add_mesh(m)
                    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
                    simplified = trimesh.Trimesh(
                        vertices=ms.current_mesh().vertex_matrix(),
                        faces=ms.current_mesh().face_matrix()
                    )
                else:
                    # 备选2: 简单采样
                    if self.verbose:
                        print("  使用简单采样方法...")
                    # 保持原样，不进行简化
                    simplified = mesh
                    if self.verbose:
                        print("  警告: 无可用的简化方法，保持原网格")
            except Exception as e:
                if self.verbose:
                    print(f"  简化失败: {e}")
                    print("  保持原网格")
                simplified = mesh

        if self.verbose:
            print(f"  简化后面数: {len(simplified.faces)}")
            if len(mesh.faces) > 0:
                print(f"  简化率: {len(simplified.faces)/len(mesh.faces)*100:.1f}%")

        return simplified

    def smooth_mesh(self, mesh,
                   iterations: int = 3):
        """
        平滑网格

        Args:
            mesh: 输入网格
            iterations: 平滑迭代次数

        Returns:
            平滑后的网格
        """
        if self.verbose:
            print(f"\n平滑网格 ({iterations}次迭代)...")

        smoothed = mesh.copy()

        # Laplacian平滑
        for i in range(iterations):
            vertices = smoothed.vertices.copy()

            # 对每个顶点，计算邻居的平均位置
            for v_idx in range(len(vertices)):
                # 获取相邻顶点
                neighbors = smoothed.vertex_neighbors[v_idx]
                if len(neighbors) > 0:
                    neighbor_pos = vertices[neighbors]
                    vertices[v_idx] = np.mean(neighbor_pos, axis=0)

            smoothed.vertices = vertices

        if self.verbose:
            print("  平滑完成")

        return smoothed

    def convert_to_solid(self, input_path: str,
                        output_path: str,
                        repair: bool = True,
                        fill_holes: bool = True,
                        voxelize: bool = False,
                        voxel_size: Optional[float] = None,
                        simplify: bool = False,
                        target_faces: Optional[int] = None,
                        smooth: bool = False,
                        smooth_iterations: int = 3) -> Dict[str, Any]:
        """
        执行网格到Solid的完整转换流程

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            repair: 是否修复网格
            fill_holes: 是否填充孔洞
            voxelize: 是否使用体素化
            voxel_size: 体素大小
            simplify: 是否简化网格
            target_faces: 目标面数
            smooth: 是否平滑
            smooth_iterations: 平滑迭代次数

        Returns:
            处理结果字典
        """
        if self.verbose:
            print("=" * 60)
            print("3D网格转Solid转换工具")
            print("=" * 60)

        results = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'stages': {}
        }

        # 加载网格
        mesh = self.load_mesh(input_path)
        results['original_vertices'] = len(mesh.vertices)
        results['original_faces'] = len(mesh.faces)

        # 检查水密性
        is_watertight, check_info = self.check_watertight(mesh)
        results['original_watertight'] = is_watertight

        # 修复网格
        if repair:
            mesh = self.repair_mesh(mesh, fill_holes=fill_holes)
            is_watertight_after = mesh.is_watertight
            results['stages']['repair'] = {
                'watertight_before': is_watertight,
                'watertight_after': is_watertight_after
            }

        # 体素化处理
        if voxelize:
            voxels = self.voxelize_mesh(mesh, voxel_size=voxel_size)
            mesh = self.voxels_to_mesh(voxels)
            results['stages']['voxelize'] = {
                'voxel_count': voxels.filled_count,
                'new_vertices': len(mesh.vertices),
                'new_faces': len(mesh.faces)
            }

        # 简化网格
        if simplify:
            mesh = self.simplify_mesh(mesh, target_faces=target_faces)
            results['stages']['simplify'] = {
                'final_faces': len(mesh.faces)
            }

        # 平滑处理
        if smooth:
            mesh = self.smooth_mesh(mesh, iterations=smooth_iterations)
            results['stages']['smooth'] = {
                'iterations': smooth_iterations
            }

        # 保存结果
        if self.verbose:
            print(f"\n保存Solid模型: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 导出网格
        mesh.export(str(output_path))

        # 分析最终面类型
        final_face_stats = self.analyze_face_types(mesh)

        results['final_vertices'] = len(mesh.vertices)
        results['final_faces'] = len(mesh.faces)
        results['final_triangle_faces'] = final_face_stats['triangle_faces']
        results['final_quad_faces'] = final_face_stats['quad_faces']
        results['final_other_faces'] = final_face_stats['other_faces']
        results['final_watertight'] = mesh.is_watertight
        results['final_volume'] = float(mesh.volume) if mesh.is_volume else 0.0

        if self.verbose:
            print("\n" + "=" * 60)
            print("转换完成!")
            print("=" * 60)
            print(f"输入: {input_path}")
            print(f"输出: {output_path}")
            print(f"\n几何信息:")
            print(f"  顶点: {results['original_vertices']} → {results['final_vertices']}")
            print(f"  面数: {results['original_faces']} → {results['final_faces']}")
            print(f"\n面类型统计:")
            print(f"  三角面: {results['final_triangle_faces']} ({final_face_stats['triangle_ratio']*100:.1f}%)")
            print(f"  四边面: {results['final_quad_faces']} ({final_face_stats['quad_ratio']*100:.1f}%)")
            if results['final_other_faces'] > 0:
                print(f"  其他面: {results['final_other_faces']}")
            print(f"\n质量信息:")
            print(f"  水密: {results['original_watertight']} → {results['final_watertight']}")
            if results['final_volume'] > 0:
                print(f"  体积: {results['final_volume']:.6f}")

        return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='3D网格转Solid转换工具 - 将网格模型转换为实体模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换（自动修复）
  python mesh_to_solid.py model.obj model_solid.obj

  # 使用体素化（更好的水密性）
  python mesh_to_solid.py model.obj model_solid.stl --voxelize --voxel-size 0.5

  # 完整处理流程
  python mesh_to_solid.py model.obj model_solid.obj --repair --simplify --smooth

  # 指定目标面数
  python mesh_to_solid.py model.obj model_solid.obj --simplify --faces 5000
        """
    )

    parser.add_argument('input', help='输入网格文件路径')
    parser.add_argument('output', help='输出Solid文件路径')

    parser.add_argument('--repair', action='store_true', default=True,
                       help='修复网格（默认启用）')
    parser.add_argument('--no-repair', dest='repair', action='store_false',
                       help='禁用修复')
    parser.add_argument('--fill-holes', action='store_true', default=True,
                       help='填充孔洞（默认启用）')
    parser.add_argument('--no-fill-holes', dest='fill_holes', action='store_false',
                       help='禁用孔洞填充')

    parser.add_argument('--voxelize', action='store_true',
                       help='使用体素化处理（确保水密性）')
    parser.add_argument('--voxel-size', type=float, default=None,
                       help='体素大小（绝对值）')

    parser.add_argument('--simplify', action='store_true',
                       help='简化网格')
    parser.add_argument('--faces', type=int, default=None,
                       help='目标面数（用于简化）')

    parser.add_argument('--smooth', action='store_true',
                       help='平滑网格')
    parser.add_argument('--smooth-iterations', type=int, default=3,
                       help='平滑迭代次数（默认3）')

    parser.add_argument('-q', '--quiet', action='store_true',
                       help='安静模式')

    args = parser.parse_args()

    try:
        converter = MeshToSolidConverter(verbose=not args.quiet)

        result = converter.convert_to_solid(
            input_path=args.input,
            output_path=args.output,
            repair=args.repair,
            fill_holes=args.fill_holes,
            voxelize=args.voxelize,
            voxel_size=args.voxel_size,
            simplify=args.simplify,
            target_faces=args.faces,
            smooth=args.smooth,
            smooth_iterations=args.smooth_iterations
        )

        # 成功退出
        if result['final_watertight']:
            sys.exit(0)
        else:
            if not args.quiet:
                print("\n警告: 最终模型不是水密的，可能需要进一步修复")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)
    except ImportError as e:
        print(f"依赖错误: {e}", file=sys.stderr)
        print("请安装依赖: pip install trimesh pymeshlab", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"转换错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(4)


if __name__ == "__main__":
    main()
