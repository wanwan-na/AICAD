"""
混元3D专用预处理器

针对混元3D (Hunyuan3D) 生成的3D模型进行优化预处理。

混元3D生成的模型常见问题：
1. 破面（非流形边、非流形顶点）
2. 孔洞（边界不闭合）
3. 重叠/重复面
4. 退化面（零面积三角形）
5. 顶点重复

本模块提供多阶段修复流程，确保输出的网格适合后续的四边面转换。
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import pymeshlab
except ImportError:
    pymeshlab = None

from .mesh_repair import (
    repair_mesh,
    repair_mesh_aggressive,
    RepairConfig,
    analyze_mesh_topology
)


@dataclass
class HunyuanPreprocessConfig:
    """混元3D预处理配置"""
    # 阶段1：几何清理
    merge_close_vertices: bool = True
    merge_threshold: float = 1e-5  # 更宽松的阈值，AI模型常有微小间隙

    # 阶段2：拓扑修复
    repair_non_manifold: bool = True
    remove_duplicate_faces: bool = True
    remove_degenerate_faces: bool = True

    # 阶段3：孔洞填充
    fill_holes: bool = True
    max_hole_vertices: int = 300  # 混元模型的孔洞可能较大
    use_ear_clipping: bool = True  # 更好的三角剖分

    # 阶段4：PyMeshLab后处理
    use_pymeshlab_repair: bool = True

    # 阶段5：验证和重试
    max_repair_iterations: int = 3
    target_watertight: bool = True  # 目标是水密网格

    # 输出选项
    verbose: bool = True


class HunyuanPreprocessor:
    """
    混元3D专用预处理器

    使用示例:
        preprocessor = HunyuanPreprocessor()
        result = preprocessor.process("hunyuan_output.glb", "cleaned.obj")
        print(f"修复了 {result['holes_filled']} 个孔洞")
    """

    def __init__(self, config: Optional[HunyuanPreprocessConfig] = None):
        self.config = config or HunyuanPreprocessConfig()

    def process(self,
                input_path: str,
                output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        处理混元3D生成的模型

        Args:
            input_path: 输入文件路径 (支持 GLB, GLTF, OBJ, STL, PLY)
            output_path: 输出文件路径（可选）

        Returns:
            处理结果字典
        """
        if trimesh is None:
            raise ImportError("需要安装 trimesh: pip install trimesh")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_preprocessed.obj"
        output_path = Path(output_path)

        cfg = self.config

        if cfg.verbose:
            print("=" * 60)
            print("混元3D 预处理器")
            print("=" * 60)
            print(f"\n输入文件: {input_path}")

        # 加载网格
        mesh = trimesh.load(str(input_path), force='mesh')
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)

        # 初始拓扑分析
        initial_topo = analyze_mesh_topology(vertices, faces)

        if cfg.verbose:
            print(f"\n初始状态:")
            print(f"  顶点数: {initial_topo['vertex_count']}")
            print(f"  面数: {initial_topo['face_count']}")
            print(f"  边界边数: {initial_topo['boundary_edge_count']}")
            print(f"  非流形边数: {initial_topo['non_manifold_edge_count']}")
            print(f"  孔洞数: {initial_topo['hole_count']}")
            print(f"  水密: {'是' if initial_topo['is_watertight'] else '否'}")
            print(f"  流形: {'是' if initial_topo['is_manifold'] else '否'}")

        # 多轮修复
        best_vertices = vertices
        best_faces = faces
        best_topo = initial_topo

        for iteration in range(cfg.max_repair_iterations):
            if cfg.verbose:
                print(f"\n--- 修复迭代 {iteration + 1}/{cfg.max_repair_iterations} ---")

            # 使用增强的mesh_repair
            repair_cfg = RepairConfig(
                fill_holes=cfg.fill_holes,
                max_hole_vertices=cfg.max_hole_vertices,
                advanced_hole_filling=True,
                use_ear_clipping=cfg.use_ear_clipping,
                repair_non_manifold=cfg.repair_non_manifold,
                remove_duplicate_faces=cfg.remove_duplicate_faces,
                remove_degenerate_faces=cfg.remove_degenerate_faces,
                merge_close_vertices=cfg.merge_close_vertices,
                merge_threshold=cfg.merge_threshold,
            )

            new_verts, new_faces = repair_mesh(best_vertices, best_faces, repair_cfg)

            # 分析修复后的拓扑
            new_topo = analyze_mesh_topology(new_verts, new_faces)

            if cfg.verbose:
                print(f"  修复后:")
                print(f"    边界边数: {new_topo['boundary_edge_count']} "
                      f"({'↓' if new_topo['boundary_edge_count'] < best_topo['boundary_edge_count'] else '↑'})")
                print(f"    非流形边数: {new_topo['non_manifold_edge_count']}")
                print(f"    孔洞数: {new_topo['hole_count']}")

            # 更新最佳结果
            best_vertices = new_verts
            best_faces = new_faces
            best_topo = new_topo

            # 检查是否达到目标
            if cfg.target_watertight and best_topo['is_watertight'] and best_topo['is_manifold']:
                if cfg.verbose:
                    print(f"\n  [OK] 达到水密流形状态!")
                break

        # 使用PyMeshLab进行额外修复
        if cfg.use_pymeshlab_repair and pymeshlab is not None:
            if cfg.verbose:
                print(f"\n--- PyMeshLab 后处理 ---")

            best_vertices, best_faces = self._pymeshlab_repair(
                best_vertices, best_faces, cfg.verbose
            )

            # 再次分析
            best_topo = analyze_mesh_topology(best_vertices, best_faces)

        # 最终分析
        final_topo = analyze_mesh_topology(best_vertices, best_faces)

        if cfg.verbose:
            print(f"\n最终状态:")
            print(f"  顶点数: {final_topo['vertex_count']}")
            print(f"  面数: {final_topo['face_count']}")
            print(f"  边界边数: {final_topo['boundary_edge_count']}")
            print(f"  孔洞数: {final_topo['hole_count']}")
            print(f"  水密: {'是' if final_topo['is_watertight'] else '否'}")
            print(f"  流形: {'是' if final_topo['is_manifold'] else '否'}")

        # 保存结果
        output_mesh = trimesh.Trimesh(vertices=best_vertices, faces=best_faces)
        output_mesh.export(str(output_path))

        if cfg.verbose:
            print(f"\n输出文件: {output_path}")

        return {
            'success': True,
            'input_path': str(input_path),
            'output_path': str(output_path),
            'initial_topology': initial_topo,
            'final_topology': final_topo,
            'holes_filled': initial_topo['hole_count'] - final_topo['hole_count'],
            'boundary_edges_reduced': initial_topo['boundary_edge_count'] - final_topo['boundary_edge_count'],
            'is_watertight': final_topo['is_watertight'],
            'is_manifold': final_topo['is_manifold'],
        }

    def _pymeshlab_repair(self,
                          vertices: np.ndarray,
                          faces: np.ndarray,
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """使用PyMeshLab进行额外修复"""
        import tempfile
        import os

        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            temp_input = f.name

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            temp_output = f.name

        try:
            # 保存
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(temp_input)

            # PyMeshLab处理
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_input)

            # 1. 移除孤立顶点
            try:
                ms.meshing_remove_unreferenced_vertices()
            except Exception:
                pass

            # 2. 移除重复面
            try:
                ms.meshing_remove_duplicate_faces()
            except Exception:
                pass

            # 3. 移除零面积面
            try:
                ms.meshing_remove_null_faces()
            except Exception:
                pass

            # 4. 修复非流形边
            try:
                ms.meshing_repair_non_manifold_edges(method=0)
            except Exception:
                pass

            try:
                ms.meshing_repair_non_manifold_edges(method=1)
            except Exception:
                pass

            # 5. 修复非流形顶点
            try:
                ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
            except Exception:
                pass

            # 6. 填充孔洞
            try:
                ms.meshing_close_holes(maxholesize=100)
            except Exception:
                pass

            # 7. 重新计算法线
            try:
                ms.compute_normal_per_vertex()
                ms.compute_normal_per_face()
                ms.meshing_re_orient_faces_coherentely()
            except Exception:
                pass

            # 保存结果
            ms.save_current_mesh(temp_output)

            # 重新加载
            result_mesh = trimesh.load(temp_output, force='mesh')
            return np.array(result_mesh.vertices), np.array(result_mesh.faces)

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_input)
                os.unlink(temp_output)
            except Exception:
                pass

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        分析混元3D生成的模型

        Args:
            file_path: 网格文件路径

        Returns:
            分析报告
        """
        if trimesh is None:
            raise ImportError("需要安装 trimesh: pip install trimesh")

        mesh = trimesh.load(str(file_path), force='mesh')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        topo = analyze_mesh_topology(vertices, faces)

        # 生成建议
        issues = []
        recommendations = []

        if not topo['is_watertight']:
            issues.append(f"存在 {topo['hole_count']} 个孔洞")
            recommendations.append("使用 --fill-holes 选项填充孔洞")

        if not topo['is_manifold']:
            issues.append(f"存在 {topo['non_manifold_edge_count']} 条非流形边")
            recommendations.append("使用预处理修复非流形几何")

        if topo['boundary_edge_count'] > 0:
            issues.append(f"存在 {topo['boundary_edge_count']} 条边界边")

        return {
            'file_path': file_path,
            'topology': topo,
            'issues': issues,
            'recommendations': recommendations,
            'needs_repair': not (topo['is_watertight'] and topo['is_manifold']),
            'severity': 'high' if not topo['is_manifold'] else
                       'medium' if not topo['is_watertight'] else 'low'
        }


def preprocess_hunyuan_mesh(input_path: str,
                            output_path: Optional[str] = None,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    便捷函数：预处理混元3D生成的模型

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        verbose: 是否打印详细信息

    Returns:
        处理结果

    示例:
        result = preprocess_hunyuan_mesh("model.glb", "model_clean.obj")
    """
    config = HunyuanPreprocessConfig(verbose=verbose)
    preprocessor = HunyuanPreprocessor(config)
    return preprocessor.process(input_path, output_path)
