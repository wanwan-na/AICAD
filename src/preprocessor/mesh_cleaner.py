"""
网格预处理模块

功能:
- 碎片移除: 删除孤立的小连通分量
- 非流形修复: 修复非流形边和顶点
- 网格简化: 减少面数同时保持形状
- 质量优化: 改善网格质量
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    import pymeshlab
except ImportError:
    pymeshlab = None

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class PreprocessConfig:
    """预处理配置参数"""
    # 碎片移除参数（设为0禁用，避免移除有效几何）
    min_component_ratio: float = 0.0  # 最小连通分量相对比例（相对于最大分量）

    # 非流形修复
    repair_non_manifold: bool = True

    # 去重参数
    remove_duplicates: bool = True
    merge_threshold: float = 1e-6  # 顶点合并距离阈值

    # 孔洞填充（默认关闭，避免改变原始拓扑）
    close_holes: bool = False
    max_hole_size: int = 30  # 最大填充孔洞的边数（降低）

    # 网格简化
    simplify: bool = False
    target_face_count: Optional[int] = None
    simplify_ratio: float = 0.5  # 简化比例（0.5表示减少到50%）

    # 法线修复
    fix_normals: bool = True


class MeshPreprocessor:
    """
    网格预处理器

    用于清理和修复AI生成的原始三角面网格，为后续重拓扑做准备。

    使用示例:
        preprocessor = MeshPreprocessor()
        result = preprocessor.process("input.obj", "output_cleaned.obj")
        print(f"处理完成: {result['output_path']}")
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        初始化预处理器

        Args:
            config: 预处理配置，为None时使用默认配置
        """
        if pymeshlab is None:
            raise ImportError("请安装 pymeshlab: pip install pymeshlab")

        self.config = config or PreprocessConfig()
        self._ms: Optional[pymeshlab.MeshSet] = None

    def process(self, input_path: str,
                output_path: Optional[str] = None,
                config: Optional[PreprocessConfig] = None) -> Dict[str, Any]:
        """
        执行完整的预处理流程

        Args:
            input_path: 输入网格文件路径
            output_path: 输出文件路径（可选，默认在输入文件名后加_cleaned）
            config: 可选的配置覆盖

        Returns:
            处理结果字典，包含统计信息
        """
        cfg = config or self.config
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 确定输出路径
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        output_path = Path(output_path)

        # 对于GLB/GLTF等带纹理的格式，先用trimesh提取纯几何数据
        suffix = input_path.suffix.lower()
        if suffix in ['.glb', '.gltf'] and trimesh is not None:
            # 使用trimesh加载，只提取几何数据
            mesh = trimesh.load(str(input_path), force='mesh')
            # 保存为临时OBJ文件（纯几何）
            temp_obj = input_path.parent / f"{input_path.stem}_temp_geometry.obj"
            mesh.export(str(temp_obj))
            # 用PyMeshLab加载OBJ
            self._ms = pymeshlab.MeshSet()
            self._ms.load_new_mesh(str(temp_obj))
            # 删除临时文件
            try:
                temp_obj.unlink()
            except Exception:
                pass
        else:
            # 创建MeshSet并加载
            self._ms = pymeshlab.MeshSet()
            self._ms.load_new_mesh(str(input_path))

        # 记录原始信息
        original_info = self._get_mesh_info()

        # 执行预处理步骤
        steps_applied = []

        # 1. 移除孤立碎片
        if cfg.min_component_ratio > 0:
            self._remove_small_components(cfg.min_component_ratio)
            steps_applied.append("remove_small_components")

        # 2. 移除重复元素
        if cfg.remove_duplicates:
            self._remove_duplicates(cfg.merge_threshold)
            steps_applied.append("remove_duplicates")

        # 3. 修复非流形
        if cfg.repair_non_manifold:
            self._repair_non_manifold()
            steps_applied.append("repair_non_manifold")

        # 4. 填充孔洞
        if cfg.close_holes:
            self._close_holes(cfg.max_hole_size)
            steps_applied.append("close_holes")

        # 5. 修复法线
        if cfg.fix_normals:
            self._fix_normals()
            steps_applied.append("fix_normals")

        # 6. 网格简化（可选）
        if cfg.simplify and (cfg.target_face_count or cfg.simplify_ratio < 1.0):
            target = cfg.target_face_count or int(original_info['face_count'] * cfg.simplify_ratio)
            self._simplify_mesh(target)
            steps_applied.append("simplify")

        # 检查网格是否有效
        processed_info = self._get_mesh_info()

        if processed_info['vertex_count'] == 0 or processed_info['face_count'] == 0:
            # 如果处理后网格为空，重新加载原始网格
            self._ms = pymeshlab.MeshSet()
            self._ms.load_new_mesh(str(input_path))
            processed_info = self._get_mesh_info()
            steps_applied = ['skipped_all_empty_result']

        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ms.save_current_mesh(str(output_path))

        return {
            'success': True,
            'input_path': str(input_path),
            'output_path': str(output_path),
            'steps_applied': steps_applied,
            'original': original_info,
            'processed': processed_info,
            'vertices_removed': original_info['vertex_count'] - processed_info['vertex_count'],
            'faces_removed': original_info['face_count'] - processed_info['face_count']
        }

    def _get_mesh_info(self) -> Dict[str, Any]:
        """获取当前网格信息"""
        mesh = self._ms.current_mesh()
        bbox = mesh.bounding_box()

        return {
            'vertex_count': mesh.vertex_number(),
            'face_count': mesh.face_number(),
            'edge_count': mesh.edge_number(),
            'bbox_diagonal': bbox.diagonal(),
            'bbox_min': [bbox.min()[i] for i in range(3)],
            'bbox_max': [bbox.max()[i] for i in range(3)]
        }

    def _remove_small_components(self, min_ratio: float):
        """移除小的孤立连通分量"""
        try:
            # 按直径比例移除小分量
            self._ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pymeshlab.PercentageValue(min_ratio * 100)
            )
        except Exception:
            # 备选方案：按面数移除
            try:
                self._ms.meshing_remove_connected_component_by_face_number(
                    mincomponentsize=100
                )
            except Exception:
                pass

    def _remove_duplicates(self, threshold: float):
        """移除重复的顶点和面"""
        try:
            # 合并相近顶点
            self._ms.meshing_merge_close_vertices(
                threshold=pymeshlab.AbsoluteValue(threshold)
            )
        except Exception:
            pass

        try:
            # 移除重复面
            self._ms.meshing_remove_duplicate_faces()
        except Exception:
            pass

        try:
            # 移除重复顶点
            self._ms.meshing_remove_duplicate_vertices()
        except Exception:
            pass

    def _repair_non_manifold(self):
        """修复非流形边和顶点（增强版）"""
        # 多次尝试修复，直到没有非流形问题或达到最大次数
        max_iterations = 5

        for _ in range(max_iterations):
            had_issues = False

            # 1. 移除孤立顶点
            try:
                self._ms.meshing_remove_unreferenced_vertices()
            except Exception:
                pass

            # 2. 移除零面积面
            try:
                self._ms.meshing_remove_null_faces()
            except Exception:
                pass

            # 3. 移除重复面
            try:
                self._ms.meshing_remove_duplicate_faces()
            except Exception:
                pass

            # 4. 修复非流形边（方法0：通过移除面）
            try:
                self._ms.meshing_repair_non_manifold_edges(method=0)
                had_issues = True
            except Exception:
                pass

            # 5. 修复非流形顶点（通过分裂）
            try:
                self._ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
                had_issues = True
            except Exception:
                pass

            # 6. 再次尝试用方法1修复非流形边（通过分裂边）
            try:
                self._ms.meshing_repair_non_manifold_edges(method=1)
            except Exception:
                pass

            if not had_issues:
                break

    def _close_holes(self, max_size: int):
        """填充小孔洞"""
        try:
            self._ms.meshing_close_holes(maxholesize=max_size)
        except Exception:
            pass

    def _fix_normals(self):
        """修复和统一法线方向"""
        try:
            # 重新计算法线
            self._ms.compute_normal_per_vertex()
            self._ms.compute_normal_per_face()
        except Exception:
            pass

        try:
            # 统一法线方向
            self._ms.meshing_re_orient_faces_coherentely()
        except Exception:
            pass

    def _simplify_mesh(self, target_faces: int):
        """简化网格到目标面数"""
        try:
            self._ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces,
                preservenormal=True,
                preservetopology=True,
                qualitythr=0.3
            )
        except Exception:
            pass

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        分析网格文件，返回问题报告

        Args:
            file_path: 网格文件路径

        Returns:
            分析报告字典
        """
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
        mesh = ms.current_mesh()

        issues = []

        # 检查非流形边
        try:
            # 通过选择非流形边来检测
            ms.compute_selection_by_non_manifold_edges_per_face()
            selected = mesh.selected_face_number()
            if selected > 0:
                issues.append({
                    'type': 'non_manifold_edges',
                    'count': selected,
                    'severity': 'high'
                })
        except Exception:
            pass

        # 检查孤立顶点
        vertex_count = mesh.vertex_number()
        face_count = mesh.face_number()

        # 基本统计
        bbox = mesh.bounding_box()

        return {
            'file_path': file_path,
            'vertex_count': vertex_count,
            'face_count': face_count,
            'bbox_diagonal': bbox.diagonal(),
            'issues': issues,
            'needs_preprocessing': len(issues) > 0,
            'recommendation': self._generate_recommendation(issues)
        }

    def _generate_recommendation(self, issues: list) -> str:
        """根据问题生成建议"""
        if not issues:
            return "网格质量良好，可直接进行重拓扑处理"

        recommendations = []
        for issue in issues:
            if issue['type'] == 'non_manifold_edges':
                recommendations.append("建议修复非流形边")
            elif issue['type'] == 'small_components':
                recommendations.append("建议移除孤立碎片")

        return "；".join(recommendations) if recommendations else "建议进行标准预处理"
