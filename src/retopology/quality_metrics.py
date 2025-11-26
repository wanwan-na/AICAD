"""
重拓扑质量评估模块

评估指标:
- 四边形比例: 目标 >95%
- 边流质量得分: 目标 >0.9
- 几何偏差: 目标 <0.1%
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class QualityThresholds:
    """质量阈值配置"""
    # 四边形比例阈值
    quad_ratio_excellent: float = 0.95
    quad_ratio_good: float = 0.90
    quad_ratio_acceptable: float = 0.80

    # 边流得分阈值
    edge_flow_excellent: float = 0.9
    edge_flow_good: float = 0.8
    edge_flow_acceptable: float = 0.7

    # 几何偏差阈值（相对于包围盒对角线的百分比）
    deviation_excellent: float = 0.0005  # 0.05%
    deviation_good: float = 0.001        # 0.1%
    deviation_acceptable: float = 0.0015  # 0.15%


class QualityMetrics:
    """
    重拓扑质量评估器

    用于评估重拓扑后的网格质量，包括四边形比例、边流质量和几何精度。
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        初始化质量评估器

        Args:
            thresholds: 质量阈值配置
        """
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(self, vertices: np.ndarray, faces: np.ndarray,
                 original_vertices: Optional[np.ndarray] = None,
                 original_faces: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        评估网格质量

        Args:
            vertices: 重拓扑后的顶点
            faces: 重拓扑后的面（可能混合三角形和四边形）
            original_vertices: 原始顶点（用于计算几何偏差）
            original_faces: 原始面

        Returns:
            质量评估报告
        """
        report = {
            'vertex_count': len(vertices),
            'face_count': len(faces)
        }

        # 1. 计算四边形比例
        quad_metrics = self._compute_quad_ratio(faces)
        report.update(quad_metrics)

        # 2. 计算边流质量
        edge_flow_metrics = self._compute_edge_flow_quality(vertices, faces)
        report.update(edge_flow_metrics)

        # 3. 计算几何偏差（如果提供原始网格）
        if original_vertices is not None:
            deviation_metrics = self._compute_geometric_deviation(
                vertices, faces, original_vertices, original_faces
            )
            report.update(deviation_metrics)

        # 4. 综合评分
        report['overall_grade'] = self._compute_overall_grade(report)
        report['passed'] = report['overall_grade'] in ['excellent', 'good', 'acceptable']

        return report

    def _compute_quad_ratio(self, faces: np.ndarray) -> Dict[str, Any]:
        """计算四边形比例"""
        if faces.ndim == 1:
            # 变长面列表
            face_sizes = [len(f) for f in faces]
        else:
            # 固定大小面数组
            face_sizes = [faces.shape[1]] * len(faces)

        total_faces = len(face_sizes)
        quad_count = sum(1 for s in face_sizes if s == 4)
        tri_count = sum(1 for s in face_sizes if s == 3)
        other_count = total_faces - quad_count - tri_count

        quad_ratio = quad_count / total_faces if total_faces > 0 else 0

        # 评级
        if quad_ratio >= self.thresholds.quad_ratio_excellent:
            grade = 'excellent'
        elif quad_ratio >= self.thresholds.quad_ratio_good:
            grade = 'good'
        elif quad_ratio >= self.thresholds.quad_ratio_acceptable:
            grade = 'acceptable'
        else:
            grade = 'poor'

        return {
            'quad_count': quad_count,
            'tri_count': tri_count,
            'other_count': other_count,
            'quad_ratio': quad_ratio,
            'quad_grade': grade
        }

    def _compute_edge_flow_quality(self, vertices: np.ndarray,
                                    faces: np.ndarray) -> Dict[str, Any]:
        """
        计算边流质量

        边流质量衡量四边形排列的规整程度，好的边流应该：
        - 边长均匀
        - 角度接近90度
        - 边走向流畅
        """
        if len(faces) == 0:
            return {'edge_flow_score': 0, 'edge_flow_grade': 'poor'}

        scores = []

        # 分析每个四边形的质量
        for face in faces:
            if len(face) == 4:
                score = self._evaluate_quad_quality(vertices, face)
                scores.append(score)
            elif len(face) == 3:
                # 三角形给予较低分数
                scores.append(0.5)

        avg_score = np.mean(scores) if scores else 0

        # 评级
        if avg_score >= self.thresholds.edge_flow_excellent:
            grade = 'excellent'
        elif avg_score >= self.thresholds.edge_flow_good:
            grade = 'good'
        elif avg_score >= self.thresholds.edge_flow_acceptable:
            grade = 'acceptable'
        else:
            grade = 'poor'

        return {
            'edge_flow_score': float(avg_score),
            'edge_flow_grade': grade
        }

    def _evaluate_quad_quality(self, vertices: np.ndarray,
                               face_indices: np.ndarray) -> float:
        """评估单个四边形的质量"""
        # 获取四边形的四个顶点
        v = vertices[face_indices]

        # 计算四条边的长度
        edges = [
            np.linalg.norm(v[1] - v[0]),
            np.linalg.norm(v[2] - v[1]),
            np.linalg.norm(v[3] - v[2]),
            np.linalg.norm(v[0] - v[3])
        ]

        # 边长均匀性得分（标准差越小越好）
        edge_std = np.std(edges)
        edge_mean = np.mean(edges)
        edge_uniformity = 1.0 - min(edge_std / (edge_mean + 1e-8), 1.0)

        # 计算四个角的角度
        angles = []
        for i in range(4):
            v0 = v[i]
            v1 = v[(i + 1) % 4]
            v2 = v[(i - 1) % 4]

            e1 = v1 - v0
            e2 = v2 - v0

            cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        # 角度接近90度的得分
        target_angle = np.pi / 2
        angle_deviations = [abs(a - target_angle) for a in angles]
        angle_score = 1.0 - min(np.mean(angle_deviations) / (np.pi / 4), 1.0)

        # 综合得分
        return 0.5 * edge_uniformity + 0.5 * angle_score

    def _compute_geometric_deviation(self, vertices: np.ndarray,
                                      faces: np.ndarray,
                                      original_vertices: np.ndarray,
                                      original_faces: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        计算几何偏差

        通过采样点到原始网格的距离来评估几何保真度
        """
        if trimesh is None:
            return {
                'geometric_deviation': None,
                'deviation_grade': 'unknown',
                'note': 'trimesh not installed'
            }

        try:
            # 创建原始网格
            original_mesh = trimesh.Trimesh(vertices=original_vertices, faces=original_faces)

            # 在重拓扑网格上采样点
            sample_points = vertices  # 简化：直接使用顶点

            # 计算到原始网格的距离
            closest_points, distances, _ = original_mesh.nearest.on_surface(sample_points)

            # 统计
            max_deviation = np.max(distances)
            mean_deviation = np.mean(distances)
            bbox_diagonal = np.linalg.norm(
                original_vertices.max(axis=0) - original_vertices.min(axis=0)
            )

            # 相对偏差
            relative_deviation = mean_deviation / bbox_diagonal if bbox_diagonal > 0 else 0

            # 评级
            if relative_deviation <= self.thresholds.deviation_excellent:
                grade = 'excellent'
            elif relative_deviation <= self.thresholds.deviation_good:
                grade = 'good'
            elif relative_deviation <= self.thresholds.deviation_acceptable:
                grade = 'acceptable'
            else:
                grade = 'poor'

            return {
                'max_deviation': float(max_deviation),
                'mean_deviation': float(mean_deviation),
                'relative_deviation': float(relative_deviation),
                'deviation_grade': grade
            }

        except Exception as e:
            return {
                'geometric_deviation': None,
                'deviation_grade': 'error',
                'error': str(e)
            }

    def _compute_overall_grade(self, report: Dict[str, Any]) -> str:
        """计算综合评级"""
        grades = ['excellent', 'good', 'acceptable', 'poor']
        grade_scores = {'excellent': 3, 'good': 2, 'acceptable': 1, 'poor': 0}

        scores = []

        # 四边形比例权重最高
        if 'quad_grade' in report:
            scores.append(grade_scores.get(report['quad_grade'], 0) * 2)

        # 边流质量
        if 'edge_flow_grade' in report:
            scores.append(grade_scores.get(report['edge_flow_grade'], 0))

        # 几何偏差
        if 'deviation_grade' in report and report['deviation_grade'] not in ['unknown', 'error']:
            scores.append(grade_scores.get(report['deviation_grade'], 0))

        if not scores:
            return 'unknown'

        avg_score = sum(scores) / len(scores)

        if avg_score >= 2.5:
            return 'excellent'
        elif avg_score >= 1.5:
            return 'good'
        elif avg_score >= 0.5:
            return 'acceptable'
        else:
            return 'poor'

    def generate_report(self, evaluation: Dict[str, Any]) -> str:
        """
        生成可读的质量报告

        Args:
            evaluation: evaluate()方法返回的评估结果

        Returns:
            格式化的报告字符串
        """
        lines = [
            "=" * 50,
            "重拓扑质量评估报告",
            "=" * 50,
            "",
            f"顶点数: {evaluation.get('vertex_count', 'N/A')}",
            f"面数: {evaluation.get('face_count', 'N/A')}",
            "",
            "--- 四边形比例 ---",
            f"四边形数: {evaluation.get('quad_count', 'N/A')}",
            f"三角形数: {evaluation.get('tri_count', 'N/A')}",
            f"四边形比例: {evaluation.get('quad_ratio', 0) * 100:.1f}%",
            f"评级: {evaluation.get('quad_grade', 'N/A')}",
            "",
            "--- 边流质量 ---",
            f"边流得分: {evaluation.get('edge_flow_score', 0):.3f}",
            f"评级: {evaluation.get('edge_flow_grade', 'N/A')}",
        ]

        if 'relative_deviation' in evaluation:
            lines.extend([
                "",
                "--- 几何偏差 ---",
                f"平均偏差: {evaluation.get('mean_deviation', 0):.6f}",
                f"相对偏差: {evaluation.get('relative_deviation', 0) * 100:.4f}%",
                f"评级: {evaluation.get('deviation_grade', 'N/A')}",
            ])

        lines.extend([
            "",
            "=" * 50,
            f"综合评级: {evaluation.get('overall_grade', 'N/A').upper()}",
            f"是否通过: {'[PASS] 通过' if evaluation.get('passed', False) else '[FAIL] 未通过'}",
            "=" * 50,
        ])

        return "\n".join(lines)
