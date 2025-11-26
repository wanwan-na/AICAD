"""
重拓扑模块测试

测试内容:
1. 网格预处理功能
2. 重拓扑核心算法
3. 质量评估指标
4. 完整流水线
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查依赖
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


def create_test_cube():
    """创建测试用立方体网格"""
    if not HAS_TRIMESH:
        return None, None

    # 立方体顶点
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    # 三角面（每个立方体面分成2个三角形）
    faces = np.array([
        # 底面
        [0, 1, 2], [0, 2, 3],
        # 顶面
        [4, 6, 5], [4, 7, 6],
        # 前面
        [0, 5, 1], [0, 4, 5],
        # 后面
        [2, 7, 3], [2, 6, 7],
        # 左面
        [0, 3, 7], [0, 7, 4],
        # 右面
        [1, 5, 6], [1, 6, 2]
    ], dtype=np.int64)

    return vertices, faces


def create_test_mesh_file(directory: str) -> str:
    """创建测试网格文件"""
    if not HAS_TRIMESH:
        return None

    vertices, faces = create_test_cube()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    file_path = Path(directory) / "test_cube.obj"
    mesh.export(str(file_path))

    return str(file_path)


class TestMeshIO(unittest.TestCase):
    """测试网格IO功能"""

    @unittest.skipIf(not HAS_TRIMESH, "trimesh not installed")
    def test_load_and_save(self):
        """测试网格加载和保存"""
        from src.utils.mesh_io import MeshIO

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            input_path = create_test_mesh_file(tmpdir)
            self.assertIsNotNone(input_path)

            # 加载
            vertices, faces = MeshIO.load_mesh(input_path)

            self.assertEqual(len(vertices), 8)  # 立方体有8个顶点
            self.assertEqual(len(faces), 12)    # 12个三角形

            # 保存
            output_path = Path(tmpdir) / "output.obj"
            MeshIO.save_mesh(str(output_path), vertices, faces)

            self.assertTrue(output_path.exists())

    @unittest.skipIf(not HAS_TRIMESH, "trimesh not installed")
    def test_get_mesh_info(self):
        """测试获取网格信息"""
        from src.utils.mesh_io import MeshIO

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = create_test_mesh_file(tmpdir)

            info = MeshIO.get_mesh_info(input_path)

            self.assertEqual(info['vertex_count'], 8)
            self.assertEqual(info['face_count'], 12)
            self.assertAlmostEqual(info['bbox_diagonal'], np.sqrt(3), places=5)


class TestMeshPreprocessor(unittest.TestCase):
    """测试网格预处理功能"""

    @unittest.skipIf(not HAS_PYMESHLAB, "pymeshlab not installed")
    def test_preprocess_basic(self):
        """测试基本预处理"""
        from src.preprocessor import MeshPreprocessor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = create_test_mesh_file(tmpdir)
            if input_path is None:
                self.skipTest("Cannot create test mesh")

            preprocessor = MeshPreprocessor()
            result = preprocessor.process(input_path)

            self.assertTrue(result['success'])
            self.assertIn('output_path', result)
            self.assertTrue(Path(result['output_path']).exists())

    @unittest.skipIf(not HAS_PYMESHLAB, "pymeshlab not installed")
    def test_analyze(self):
        """测试网格分析"""
        from src.preprocessor import MeshPreprocessor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = create_test_mesh_file(tmpdir)
            if input_path is None:
                self.skipTest("Cannot create test mesh")

            preprocessor = MeshPreprocessor()
            analysis = preprocessor.analyze(input_path)

            self.assertEqual(analysis['vertex_count'], 8)
            self.assertEqual(analysis['face_count'], 12)
            self.assertIn('recommendation', analysis)


class TestQualityMetrics(unittest.TestCase):
    """测试质量评估功能"""

    def test_quad_ratio_calculation(self):
        """测试四边形比例计算"""
        from src.retopology.quality_metrics import QualityMetrics

        metrics = QualityMetrics()

        # 创建测试数据：混合三角形和四边形
        vertices = np.random.rand(10, 3)

        # 全是三角形
        tri_faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = metrics.evaluate(vertices, tri_faces)
        self.assertEqual(result['quad_ratio'], 0)
        self.assertEqual(result['quad_grade'], 'poor')

    def test_quality_report_generation(self):
        """测试质量报告生成"""
        from src.retopology.quality_metrics import QualityMetrics

        metrics = QualityMetrics()

        # 模拟评估结果
        evaluation = {
            'vertex_count': 100,
            'face_count': 50,
            'quad_count': 45,
            'tri_count': 5,
            'quad_ratio': 0.9,
            'quad_grade': 'good',
            'edge_flow_score': 0.85,
            'edge_flow_grade': 'good',
            'overall_grade': 'good',
            'passed': True
        }

        report = metrics.generate_report(evaluation)

        self.assertIn('四边形比例', report)
        self.assertIn('90.0%', report)
        self.assertIn('综合评级', report)


class TestRetopologyProcessor(unittest.TestCase):
    """测试重拓扑处理器"""

    @unittest.skipIf(not HAS_PYMESHLAB, "pymeshlab not installed")
    def test_basic_retopology(self):
        """测试基本重拓扑功能"""
        from src.retopology import RetopologyProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = create_test_mesh_file(tmpdir)
            if input_path is None:
                self.skipTest("Cannot create test mesh")

            processor = RetopologyProcessor()
            result = processor.process(input_path)

            self.assertTrue(result['success'])
            self.assertIn('quality', result)
            self.assertIn('quad_ratio', result['quality'])

    @unittest.skipIf(not HAS_PYMESHLAB, "pymeshlab not installed")
    def test_retopology_with_config(self):
        """测试带配置的重拓扑"""
        from src.retopology import RetopologyProcessor, RetopologyConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = create_test_mesh_file(tmpdir)
            if input_path is None:
                self.skipTest("Cannot create test mesh")

            config = RetopologyConfig(
                target_face_count=20,
                max_retries=2
            )

            processor = RetopologyProcessor(config)
            result = processor.process(input_path)

            self.assertTrue(result['success'])


class TestPipeline(unittest.TestCase):
    """测试完整流水线"""

    @unittest.skipIf(not (HAS_PYMESHLAB and HAS_TRIMESH),
                     "Required dependencies not installed")
    def test_full_pipeline(self):
        """测试完整重拓扑流水线"""
        from main import retopology_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = create_test_mesh_file(tmpdir)
            if input_path is None:
                self.skipTest("Cannot create test mesh")

            output_path = Path(tmpdir) / "output_quad.obj"

            result = retopology_pipeline(
                input_path=input_path,
                output_path=str(output_path),
                verbose=False
            )

            self.assertIn('output_path', result)
            self.assertIn('quality', result)
            self.assertTrue(Path(result['output_path']).exists())


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        from src.retopology import RetopologyProcessor

        processor = RetopologyProcessor()

        with self.assertRaises(FileNotFoundError):
            processor.process("nonexistent_file.obj")

    @unittest.skipIf(not HAS_PYMESHLAB, "pymeshlab not installed")
    def test_empty_config(self):
        """测试空配置"""
        from src.retopology import RetopologyConfig

        config = RetopologyConfig()

        self.assertIsNone(config.target_face_count)
        self.assertEqual(config.auto_face_ratio, 0.4)
        self.assertEqual(config.max_retries, 3)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestMeshIO))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestRetopologyProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
