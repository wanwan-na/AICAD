"""
Instant Meshes 集成模块

Instant Meshes 是一个强大的四边形重拓扑工具
GitHub: https://github.com/wjakob/instant-meshes

两种使用方式:
1. 命令行调用 (需要预先安装 Instant Meshes)
2. Python 绑定 (pyinstantmeshes, 如果可用)

安装 Instant Meshes:
- Windows: 下载预编译版本 https://github.com/wjakob/instant-meshes/releases
- 将 Instant Meshes.exe 放到系统 PATH 或指定路径
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import shutil

try:
    import numpy as np
except ImportError:
    np = None

try:
    import trimesh
except ImportError:
    trimesh = None


class InstantMeshesWrapper:
    """
    Instant Meshes 命令行封装

    提供四边形重拓扑功能，生成高质量四边形网格
    """

    # 默认可执行文件名
    EXECUTABLE_NAMES = [
        'Instant Meshes.exe',
        'Instant Meshes',
        'instant-meshes.exe',
        'instant-meshes',
        'InstantMeshes.exe',
        'InstantMeshes'
    ]

    def __init__(self, executable_path: Optional[str] = None):
        """
        初始化 Instant Meshes 封装

        Args:
            executable_path: Instant Meshes 可执行文件路径
                           如果为 None，会自动搜索
        """
        self.executable_path = executable_path or self._find_executable()

    def _find_executable(self) -> Optional[str]:
        """搜索 Instant Meshes 可执行文件"""
        # 搜索路径
        search_paths = [
            Path.cwd(),
            Path.cwd() / 'tools',
            Path.cwd() / 'bin',
            Path.home() / 'InstantMeshes',
            Path('C:/Program Files/Instant Meshes'),
            Path('C:/Program Files (x86)/Instant Meshes'),
        ]

        # 添加 PATH 环境变量中的路径
        path_env = os.environ.get('PATH', '')
        for p in path_env.split(os.pathsep):
            if p:
                search_paths.append(Path(p))

        # 搜索
        for search_path in search_paths:
            for exe_name in self.EXECUTABLE_NAMES:
                exe_path = search_path / exe_name
                if exe_path.exists():
                    return str(exe_path)

        return None

    def is_available(self) -> bool:
        """检查 Instant Meshes 是否可用"""
        return self.executable_path is not None and Path(self.executable_path).exists()

    def retopologize(self,
                     input_path: str,
                     output_path: Optional[str] = None,
                     target_vertex_count: Optional[int] = None,
                     target_face_count: Optional[int] = None,
                     crease_angle: float = 30.0,
                     smooth_iterations: int = 2,
                     deterministic: bool = True) -> Dict[str, Any]:
        """
        执行四边形重拓扑

        Args:
            input_path: 输入网格文件路径 (OBJ)
            output_path: 输出路径
            target_vertex_count: 目标顶点数
            target_face_count: 目标面数 (会转换为顶点数)
            crease_angle: 锐边角度阈值 (度)
            smooth_iterations: 平滑迭代次数
            deterministic: 是否使用确定性模式

        Returns:
            处理结果字典
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'Instant Meshes not found',
                'hint': 'Please download from https://github.com/wjakob/instant-meshes/releases'
            }

        input_path = Path(input_path)
        if not input_path.exists():
            return {'success': False, 'error': f'Input file not found: {input_path}'}

        # 确定输出路径
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_quad.obj"
        output_path = Path(output_path)

        # 估算目标顶点数
        if target_vertex_count is None and target_face_count is not None:
            # 四边形网格: 顶点数约为面数的一半
            target_vertex_count = target_face_count // 2

        # 构建命令
        cmd = [self.executable_path]

        # 输入输出
        cmd.extend(['-o', str(output_path)])

        # 目标顶点数
        if target_vertex_count:
            cmd.extend(['-v', str(target_vertex_count)])

        # 锐边角度
        cmd.extend(['-c', str(crease_angle)])

        # 平滑迭代
        cmd.extend(['-S', str(smooth_iterations)])

        # 确定性模式
        if deterministic:
            cmd.append('-d')

        # 输入文件 (放最后)
        cmd.append(str(input_path))

        # 执行
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0 and output_path.exists():
                # 分析输出结果
                return self._analyze_output(input_path, output_path)
            else:
                return {
                    'success': False,
                    'error': f'Instant Meshes failed: {result.stderr}',
                    'stdout': result.stdout,
                    'returncode': result.returncode
                }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Instant Meshes timeout (>5min)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _analyze_output(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """分析输出网格"""
        result = {
            'success': True,
            'input_path': str(input_path),
            'output_path': str(output_path)
        }

        if trimesh is not None:
            try:
                mesh = trimesh.load(str(output_path), force='mesh')
                result['vertex_count'] = len(mesh.vertices)
                result['face_count'] = len(mesh.faces)

                # 计算四边形比例 (Instant Meshes 输出应该是纯四边形)
                # 但 trimesh 会自动三角化，所以这里无法直接判断
                result['note'] = 'Mesh loaded as triangulated (trimesh auto-converts)'

            except Exception as e:
                result['analysis_error'] = str(e)

        return result


def quick_quad_retopo(input_path: str,
                      output_path: Optional[str] = None,
                      target_faces: int = 5000) -> Dict[str, Any]:
    """
    快速四边形重拓扑

    Args:
        input_path: 输入网格
        output_path: 输出路径
        target_faces: 目标面数

    Returns:
        处理结果
    """
    wrapper = InstantMeshesWrapper()

    if wrapper.is_available():
        return wrapper.retopologize(
            input_path,
            output_path,
            target_face_count=target_faces
        )
    else:
        return {
            'success': False,
            'error': 'Instant Meshes not installed',
            'hint': 'Download from: https://github.com/wjakob/instant-meshes/releases'
        }


def check_instant_meshes() -> Dict[str, Any]:
    """
    检查 Instant Meshes 安装状态

    Returns:
        状态信息
    """
    wrapper = InstantMeshesWrapper()

    if wrapper.is_available():
        return {
            'installed': True,
            'path': wrapper.executable_path,
            'message': 'Instant Meshes is available'
        }
    else:
        return {
            'installed': False,
            'path': None,
            'message': 'Instant Meshes not found',
            'install_hint': '''
To install Instant Meshes:
1. Download from: https://github.com/wjakob/instant-meshes/releases
2. Extract to a folder
3. Either:
   a) Add the folder to system PATH, or
   b) Copy "Instant Meshes.exe" to the AI_CAD/tools folder, or
   c) Specify the path when creating InstantMeshesWrapper
'''
        }
