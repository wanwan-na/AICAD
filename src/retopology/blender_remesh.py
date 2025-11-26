"""
Blender 四边面重拓扑模块

使用 Blender 的 Remesh 功能将三角面网格转换为四边面网格。

Blender 提供多种 Remesh 方法:
1. QuadriFlow - 生成纯四边面，适合CAD
2. Voxel Remesh - 基于体素，可产生更均匀的网格
3. Sharp/Smooth/Blocks - 传统 Remesh 方法

使用方法:
    remesher = BlenderRemesh(blender_path="path/to/blender.exe")
    result = remesher.remesh_to_quads(input_mesh, output_path)
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Blender Python 脚本模板
BLENDER_REMESH_SCRIPT = '''
import bpy
import sys
import json

def remesh_to_quads(input_path, output_path, method, target_faces, smooth_iterations):
    """在 Blender 中执行四边面重拓扑"""

    # 清除场景
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 导入模型
    ext = input_path.lower().split('.')[-1]
    if ext == 'obj':
        bpy.ops.wm.obj_import(filepath=input_path)
    elif ext == 'stl':
        bpy.ops.wm.stl_import(filepath=input_path)
    elif ext == 'ply':
        bpy.ops.wm.ply_import(filepath=input_path)
    elif ext in ['glb', 'gltf']:
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath=input_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # 获取导入的物体
    obj = None
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            obj = o
            break

    if obj is None:
        raise RuntimeError("No mesh object found after import")

    # 选择并激活物体
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # 记录原始信息
    original_verts = len(obj.data.vertices)
    original_faces = len(obj.data.polygons)

    # 根据方法执行重拓扑
    if method == 'quadriflow':
        # QuadriFlow - 生成纯四边面
        bpy.ops.object.quadriflow_remesh(
            use_mesh_symmetry=False,
            use_preserve_sharp=True,
            use_preserve_boundary=True,
            preserve_paint_mask=False,
            smooth_normals=True,
            target_faces=target_faces,
            mode='FACES',
            seed=0
        )

    elif method == 'voxel':
        # Voxel Remesh + Decimate to quads
        # 先添加 Remesh 修改器
        remesh_mod = obj.modifiers.new(name='Remesh', type='REMESH')
        remesh_mod.mode = 'VOXEL'
        # 根据目标面数计算 voxel 大小
        bbox = obj.dimensions
        volume = bbox.x * bbox.y * bbox.z
        voxel_size = (volume / target_faces) ** (1/3) * 2
        remesh_mod.voxel_size = max(0.01, voxel_size)
        remesh_mod.adaptivity = 0.0
        remesh_mod.use_smooth_shade = True

        # 应用修改器
        bpy.ops.object.modifier_apply(modifier='Remesh')

        # 转为编辑模式，将三角面转为四边面
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=0.698132,  # 40度
            shape_threshold=0.698132,
            uvs=False,
            vcols=False,
            seam=False,
            sharp=False,
            materials=False
        )
        bpy.ops.object.mode_set(mode='OBJECT')

    elif method == 'decimate_planar':
        # 使用 Decimate 的 Planar 模式
        decimate_mod = obj.modifiers.new(name='Decimate', type='DECIMATE')
        decimate_mod.decimate_type = 'DISSOLVE'
        decimate_mod.angle_limit = 0.0872665  # 5度
        decimate_mod.use_dissolve_boundaries = False

        # 应用修改器
        bpy.ops.object.modifier_apply(modifier='Decimate')

        # 转为编辑模式，将三角面转为四边面
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=0.698132,
            shape_threshold=0.698132
        )
        bpy.ops.object.mode_set(mode='OBJECT')

    elif method == 'tris_to_quads':
        # 直接使用 Tris to Quads 操作
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=0.698132,
            shape_threshold=0.698132,
            uvs=False,
            vcols=False,
            seam=False,
            sharp=False,
            materials=False
        )
        bpy.ops.object.mode_set(mode='OBJECT')

    else:
        raise ValueError(f"Unknown method: {method}")

    # 可选：平滑处理
    if smooth_iterations > 0:
        smooth_mod = obj.modifiers.new(name='Smooth', type='SMOOTH')
        smooth_mod.iterations = smooth_iterations
        smooth_mod.factor = 0.5
        bpy.ops.object.modifier_apply(modifier='Smooth')

    # 统计结果
    mesh = obj.data
    final_verts = len(mesh.vertices)
    final_faces = len(mesh.polygons)

    # 统计面类型
    tri_count = 0
    quad_count = 0
    ngon_count = 0

    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            tri_count += 1
        elif len(poly.vertices) == 4:
            quad_count += 1
        else:
            ngon_count += 1

    # 导出结果
    out_ext = output_path.lower().split('.')[-1]
    if out_ext == 'obj':
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=True,
            export_triangulated_mesh=False,  # 保持四边面
            export_normals=True,
            export_uv=True
        )
    elif out_ext == 'stl':
        bpy.ops.wm.stl_export(filepath=output_path, export_selected_objects=True)
    elif out_ext == 'ply':
        bpy.ops.wm.ply_export(filepath=output_path, export_selected_objects=True)
    elif out_ext in ['glb', 'gltf']:
        bpy.ops.export_scene.gltf(filepath=output_path, use_selection=True)
    elif out_ext == 'fbx':
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True)

    # 返回结果统计
    result = {
        'success': True,
        'original_vertices': original_verts,
        'original_faces': original_faces,
        'final_vertices': final_verts,
        'final_faces': final_faces,
        'tri_count': tri_count,
        'quad_count': quad_count,
        'ngon_count': ngon_count,
        'quad_ratio': quad_count / final_faces if final_faces > 0 else 0,
        'output_path': output_path
    }

    return result


if __name__ == '__main__':
    # 从命令行参数读取配置
    args = sys.argv[sys.argv.index('--') + 1:]
    config = json.loads(args[0])

    try:
        result = remesh_to_quads(
            input_path=config['input_path'],
            output_path=config['output_path'],
            method=config.get('method', 'quadriflow'),
            target_faces=config.get('target_faces', 5000),
            smooth_iterations=config.get('smooth_iterations', 0)
        )
        print("BLENDER_RESULT:" + json.dumps(result))
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        print("BLENDER_RESULT:" + json.dumps(error_result))
        sys.exit(1)
'''


class BlenderRemesh:
    """
    使用 Blender 进行四边面重拓扑

    支持的方法:
    - quadriflow: QuadriFlow 算法，生成纯四边面（推荐用于CAD）
    - voxel: 体素重拓扑后转四边面
    - tris_to_quads: 直接将三角面合并为四边面
    - decimate_planar: 先简化平面区域再转四边面
    """

    # 常见的 Blender 安装路径
    DEFAULT_BLENDER_PATHS = [
        # 用户下载目录（便携版）
        r"D:\Downloads\blender-4.4.3-windows-x64\blender.exe",
        # Windows 标准安装
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        # macOS
        "/Applications/Blender.app/Contents/MacOS/Blender",
        # Linux
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]

    def __init__(self, blender_path: Optional[str] = None):
        """
        初始化 Blender 重拓扑器

        Args:
            blender_path: Blender 可执行文件路径，如不指定则自动查找
        """
        self.blender_path = blender_path or self._find_blender()

        if self.blender_path is None:
            raise RuntimeError(
                "未找到 Blender。请安装 Blender 或手动指定路径。\n"
                "下载地址: https://www.blender.org/download/"
            )

    def _find_blender(self) -> Optional[str]:
        """自动查找 Blender 安装路径"""
        for path in self.DEFAULT_BLENDER_PATHS:
            if os.path.exists(path):
                return path

        # 尝试通过 which/where 命令查找
        try:
            if sys.platform == 'win32':
                result = subprocess.run(
                    ['where', 'blender'],
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    ['which', 'blender'],
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                path = result.stdout.strip().split('\n')[0]
                if os.path.exists(path):
                    return path
        except Exception:
            pass

        return None

    def get_blender_version(self) -> str:
        """获取 Blender 版本"""
        try:
            result = subprocess.run(
                [self.blender_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # 解析版本信息
                lines = result.stdout.strip().split('\n')
                return lines[0] if lines else "Unknown"
        except Exception as e:
            return f"Error: {e}"

        return "Unknown"

    def remesh_to_quads(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        method: str = 'quadriflow',
        target_faces: int = 5000,
        smooth_iterations: int = 0,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        将网格转换为四边面

        Args:
            input_path: 输入网格文件路径
            output_path: 输出文件路径（可选，默认在同目录生成）
            method: 重拓扑方法
                - 'quadriflow': QuadriFlow（推荐，纯四边面）
                - 'voxel': 体素重拓扑
                - 'tris_to_quads': 三角面合并
                - 'decimate_planar': 平面简化
            target_faces: 目标面数（仅对 quadriflow 和 voxel 有效）
            smooth_iterations: 平滑迭代次数
            timeout: 超时时间（秒）

        Returns:
            包含转换结果的字典
        """
        input_path = str(Path(input_path).resolve())

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 生成输出路径
        if output_path is None:
            input_p = Path(input_path)
            output_path = str(input_p.parent / f"{input_p.stem}_quad{input_p.suffix}")

        output_path = str(Path(output_path).resolve())

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建临时脚本文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(BLENDER_REMESH_SCRIPT)
            script_path = f.name

        try:
            # 准备配置
            config = {
                'input_path': input_path,
                'output_path': output_path,
                'method': method,
                'target_faces': target_faces,
                'smooth_iterations': smooth_iterations
            }

            # 运行 Blender
            cmd = [
                self.blender_path,
                '--background',
                '--python', script_path,
                '--',
                json.dumps(config)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # 解析输出
            output = result.stdout + result.stderr

            # 查找结果标记
            for line in output.split('\n'):
                if line.startswith('BLENDER_RESULT:'):
                    result_json = line[len('BLENDER_RESULT:'):]
                    return json.loads(result_json)

            # 未找到结果标记，返回错误
            return {
                'success': False,
                'error': f"Blender 执行异常: {output[:500]}",
                'returncode': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Blender 执行超时 ({timeout}秒)"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # 清理临时脚本
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def batch_remesh(
        self,
        input_paths: list,
        output_dir: str,
        method: str = 'quadriflow',
        target_faces: int = 5000,
        smooth_iterations: int = 0
    ) -> list:
        """
        批量重拓扑

        Args:
            input_paths: 输入文件路径列表
            output_dir: 输出目录
            method: 重拓扑方法
            target_faces: 目标面数
            smooth_iterations: 平滑迭代次数

        Returns:
            结果列表
        """
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in input_paths:
            input_p = Path(input_path)
            output_path = output_dir / f"{input_p.stem}_quad{input_p.suffix}"

            result = self.remesh_to_quads(
                input_path=input_path,
                output_path=str(output_path),
                method=method,
                target_faces=target_faces,
                smooth_iterations=smooth_iterations
            )
            result['input_path'] = input_path
            results.append(result)

        return results


def check_blender_available() -> Tuple[bool, str]:
    """
    检查 Blender 是否可用

    Returns:
        (是否可用, 版本信息或错误信息)
    """
    try:
        remesher = BlenderRemesh()
        version = remesher.get_blender_version()
        return True, version
    except RuntimeError as e:
        return False, str(e)


def remesh_to_quads_with_blender(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = 'quadriflow',
    target_faces: int = 5000,
    blender_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：使用 Blender 将网格转换为四边面

    Args:
        input_path: 输入网格路径
        output_path: 输出路径（可选）
        method: 方法 ('quadriflow', 'voxel', 'tris_to_quads', 'decimate_planar')
        target_faces: 目标面数
        blender_path: Blender 路径（可选）

    Returns:
        转换结果字典
    """
    remesher = BlenderRemesh(blender_path)
    return remesher.remesh_to_quads(
        input_path=input_path,
        output_path=output_path,
        method=method,
        target_faces=target_faces
    )


# 示例使用
if __name__ == '__main__':
    # 检查 Blender
    available, info = check_blender_available()
    print(f"Blender 可用: {available}")
    print(f"信息: {info}")

    if available and len(sys.argv) > 1:
        input_file = sys.argv[1]
        result = remesh_to_quads_with_blender(
            input_path=input_file,
            method='quadriflow',
            target_faces=5000
        )
        print(f"\n转换结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
