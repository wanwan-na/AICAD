"""
AI_CAD 重拓扑主程序

完整的重拓扑流水线：预处理 → 重拓扑 → 质量验证

针对混元3D (Hunyuan3D) 等AI生成的3D模型进行优化。

使用方法:
    python main.py input.obj output_quad.obj
    python main.py input.obj  # 自动生成输出文件名
    python main.py input.glb --hunyuan  # 使用混元3D专用预处理
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from src.preprocessor import MeshPreprocessor
from src.preprocessor.mesh_repair import repair_mesh, RepairConfig, analyze_mesh_topology
from src.preprocessor.hunyuan_preprocessor import HunyuanPreprocessor, HunyuanPreprocessConfig
from src.retopology import RetopologyProcessor, RetopologyConfig


def retopology_pipeline(input_path: str,
                        output_path: Optional[str] = None,
                        skip_preprocess: bool = False,
                        fill_holes: bool = False,
                        hunyuan_mode: bool = False,
                        aggressive_repair: bool = False,
                        target_face_count: Optional[int] = None,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    完整重拓扑流水线

    流程:
    1. 混元3D专用预处理（可选，推荐用于混元3D输出）
    2. 孔洞填充（可选，用于AI生成的模型）
    3. 网格预处理（清理、修复）
    4. 重拓扑（三角面→四边形）
    5. 质量验证

    Args:
        input_path: 输入三角面网格文件路径
        output_path: 输出四边形网格路径（可选）
        skip_preprocess: 是否跳过预处理
        fill_holes: 是否填充孔洞（推荐用于AI生成的模型）
        hunyuan_mode: 是否使用混元3D专用预处理（推荐用于混元3D输出）
        aggressive_repair: 是否使用激进修复模式
        target_face_count: 目标面数（可选）
        verbose: 是否打印详细信息

    Returns:
        处理结果字典
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if verbose:
        print("=" * 60)
        print("AI_CAD 重拓扑流水线")
        print("=" * 60)
        print(f"\n输入文件: {input_path}")

    results = {
        'input_path': str(input_path),
        'stages': {}
    }

    current_mesh_path = str(input_path)

    # ===== 阶段0: 混元3D专用预处理（可选）=====
    if hunyuan_mode:
        if verbose:
            print("\n" + "-" * 40)
            print("阶段 0: 混元3D专用预处理")
            print("-" * 40)

        hunyuan_config = HunyuanPreprocessConfig(
            merge_close_vertices=True,
            merge_threshold=1e-5,
            repair_non_manifold=True,
            remove_duplicate_faces=True,
            remove_degenerate_faces=True,
            fill_holes=True,
            max_hole_vertices=500,
            use_ear_clipping=True,
            use_pymeshlab_repair=True,
            max_repair_iterations=3,
            verbose=verbose
        )

        hunyuan_preprocessor = HunyuanPreprocessor(hunyuan_config)
        hunyuan_result = hunyuan_preprocessor.process(current_mesh_path)

        current_mesh_path = hunyuan_result['output_path']
        results['stages']['hunyuan_preprocess'] = hunyuan_result

        if verbose:
            print(f"\n  [OK] 混元3D预处理完成")
            print(f"    填充孔洞: {hunyuan_result['holes_filled']} 个")
            print(f"    水密: {'是' if hunyuan_result['is_watertight'] else '否'}")
            print(f"    流形: {'是' if hunyuan_result['is_manifold'] else '否'}")

    # ===== 阶段0.5: 激进修复模式（可选）=====
    if aggressive_repair and not hunyuan_mode:
        if verbose:
            print("\n" + "-" * 40)
            print("阶段 0: 激进网格修复")
            print("-" * 40)

        import trimesh
        from src.preprocessor.mesh_repair import repair_mesh_aggressive

        mesh = trimesh.load(str(current_mesh_path), force='mesh')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # 分析原始拓扑
        original_topo = analyze_mesh_topology(vertices, faces)
        if verbose:
            print(f"  原始孔洞数: {original_topo['hole_count']}")
            print(f"  原始边界边数: {original_topo['boundary_edge_count']}")

        # 激进修复
        new_verts, new_faces = repair_mesh_aggressive(vertices, faces)

        # 分析修复后
        final_topo = analyze_mesh_topology(new_verts, new_faces)
        if verbose:
            print(f"  修复后孔洞数: {final_topo['hole_count']}")
            print(f"  修复后边界边数: {final_topo['boundary_edge_count']}")

        # 保存修复后的网格
        repaired_path = input_path.parent / f"{input_path.stem}_repaired.obj"
        repaired_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
        repaired_mesh.export(str(repaired_path))

        current_mesh_path = str(repaired_path)
        results['stages']['aggressive_repair'] = {
            'original_holes': original_topo['hole_count'],
            'final_holes': final_topo['hole_count'],
            'output_path': str(repaired_path)
        }

        if verbose:
            print(f"\n  [OK] 激进修复完成")

    # ===== 阶段1: 孔洞填充（可选，独立于混元预处理）=====
    if fill_holes and not hunyuan_mode:
        if verbose:
            print("\n" + "-" * 40)
            print("阶段 0: 孔洞填充")
            print("-" * 40)

        import trimesh
        import numpy as np

        # 加载网格
        mesh = trimesh.load(str(input_path), force='mesh')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        if verbose:
            print(f"  原始顶点数: {len(vertices)}")
            print(f"  原始面数: {len(faces)}")

        # 填充孔洞
        repair_config = RepairConfig(fill_holes=True, max_hole_vertices=150)
        new_verts, new_faces = repair_mesh(vertices, faces, repair_config)

        if verbose:
            print(f"  填充后顶点数: {len(new_verts)}")
            print(f"  填充后面数: {len(new_faces)}")

        # 保存填充后的网格
        filled_path = input_path.parent / f"{input_path.stem}_filled.obj"
        filled_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
        filled_mesh.export(str(filled_path))

        current_mesh_path = str(filled_path)
        results['stages']['fill_holes'] = {
            'original_faces': len(faces),
            'filled_faces': len(new_faces),
            'output_path': str(filled_path)
        }

        if verbose:
            print(f"\n  [OK] 孔洞填充完成")

    # ===== 阶段1: 预处理 =====
    if not skip_preprocess:
        if verbose:
            print("\n" + "-" * 40)
            print("阶段 1/2: 网格预处理")
            print("-" * 40)

        preprocessor = MeshPreprocessor()

        # 分析网格
        analysis = preprocessor.analyze(current_mesh_path)
        if verbose:
            print(f"  原始顶点数: {analysis['vertex_count']}")
            print(f"  原始面数: {analysis['face_count']}")
            if analysis['issues']:
                print(f"  发现问题: {len(analysis['issues'])} 个")
            print(f"  建议: {analysis['recommendation']}")

        # 执行预处理
        cleaned_path = input_path.parent / f"{input_path.stem}_cleaned.obj"
        preprocess_result = preprocessor.process(current_mesh_path, str(cleaned_path))

        results['stages']['preprocess'] = preprocess_result

        if verbose:
            print(f"\n  [OK] 预处理完成")
            print(f"    移除顶点: {preprocess_result['vertices_removed']}")
            print(f"    移除面: {preprocess_result['faces_removed']}")
            print(f"    输出: {preprocess_result['output_path']}")

        current_mesh_path = preprocess_result['output_path']

    # ===== 阶段2: 重拓扑 =====
    if verbose:
        print("\n" + "-" * 40)
        print("阶段 2/2: 重拓扑处理")
        print("-" * 40)

    # 配置重拓扑参数
    config = RetopologyConfig(
        target_face_count=target_face_count,
        auto_face_ratio=0.4,
        sharp_angle_threshold=30.0,
        max_retries=3
    )

    processor = RetopologyProcessor(config)

    # 确定最终输出路径
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_quad.obj"

    # 执行重拓扑
    retopo_result = processor.process(current_mesh_path, str(output_path), config)

    results['stages']['retopology'] = retopo_result
    results['output_path'] = retopo_result['output_path']
    results['quality'] = retopo_result['quality']

    if verbose:
        print(f"\n  [OK] 重拓扑完成 (尝试 {retopo_result['attempts']} 次)")
        print(f"\n{retopo_result['quality_report']}")

    # ===== 清理临时文件 =====
    if not skip_preprocess:
        try:
            Path(cleaned_path).unlink()
            if verbose:
                print("\n  [OK] 已清理临时文件")
        except Exception:
            pass

    # ===== 最终总结 =====
    if verbose:
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"输出文件: {results['output_path']}")
        print(f"四边形比例: {results['quality']['quad_ratio']*100:.1f}%")
        print(f"综合评级: {results['quality']['overall_grade'].upper()}")

    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='AI_CAD 重拓扑工具 - 将三角面网格转换为四边形网格（针对混元3D等AI生成模型优化）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py model.obj                    # 自动输出 model_quad.obj
  python main.py model.obj output.obj         # 指定输出文件
  python main.py model.obj -f 5000            # 指定目标面数
  python main.py model.obj --skip-preprocess  # 跳过预处理

  # 混元3D (Hunyuan3D) 专用预处理（推荐）:
  python main.py hunyuan_output.glb --hunyuan

  # 激进修复模式（用于严重破损的模型）:
  python main.py model.obj --aggressive
        """
    )

    parser.add_argument('input', help='输入网格文件路径 (OBJ/STL/PLY/GLB/GLTF)')
    parser.add_argument('output', nargs='?', default=None, help='输出文件路径 (可选)')
    parser.add_argument('-f', '--faces', type=int, default=None,
                        help='目标面数 (默认自动计算)')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过预处理步骤')
    parser.add_argument('--fill-holes', action='store_true',
                        help='填充孔洞')
    parser.add_argument('--hunyuan', action='store_true',
                        help='使用混元3D专用预处理（推荐用于Hunyuan3D输出，会自动修复破面和孔洞）')
    parser.add_argument('--aggressive', action='store_true',
                        help='使用激进修复模式（用于严重破损的模型）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='安静模式，减少输出')

    args = parser.parse_args()

    try:
        result = retopology_pipeline(
            input_path=args.input,
            output_path=args.output,
            skip_preprocess=args.skip_preprocess,
            fill_holes=args.fill_holes,
            hunyuan_mode=args.hunyuan,
            aggressive_repair=args.aggressive,
            target_face_count=args.faces,
            verbose=not args.quiet
        )

        # 返回状态码
        if result['quality']['passed']:
            sys.exit(0)
        else:
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)
    except ImportError as e:
        print(f"依赖错误: {e}", file=sys.stderr)
        print("请运行: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"处理错误: {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
