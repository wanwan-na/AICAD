"""
AI_CAD 处理流程主程序

使用方法:
    python run_pipeline.py input.glb
    python run_pipeline.py input.glb --output output.obj
    python run_pipeline.py input.glb --method blender --target-faces 5000
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='混元3D模型 → 四边面网格转换')
    parser.add_argument('input', help='输入文件路径 (.glb, .obj, .stl, .ply)')
    parser.add_argument('--output', '-o', help='输出文件路径（默认自动生成）')
    parser.add_argument('--method', '-m', choices=['blender', 'builtin'],
                        default='builtin', help='四边面转换方法: blender(推荐) 或 builtin')
    parser.add_argument('--target-faces', '-t', type=int, default=5000,
                        help='目标面数（默认5000）')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过预处理步骤')
    parser.add_argument('--blender-path', help='Blender可执行文件路径')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认输出与输入相同格式，如果是glb/gltf则保持，否则用obj
        if input_path.suffix.lower() in ['.glb', '.gltf']:
            output_path = input_path.parent / f"{input_path.stem}_quad.glb"
        else:
            output_path = input_path.parent / f"{input_path.stem}_quad.obj"

    print("=" * 50)
    print("AI_CAD 处理流程")
    print("=" * 50)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"转换方法: {args.method}")
    print(f"目标面数: {args.target_faces}")
    print("=" * 50)

    current_file = str(input_path)

    # ===== 步骤1: 预处理 =====
    if not args.skip_preprocess:
        print("\n【步骤1/2】预处理 - 修复破面和孔洞...")
        try:
            from src.preprocessor import HunyuanPreprocessor

            preprocessor = HunyuanPreprocessor()
            cleaned_path = input_path.parent / f"{input_path.stem}_cleaned.obj"

            result = preprocessor.process(current_file, str(cleaned_path))

            print(f"  ✓ 预处理完成")
            print(f"    - 原始顶点: {result.get('original_vertices', 'N/A')}")
            print(f"    - 原始面数: {result.get('original_faces', 'N/A')}")
            print(f"    - 填充孔洞: {result.get('holes_filled', 0)}")
            print(f"    - 输出文件: {cleaned_path}")

            current_file = str(cleaned_path)

        except Exception as e:
            print(f"  ✗ 预处理失败: {e}")
            print("  继续使用原始文件...")
    else:
        print("\n【步骤1/2】跳过预处理")

    # ===== 步骤2: 四边面转换 =====
    print(f"\n【步骤2/2】四边面转换 ({args.method})...")

    if args.method == 'blender':
        # 使用 Blender QuadriFlow
        try:
            from src.retopology import BlenderRemesh, check_blender_available

            available, info = check_blender_available()
            if not available:
                print(f"  ✗ Blender 不可用: {info}")
                print("  请安装 Blender: https://www.blender.org/download/")
                print("  或使用 --method builtin 切换到内置方法")
                sys.exit(1)

            print(f"  使用 {info}")

            remesher = BlenderRemesh(args.blender_path)
            result = remesher.remesh_to_quads(
                input_path=current_file,
                output_path=str(output_path),
                method='quadriflow',
                target_faces=args.target_faces
            )

            if result.get('success'):
                print(f"  ✓ 四边面转换完成")
                print(f"    - 顶点数: {result.get('final_vertices', 'N/A')}")
                print(f"    - 面数: {result.get('final_faces', 'N/A')}")
                print(f"    - 四边形: {result.get('quad_count', 0)}")
                print(f"    - 三角形: {result.get('tri_count', 0)}")
                print(f"    - 四边形比例: {result.get('quad_ratio', 0):.1%}")
            else:
                print(f"  ✗ 转换失败: {result.get('error', '未知错误')}")
                sys.exit(1)

        except ImportError as e:
            print(f"  ✗ 导入错误: {e}")
            sys.exit(1)

    else:
        # 使用内置转换器
        try:
            from src.retopology import RetopologyProcessor, RetopologyConfig

            config = RetopologyConfig(
                target_face_count=args.target_faces,
                min_quad_ratio=0.5
            )

            processor = RetopologyProcessor(config)
            result = processor.process(current_file, str(output_path))

            if result.get('success'):
                quality = result.get('quality', {})
                print(f"  ✓ 四边面转换完成")
                print(f"    - 顶点数: {quality.get('vertex_count', 'N/A')}")
                print(f"    - 面数: {quality.get('face_count', 'N/A')}")
                print(f"    - 四边形: {quality.get('quad_count', 0)}")
                print(f"    - 三角形: {quality.get('tri_count', 0)}")
                print(f"    - 四边形比例: {quality.get('quad_ratio', 0):.1%}")
                print(f"    - 质量评级: {quality.get('quad_grade', 'N/A')}")
            else:
                print(f"  ✗ 转换失败")
                sys.exit(1)

        except ImportError as e:
            print(f"  ✗ 导入错误: {e}")
            print("  请确保已安装依赖: pip install pymeshlab trimesh numpy")
            sys.exit(1)

    # ===== 完成 =====
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"输出文件: {output_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
