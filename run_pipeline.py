"""
AI_CAD 处理流程主程序

使用方法:
    python run_pipeline.py model.glb                # 从 inputs/ 读取，输出到 outputs/
    python run_pipeline.py model.glb -m blender     # 使用 Blender 方法
    python run_pipeline.py model.glb -t 5000        # 指定目标面数
    python run_pipeline.py --list                   # 列出 inputs/ 中的文件
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


# 项目根目录
PROJECT_ROOT = Path(__file__).parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_next_output_folder(base_name: str) -> Path:
    """生成新的输出文件夹，格式: outputs/{文件名}_{序号}_{时间戳}/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 查找已有的文件夹数量
    existing = list(OUTPUTS_DIR.glob(f"{base_name}_*"))
    next_num = len(existing) + 1

    folder_name = f"{base_name}_{next_num:03d}_{timestamp}"
    output_folder = OUTPUTS_DIR / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder


def list_input_files():
    """列出 inputs 文件夹中的可用文件"""
    if not INPUTS_DIR.exists():
        print(f"inputs 文件夹不存在: {INPUTS_DIR}")
        return []

    extensions = ['.glb', '.gltf', '.obj', '.stl', '.ply']
    files = []
    for ext in extensions:
        files.extend(INPUTS_DIR.glob(f"*{ext}"))

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='混元3D模型 → 四边面网格转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_pipeline.py model.glb              # 处理 inputs/model.glb
    python run_pipeline.py model.glb -m blender   # 使用 Blender 方法
    python run_pipeline.py --list                 # 列出可用文件
        """
    )
    parser.add_argument('input', nargs='?', help='输入文件名（从 inputs/ 文件夹读取）')
    parser.add_argument('--list', '-l', action='store_true', help='列出 inputs/ 中的可用文件')
    parser.add_argument('--method', '-m', choices=['blender', 'builtin'],
                        default='builtin', help='四边面转换方法: blender(推荐) 或 builtin')
    parser.add_argument('--target-faces', '-t', type=int, default=5000,
                        help='目标面数（默认5000）')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过预处理步骤')
    parser.add_argument('--blender-path', help='Blender可执行文件路径')

    args = parser.parse_args()

    # 列出文件模式
    if args.list:
        files = list_input_files()
        if files:
            print(f"\ninputs/ 文件夹中的可用文件 ({len(files)} 个):\n")
            for i, f in enumerate(files, 1):
                size_kb = f.stat().st_size / 1024
                print(f"  {i}. {f.name} ({size_kb:.1f} KB)")
            print(f"\n使用方法: python run_pipeline.py <文件名>")
        else:
            print("inputs/ 文件夹中没有找到支持的文件")
        sys.exit(0)

    # 检查输入参数
    if not args.input:
        parser.print_help()
        print("\n提示: 使用 --list 查看可用文件")
        sys.exit(1)

    # 确定输入路径（从 inputs/ 文件夹读取）
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # 如果是相对路径，从 inputs/ 文件夹查找
        input_path = INPUTS_DIR / args.input

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        print(f"\n提示: 请将文件放入 inputs/ 文件夹，或使用 --list 查看可用文件")
        sys.exit(1)

    # 创建新的输出文件夹
    output_folder = get_next_output_folder(input_path.stem)

    # 确定输出文件路径
    output_path = output_folder / f"{input_path.stem}_quad.obj"

    print("=" * 50)
    print("AI_CAD 处理流程")
    print("=" * 50)
    print(f"输入文件: {input_path}")
    print(f"输出文件夹: {output_folder}")
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
            cleaned_path = output_folder / f"{input_path.stem}_cleaned.obj"

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
    print(f"输出文件夹: {output_folder}")
    print(f"输出文件: {output_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
