"""
AI_CAD 环境检查脚本

检查所有依赖是否正确安装
"""

import sys


def check_dependencies():
    """检查依赖库"""
    print("=" * 50)
    print("AI_CAD 环境检查")
    print("=" * 50)
    print()

    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'trimesh': 'trimesh',
        'pymeshlab': 'pymeshlab',
        'open3d': 'open3d',
        'tqdm': 'tqdm',
        'yaml': 'pyyaml'
    }

    all_ok = True

    print("核心依赖检查:")
    print("-" * 30)

    for name, package in dependencies.items():
        try:
            __import__(name)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            if package in ['trimesh', 'pymeshlab']:
                all_ok = False

    print()

    # 检查 Instant Meshes
    print("可选工具检查:")
    print("-" * 30)

    try:
        from src.retopology import check_instant_meshes
        status = check_instant_meshes()
        if status['installed']:
            print(f"  [OK] Instant Meshes: {status['path']}")
        else:
            print(f"  [NOT FOUND] Instant Meshes")
            print("    提示: 下载 https://github.com/wjakob/instant-meshes/releases")
    except Exception as e:
        print(f"  [ERROR] Instant Meshes check failed: {e}")

    print()

    # 测试基本功能
    print("功能测试:")
    print("-" * 30)

    try:
        from src.preprocessor import MeshPreprocessor
        print("  [OK] MeshPreprocessor 可导入")
    except Exception as e:
        print(f"  [ERROR] MeshPreprocessor: {e}")
        all_ok = False

    try:
        from src.retopology import RetopologyProcessor
        print("  [OK] RetopologyProcessor 可导入")
    except Exception as e:
        print(f"  [ERROR] RetopologyProcessor: {e}")
        all_ok = False

    try:
        from src.utils import MeshIO
        print("  [OK] MeshIO 可导入")
    except Exception as e:
        print(f"  [ERROR] MeshIO: {e}")
        all_ok = False

    print()
    print("=" * 50)

    if all_ok:
        print("环境检查通过! 可以开始使用 AI_CAD")
        print()
        print("使用方法:")
        print("  python main.py <input.obj> [output.obj]")
        print()
        print("示例:")
        print("  python main.py model.obj")
        print("  python main.py model.obj result_quad.obj -f 5000")
        return 0
    else:
        print("环境检查未通过，请安装缺失的依赖:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(check_dependencies())
