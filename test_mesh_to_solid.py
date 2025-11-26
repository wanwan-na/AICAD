"""
mesh_to_solid.py 功能测试脚本

演示各种使用场景
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示结果"""
    print("\n" + "="*60)
    print(f"测试: {description}")
    print("="*60)
    print(f"命令: {' '.join(cmd)}")
    print("-"*60)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"✓ 成功")
    else:
        print(f"✗ 失败 (退出码: {result.returncode})")

    return result.returncode


def main():
    """主测试函数"""

    # 检查Python解释器
    python_exe = Path(".venv/Scripts/python.exe")
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    # 检查输入文件
    test_input = "test_input.obj"
    if not Path(test_input).exists():
        # 查找任何可用的obj文件
        obj_files = list(Path(".").glob("*.obj"))
        if obj_files:
            test_input = str(obj_files[0])
            print(f"使用测试文件: {test_input}")
        else:
            print("错误: 找不到测试输入文件")
            return 1

    tests = [
        # 测试1: 基本转换
        {
            "cmd": [str(python_exe), "mesh_to_solid.py",
                   test_input, "output_basic.obj", "--no-fill-holes"],
            "description": "基本转换（不填充孔洞）"
        },

        # 测试2: 带修复的转换
        {
            "cmd": [str(python_exe), "mesh_to_solid.py",
                   test_input, "output_repaired.obj", "--repair", "--no-fill-holes"],
            "description": "修复网格"
        },

        # 测试3: 填充孔洞
        {
            "cmd": [str(python_exe), "mesh_to_solid.py",
                   test_input, "output_filled.obj", "--fill-holes"],
            "description": "填充孔洞"
        },

        # 测试4: 简化网格
        {
            "cmd": [str(python_exe), "mesh_to_solid.py",
                   test_input, "output_simplified.obj",
                   "--simplify", "--faces", "5000", "--no-fill-holes"],
            "description": "简化网格（目标5000面）"
        },

        # 测试5: 平滑处理
        {
            "cmd": [str(python_exe), "mesh_to_solid.py",
                   test_input, "output_smooth.obj",
                   "--smooth", "--smooth-iterations", "3", "--no-fill-holes"],
            "description": "平滑处理（3次迭代）"
        },

        # 测试6: 组合处理
        {
            "cmd": [str(python_exe), "mesh_to_solid.py",
                   test_input, "output_combined.obj",
                   "--repair", "--simplify", "--faces", "8000",
                   "--smooth", "--no-fill-holes"],
            "description": "组合处理（修复+简化+平滑）"
        },

        # 测试7: 查看帮助
        {
            "cmd": [str(python_exe), "mesh_to_solid.py", "--help"],
            "description": "显示帮助信息"
        },
    ]

    print("\n" + "#"*60)
    print("# mesh_to_solid.py 功能测试套件")
    print("#"*60)
    print(f"\n测试输入文件: {test_input}")
    print(f"Python解释器: {python_exe}")

    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n>>> 测试 {i}/{len(tests)}")
        ret = run_command(test["cmd"], test["description"])
        results.append((test["description"], ret))

    # 总结
    print("\n" + "#"*60)
    print("# 测试总结")
    print("#"*60)

    success_count = sum(1 for _, ret in results if ret == 0)
    total_count = len(results)

    print(f"\n通过: {success_count}/{total_count}")
    print("\n详细结果:")
    for desc, ret in results:
        status = "✓ 通过" if ret == 0 else "✗ 失败"
        print(f"  {status} - {desc}")

    # 检查生成的文件
    print("\n生成的文件:")
    output_files = list(Path(".").glob("output_*.obj"))
    for f in sorted(output_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
