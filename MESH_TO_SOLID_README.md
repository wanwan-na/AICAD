# 3D网格转Solid工具使用说明

## 功能简介

`mesh_to_solid.py` 是一个将3D网格模型（三角网格或四边形网格）转换为实体模型（Solid）的工具。该工具可以：

- 检查和修复网格水密性
- 填充孔洞使模型成为封闭实体
- 体素化处理以确保水密性
- 简化网格减少面数
- 平滑网格表面
- **详细统计三角面和四边面数量** 🆕

## 安装依赖

基础依赖（必需）：
```bash
pip install trimesh scipy numpy
```

高级功能依赖（可选）：
```bash
# 孔洞填充功能
pip install pymeshlab

# 体素化功能
pip install scikit-image
```

## 使用方法

### 基本用法

最简单的用法，自动修复并转换为solid：
```bash
python mesh_to_solid.py input.obj output_solid.obj
```

### 填充孔洞

适用于有孔洞的模型（如AI生成的模型）：
```bash
python mesh_to_solid.py input.obj output_solid.obj --fill-holes
```

### 体素化处理

使用体素化可以确保更好的水密性：
```bash
python mesh_to_solid.py input.obj output_solid.obj --voxelize
```

指定体素大小：
```bash
python mesh_to_solid.py input.obj output_solid.obj --voxelize --voxel-size 0.5
```

### 简化网格

减少面数以提高性能：
```bash
python mesh_to_solid.py input.obj output_solid.obj --simplify --faces 5000
```

### 平滑处理

平滑网格表面：
```bash
python mesh_to_solid.py input.obj output_solid.obj --smooth --smooth-iterations 5
```

### 组合使用

完整的处理流程：
```bash
python mesh_to_solid.py input.obj output_solid.obj --repair --fill-holes --simplify --smooth
```

## 参数说明

### 位置参数
- `input`: 输入网格文件路径（支持OBJ、STL、PLY等格式）
- `output`: 输出Solid文件路径

### 可选参数

#### 修复选项
- `--repair`: 修复网格（默认启用）
- `--no-repair`: 禁用修复
- `--fill-holes`: 填充孔洞（默认启用）
- `--no-fill-holes`: 禁用孔洞填充

#### 体素化选项
- `--voxelize`: 使用体素化处理
- `--voxel-size FLOAT`: 体素大小（绝对值）

#### 简化选项
- `--simplify`: 简化网格
- `--faces INT`: 目标面数

#### 平滑选项
- `--smooth`: 平滑网格
- `--smooth-iterations INT`: 平滑迭代次数（默认3）

#### 其他选项
- `-q, --quiet`: 安静模式，减少输出信息

## 使用场景

### 场景1：AI生成的模型转CAD
AI生成的3D模型通常有孔洞和不规则的拓扑结构，需要修复后才能在CAD软件中使用：
```bash
python mesh_to_solid.py ai_model.glb cad_solid.obj --fill-holes --smooth
```

### 场景2：高精度模型简化
原始扫描或生成的模型面数过多，需要简化：
```bash
python mesh_to_solid.py high_poly.obj low_poly.obj --simplify --faces 10000
```

### 场景3：确保水密性
需要确保模型完全水密以进行3D打印或仿真：
```bash
python mesh_to_solid.py model.obj watertight.obj --voxelize --fill-holes
```

### 场景4：表面质量优化
优化模型表面使其更光滑：
```bash
python mesh_to_solid.py rough.obj smooth.obj --smooth --smooth-iterations 10
```

## 工作流集成

### 与现有重拓扑工具配合使用

1. 先使用主程序将三角网格转换为四边形：
```bash
python main.py input.obj quad_mesh.obj --hunyuan
```

2. 再将四边形网格转换为solid：
```bash
python mesh_to_solid.py quad_mesh.obj final_solid.obj --repair
```

### 完整的AI模型处理流程

```bash
# 步骤1: 重拓扑（三角网格→四边形网格）
python main.py ai_model.glb quad_model.obj --hunyuan

# 步骤2: 转换为solid（修复+填充孔洞+优化）
python mesh_to_solid.py quad_model.obj solid_model.obj --fill-holes --smooth

# 步骤3: 如果需要简化
python mesh_to_solid.py solid_model.obj final_model.obj --simplify --faces 5000
```

## 输出说明

转换完成后，工具会显示：
- 输入/输出文件路径
- 顶点数和面数的变化
- **三角面和四边面的详细统计** 🆕
- 水密性状态（转换前后）
- 模型体积（如果是水密的）

### 加载时显示
```
加载网格: test_input.obj
  顶点数: 22447
  面数: 40000
    三角面: 40000 (100.0%)
    四边面: 0 (0.0%)
  水密: 否
  流形: 否
```

### 完成时显示
```
============================================================
转换完成!
============================================================
输入: test_input.obj
输出: test_final_demo.obj

几何信息:
  顶点: 22447 → 8724
  面数: 40000 → 15000

面类型统计:
  三角面: 15000 (100.0%)
  四边面: 0 (0.0%)

质量信息:
  水密: False → False
```

### 面类型说明

- **三角面**: 由3个顶点组成的面，最常见的面类型
- **四边面**: 由4个顶点组成的面，常见于CAD建模和重拓扑后的模型
- **其他面**: N边形面（N>4），较少见

**注意**: trimesh库默认会将所有多边形面转换为三角面，因此输出通常是100%三角面。如果需要保留四边面结构，建议使用项目中的quad_converter工具。

## 注意事项

1. **水密性**: 如果原始模型有大量孔洞，建议使用 `--fill-holes` 或 `--voxelize`
2. **性能**: 体素化处理较慢，但能保证最好的水密性
3. **面数**: 简化操作会降低模型精度，请根据实际需求调整目标面数
4. **平滑**: 过多的平滑迭代可能会丢失细节特征

## 错误处理

### 依赖缺失
如果看到导入错误，请安装相应的依赖库：
```bash
pip install trimesh pymeshlab scipy scikit-image
```

### 模型不水密
如果转换后仍不水密，尝试：
```bash
python mesh_to_solid.py input.obj output.obj --voxelize --voxel-size 0.3
```

## 技术细节

### 修复流程
1. 合并重复顶点
2. 移除退化面
3. 移除重复面
4. 修复法向量
5. 填充孔洞（如果启用）

### 体素化算法
使用Marching Cubes算法将体素网格转换回三角网格，确保生成的模型是水密的。

### 简化算法
使用二次误差度量（Quadric Error Metrics）进行网格简化，在减少面数的同时保持模型形状。

## 示例图

处理前后对比：
- 原始模型：可能有孔洞、破面
- 处理后模型：水密、干净、优化过的实体模型

## 常见问题

**Q: 转换后文件很大怎么办？**
A: 使用 `--simplify` 参数减少面数

**Q: 模型细节丢失了？**
A: 减少平滑迭代次数或提高简化的目标面数

**Q: 转换很慢？**
A: 禁用体素化，或增大体素尺寸

**Q: 模型还是不水密？**
A: 尝试使用 `--voxelize` 配合较小的体素尺寸

## 版本信息

- 版本: 1.1 (新增面类型统计功能)
- 作者: AI_CAD Team
- 依赖: trimesh >= 4.0.0, pymeshlab >= 2023.12, scipy >= 1.11.0

## 更新日志

### v1.1 (2025-11-25)
- ✨ 新增功能：详细的三角面和四边面统计
- 📊 加载时显示面类型分布
- 📊 转换完成时显示面类型变化
- 📈 显示各类型面的数量和百分比

### v1.0 (2025-11-25)
- 🎉 初始版本发布
- ✅ 网格修复和水密性检查
- ✅ 孔洞填充功能
- ✅ 体素化转换
- ✅ 网格简化和平滑
