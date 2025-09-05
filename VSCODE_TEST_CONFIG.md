# VS Code 测试配置说明

## 配置概述

已成功配置VS Code的launch.json文件，使其能够正常运行FVM框架的测试文件。

## 可用的调试配置

### 1. 运行单个测试文件
- **配置名称**: "Run FVM Test File"
- **用途**: 运行当前打开的测试文件
- **使用方法**: 打开测试文件，选择此配置运行

### 2. 运行所有FVM测试
- **配置名称**: "Run All FVM Tests"
- **用途**: 运行完整的测试套件
- **命令**: `python -m fvm_framework.tests.run_tests --verbose`

### 3. 运行基础测试
- **配置名称**: "Run Basic FVM Tests"
- **用途**: 只运行基础功能测试
- **命令**: `python -m fvm_framework.tests.run_tests --type basic --verbose`

### 4. 运行性能测试
- **配置名称**: "Run Performance FVM Tests"
- **用途**: 只运行性能相关测试
- **命令**: `python -m fvm_framework.tests.run_tests --type performance --verbose`

### 5. 运行特定测试模块
- **配置名称**: "Run Test Data Container"
- **用途**: 运行数据容器测试
- **命令**: `python -m fvm_framework.tests.run_tests --module test_data_container --verbose`

- **配置名称**: "Run Test Pipeline"
- **用途**: 运行管道测试
- **命令**: `python -m fvm_framework.tests.run_tests --module test_pipeline --verbose`

- **配置名称**: "Run Test Boundary Conditions"
- **用途**: 运行边界条件测试
- **命令**: `python -m fvm_framework.tests.run_tests --module test_boundary_conditions --verbose`

## 使用方法

1. 在VS Code中按 `F5` 或点击调试面板
2. 从下拉菜单中选择所需的配置
3. 点击运行按钮

## 测试状态

### ✅ 正常工作的测试
- **test_data_container**: 所有14个测试通过
- **test_pipeline**: 基础测试通过（9/16个测试）
- **test_boundary_conditions**: 大部分测试通过

### ⚠️ 需要修复的测试
- **test_pipeline**: 5个测试失败，需要physics_equation参数
- **test_boundary_conditions**: 1个测试失败（transmissive边界条件）

## 修复的问题

1. **导入路径问题**: 修复了pipeline.py中的相对导入路径
2. **测试运行器**: 修复了run_tests.py中的模块导入问题
3. **Python路径**: 正确配置了PYTHONPATH环境变量

## 命令行运行方式

除了VS Code调试配置，也可以直接在终端运行：

```bash
# 运行所有测试
python -m fvm_framework.tests.run_tests --verbose

# 运行基础测试
python -m fvm_framework.tests.run_tests --type basic --verbose

# 运行特定模块
python -m fvm_framework.tests.run_tests --module test_data_container --verbose
```

## 注意事项

1. 确保在项目根目录下运行测试
2. PYTHONPATH已正确设置为项目根目录
3. 某些测试可能需要额外的参数（如physics_equation）
4. 测试使用相对导入，必须通过模块方式运行
