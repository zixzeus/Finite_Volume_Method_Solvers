# 调试配置使用指南

本项目已配置了完整的Python调试环境，包括VS Code调试配置、专用调试脚本和自动化测试套件。

## 📁 调试文件结构

```
Finite_Volume_Method_Solvers/
├── .vscode/
│   ├── launch.json          # VS Code调试配置
│   └── settings.json        # VS Code项目设置
├── .env                     # 环境变量配置
├── debug_*.py              # 专用调试脚本
├── run_debug_tests.py      # 主调试测试脚本
└── DEBUG_GUIDE.md          # 本文档
```

## 🚀 快速开始

### 1. 环境检查

#### Conda环境配置
项目配置为使用conda的base环境：
```bash
# 确认conda环境
conda env list
# 应该显示: base * /Users/ericzeus/anaconda3

# 测试环境配置
/Users/ericzeus/anaconda3/bin/python test_conda_env.py

# 运行完整测试套件  
python run_debug_tests.py
```

#### 自动检查内容：
- Python版本 (当前: 3.11.8)
- 必需包 (numpy 1.26.4, matplotlib 3.10.0, scipy 1.15.1)
- 框架模块完整性
- 创建输出目录

### 2. VS Code调试

#### 预配置的调试选项：
- **Framework Demo** - 运行完整框架演示
- **Blast Wave Test** - 爆炸波测试
- **Kelvin-Helmholtz Test** - KH不稳定性测试
- **DG Method Test** - 间断伽辽金方法测试
- **Riemann Solver Test** - Riemann求解器测试
- **Current File** - 调试当前打开文件

#### 使用方法：
1. 在VS Code中打开项目
2. 按 `F5` 或点击调试按钮
3. 选择相应的调试配置
4. 设置断点并开始调试

### 3. 命令行调试

```bash
# 运行单个调试脚本
python debug_blast_wave.py          # 爆炸波调试
python debug_riemann_solvers.py     # Riemann求解器调试
python debug_dg_method.py           # DG方法调试
python debug_kh_instability.py      # KH不稳定性调试

# 运行框架演示
python framework_usage_guide.py
```

## 🔧 调试脚本功能

### debug_blast_wave.py
- ✅ 数据结构测试
- ✅ 欧拉方程验证
- ✅ 空间离散化测试
- ✅ 时间积分测试
- ✅ 完整爆炸波设置
- ✅ 可视化生成
- ✅ 迷你仿真（5步）

### debug_riemann_solvers.py
- ✅ 求解器工厂测试
- ✅ 标准Riemann问题
- ✅ 多求解器对比
- ✅ 通量计算接口
- ✅ 性能基准测试
- ✅ 结果可视化

### debug_dg_method.py
- ✅ DG数据容器测试
- ✅ 基函数评估
- ✅ 质量矩阵验证
- ✅ 积分规则检查
- ✅ 空间残差计算
- ✅ 收敛性测试

### debug_kh_instability.py
- ✅ KH参数设置
- ✅ 初始条件验证
- ✅ 涡量计算
- ✅ 混合测量
- ✅ 完整KH仿真
- ✅ 发展过程可视化

## 📊 输出文件

调试脚本会在项目根目录生成以下文件：

```
debug_blast_wave_initial.png     # 爆炸波初始条件
debug_kh_initial.png            # KH不稳定性初始条件
debug_kh_final.png              # KH不稳定性最终状态
debug_riemann_comparison.png    # Riemann求解器对比
debug_dg_methods.png            # DG方法对比
framework_demo_results.png      # 框架演示结果
```

## 🛠️ 高级调试选项

### 1. Conda环境配置
当前配置使用conda base环境：
```bash
# Conda环境信息
CONDA_PREFIX=/Users/ericzeus/anaconda3
CONDA_DEFAULT_ENV=base
Python: /Users/ericzeus/anaconda3/bin/python (3.11.8)

# 包版本
numpy: 1.26.4
matplotlib: 3.10.0  
scipy: 1.15.1
```

### 2. 环境变量配置 (.env)
```bash
PYTHONPATH=/Users/ericzeus/Finite_Volume_Method_Solvers:/Users/ericzeus/Finite_Volume_Method_Solvers/fvm_framework
CONDA_DEFAULT_ENV=base
CONDA_PREFIX=/Users/ericzeus/anaconda3
NUMPY_THREADS=4
PYTHONWARNINGS=default
PYTHON_DEBUG=1
MPLBACKEND=Agg
```

### 3. VS Code设置 (.vscode/settings.json)
- Python解释器路径
- Linting配置 (pylint)
- 格式化设置 (black)
- 测试配置 (pytest)
- 分析路径配置

### 4. 调试启动配置 (.vscode/launch.json)
- 使用最新的debugpy调试器
- 集成终端输出
- 环境变量自动加载
- 参数化调试支持

## 🔍 常见调试场景

### 1. 算法验证
```bash
# 测试特定算法
python debug_riemann_solvers.py

# 检查输出中的:
# - 通量值对比
# - 数值稳定性
# - 性能指标
```

### 2. 数值精度检查
```bash
# 运行DG收敛性测试
python debug_dg_method.py

# 检查L2误差收敛率
```

### 3. 物理正确性验证
```bash
# 运行物理测试案例
python debug_blast_wave.py
python debug_kh_instability.py

# 检查:
# - 守恒性
# - 物理合理性
# - 边界条件
```

### 4. 完整系统测试
```bash
# 运行所有测试
python run_debug_tests.py

# 获得全面的系统健康报告
```

## 🐛 故障排除

### 常见问题：

1. **导入错误**
   ```
   解决: 检查PYTHONPATH设置，确保fvm_framework在路径中
   ```

2. **numpy/matplotlib错误**
   ```bash
   pip install numpy matplotlib scipy
   ```

3. **权限问题**
   ```bash
   # 确保有写入权限创建输出文件
   chmod +w /path/to/project/directory
   ```

4. **VS Code调试不工作**
   ```
   - 安装Python扩展
   - 检查Python解释器路径
   - 重新加载VS Code窗口
   ```

### 调试提示：

1. **设置断点** - 在关键数值计算处设置断点
2. **检查数组形状** - 使用 `print(array.shape)` 验证数据结构
3. **监视变量** - 在VS Code中添加变量到监视列表
4. **单步执行** - 使用F10/F11逐步执行算法
5. **条件断点** - 在特定条件下暂停 (例如 `np.isnan(flux).any()`)

## 📈 性能分析

调试脚本包含性能测试功能：

```python
# Riemann求解器性能对比
python debug_riemann_solvers.py
# 输出: 每次调用的微秒时间

# 内存使用监控
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 🎯 调试最佳实践

1. **从简单开始** - 先运行基本的debug_blast_wave.py
2. **逐步复杂** - 成功后再测试其他组件
3. **检查输出** - 仔细查看生成的图像文件
4. **记录问题** - 记录错误信息和复现步骤
5. **系统测试** - 定期运行完整测试套件

---

通过这个调试配置，你可以：
- 🔍 深入理解算法实现
- 🐛 快速定位和修复问题  
- 📊 验证数值结果正确性
- ⚡ 优化代码性能
- 🎯 确保框架稳定性