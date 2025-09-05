# 统一空间离散方法架构指南

## 概述

我已经重新设计了FVM框架的空间离散方法架构，将Riemann solver和finite volume方法统一到一个`SpatialDiscretization`框架下。这样可以更灵活地使用不同的空间离散方法，包括TVDLF和Lax-Friedrichs等。

## 新架构的主要特点

### 1. 统一的空间离散接口 (`spatial/spatial_discretization.py`)

- **`SpatialDiscretization`**: 所有空间离散方法的抽象基类
- **`FiniteVolumeScheme`**: 有限体积方法的基类
- **`RiemannBasedScheme`**: 基于Riemann solver方法的基类

### 2. 具体的空间离散方法

- **`LaxFriedrichsScheme`**: 一阶Lax-Friedrichs方法
- **`TVDLFScheme`**: 二阶TVD Lax-Friedrichs方法（支持flux limiter）
- **`HLLRiemannScheme`**: 基于HLL Riemann solver的方法

### 3. 工厂模式

```python
from spatial.spatial_discretization import SpatialDiscretizationFactory

# 创建Lax-Friedrichs方法
lf_scheme = SpatialDiscretizationFactory.create('lax_friedrichs')

# 创建TVDLF方法，使用minmod限制器
tvdlf_scheme = SpatialDiscretizationFactory.create('tvdlf', limiter='minmod')

# 创建HLL方法
hll_scheme = SpatialDiscretizationFactory.create('hll')
```

## 修改的组件

### 1. Pipeline架构 (`core/pipeline.py`)

#### FluxStage 修改
```python
# 旧版本：只能使用Riemann solver
FluxStage(riemann_solver='hllc')

# 新版本：可以使用任意空间离散方法
FluxStage(spatial_scheme='lax_friedrichs')
FluxStage(spatial_scheme='tvdlf', limiter='minmod')
FluxStage(spatial_scheme='hll')
```

#### TemporalStage 修改
```python
# 旧版本：硬编码使用特定Riemann solver
TemporalStage(scheme='rk3')

# 新版本：可以指定空间离散方法
TemporalStage(scheme='rk3', spatial_scheme='lax_friedrichs')
TemporalStage(scheme='rk3', spatial_scheme='tvdlf', limiter='minmod')
```

### 2. FVMPipeline构造函数
```python
# 旧版本
pipeline = FVMPipeline(
    boundary_type='periodic',
    riemann_solver='hllc',
    time_scheme='rk3'
)

# 新版本
pipeline = FVMPipeline(
    boundary_type='periodic',
    spatial_scheme='lax_friedrichs',  # 或 'tvdlf', 'hll' 等
    time_scheme='rk3',
    limiter='minmod'  # 适用于TVDLF
)
```

### 3. Solver配置 (`solver.py`)
```python
# 旧版本配置
config = {
    'numerical': {
        'riemann_solver': 'hllc',
        'time_integrator': 'rk3',
        'cfl_number': 0.5,
        'boundary_type': 'periodic'
    }
}

# 新版本配置
config = {
    'numerical': {
        'spatial_scheme': 'lax_friedrichs',  # 统一的空间离散方法
        'time_integrator': 'rk3',
        'cfl_number': 0.5,
        'boundary_type': 'periodic',
        'spatial_params': {'limiter': 'minmod'}  # 方法特定参数
    }
}
```

## 支持的空间离散方法

### 1. Lax-Friedrichs (`'lax_friedrichs'`)
- **特点**: 一阶精度，稳定但有较大数值耗散
- **适用**: 所有物理方程
- **参数**: 无额外参数

### 2. TVDLF (`'tvdlf'`)
- **特点**: 二阶精度，TVD性质，较小数值耗散
- **适用**: 所有物理方程
- **参数**: `limiter` ('minmod', 'superbee', 'van_leer', 'mc')

### 3. HLL Riemann (`'hll'`)
- **特点**: 基于Riemann solver，适中精度
- **适用**: 主要用于Euler和MHD方程
- **参数**: 无额外参数

## 使用示例

### 1. 基本使用
```python
from examples.unified_test_runner import UnifiedTestRunner, TestConfiguration

# 创建测试配置
config = TestConfiguration(
    name="Euler_TVDLF_Test",
    physics="euler",
    spatial_scheme="tvdlf",        # 使用TVDLF方法
    time_integrator="rk3",
    grid_size=(50, 50),
    domain_size=(1.0, 1.0),
    final_time=0.2,
    cfl_number=0.3,
    boundary_type="periodic",
    physics_params={"gamma": 1.4},
    initial_condition="gaussian",
    initial_params={"center_x": 0.5, "center_y": 0.5, "width": 0.1, "amplitude": 1.0},
    spatial_params={"limiter": "minmod"}  # TVDLF参数
)

# 运行测试
runner = UnifiedTestRunner(config)
result = runner.run_simulation()
```

### 2. 比较不同方法
```python
from examples.unified_test_runner import run_all_tests

# 运行预定义的测试案例，比较Lax-Friedrichs和TVDLF
results = run_all_tests()
```

## 测试案例

新的测试框架 (`examples/unified_test_runner.py`) 提供了以下测试案例：

1. **Euler + Lax-Friedrichs**: 均匀流初始条件
2. **Euler + TVDLF**: 均匀流初始条件，minmod限制器
3. **Advection + Lax-Friedrichs**: 高斯脉冲初始条件
4. **Advection + TVDLF**: 高斯脉冲初始条件，minmod限制器

## 优势

1. **统一接口**: 所有空间离散方法使用相同的接口
2. **灵活扩展**: 容易添加新的空间离散方法
3. **参数化**: 支持方法特定的参数配置
4. **解耦设计**: Pipeline不再依赖特定的Riemann solver
5. **易于测试**: 可以轻松比较不同方法的性能

## 兼容性

- 保持了与现有物理方程的兼容性
- 现有的Riemann solver可以通过`RiemannBasedScheme`包装使用
- 配置格式更新，但保持向后兼容的可能性

## 下一步扩展

1. 添加更多空间离散方法（WENO, DG等）
2. 优化性能关键路径
3. 添加自适应网格细化支持
4. 完善边界条件处理