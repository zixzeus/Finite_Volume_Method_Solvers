# Spatial Discretization Methods

这个目录包含了所有空间离散化方法的实现，现在采用模块化设计，每种方法都在独立的文件中。

## 文件结构

### 核心架构
- `base.py` - 所有空间离散方法的抽象基类
- `factory.py` - 统一的工厂方法，创建所有类型的空间离散方案

### 有限体积方法
- `lax_friedrichs.py` - Lax-Friedrichs方法（一阶，稳定但有耗散）
- `tvd_lax_friedrichs.py` - TVD Lax-Friedrichs方法（二阶，支持flux limiter）

### Riemann求解器方法
- `riemann_schemes.py` - HLL/HLLC/HLLD Riemann求解器的包装器
- `riemann_solvers.py` - 原始Riemann求解器实现（底层）

### 高阶方法
- `dg_scheme.py` - 间断Galerkin方法的统一接口
- `discontinuous_galerkin.py` - 原始DG实现（底层）

## 使用方式

### 基本使用
```python
from spatial.factory import SpatialDiscretizationFactory

# 创建Lax-Friedrichs方法
lf_scheme = SpatialDiscretizationFactory.create('lax_friedrichs')

# 创建TVDLF方法
tvdlf_scheme = SpatialDiscretizationFactory.create('tvdlf', limiter='minmod')

# 创建DG方法
dg_scheme = SpatialDiscretizationFactory.create('dg', polynomial_order=2)
```

### 查看所有可用方法
```python
from spatial.factory import SpatialDiscretizationFactory
SpatialDiscretizationFactory.print_available_schemes()
```

## 支持的方法

### 有限体积方法
- `lax_friedrichs` - 一阶Lax-Friedrichs（别名：lf, lax_f）
- `tvdlf` - 二阶TVD Lax-Friedrichs（别名：tvd_lf, tvd_lax_friedrichs）
  - 参数：`limiter` ('minmod', 'superbee', 'van_leer', 'mc')

### Riemann求解器方法
- `hll` - HLL Riemann求解器（别名：hll_riemann）
- `hllc` - HLLC Riemann求解器（别名：hllc_riemann）
- `hlld` - HLLD Riemann求解器（别名：hlld_riemann）

### 间断Galerkin方法
- `dg` - 通用DG方法（别名：discontinuous_galerkin）
  - 参数：`polynomial_order` (0,1,2), `riemann_solver`
- `dg_p0` - P0 DG（等价于有限体积）（别名：dg0）
- `dg_p1` - P1 DG（分片线性）（别名：dg1）
- `dg_p2` - P2 DG（分片二次）（别名：dg2）

## 架构优势

1. **模块化设计** - 每种方法在独立文件中，便于维护和扩展
2. **统一接口** - 所有方法都实现相同的`SpatialDiscretization`接口
3. **灵活参数** - 支持方法特定的参数配置
4. **易于扩展** - 添加新方法只需继承基类并注册到工厂
5. **清晰分层** - 抽象层、具体实现层和工厂层分离

## 扩展新方法

要添加新的空间离散方法：

1. 在相应文件中创建新类，继承适当的基类
2. 实现`compute_fluxes`方法
3. 在`factory.py`中注册新方法
4. 可选：添加方法特定的参数和别名

这种设计让代码更加模块化，每个文件职责单一，易于理解和维护。