# FVM Framework Tutorial

This tutorial guides you through using the FVM Framework for computational fluid dynamics simulations, from basic concepts to advanced applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Concepts](#basic-concepts)
3. [Your First Simulation](#your-first-simulation)
4. [Understanding the Pipeline](#understanding-the-pipeline)
5. [Working with Different Physics](#working-with-different-physics)
6. [Boundary Conditions](#boundary-conditions)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Examples](#advanced-examples)

## Getting Started

### Installation

Make sure you have Python 3.8+ installed, then:

```bash
# Basic installation
pip install numpy

# For visualization
pip install matplotlib

# For performance monitoring  
pip install psutil

# Add framework to path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Verify Installation

```python
import fvm_framework
print(fvm_framework.get_info())
```

## Basic Concepts

### Finite Volume Method

The finite volume method discretizes conservation laws:

```
∂U/∂t + ∇·F(U) = S(U)
```

Where:
- `U` = conserved variables [ρ, ρu, ρv, ρw, E]
- `F(U)` = flux function
- `S(U)` = source terms

### Pipeline Architecture

The framework uses a 5-stage pipeline:

```
State → Boundary → Reconstruction → Flux → Source → Temporal → New State
```

Each stage is optimized for:
- Cache efficiency
- Vectorization (SIMD)
- Memory bandwidth utilization

### Structure of Arrays (SoA)

Traditional approach (slow):
```python
# Array of Structures - cache unfriendly
cells = [Cell(rho, u, v, w, E) for _ in range(N)]
for cell in cells:
    cell.rho += dt * residual  # Scattered memory access
```

FVM Framework approach (fast):
```python
# Structure of Arrays - vectorizable  
state = np.zeros((5, nx, ny))  # [rho, rho_u, rho_v, rho_w, E]
state[0] += dt * residual[0]   # Contiguous memory access
```

## Your First Simulation

Let's simulate a simple acoustic wave:

```python
from fvm_framework import FVMSolver
import numpy as np

# Step 1: Configure the solver
config = {
    'grid': {
        'nx': 100, 'ny': 50,      # Grid resolution
        'dx': 0.01, 'dy': 0.02,   # Cell sizes
        'x_min': 0.0, 'y_min': 0.0
    },
    'numerical': {
        'riemann_solver': 'hllc',  # Accurate shock capturing
        'time_integrator': 'rk3',  # Third-order accuracy
        'cfl_number': 0.5,         # Stability
        'boundary_type': 'periodic'
    },
    'simulation': {
        'final_time': 0.5,
        'output_interval': 0.1
    }
}

# Step 2: Create solver
solver = FVMSolver(config)

# Step 3: Set initial conditions
def acoustic_wave_ic(x, y):
    gamma = 1.4
    
    # Base state
    rho0 = 1.0
    u0 = v0 = w0 = 0.0
    p0 = 1.0
    
    # Wave perturbation
    amplitude = 0.01
    wavelength = 0.5
    k = 2 * np.pi / wavelength
    
    # Acoustic wave: small density and velocity perturbations
    rho = rho0 + amplitude * np.sin(k * x)
    u = (amplitude / rho0) * np.sin(k * x)  # Velocity perturbation
    v = w = 0.0
    p = p0  # Constant pressure for acoustic wave
    
    # Convert to conservative variables
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2 + w**2)
    return np.array([rho, rho*u, rho*v, rho*w, E])

solver.set_initial_conditions(acoustic_wave_ic)

# Step 4: Run simulation
print("Running acoustic wave simulation...")
solver.solve()

# Step 5: Analyze results
solution = solver.get_solution()
print(f"Simulation completed:")
print(f"  Final time: {solution['current_time']:.4f}")
print(f"  Time steps: {solution['time_step']}")
print(f"  Density range: [{np.min(solution['density']):.6f}, {np.max(solution['density']):.6f}]")

# Step 6: Basic visualization (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot density
    plt.subplot(1, 2, 1)
    plt.contourf(solution['x'], solution['y'], solution['density'], levels=20)
    plt.colorbar(label='Density')
    plt.title('Density Field')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Plot velocity
    plt.subplot(1, 2, 2)
    plt.contourf(solution['x'], solution['y'], solution['velocity_x'], levels=20)
    plt.colorbar(label='X-Velocity')
    plt.title('X-Velocity Field')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Matplotlib not available - skipping visualization")
```

## Understanding the Pipeline

### Data Flow

```python
from fvm_framework.core import FVMDataContainer2D, GridGeometry

# Create data container
geometry = GridGeometry(nx=50, ny=50, dx=0.02, dy=0.02)
data = FVMDataContainer2D(geometry)

# The data container uses SoA layout:
print(f"State shape: {data.state.shape}")        # (5, 50, 50)
print(f"Flux X shape: {data.flux_x.shape}")      # (5, 51, 50) 
print(f"Flux Y shape: {data.flux_y.shape}")      # (5, 50, 51)

# Access conservative variables
density = data.state[0]      # ρ
momentum_x = data.state[1]   # ρu
momentum_y = data.state[2]   # ρv
momentum_z = data.state[3]   # ρw  
energy = data.state[4]       # E

# Get primitive variables (computed automatically)
primitives = data.get_primitives(gamma=1.4)
rho = primitives[0]          # density
u = primitives[1]            # x-velocity
v = primitives[2]            # y-velocity
w = primitives[3]            # z-velocity
p = primitives[4]            # pressure
```

### Pipeline Stages

Each stage can be used independently:

```python
from fvm_framework.core.pipeline import BoundaryStage, FluxStage, TemporalStage

# Initialize stages
boundary_stage = BoundaryStage('periodic')
flux_stage = FluxStage('hllc')
temporal_stage = TemporalStage('rk3')

# Process data through stages
boundary_stage.process(data)
flux_stage.process(data, gamma=1.4)
temporal_stage.process(data, dt=0.001)

# Get performance statistics
print(f"Boundary stage time: {boundary_stage.get_performance_stats()}")
```

## Working with Different Physics

### Euler Equations (Compressible Flow)

The default physics model - ideal gas dynamics:

```python
def compressible_flow_ic(x, y):
    gamma = 1.4  # Heat capacity ratio for air
    
    # Create a pressure-driven flow
    if x < 0.5:
        rho, p = 1.0, 2.0      # High pressure region
    else:
        rho, p = 0.5, 1.0      # Low pressure region
    
    u = v = w = 0.0  # Initially at rest
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2 + w**2)
    
    return np.array([rho, rho*u, rho*v, rho*w, E])

solver.set_initial_conditions(compressible_flow_ic)
```

### Adding Source Terms

For problems with gravity, rotation, or other forces:

```python
from fvm_framework.temporal import ResidualFunction

def gravity_source(data, **kwargs):
    """Add gravitational source term"""
    gravity = kwargs.get('gravity', -9.81)
    source = np.zeros_like(data.state)
    
    # Add gravity to y-momentum and energy equations
    rho = data.state[0]
    source[2] = rho * gravity        # ρ * g_y
    source[4] = rho * data.state[2] / rho * gravity  # ρ * v * g_y
    
    return source

# Create residual function with source terms
residual_fn = ResidualFunction(flux_computer, gravity_source)
```

### Custom Physics Models

```python
def custom_flux_function(u_left, u_right, direction, gamma):
    """Custom flux computation"""
    # Your custom physics model
    # Return flux vector
    pass

# Integrate into framework
from fvm_framework.spatial import RiemannSolver

class CustomSolver(RiemannSolver):
    def solve(self, u_left, u_right, direction, gamma=1.4):
        return custom_flux_function(u_left, u_right, direction, gamma)
```

## Boundary Conditions

### Basic Boundary Types

```python
from fvm_framework.boundary import BoundaryManager

# Create boundary manager
bm = BoundaryManager()

# Set different boundaries
bm.set_boundary('left', 'inflow')
bm.set_boundary('right', 'outflow') 
bm.set_boundary('bottom', 'reflective')
bm.set_boundary('top', 'transmissive')

# Apply to solver
solver.boundary_manager = bm
```

### Physics-Based Boundaries

```python
from fvm_framework.boundary import EulerBoundaryConditions

# Subsonic inflow (specify total conditions)
inflow_bc = EulerBoundaryConditions.subsonic_inflow(
    rho_inf=1.225,     # kg/m³
    u_inf=100.0,       # m/s
    v_inf=0.0,
    w_inf=0.0,
    p_inf=101325.0     # Pa
)

# Subsonic outflow (specify back pressure)
outflow_bc = EulerBoundaryConditions.subsonic_outflow(
    p_back=101325.0    # Pa
)

# Apply boundaries
solver.set_boundary_conditions({
    'left': inflow_bc,
    'right': outflow_bc,
    'top': 'reflective',
    'bottom': 'reflective'
})
```

### Custom Boundary Conditions

```python
from fvm_framework.boundary import CustomBC

def moving_wall_bc(data, **kwargs):
    """Moving wall boundary condition"""
    wall_velocity = kwargs.get('wall_velocity', [0.0, 0.0, 0.0])
    
    # Bottom wall (y=0) - moving wall
    rho_wall = data.state[0, :, 0]
    data.state[1, :, 0] = rho_wall * wall_velocity[0]  # ρu
    data.state[2, :, 0] = rho_wall * wall_velocity[1]  # ρv
    data.state[3, :, 0] = rho_wall * wall_velocity[2]  # ρw

# Create custom boundary
moving_wall = CustomBC("MovingWall", moving_wall_bc)

# Use in simulation
solver.boundary_manager.set_boundary('bottom', moving_wall)
solver.solve(wall_velocity=[10.0, 0.0, 0.0])
```

## Performance Optimization

### Profiling Your Simulation

```python
from fvm_framework.utils import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile a simulation run
metrics = profiler.profile_simulation(solver, final_time=0.1)

print(f"Performance: {metrics.cell_updates_per_second/1000:.1f} kcell-updates/s")
print(f"Memory usage: {metrics.peak_memory_mb:.1f} MB")
print(f"Efficiency: {metrics.cpu_time/metrics.wall_time*100:.1f}%")
```

### Optimization Guidelines

1. **Choose appropriate CFL number**:
   ```python
   # Conservative (stable)
   'cfl_number': 0.3
   
   # Aggressive (faster but less stable)
   'cfl_number': 0.8
   ```

2. **Select optimal Riemann solver**:
   ```python
   # Most robust (slower)
   'riemann_solver': 'hll'
   
   # Best accuracy/speed balance
   'riemann_solver': 'hllc'
   
   # Automatically adaptive
   'riemann_solver': 'adaptive'
   ```

3. **Grid size considerations**:
   ```python
   # Good for performance (powers of 2)
   'nx': 256, 'ny': 128
   
   # Avoid odd numbers if possible
   'nx': 255, 'ny': 127  # Slower
   ```

4. **Memory optimization**:
   ```python
   # Monitor memory usage
   import psutil
   process = psutil.Process()
   print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")
   ```

### Benchmarking Different Configurations

```python
from fvm_framework.utils import benchmark_solver

# Comprehensive benchmark
results = benchmark_solver(
    grid_sizes=[50, 100, 200],
    riemann_solvers=['hll', 'hllc'],
    time_integrators=['rk3', 'rk4'],
    final_time=0.01
)

# Analyze results
successful = results['summary']['successful']
print(f"Successful configurations: {successful}")
```

## Advanced Examples

### Kelvin-Helmholtz Instability

Shear flow instability with vortex formation:

```python
from fvm_framework.examples import KelvinHelmholtzInstability

# Create KH instability problem
kh = KelvinHelmholtzInstability(shear_width=0.05)

# Run simulation to see vortex development
solution = kh.run(final_time=2.0)

# The solution will show vortex formation due to shear instability
kh.plot_solution(variable='density')
```

### Shock-Boundary Layer Interaction

```python
def shock_boundary_layer_ic(x, y):
    """Initial conditions for shock-BL interaction"""
    gamma = 1.4
    
    # Boundary layer profile (simplified)
    u_inf = 2.0  # Mach 2 flow
    delta = 0.1  # Boundary layer thickness
    
    if y < delta:
        # Boundary layer
        u = u_inf * (y / delta) ** (1/7)  # Power law profile
    else:
        # Freestream
        u = u_inf
    
    # Add shock
    if x > 0.5:
        # Post-shock conditions (approximate)
        rho = 2.67
        p = 4.5
        u *= 0.375  # Velocity reduction
    else:
        # Pre-shock conditions  
        rho = 1.0
        p = 1.0
    
    v = w = 0.0
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2 + w**2)
    
    return np.array([rho, rho*u, rho*v, rho*w, E])

# Setup solver for shock-BL interaction
config = {
    'grid': {'nx': 400, 'ny': 100, 'dx': 0.0025, 'dy': 0.002},
    'numerical': {
        'riemann_solver': 'hllc',
        'time_integrator': 'rk3',
        'cfl_number': 0.4,
        'boundary_type': 'mixed'  # Different BCs on different boundaries
    },
    'simulation': {'final_time': 0.5}
}

solver = FVMSolver(config)

# Set boundary conditions
solver.set_boundary_conditions({
    'left': 'inflow',     # Supersonic inflow
    'right': 'outflow',   # Subsonic outflow  
    'bottom': 'reflective',  # Wall
    'top': 'transmissive'    # Freestream
})

solver.set_initial_conditions(shock_boundary_layer_ic)
solver.solve()
```

### Multi-scale Flow Problem

```python
def multi_scale_ic(x, y):
    """Initial conditions with multiple length scales"""
    gamma = 1.4
    
    # Large-scale background flow
    u_bg = 0.5 * np.sin(2 * np.pi * x)
    v_bg = 0.3 * np.cos(2 * np.pi * y)
    
    # Small-scale turbulent fluctuations  
    u_turb = 0.1 * np.random.randn() * np.sin(20 * np.pi * x)
    v_turb = 0.1 * np.random.randn() * np.cos(20 * np.pi * y)
    
    # Medium-scale vortices
    r = np.sqrt((x-0.5)**2 + (y-0.5)**2)
    theta = np.arctan2(y-0.5, x-0.5)
    if r < 0.2:
        u_vortex = -0.5 * np.sin(theta) * np.exp(-r/0.1)
        v_vortex = 0.5 * np.cos(theta) * np.exp(-r/0.1)
    else:
        u_vortex = v_vortex = 0.0
    
    # Combine scales
    rho = 1.0 + 0.1 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)
    u = u_bg + u_turb + u_vortex
    v = v_bg + v_turb + v_vortex
    w = 0.0
    p = 1.0
    
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2 + w**2)
    return np.array([rho, rho*u, rho*v, rho*w, E])

# High-resolution solver for multi-scale problem
solver = FVMSolver({
    'grid': {'nx': 512, 'ny': 512, 'dx': 2.0/512, 'dy': 2.0/512},
    'numerical': {
        'riemann_solver': 'hllc',
        'time_integrator': 'rk4',  # High-order for accuracy
        'cfl_number': 0.3,         # Conservative for stability
    }
})

solver.set_initial_conditions(multi_scale_ic)
```

## Best Practices

### 1. Problem Setup
- Start with simple initial conditions
- Use known analytical solutions for validation
- Check conservation properties
- Verify boundary condition implementation

### 2. Numerical Choices
- Use HLLC for most problems (good accuracy/robustness balance)
- Start with RK3 time integration
- Begin with conservative CFL numbers (0.3-0.5)
- Use periodic boundaries for testing

### 3. Performance
- Profile your simulations to identify bottlenecks
- Use appropriate grid sizes (avoid very small/large aspect ratios)
- Monitor memory usage for large problems
- Consider adaptive time stepping for efficiency

### 4. Debugging
- Check for negative densities/pressures
- Verify initial conditions satisfy governing equations
- Use visualization to identify problems
- Start with small grid sizes for debugging

### 5. Production Runs
- Save intermediate results for long simulations
- Use performance monitoring
- Plan for computational resources needed
- Document your configuration choices

This tutorial should give you a solid foundation for using the FVM Framework. For more advanced topics, consult the API documentation and example problems.