# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install the framework in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[dev]"

# Basic installation with minimal dependencies
pip install -e .
```

### Testing Commands
```bash
# Run all tests
python fvm_framework/drivers/run_tests.py

# Run specific test types
python fvm_framework/drivers/run_tests.py --type basic
python fvm_framework/drivers/run_tests.py --type performance

# Run unit tests for specific components
python fvm_framework/drivers/test_pipeline.py
python fvm_framework/drivers/test_data_container.py
python fvm_framework/drivers/test_boundary_conditions.py

# Run comparison tests (physics validation)
python fvm_framework/drivers/run_comparison_tests.py
```

### Running Physics Comparisons
```bash
# Individual physics equation tests
python fvm_framework/drivers/driver_euler_comparison.py
python fvm_framework/drivers/driver_burgers_comparison.py
python fvm_framework/drivers/driver_advection_comparison.py
python fvm_framework/drivers/driver_sound_wave_comparison.py
python fvm_framework/drivers/driver_mhd_comparison.py
```

## Architecture Overview

### Pipeline Data-Driven Design
The FVM framework uses a 5-stage computational pipeline architecture:

1. **Boundary Stage**: Apply boundary conditions using `BoundaryManager`
2. **Reconstruction Stage**: Reconstruct interface values (constant, MUSCL, WENO, slope limiter)
3. **Flux Stage**: Calculate numerical fluxes using Riemann solvers (HLL, HLLC, HLLD)  
4. **Source Stage**: Evaluate source terms (physics-dependent)
5. **Temporal Stage**: Advance solution in time (Euler, RK2, RK3, RK4)

### Core Components

**Data Container** (`FVMDataContainer2D`):
- Structure of Arrays (SoA) memory layout for SIMD optimization
- Automatic primitive variable calculation (density, pressure, velocity)
- Block-wise data access for cache efficiency
- Shape: `(n_variables, nx, ny)` where variables are `[rho, rho_u, rho_v, rho_w, E]`

**Pipeline Framework** (`FVMPipeline`):
- Orchestrates computation stages in sequence
- Built-in performance monitoring with `PipelineMonitor`
- Configurable stage implementations via factories
- Located in `fvm_framework/core/pipeline.py`

**Factory Pattern**:
- `RiemannSolverFactory` in `fvm_framework/spatial/riemann_solvers.py`
- `TimeIntegratorFactory` in `fvm_framework/temporal/time_integrators.py`
- `ReconstructionFactory` in `fvm_framework/spatial/reconstruction/factory.py`
- `FluxCalculatorFactory` in `fvm_framework/spatial/flux_calculation/factory.py`

### Key Directory Structure
```
fvm_framework/
├── core/                    # Core pipeline and data management
│   ├── pipeline.py         # Pipeline orchestration and stages
│   ├── data_container.py   # SoA data layout and management
│   └── solver.py          # High-level solver interface (under refactoring)
├── spatial/                # Spatial discretization methods
│   ├── reconstruction/     # Interface value reconstruction schemes
│   ├── flux_calculation/   # Flux calculation methods  
│   └── riemann_solvers.py # Riemann solver implementations
├── temporal/               # Time integration schemes
├── boundary/               # Boundary condition implementations  
├── physics/                # Physics equation definitions
├── testcases/             # Standard test problems and validation
└── drivers/               # Test drivers and comparison scripts
```

### Physics Equations Supported
- **Advection**: Linear advection equation
- **Burgers**: Nonlinear Burgers equation  
- **Euler**: Compressible Euler equations (gas dynamics)
- **Sound Wave**: Linearized sound wave equations
- **MHD**: Magnetohydrodynamics equations

Each physics class inherits from `PhysicsBase` and implements:
- `conservative_to_primitive()`: Convert conservative to primitive variables
- `primitive_to_conservative()`: Convert primitive to conservative variables  
- `compute_max_wave_speed()`: Calculate maximum wave speed for CFL condition

## Development Guidelines

### Code Conventions
- Follow the existing SoA (Structure of Arrays) data layout pattern
- Use factory patterns for creating numerical schemes
- Implement new stages by inheriting from `ComputationStage`
- Physics equations should inherit from `PhysicsBase`
- All new reconstruction schemes inherit from `ReconstructionScheme`
- All new flux methods inherit from `FluxCalculator`

### Testing Requirements
- Unit tests go in `fvm_framework/drivers/test_*.py`
- Physics validation tests go in `fvm_framework/testcases/`
- Comparison drivers in `fvm_framework/drivers/driver_*_comparison.py`
- Always run tests after making changes to numerical schemes

### Performance Considerations
- The framework is optimized for vectorization and cache efficiency
- Data is stored in SoA format: `state[variable_index, i, j]`
- Boundary conditions are applied in-place to minimize memory allocation
- Pipeline stages are timed automatically for performance analysis

### Common Debugging
- Use `PipelineMonitor` for performance profiling
- Check CFL conditions if simulation becomes unstable
- Verify boundary condition setup for physics problems
- Use the comparison drivers to validate new numerical methods

## File Organization Notes

- **Solver refactoring**: The main `FVMSolver` class is currently being refactored (see commented imports in `__init__.py`)
- **Test structure**: Tests are distributed between `drivers/test_*.py` (unit tests) and `testcases/` (physics validation)
- **Chinese documentation**: Some documentation files are in Chinese and provide detailed architecture explanations
- **Deleted files**: Several summary and guide files have been marked for deletion in git status