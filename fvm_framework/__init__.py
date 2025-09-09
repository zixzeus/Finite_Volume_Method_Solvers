"""
FVM Framework - High-Performance 2D Finite Volume Method Solver

A pipeline data-driven architecture for finite volume method simulations
optimized for modern hardware with vectorization and parallel computing support.

Author: FVM Framework Development Team
License: MIT
Version: 1.0.0
"""

# Core components
from .core.data_container import FVMDataContainer2D, GridGeometry
from .core.pipeline import FVMPipeline, PipelineMonitor

# Boundary conditions
from .boundary.boundary_conditions import (
    BoundaryManager, EulerBoundaryConditions,
    PeriodicBC, ReflectiveBC, TransmissiveBC, InflowBC
)

# Spatial discretization - Modular Framework
from . import spatial
from .spatial.riemann_solvers import (
    RiemannSolverFactory, HLLSolver, HLLCSolver, HLLDSolver
)

# Temporal integration
from .temporal.time_integrators import (
    TimeIntegratorFactory, ForwardEuler, RungeKutta2, RungeKutta3, RungeKutta4
)

# Complete solver interface
# from .core.solver import FVMSolver, create_blast_wave_solver, create_shock_tube_solver

__version__ = "1.0.0"
__author__ = "FVM Framework Development Team"
__license__ = "MIT"

__all__ = [
    # Core
    'FVMDataContainer2D',
    'GridGeometry', 
    'FVMPipeline',
    'PipelineMonitor',
    
    # Boundary conditions
    'BoundaryManager',
    'EulerBoundaryConditions',
    'PeriodicBC',
    'ReflectiveBC', 
    'TransmissiveBC',
    'InflowBC',
    
    # Spatial methods
    'spatial',
    'RiemannSolverFactory',
    'HLLSolver',
    'HLLCSolver',
    'HLLDSolver',
    
    # Temporal methods
    'TimeIntegratorFactory',
    'ForwardEuler',
    'RungeKutta2',
    'RungeKutta3', 
    'RungeKutta4',
    
    # Complete solver (commented out for now)
    # 'FVMSolver',
    # 'create_blast_wave_solver',
    # 'create_shock_tube_solver'
]


def get_version():
    """Get framework version"""
    return __version__


def get_info():
    """Get framework information"""
    return {
        'name': 'FVM Framework',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'description': 'High-performance 2D finite volume method solver',
        'architecture': 'Pipeline data-driven with Structure of Arrays',
        'features': [
            'Multiple Riemann solvers (HLL, HLLC, HLLD)',
            'Various time integrators (Euler, RK2, RK3, RK4)',
            'Comprehensive boundary conditions',
            'Pipeline-based architecture',
            'Performance monitoring',
            'Vectorization-friendly data layout'
        ]
    }


def print_info():
    """Print framework information"""
    info = get_info()
    print(f"{info['name']} v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"License: {info['license']}")
    print(f"Architecture: {info['architecture']}")
    print("\nFeatures:")
    for feature in info['features']:
        print(f"  • {feature}")


# Quick start examples and tutorials (commented out due to solver refactoring)


# Import numpy if available for examples
try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False
    print("Warning: NumPy not available. Some examples may not work.")


def run_tests():
    """Run the framework test suite"""
    try:
        from .tests import run_all_tests
        return run_all_tests()
    except ImportError:
        print("Test suite not available. Install in development mode to run tests.")
        return None


if __name__ == "__main__":
    print_info()