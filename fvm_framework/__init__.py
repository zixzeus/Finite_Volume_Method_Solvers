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

# Spatial discretization
from .spatial.finite_volume import LaxFriedrichs, TVDLF, UpwindScheme
from .spatial.riemann_solvers import (
    RiemannSolverFactory, HLLSolver, HLLCSolver, HLLDSolver
)

# Temporal integration
from .temporal.time_integrators import (
    TimeIntegratorFactory, ForwardEuler, RungeKutta2, RungeKutta3, RungeKutta4
)

# Complete solver interface
from .solver import FVMSolver, create_blast_wave_solver, create_shock_tube_solver

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
    'LaxFriedrichs',
    'TVDLF',
    'UpwindScheme',
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
    
    # Complete solver
    'FVMSolver',
    'create_blast_wave_solver',
    'create_shock_tube_solver'
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


# Quick start examples and tutorials
class QuickStart:
    """Quick start examples for the FVM framework"""
    
    @staticmethod
    def blast_wave_example():
        """
        Example: 2D blast wave simulation
        
        This creates a blast wave problem with high pressure in the center
        and low pressure in the surrounding region.
        """
        print("FVM Framework - Blast Wave Example")
        print("=" * 40)
        
        # Create solver
        solver = create_blast_wave_solver(nx=100, ny=100)
        
        # Set initial conditions - blast wave
        def blast_wave_ic(x, y):
            r = np.sqrt(x**2 + y**2)
            if r < 0.1:
                # High pressure region
                rho = 1.0
                p = 10.0
            else:
                # Low pressure region  
                rho = 0.125
                p = 0.1
            
            # Conservative variables [rho, rho*u, rho*v, rho*w, E]
            gamma = 1.4
            u = v = w = 0.0  # Initially at rest
            E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
            
            return np.array([rho, rho*u, rho*v, rho*w, E])
        
        solver.set_initial_conditions(blast_wave_ic)
        
        print("Initial conditions set - circular blast wave")
        print("Grid: 100 x 100")
        print("Riemann solver: HLLC")
        print("Time integrator: RK3")
        print("Boundary conditions: Transmissive")
        print("\nTo run simulation: solver.solve()")
        
        return solver
    
    @staticmethod
    def shock_tube_example():
        """
        Example: 1D shock tube (Sod's problem) in 2D domain
        
        Classic Riemann problem with left and right states.
        """
        print("FVM Framework - Shock Tube Example")
        print("=" * 40)
        
        # Create solver
        solver = create_shock_tube_solver(nx=200, ny=4)
        
        # Set initial conditions - Sod shock tube
        def shock_tube_ic(x, y):
            if x < 0.5:
                # Left state
                rho, p = 1.0, 1.0
            else:
                # Right state
                rho, p = 0.125, 0.1
            
            # Conservative variables
            gamma = 1.4
            u = v = w = 0.0  # Initially at rest
            E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
            
            return np.array([rho, rho*u, rho*v, rho*w, E])
        
        solver.set_initial_conditions(shock_tube_ic)
        
        print("Initial conditions set - Sod shock tube")
        print("Grid: 200 x 4") 
        print("Domain: [0, 1] x [0, 0.04]")
        print("Riemann solver: HLLC")
        print("Time integrator: RK3")
        print("Boundary conditions: Transmissive")
        print("\nTo run simulation: solver.solve()")
        
        return solver


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