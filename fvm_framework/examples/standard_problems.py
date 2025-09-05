"""
Standard Test Problems for FVM Framework

This module implements classical test problems for validating finite volume
method implementations, including blast waves, shock tubes, and flow instabilities.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict, Any
import matplotlib.pyplot as plt

from solver import FVMSolver, create_blast_wave_solver, create_shock_tube_solver
from core.data_container import GridGeometry


class TestProblem:
    """Base class for test problems"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.solver: Optional[FVMSolver] = None
        self.solution: Optional[Dict[str, np.ndarray]] = None
        
    def setup_solver(self) -> FVMSolver:
        """Set up the solver for this test problem"""
        raise NotImplementedError
        
    def initial_conditions(self, x: float, y: float) -> np.ndarray:
        """Define initial conditions at point (x, y)"""
        raise NotImplementedError
        
    def run(self, final_time: Optional[float] = None, silent: bool = False) -> Dict[str, np.ndarray]:
        """Run the test problem"""
        if self.solver is None:
            self.solver = self.setup_solver()
            
        self.solver.set_initial_conditions(self.initial_conditions)
        
        if not silent:
            print(f"Running {self.name}")
            print(f"Description: {self.description}")
            
        self.solver.solve(final_time)
        self.solution = self.solver.get_solution()
        
        return self.solution
    
    def plot_solution(self, variable: str = 'density', save_path: Optional[str] = None):
        """Plot the solution"""
        if self.solution is None:
            raise RuntimeError("Must run simulation before plotting")
            
        plt.figure(figsize=(10, 8))
        
        if variable in self.solution:
            plt.contourf(self.solution['x'], self.solution['y'], 
                        self.solution[variable], levels=50, cmap='viridis')
            plt.colorbar(label=variable.title())
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{self.name}: {variable.title()} at t = {self.solution["current_time"]:.4f}')
            plt.axis('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
        else:
            print(f"Variable {variable} not available in solution")


class SodShockTube(TestProblem):
    """
    Sod's shock tube problem.
    
    Classic 1D Riemann problem with known analytical solution.
    Left state: rho=1.0, p=1.0, u=0
    Right state: rho=0.125, p=0.1, u=0
    """
    
    def __init__(self):
        super().__init__(
            "Sod Shock Tube",
            "1D shock tube with exact Riemann solution"
        )
        self.gamma = 1.4
        
    def setup_solver(self) -> FVMSolver:
        """Set up shock tube solver"""
        config = {
            'grid': {
                'nx': 400, 'ny': 4,
                'dx': 1.0/400, 'dy': 0.01,
                'x_min': 0.0, 'y_min': -0.02
            },
            'numerical': {
                'riemann_solver': 'hllc',
                'time_integrator': 'rk3',
                'cfl_number': 0.9,
                'boundary_type': 'transmissive'
            },
            'simulation': {
                'final_time': 0.2,
                'output_interval': 0.05,
                'monitor_interval': 50
            }
        }
        return FVMSolver(config)
    
    def initial_conditions(self, x: float, y: float) -> np.ndarray:
        """Sod shock tube initial conditions"""
        if x < 0.5:
            # Left state
            rho, p = 1.0, 1.0
        else:
            # Right state
            rho, p = 0.125, 0.1
            
        # At rest initially
        u = v = w = 0.0
        
        # Total energy
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    def plot_1d_comparison(self, analytical: bool = True, save_path: Optional[str] = None):
        """Plot 1D comparison with analytical solution"""
        if self.solution is None:
            raise RuntimeError("Must run simulation before plotting")
            
        # Extract 1D slice along x-direction
        j_mid = self.solution['density'].shape[1] // 2
        x_1d = self.solution['x'][:, j_mid]
        density_1d = self.solution['density'][:, j_mid]
        pressure_1d = self.solution['pressure'][:, j_mid]
        velocity_1d = self.solution['velocity_x'][:, j_mid]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Density
        ax1.plot(x_1d, density_1d, 'b-', linewidth=2, label='FVM')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.set_title('Density')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Velocity
        ax2.plot(x_1d, velocity_1d, 'r-', linewidth=2, label='FVM')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Pressure
        ax3.plot(x_1d, pressure_1d, 'g-', linewidth=2, label='FVM')
        ax3.set_xlabel('x')
        ax3.set_ylabel('Pressure')
        ax3.set_title('Pressure')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.suptitle(f'Sod Shock Tube at t = {self.solution["current_time"]:.4f}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class CircularBlastWave(TestProblem):
    """
    2D circular blast wave problem.
    
    High pressure circular region embedded in low pressure medium.
    Tests shock propagation and cylindrical symmetry preservation.
    """
    
    def __init__(self, blast_radius: float = 0.1):
        super().__init__(
            "Circular Blast Wave",
            f"2D blast wave with initial radius {blast_radius}"
        )
        self.blast_radius = blast_radius
        self.gamma = 1.4
        
    def setup_solver(self) -> FVMSolver:
        """Set up blast wave solver"""
        return create_blast_wave_solver(nx=200, ny=200, domain_size=2.0)
    
    def initial_conditions(self, x: float, y: float) -> np.ndarray:
        """Circular blast wave initial conditions"""
        r = np.sqrt(x**2 + y**2)
        
        if r < self.blast_radius:
            # High pressure region
            rho = 1.0
            p = 10.0
        else:
            # Low pressure region
            rho = 0.125
            p = 0.1
            
        # At rest initially
        u = v = w = 0.0
        
        # Total energy
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])


class KelvinHelmholtzInstability(TestProblem):
    """
    Kelvin-Helmholtz instability problem.
    
    Shear flow instability with density and velocity jumps.
    Tests the ability to capture flow instabilities and vortex formation.
    """
    
    def __init__(self, shear_width: float = 0.05):
        super().__init__(
            "Kelvin-Helmholtz Instability",
            "Shear flow instability with vortex formation"
        )
        self.shear_width = shear_width
        self.gamma = 1.4
        
    def setup_solver(self) -> FVMSolver:
        """Set up KH instability solver"""
        config = {
            'grid': {
                'nx': 256, 'ny': 128,
                'dx': 2.0/256, 'dy': 1.0/128,
                'x_min': 0.0, 'y_min': 0.0
            },
            'numerical': {
                'riemann_solver': 'hllc',
                'time_integrator': 'rk3',
                'cfl_number': 0.4,
                'boundary_type': 'periodic'  # Periodic in x, walls in y would be better
            },
            'simulation': {
                'final_time': 2.0,
                'output_interval': 0.2,
                'monitor_interval': 100
            }
        }
        return FVMSolver(config)
    
    def initial_conditions(self, x: float, y: float) -> np.ndarray:
        """KH instability initial conditions"""
        # Interface at y = 0.5
        y_interface = 0.5
        
        # Smooth transition using hyperbolic tangent
        rho1, rho2 = 2.0, 1.0
        u1, u2 = -0.5, 0.5
        p = 2.5  # Constant pressure
        
        # Smooth profiles
        rho = 0.5 * (rho1 + rho2) + 0.5 * (rho1 - rho2) * np.tanh((y - y_interface) / self.shear_width)
        u = 0.5 * (u1 + u2) + 0.5 * (u1 - u2) * np.tanh((y - y_interface) / self.shear_width)
        
        # Add small perturbation to trigger instability
        perturbation_amplitude = 0.01
        v = perturbation_amplitude * np.sin(4 * np.pi * x) * np.exp(-(y - y_interface)**2 / (2 * self.shear_width**2))
        w = 0.0
        
        # Total energy
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])


class RayleighTaylorInstability(TestProblem):
    """
    Rayleigh-Taylor instability problem.
    
    Heavy fluid over light fluid in gravitational field.
    Tests buoyancy-driven instabilities.
    """
    
    def __init__(self, gravity: float = -0.1):
        super().__init__(
            "Rayleigh-Taylor Instability", 
            "Heavy fluid over light fluid instability"
        )
        self.gravity = gravity
        self.gamma = 1.4
        
    def setup_solver(self) -> FVMSolver:
        """Set up RT instability solver"""
        config = {
            'grid': {
                'nx': 128, 'ny': 256,
                'dx': 1.0/128, 'dy': 2.0/256,
                'x_min': 0.0, 'y_min': 0.0
            },
            'numerical': {
                'riemann_solver': 'hllc',
                'time_integrator': 'rk3',
                'cfl_number': 0.3,
                'boundary_type': 'reflective'  # Walls on all sides
            },
            'simulation': {
                'final_time': 8.0,
                'output_interval': 0.5,
                'monitor_interval': 100
            }
        }
        return FVMSolver(config)
    
    def initial_conditions(self, x: float, y: float) -> np.ndarray:
        """RT instability initial conditions"""
        # Interface at y = 1.0
        y_interface = 1.0
        interface_width = 0.05
        
        # Heavy fluid above, light fluid below
        if y > y_interface:
            rho = 2.0
        else:
            rho = 1.0
            
        # Smooth the interface
        rho = 1.5 + 0.5 * np.tanh((y - y_interface) / interface_width)
        
        # Initially at rest
        u = v = w = 0.0
        
        # Add small perturbation to trigger instability
        perturbation_amplitude = 0.01
        v = perturbation_amplitude * (1 + np.cos(4 * np.pi * x)) * np.exp(-(y - y_interface)**2 / (2 * interface_width**2))
        
        # Hydrostatic pressure (approximately)
        p = 1.0 + rho * abs(self.gravity) * (2.0 - y)
        
        # Total energy
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])


class DoubleMachReflection(TestProblem):
    """
    Double Mach reflection problem.
    
    Strong shock reflecting off a wedge, creating complex shock interactions.
    Standard test for high-resolution shock-capturing schemes.
    """
    
    def __init__(self):
        super().__init__(
            "Double Mach Reflection",
            "Strong shock reflection with complex interactions"
        )
        self.gamma = 1.4
        self.mach_number = 10.0
        
    def setup_solver(self) -> FVMSolver:
        """Set up double Mach reflection solver"""
        config = {
            'grid': {
                'nx': 480, 'ny': 120,
                'dx': 4.0/480, 'dy': 1.0/120,
                'x_min': 0.0, 'y_min': 0.0
            },
            'numerical': {
                'riemann_solver': 'hllc',
                'time_integrator': 'rk3',
                'cfl_number': 0.6,
                'boundary_type': 'mixed'  # Need custom boundary conditions
            },
            'simulation': {
                'final_time': 0.2,
                'output_interval': 0.05,
                'monitor_interval': 50
            }
        }
        return FVMSolver(config)
    
    def initial_conditions(self, x: float, y: float) -> np.ndarray:
        """Double Mach reflection initial conditions"""
        # Shock angle and position
        shock_angle = np.pi / 3.0  # 60 degrees
        x_shock = x + y / np.tan(shock_angle)
        
        if x_shock < 1.0/6.0:
            # Pre-shock state
            rho = 8.0
            u = 8.25 * np.cos(shock_angle)
            v = -8.25 * np.sin(shock_angle)
            p = 116.5
        else:
            # Post-shock state
            rho = 1.4
            u = 0.0
            v = 0.0
            p = 1.0
            
        w = 0.0
        
        # Total energy
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])


# Test suite runner
class TestSuite:
    """Test suite for running multiple standard problems"""
    
    def __init__(self):
        self.problems = {
            'sod': SodShockTube(),
            'blast': CircularBlastWave(),
            'kh': KelvinHelmholtzInstability(),
            'rt': RayleighTaylorInstability(),
            'dmr': DoubleMachReflection()
        }
        
    def run_problem(self, problem_name: str, **kwargs):
        """Run a specific test problem"""
        if problem_name not in self.problems:
            print(f"Unknown problem: {problem_name}")
            print(f"Available problems: {list(self.problems.keys())}")
            return None
            
        problem = self.problems[problem_name]
        return problem.run(**kwargs)
    
    def run_all(self, quick: bool = True):
        """Run all test problems"""
        results = {}
        
        print("Running FVM Framework Test Suite")
        print("=" * 50)
        
        for name, problem in self.problems.items():
            print(f"\nRunning {name}...")
            try:
                if quick and name in ['rt', 'dmr']:
                    # Skip long-running tests in quick mode
                    print(f"Skipping {name} (long-running test)")
                    continue
                    
                final_time = 0.1 if quick else None
                results[name] = problem.run(final_time=final_time, silent=True)
                print(f"✓ {name} completed successfully")
                
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                results[name] = None
        
        return results
    
    def list_problems(self):
        """List available test problems"""
        print("Available Test Problems:")
        print("-" * 30)
        for name, problem in self.problems.items():
            print(f"{name:10s}: {problem.description}")


def run_validation_suite():
    """Run validation test suite"""
    suite = TestSuite()
    return suite.run_all(quick=False)


def run_quick_tests():
    """Run quick validation tests"""
    suite = TestSuite()
    return suite.run_all(quick=True)


if __name__ == "__main__":
    # Demo: run Sod shock tube
    print("FVM Framework - Running Sod Shock Tube Demo")
    
    sod = SodShockTube()
    sod.run()
    
    try:
        sod.plot_1d_comparison()
    except ImportError:
        print("Matplotlib not available - cannot plot results")