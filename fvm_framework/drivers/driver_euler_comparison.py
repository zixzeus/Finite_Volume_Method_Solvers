"""
Euler Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D Euler equations,
testing various Riemann solvers and reconstruction methods for compressible flow problems.

Physics: ∂U/∂t + ∂F(U)/∂x + ∂G(U)/∂y = 0
Where U = [ρ, ρu, ρv, ρw, E]
"""

import numpy as np
import os
import time
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvm_framework.utils import FVMPlotter, create_physics_specific_plotter

from fvm_framework.core.solver import FVMSolver
from fvm_framework.testcases.testsuite_euler_2D import get_euler_test_case, list_euler_test_cases


@dataclass
class EulerComparisonParameters:
    """Parameters for Euler equation comparison tests"""
    # Grid parameters
    nx: int = 64
    ny: int = 64
    domain_size: float = 1.0
    
    # Physics parameters
    gamma: float = 1.4
    
    # Simulation parameters
    final_time: float = 0.2
    cfl_number: float = 0.3
    outputtimes: Optional[List[float]] = None
    
    # Test cases to run
    test_cases: Optional[List[str]] = None
    
    # Spatial methods to compare
    spatial_methods: Optional[List[Dict[str, Any]]] = None
    
    # Output parameters
    output_dir: str = "comparison_results"
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self):
        if self.test_cases is None:
            self.test_cases = ['sod_shock_tube', 'double_mach_reflection', 'explosion_2d']
        
        if self.outputtimes is None:
            # Create 5 equally spaced time points from 0 to final_time
            self.outputtimes = [i * self.final_time / 4 for i in range(5)]
        
        if self.spatial_methods is None:
            self.spatial_methods = [
                {
                    'name': 'Lax-Friedrichs + Constant',
                    'reconstruction_type': 'constant',
                    'flux_type': 'lax_friedrichs',
                    'color': 'blue',
                    'linestyle': '-'
                },
                {
                    'name': 'HLL + Slope Limiter',
                    'reconstruction_type': 'slope_limiter',
                    'flux_type': 'hll',
                    'flux_params': {'riemann_solver': 'hll'},
                    'color': 'red',
                    'linestyle': '--'
                },
                {
                    'name': 'HLLC + Slope Limiter',
                    'reconstruction_type': 'slope_limiter',
                    'flux_type': 'hllc',
                    'flux_params': {'riemann_solver': 'hllc'},
                    'color': 'green',
                    'linestyle': '-.'
                },
                {
                    'name': 'HLLC + WENO5',
                    'reconstruction_type': 'weno5',
                    'flux_type': 'hllc',
                    'flux_params': {'riemann_solver': 'hllc'},
                    'color': 'purple',
                    'linestyle': ':'
                }
            ]


class EulerComparison:
    """Main comparison test class for Euler equations"""
    
    def __init__(self, params: EulerComparisonParameters):
        self.params = params
        self.results = {}
        self.timing_results = {}
        
        # Create output directory
        if self.params.save_plots:
            os.makedirs(self.params.output_dir, exist_ok=True)
    
    def create_solver_config(self, method: Dict[str, Any]) -> Dict[str, Any]:
        """Create solver configuration for a specific method"""
        return {
            'grid': {
                'nx': self.params.nx,
                'ny': self.params.ny,
                'dx': self.params.domain_size / self.params.nx,
                'dy': self.params.domain_size / self.params.ny,
                'x_min': 0.0,
                'y_min': 0.0
            },
            'physics': {
                'equation': 'euler',
                'params': {
                    'gamma': self.params.gamma
                }
            },
            'numerical': {
                'reconstruction_type': method['reconstruction_type'],
                'flux_type': method['flux_type'],
                'time_scheme': 'rk3',
                'cfl_number': self.params.cfl_number,
                'boundary_type': 'transmissive',
                'flux_params': method.get('flux_params', {})
            },
            'simulation': {
                'final_time': self.params.final_time,
                'output_interval': self.params.final_time,
                'monitor_interval': 100,
                'outputtimes': self.params.outputtimes
            }
        }
    
    def run_single_test(self, test_case: str, method: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Run a single test with specified method and test case"""
        print(f"  Running {method['name']} with {test_case}")
        
        try:
            # Create solver
            config = self.create_solver_config(method)
            solver = FVMSolver(config)
            
            # Get initial conditions
            initial_state = get_euler_test_case(test_case, self.params.nx, self.params.ny)
            
            # Set initial conditions
            solver.set_initial_conditions(initial_state)
            
            # Run simulation with timing
            start_time = time.perf_counter()
            # Run simulation (solver will automatically collect time series data)
            solver.solve()
            end_time = time.perf_counter()
            
            # Get final solution and time series data
            solution = solver.get_solution()
            time_series_data = solver.get_time_series()
            stats = solver.get_statistics()
            
            timing_info = {
                'total_time': end_time - start_time,
                'total_steps': solution['time_step'],
                'avg_time_per_step': (end_time - start_time) / max(solution['time_step'], 1)
            }
            
            # Package solution with time series data
            solution_with_series = {
                'final_solution': solution,
                'time_series': time_series_data,
                'stats': stats
            }
            
            return solution_with_series, timing_info
            
        except Exception as e:
            print(f"    Error in simulation: {e}")
            return None, {'error': str(e)}
    
    def run_comparison_test(self, test_case: str):
        """Run comparison test for a specific test case"""
        print(f"\n=== Running Euler Comparison: {test_case} ===")
        
        test_results = {}
        test_timings = {}
        
        # Get initial condition for reference
        initial_state = get_euler_test_case(test_case, self.params.nx, self.params.ny)
        
        for method in self.params.spatial_methods:
            # Skip WENO5 for blast_wave test case (too unstable for strong shocks)
            if test_case == 'blast_wave' and method.get('reconstruction_type') == 'weno5':
                print(f"    Skipping {method['name']} for blast_wave (WENO5 not suitable for strong shocks)")
                continue
                
            solution, timing = self.run_single_test(test_case, method)
            
            if solution is not None:
                test_results[method['name']] = solution
                test_timings[method['name']] = timing
        
        self.results[test_case] = {
            'initial_condition': initial_state,
            'solutions': test_results,
            'timings': test_timings
        }
    
    def compute_conservation_errors(self, test_case: str) -> Dict[str, Dict[str, float]]:
        """Compute conservation errors for mass, momentum, and energy"""
        if test_case not in self.results:
            return {}
        
        errors = {}
        initial_condition = self.results[test_case]['initial_condition']
        
        # Compute initial conserved quantities
        rho_i = initial_condition[0]
        rho_u_i = initial_condition[1]
        rho_v_i = initial_condition[2]
        E_i = initial_condition[4]
        
        initial_mass = np.sum(rho_i)
        initial_mom_x = np.sum(rho_u_i)
        initial_mom_y = np.sum(rho_v_i)
        initial_energy = np.sum(E_i)
        
        for method_name, solution in self.results[test_case]['solutions'].items():
            if 'conservative' in solution:
                final_state = solution['conservative']
            else:
                final_state = solution['final_solution']['conservative']
            
            # Compute final conserved quantities
            rho_f = final_state[0]
            rho_u_f = final_state[1]
            rho_v_f = final_state[2]
            E_f = final_state[4]
            
            final_mass = np.sum(rho_f)
            final_mom_x = np.sum(rho_u_f)
            final_mom_y = np.sum(rho_v_f)
            final_energy = np.sum(E_f)
            
            # Compute relative errors
            mass_error = abs(final_mass - initial_mass) / abs(initial_mass)
            mom_x_error = abs(final_mom_x - initial_mom_x) / abs(initial_mom_x + 1e-15)
            mom_y_error = abs(final_mom_y - initial_mom_y) / abs(initial_mom_y + 1e-15)
            energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
            
            errors[method_name] = {
                'mass': mass_error,
                'momentum_x': mom_x_error,
                'momentum_y': mom_y_error,
                'energy': energy_error,
                'total': mass_error + mom_x_error + mom_y_error + energy_error
            }
        
        return errors
    
    def compute_primitive_variables(self, conservative_state: np.ndarray, gamma: float = 1.4) -> np.ndarray:
        """Convert conservative to primitive variables"""
        rho = conservative_state[0]
        rho_u = conservative_state[1]
        rho_v = conservative_state[2]
        rho_w = conservative_state[3]
        E = conservative_state[4]
        
        # Avoid division by zero
        rho = np.maximum(rho, 1e-15)
        
        u = rho_u / rho
        v = rho_v / rho
        w = rho_w / rho
        
        # Compute pressure
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        internal_energy = E - kinetic_energy
        pressure = (gamma - 1.0) * internal_energy
        pressure = np.maximum(pressure, 1e-15)  # Ensure positive pressure
        
        return np.array([rho, u, v, w, pressure])
    
    def plot_comparison(self, test_case: str):
        """Generate comparison plots for a test case"""
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('euler')
        
        # Use common multi-variable comparison plotting
        plotter.plot_multi_variable_comparison(
            test_case=test_case,
            results=self.results,
            variables=physics_config['variables'],
            title_suffix=physics_config['title_suffix']
        )
        
        # Also generate conservation error plot if available
        conservation_errors = self.compute_conservation_errors(test_case) 
        if conservation_errors:
            plotter.plot_conservation_errors(
                conservation_errors=conservation_errors,
                test_case=test_case,
                error_types=physics_config['conservation_errors']
            )
        
    def plot_time_series(self, test_case: str, method_name: str):
        """Generate time series plots for specified output times"""
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('euler')
        
        # Multi-variable time series showing all 5 Euler variables
        plotter.plot_multi_variable_time_series(
            test_case=test_case,
            method_name=method_name,
            results=self.results,
            variables=physics_config['variables']
        )
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*60)
        print("EULER EQUATION COMPARISON SUMMARY")
        print("="*60)
        
        for test_case in self.params.test_cases:
            if test_case not in self.results:
                continue
                
            print(f"\nTest Case: {test_case}")
            print("-" * 40)
            
            # Print timings
            timings = self.results[test_case]['timings']
            if timings:
                print("Computation Times:")
                for method_name, timing in timings.items():
                    if 'error' not in timing:
                        print(f"  {method_name:25s}: {timing['total_time']:6.3f}s "
                              f"({timing['total_steps']} steps, {timing['avg_time_per_step']:.6f}s/step)")
                    else:
                        print(f"  {method_name:25s}: FAILED ({timing['error']})")
            
            # Print conservation errors
            errors = self.compute_conservation_errors(test_case)
            if errors:
                print("Conservation Errors:")
                for method_name, error in errors.items():
                    print(f"  {method_name:25s}: Mass: {error['mass']:8.2e}, "
                          f"Energy: {error['energy']:8.2e}, Total: {error['total']:8.2e}")
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("Starting Euler Equation Comparison Tests")
        print(f"Grid: {self.params.nx} × {self.params.ny}")
        
        if self.params.spatial_methods is None or self.params.test_cases is None:
            print("Error: Missing spatial methods or test cases configuration")
            return
        
        print(f"Methods: {len(self.params.spatial_methods)}")
        print(f"Test cases: {len(self.params.test_cases)}")
        
        for test_case in self.params.test_cases:
            self.run_comparison_test(test_case)
            self.plot_comparison(test_case)
            
            # Generate time series plots for each method
            for method in self.params.spatial_methods:
                method_name = method['name']
                if test_case in self.results and method_name in self.results[test_case]['solutions']:
                    self.plot_time_series(test_case, method_name)
        
        self.print_summary()


def main():
    """Main function to run Euler comparison tests"""
    # Create comparison parameters
    params = EulerComparisonParameters(
        nx=64,
        ny=64,
        final_time=0.15,
        cfl_number=0.3,
        test_cases=['sod_shock_tube', 'blast_wave'],  # Start with simpler cases
        outputtimes=[0.0, 0.0375, 0.075, 0.1125, 0.15],  # 5 equally spaced points from 0 to final_time
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = EulerComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()