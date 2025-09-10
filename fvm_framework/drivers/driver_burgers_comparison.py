"""
Burgers Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D Burgers equations,
testing nonlinear wave propagation, shock formation, and viscous effects.

Physics: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
         ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v
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
from fvm_framework.testcases.testsuite_burgers_2D import get_burgers_test_case, list_burgers_test_cases


@dataclass
class BurgersComparisonParameters:
    """Parameters for Burgers equation comparison tests"""
    # Grid parameters
    nx: int = 64
    ny: int = 64
    domain_size: float = 1.0
    
    # Physics parameters
    viscosity: float = 0.01
    
    # Simulation parameters
    final_time: float = 0.5
    cfl_number: float = 0.3
    outputtimes: Optional[List[float]] = None
    
    # Test cases to run
    test_cases: Optional[List[str]] = None
    
    # Spatial methods to compare
    spatial_methods: List[Dict[str, Any]] = None
    
    # Output parameters
    output_dir: str = "comparison_results"
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self):
        if self.test_cases is None:
            self.test_cases = ['smooth_sine_wave', 'gaussian_vortex', 'taylor_green_vortex']
        
        if self.outputtimes is None:
            # Create 5 equally spaced time points from 0 to final_time
            self.outputtimes = [i * self.final_time / 4 for i in range(5)]
        
        if self.spatial_methods is None:
            self.spatial_methods = [
                {
                    'name': 'First Order (Constant)',
                    'reconstruction_type': 'constant',
                    'flux_type': 'lax_friedrichs',
                    'color': 'blue',
                    'linestyle': '-'
                },
                {
                    'name': 'Lax-Friedrichs (Van Leer Limiter)',
                    'reconstruction_type': 'slope_limiter',
                    'reconstruction_params': {'limiter': 'van_leer'},
                    'flux_type': 'lax_friedrichs',
                    'color': 'green',
                    'linestyle': '-.'
                },
                {
                    'name': 'Lax-Friedrichs (Superbee Limiter)',
                    'reconstruction_type': 'slope_limiter',
                    'reconstruction_params': {'limiter': 'superbee'},
                    'flux_type': 'lax_friedrichs',
                    'color': 'purple',
                    'linestyle': ':'
                }
            ]


class BurgersComparison:
    """Main comparison test class for Burgers equations"""
    
    def __init__(self, params: BurgersComparisonParameters):
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
                'equation': 'burgers',
                'params': {
                    'viscosity': self.params.viscosity
                }
            },
            'numerical': {
                'reconstruction_type': method['reconstruction_type'],
                'flux_type': method['flux_type'],
                'time_scheme': 'euler',
                'cfl_number': self.params.cfl_number,
                'boundary_type': 'periodic',
                'flux_params': method.get('flux_params', {}),
                'reconstruction_params': method.get('reconstruction_params', {})
            },
            'simulation': {
                'final_time': self.params.final_time,
                'output_interval': self.params.final_time,
                'monitor_interval': 1000,
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
            initial_state = get_burgers_test_case(test_case, self.params.nx, self.params.ny)
            
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
        print(f"\n=== Running Burgers Comparison: {test_case} ===")
        
        test_results = {}
        test_timings = {}
        
        # Get initial condition for reference
        initial_state = get_burgers_test_case(test_case, self.params.nx, self.params.ny)
        
        for method in self.params.spatial_methods:
            solution, timing = self.run_single_test(test_case, method)
            
            if solution is not None:
                test_results[method['name']] = solution
                test_timings[method['name']] = timing
        
        self.results[test_case] = {
            'initial_condition': initial_state,
            'solutions': test_results,
            'timings': test_timings
        }
    
    def compute_energy_dissipation(self, test_case: str) -> Dict[str, Dict[str, float]]:
        """Compute energy dissipation and enstrophy for vorticity-based cases"""
        if test_case not in self.results:
            return {}
        
        metrics = {}
        initial_condition = self.results[test_case]['initial_condition']
        
        # Compute initial kinetic energy
        u_i, v_i = initial_condition[0], initial_condition[1]
        initial_kinetic_energy = 0.5 * np.sum(u_i**2 + v_i**2)
        
        # Compute initial enstrophy (vorticity squared)
        dx = self.params.domain_size / self.params.nx
        dy = self.params.domain_size / self.params.ny
        
        du_dy_i = np.gradient(u_i, dy, axis=1)
        dv_dx_i = np.gradient(v_i, dx, axis=0)
        vorticity_i = dv_dx_i - du_dy_i
        initial_enstrophy = 0.5 * np.sum(vorticity_i**2)
        
        for method_name, solution in self.results[test_case]['solutions'].items():
            if 'conservative' in solution:
                final_state = solution['conservative']
            else:
                final_state = solution['final_solution']['conservative']
            u_f, v_f = final_state[0], final_state[1]
            
            # Compute final kinetic energy
            final_kinetic_energy = 0.5 * np.sum(u_f**2 + v_f**2)
            
            # Compute final enstrophy
            du_dy_f = np.gradient(u_f, dy, axis=1)
            dv_dx_f = np.gradient(v_f, dx, axis=0)
            vorticity_f = dv_dx_f - du_dy_f
            final_enstrophy = 0.5 * np.sum(vorticity_f**2)
            
            # Compute dissipation rates
            energy_dissipation = (initial_kinetic_energy - final_kinetic_energy) / initial_kinetic_energy
            enstrophy_dissipation = (initial_enstrophy - final_enstrophy) / (initial_enstrophy + 1e-15)
            
            metrics[method_name] = {
                'energy_dissipation': energy_dissipation,
                'enstrophy_dissipation': enstrophy_dissipation,
                'final_kinetic_energy': final_kinetic_energy,
                'final_enstrophy': final_enstrophy
            }
        
        return metrics

    
    def plot_time_series(self, test_case: str, method_name: str):
        """Generate time series plots for specified output times"""
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('burgers')
        
        
        # Multi-variable time series showing both u and v components

        plotter.plot_multi_variable_time_series(
            test_case=test_case,
            method_name=method_name,
            results=self.results,
            variables=physics_config['variables']
        )
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*60)
        print("BURGERS EQUATION COMPARISON SUMMARY")
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
            
            # Print energy metrics
            energy_metrics = self.compute_energy_dissipation(test_case)
            if energy_metrics:
                print("Energy Dissipation:")
                for method_name, metrics in energy_metrics.items():
                    print(f"  {method_name:25s}: Energy: {metrics['energy_dissipation']:6.3f}, "
                          f"Enstrophy: {metrics['enstrophy_dissipation']:6.3f}")
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("Starting Burgers Equation Comparison Tests")
        print(f"Grid: {self.params.nx} × {self.params.ny}")
        print(f"Viscosity: {self.params.viscosity}")
        print(f"Methods: {len(self.params.spatial_methods)}")
        print(f"Test cases: {len(self.params.test_cases)}")
        
        for test_case in self.params.test_cases:
            self.run_comparison_test(test_case)
            
            # Generate time series plots for each method
            for method in self.params.spatial_methods:
                method_name = method['name']
                if test_case in self.results and method_name in self.results[test_case]['solutions']:
                    self.plot_time_series(test_case, method_name)
        
        self.print_summary()


def main():
    """Main function to run Burgers comparison tests"""
    # Create comparison parameters
    params = BurgersComparisonParameters(
        nx=64,
        ny=64,
        final_time=0.3,
        cfl_number=0.3,
        viscosity=0.01,
        test_cases=['smooth_sine_wave', 'gaussian_vortex', 'taylor_green_vortex'],
        outputtimes=[0.0, 0.075, 0.15, 0.225, 0.3],  # 5 equally spaced points from 0 to final_time
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = BurgersComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()