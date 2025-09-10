"""
Euler Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D Euler equations,
testing various Riemann solvers and reconstruction methods for compressible flow problems.

Physics: ∂U/∂t + ∂F(U)/∂x + ∂G(U)/∂y = 0
Where U = [ρ, ρu, ρv, ρw, E]
"""

import numpy as np
import matplotlib.pyplot as plt
from fvm_framework.utils import FVMPlotter, create_physics_specific_plotter
import os
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    # Test cases to run
    test_cases: List[str] = None
    
    # Spatial methods to compare
    spatial_methods: List[Dict[str, Any]] = None
    
    # Output parameters
    output_dir: str = "comparison_results"
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self):
        if self.test_cases is None:
            self.test_cases = ['sod_shock_tube', 'double_mach_reflection', 'explosion_2d']
        
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
                    'name': 'HLLD + WENO5',
                    'reconstruction_type': 'weno5',
                    'flux_type': 'hlld',
                    'flux_params': {'riemann_solver': 'hlld'},
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
                'output_interval': self.params.final_time / 5,
                'monitor_interval': 100
            }
        }
    
    def run_single_test(self, test_case: str, method: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            solver.solve()
            end_time = time.perf_counter()
            
            # Get final solution
            solution = solver.get_solution()
            stats = solver.get_statistics()
            
            timing_info = {
                'total_time': end_time - start_time,
                'total_steps': solution['time_step'],
                'avg_time_per_step': (end_time - start_time) / max(solution['time_step'], 1),
                'pipeline_performance': stats['pipeline_performance']
            }
            
            return solution, timing_info
            
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
            final_state = solution['conservative']
            
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
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Euler Equation Comparison: {test_case}', fontsize=16)
        
        # Create coordinate arrays
        x = np.linspace(0, self.params.domain_size, self.params.nx)
        y = np.linspace(0, self.params.domain_size, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Plot initial condition (density)
        initial_state = self.results[test_case]['initial_condition']
        initial_primitives = self.compute_primitive_variables(initial_state, self.params.gamma)
        
        im0 = axes[0,0].contourf(X, Y, initial_primitives[0], levels=20, cmap='viridis')
        axes[0,0].set_title('Initial Density')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0,0])
        
        # Density cross-section comparison
        y_mid_idx = self.params.ny // 2
        axes[0,1].plot(x, initial_primitives[0, :, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        # Plot all methods - density
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_primitives = self.compute_primitive_variables(solution['conservative'], self.params.gamma)
                
                axes[0,1].plot(x, final_primitives[0, :, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[0,1].set_title(f'Density Cross-section (y = {self.params.domain_size/2:.1f})')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('ρ')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Pressure cross-section comparison
        axes[0,2].plot(x, initial_primitives[4, :, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_primitives = self.compute_primitive_variables(solution['conservative'], self.params.gamma)
                
                axes[0,2].plot(x, final_primitives[4, :, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[0,2].set_title(f'Pressure Cross-section (y = {self.params.domain_size/2:.1f})')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('p')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Conservation error comparison
        errors = self.compute_conservation_errors(test_case)
        if errors:
            method_names = list(errors.keys())
            total_errors = [errors[name]['total'] for name in method_names]
            
            bars = axes[1,0].bar(range(len(method_names)), total_errors)
            axes[1,0].set_title('Total Conservation Error')
            axes[1,0].set_xlabel('Method')
            axes[1,0].set_ylabel('Total Error')
            axes[1,0].set_xticks(range(len(method_names)))
            axes[1,0].set_xticklabels(method_names, rotation=45, ha='right')
            axes[1,0].set_yscale('log')
            axes[1,0].grid(True, alpha=0.3)
            
            # Color bars according to method colors
            for i, bar in enumerate(bars):
                if i < len(self.params.spatial_methods):
                    bar.set_color(self.params.spatial_methods[i]['color'])
        
        # Timing comparison
        timings = self.results[test_case]['timings']
        if timings:
            method_names = list(timings.keys())
            compute_times = [timings[name]['total_time'] for name in method_names 
                           if 'error' not in timings[name]]
            
            if compute_times:
                bars = axes[1,1].bar(range(len(method_names)), compute_times)
                axes[1,1].set_title('Computation Time Comparison')
                axes[1,1].set_xlabel('Method')
                axes[1,1].set_ylabel('Time (seconds)')
                axes[1,1].set_xticks(range(len(method_names)))
                axes[1,1].set_xticklabels(method_names, rotation=45, ha='right')
                axes[1,1].grid(True, alpha=0.3)
                
                # Color bars
                for i, bar in enumerate(bars):
                    if i < len(self.params.spatial_methods):
                        bar.set_color(self.params.spatial_methods[i]['color'])
        
        # Step count comparison
        if timings:
            method_names = list(timings.keys())
            step_counts = [timings[name]['total_steps'] for name in method_names 
                          if 'error' not in timings[name]]
            
            if step_counts:
                bars = axes[1,2].bar(range(len(method_names)), step_counts)
                axes[1,2].set_title('Time Steps Comparison')
                axes[1,2].set_xlabel('Method')
                axes[1,2].set_ylabel('Number of Steps')
                axes[1,2].set_xticks(range(len(method_names)))
                axes[1,2].set_xticklabels(method_names, rotation=45, ha='right')
                axes[1,2].grid(True, alpha=0.3)
                
                # Color bars
                for i, bar in enumerate(bars):
                    if i < len(self.params.spatial_methods):
                        bar.set_color(self.params.spatial_methods[i]['color'])
        
    def plot_time_series(self, test_case: str, method_name: str):
        """Generate time series plots for specified output times"""
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('euler')
        
        # Generate both single variable (density) and multi-variable time series
        # Single variable (density) for compatibility
        plotter.plot_time_series(
            test_case=test_case,
            method_name=method_name,
            results=self.results,
            variable_index=0,  # Density
            variable_name="Density"
        )
        
        # Multi-variable time series showing all 5 Euler variables
        if 'time_series_variables' in physics_config:
            plotter.plot_multi_variable_time_series(
                test_case=test_case,
                method_name=method_name,
                results=self.results,
                variables=physics_config['time_series_variables']
            )
        
        plt.tight_layout()
        
        if self.params.save_plots:
            filename = os.path.join(self.params.output_dir, f'euler_comparison_{test_case}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {filename}")
        
        if self.params.show_plots:
            plt.show()
        else:
            plt.close()
    
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
        print(f"Methods: {len(self.params.spatial_methods)}")
        print(f"Test cases: {len(self.params.test_cases)}")
        
        for test_case in self.params.test_cases:
            self.run_comparison_test(test_case)
            self.plot_comparison(test_case)
        
        self.print_summary()


def main():
    """Main function to run Euler comparison tests"""
    # Create comparison parameters
    params = EulerComparisonParameters(
        nx=64,
        ny=64,
        final_time=0.15,
        cfl_number=0.3,
        test_cases=['sod_shock_tube', 'explosion_2d'],  # Start with simpler cases
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = EulerComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()