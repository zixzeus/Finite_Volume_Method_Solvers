"""
Advection Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D advection equations,
following the structure of driver files that test multiple methods and generate comparison plots.

Physics: ∂u/∂t + a∂u/∂x + b∂u/∂y = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvm_framework.core.solver import FVMSolver
from fvm_framework.testcases.testsuite_advection_2D import get_advection_test_case, list_advection_test_cases


@dataclass
class ComparisonParameters:
    """Parameters for comparison tests"""
    # Grid parameters
    nx: int = 100
    ny: int = 100
    domain_size: float = 1.0
    
    # Physics parameters
    advection_x: float = 1.0
    advection_y: float = 1.0
    
    # Simulation parameters
    final_time: float = 0.5
    cfl_number: float = 0.4
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
            self.test_cases = ['gaussian_pulse', 'square_wave', 'sine_wave']
        
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


class AdvectionComparison:
    """Main comparison test class for advection equations"""
    
    def __init__(self, params: ComparisonParameters):
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
                'equation': 'advection',
                'params': {
                    'advection_x': self.params.advection_x,
                    'advection_y': self.params.advection_y
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
        
        # Create solver
        config = self.create_solver_config(method)
        solver = FVMSolver(config)
        
        # Get initial conditions
        initial_state = get_advection_test_case(test_case, self.params.nx, self.params.ny)
        
        # Set initial conditions
        solver.set_initial_conditions(initial_state)
        
        # Run simulation with timing
        start_time = time.perf_counter()
        try:
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
        print(f"\n=== Running Advection Comparison: {test_case} ===")
        
        test_results = {}
        test_timings = {}
        
        # Get analytical initial condition for reference
        initial_state = get_advection_test_case(test_case, self.params.nx, self.params.ny)
        
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
    
    def compute_errors(self, test_case: str) -> Dict[str, Dict[str, float]]:
        """Compute error norms comparing to exact advection (for simple cases)"""
        if test_case not in self.results:
            return {}
        
        errors = {}
        initial_condition = self.results[test_case]['initial_condition']
        
        for method_name, solution in self.results[test_case]['solutions'].items():
            final_state = solution['final_solution']['conservative']
            
            # For linear advection, compute L1, L2, Linf norms
            diff = final_state - initial_condition  # Simplified - exact solution would need advection
            
            l1_norm = np.mean(np.abs(diff))
            l2_norm = np.sqrt(np.mean(diff**2))
            linf_norm = np.max(np.abs(diff))
            
            errors[method_name] = {
                'L1': l1_norm,
                'L2': l2_norm,
                'Linf': linf_norm
            }
        
        return errors
    
    def plot_comparison(self, test_case: str):
        """Generate comparison plots for a test case"""
        if test_case not in self.results:
            print(f"No results found for test case: {test_case}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Advection Equation Comparison: {test_case}', fontsize=16)
        
        # Plot initial condition
        initial_state = self.results[test_case]['initial_condition'][0]  # [u] for advection
        x = np.linspace(0, self.params.domain_size, self.params.nx)
        y = np.linspace(0, self.params.domain_size, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Initial condition contour - make it square with equal aspect ratio
        im0 = axes[0].contourf(X, Y, initial_state, levels=20, cmap='viridis')
        axes[0].set_title('Initial Condition')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_xlim(0, self.params.domain_size)
        axes[0].set_ylim(0, self.params.domain_size)
        axes[0].set_aspect('equal', adjustable='box')
        plt.colorbar(im0, ax=axes[0])
        
        # Cross-section comparison at y = domain_size/2
        y_mid_idx = self.params.ny // 2
        axes[1].plot(x, initial_state[:, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        # Plot all methods
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_state = solution['final_solution']['conservative'][0]  # [u] for advection
                
                # Cross-section
                axes[1].plot(x, final_state[:, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[1].set_title(f'Cross-section at y = {self.params.domain_size/2:.1f}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('u')
        axes[1].set_xlim(0, self.params.domain_size)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.params.save_plots:
            filename = os.path.join(self.params.output_dir, f'advection_comparison_{test_case}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {filename}")
        
        if self.params.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_time_series(self, test_case: str, method_name: str):
        """Generate time series plots for specified output times"""
        if test_case not in self.results:
            print(f"No results found for test case: {test_case}")
            return
        
        if method_name not in self.results[test_case]['solutions']:
            print(f"No solution found for method: {method_name}")
            return
        
        solution_data = self.results[test_case]['solutions'][method_name]
        time_series = solution_data.get('time_series')
        
        if time_series is None:
            print(f"No time series data available for {method_name}")
            return
        
        # Setup grid for plotting
        x = np.linspace(0, self.params.domain_size, self.params.nx)
        y = np.linspace(0, self.params.domain_size, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Determine number of subplots based on outputtimes
        if self.params.outputtimes is None:
            print("No output times specified")
            return
            
        n_times = len(self.params.outputtimes)
        cols = min(5, n_times)  # Maximum 5 columns per row
        
        # Create figure with 2 rows: top for contour plots, bottom for cross-sections
        fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
        if cols == 1:
            # If only one column, axes is (2,) array
            contour_axes = [axes[0]]
            cross_axes = [axes[1]]
        else:
            # If multiple columns, axes is (2, cols) array
            contour_axes = axes[0]  # Top row for contour plots
            cross_axes = axes[1]    # Bottom row for cross-sections
        
        fig.suptitle(f'{test_case} - {method_name} - Time Evolution', fontsize=16)
        
        # Find time series data closest to each output time
        y_mid_idx = self.params.ny // 2  # Middle y-index for cross-section
        
        # First pass: collect all cross-section data to determine consistent y-axis range
        cross_section_data = []
        y_min, y_max = float('inf'), float('-inf')
        
        for i, target_time in enumerate(self.params.outputtimes):
            if i < cols:
                if 'times' in time_series and 'states' in time_series:
                    times = time_series['times']
                    states = time_series['states']
                    time_idx = np.argmin(np.abs(np.array(times) - target_time))
                    state = states[time_idx][0]
                    cross_data = state[:, y_mid_idx]
                else:
                    if target_time == 0.0:
                        state = self.results[test_case]['initial_condition'][0]
                    else:
                        state = solution_data['final_solution']['conservative'][0]
                    cross_data = state[:, y_mid_idx]
                
                cross_section_data.append(cross_data)
                y_min = min(y_min, cross_data.min())
                y_max = max(y_max, cross_data.max())
        
        # Add small margin to y-axis range
        y_range = y_max - y_min
        if y_range > 0:
            y_min -= 0.05 * y_range
            y_max += 0.05 * y_range
        else:
            y_min -= 0.1
            y_max += 0.1
        
        # Second pass: create plots with consistent y-axis scaling
        for i, target_time in enumerate(self.params.outputtimes):
            if i < cols:  # Only plot for available columns
                # Find closest time in time series
                if 'times' in time_series and 'states' in time_series:
                    times = time_series['times']
                    states = time_series['states']
                    
                    # Find closest time index
                    time_idx = np.argmin(np.abs(np.array(times) - target_time))
                    actual_time = times[time_idx]
                    state = states[time_idx][0]  # [u] for advection
                    
                    # Top row: Create contour plot
                    if cols == 1:
                        contour_ax = contour_axes[0]
                        cross_ax = cross_axes[0]
                    else:
                        contour_ax = contour_axes[i]
                        cross_ax = cross_axes[i]
                    
                    im = contour_ax.contourf(X, Y, state, levels=20, cmap='viridis')
                    contour_ax.set_title(f't = {actual_time:.3f}')
                    contour_ax.set_xlabel('x')
                    contour_ax.set_ylabel('y')
                    contour_ax.set_xlim(0, self.params.domain_size)
                    contour_ax.set_ylim(0, self.params.domain_size)
                    contour_ax.set_aspect('equal', adjustable='box')
                    plt.colorbar(im, ax=contour_ax)
                    
                    # Bottom row: Cross-section plot
                    cross_ax.plot(x, cross_section_data[i], 'b-', linewidth=2)
                    cross_ax.set_title(f'Cross-section at y = {self.params.domain_size/2:.1f}')
                    cross_ax.set_xlabel('x')
                    cross_ax.set_ylabel('u')
                    cross_ax.set_xlim(0, self.params.domain_size)
                    cross_ax.set_ylim(y_min, y_max)  # Use consistent y-axis scaling
                    cross_ax.grid(True, alpha=0.3)
                    
                else:
                    # Fallback: use initial condition for t=0, final for others
                    if cols == 1:
                        contour_ax = contour_axes[0]
                        cross_ax = cross_axes[0]
                    else:
                        contour_ax = contour_axes[i]
                        cross_ax = cross_axes[i]
                    
                    if target_time == 0.0:
                        state = self.results[test_case]['initial_condition'][0]
                        contour_ax.contourf(X, Y, state, levels=20, cmap='viridis')
                        contour_ax.set_title(f't = 0.000 (Initial)')
                        cross_ax.plot(x, state[:, y_mid_idx], 'b-', linewidth=2)
                    else:
                        state = solution_data['final_solution']['conservative'][0]
                        contour_ax.contourf(X, Y, state, levels=20, cmap='viridis')
                        contour_ax.set_title(f't = {self.params.final_time:.3f} (Final)')
                        cross_ax.plot(x, state[:, y_mid_idx], 'b-', linewidth=2)
                    
                    contour_ax.set_xlabel('x')
                    contour_ax.set_ylabel('y')
                    contour_ax.set_xlim(0, self.params.domain_size)
                    contour_ax.set_ylim(0, self.params.domain_size)
                    contour_ax.set_aspect('equal', adjustable='box')
                    
                    cross_ax.set_title(f'Cross-section at y = {self.params.domain_size/2:.1f}')
                    cross_ax.set_xlabel('x')
                    cross_ax.set_ylabel('u')
                    cross_ax.set_xlim(0, self.params.domain_size)
                    cross_ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_times, cols):
            if cols == 1:
                if n_times == 0:
                    contour_axes[0].set_visible(False)
                    cross_axes[0].set_visible(False)
            else:
                contour_axes[i].set_visible(False)
                cross_axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if self.params.save_plots:
            filename = os.path.join(self.params.output_dir, 
                                   f'{test_case}_{method_name.replace(" ", "_").replace("(", "").replace(")", "").lower()}_time_series.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved time series plot: {filename}")
        
        if self.params.show_plots:
            plt.show()
        else:
            plt.close()
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*60)
        print("ADVECTION EQUATION COMPARISON SUMMARY")
        print("="*60)
        
        if self.params.test_cases is None:
            return
            
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
                        steps = self.results[test_case]['solutions'][method_name]['final_solution']['time_step']
                        print(f"  {method_name:25s}: {timing['total_time']:6.3f}s "
                              f"({steps} steps)")
                    else:
                        print(f"  {method_name:25s}: FAILED ({timing['error']})")
            
            # Print errors
            errors = self.compute_errors(test_case)
            if errors:
                print("L2 Errors:")
                for method_name, error in errors.items():
                    print(f"  {method_name:25s}: {error['L2']:10.2e}")
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("Starting Advection Equation Comparison Tests")
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
    """Main function to run advection comparison tests"""
    # Create comparison parameters
    params = ComparisonParameters(
        nx=100,
        ny=100,
        final_time=0.3,
        cfl_number=0.4,
        test_cases=['gaussian_pulse', 'square_wave', 'sine_wave'],
        outputtimes=[0.0, 0.075, 0.15, 0.225, 0.3],  # 5 equally spaced points from 0 to final_time
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = AdvectionComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()