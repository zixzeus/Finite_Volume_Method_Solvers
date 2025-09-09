"""
Burgers Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D Burgers equations,
testing nonlinear wave propagation, shock formation, and viscous effects.

Physics: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
         ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            self.test_cases = ['smooth_sine_wave', 'gaussian_vortex', 'shock_formation', 'taylor_green_vortex']
        
        if self.spatial_methods is None:
            self.spatial_methods = [
                {
                    'name': 'First Order + Lax-Friedrichs',
                    'reconstruction_type': 'constant',
                    'flux_type': 'lax_friedrichs',
                    'color': 'blue',
                    'linestyle': '-'
                },
                {
                    'name': 'Slope Limiter + Lax-Friedrichs',
                    'reconstruction_type': 'slope_limiter',
                    'flux_type': 'lax_friedrichs',
                    'color': 'red',
                    'linestyle': '--'
                },
                {
                    'name': 'Second Order + HLL',
                    'reconstruction_type': 'slope_limiter',
                    'flux_type': 'hll',
                    'flux_params': {'riemann_solver': 'hll'},
                    'color': 'green',
                    'linestyle': '-.'
                },
                {
                    'name': 'WENO5 + HLLC',
                    'reconstruction_type': 'weno5',
                    'flux_type': 'hllc',
                    'flux_params': {'riemann_solver': 'hllc'},
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
                'time_scheme': 'rk3',
                'cfl_number': self.params.cfl_number,
                'boundary_type': 'periodic',
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
            initial_state = get_burgers_test_case(test_case, self.params.nx, self.params.ny)
            
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
            final_state = solution['conservative']
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
    
    def compute_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute vorticity field ω = ∇ × v"""
        dx = self.params.domain_size / self.params.nx
        dy = self.params.domain_size / self.params.ny
        
        du_dy = np.gradient(u, dy, axis=1)
        dv_dx = np.gradient(v, dx, axis=0)
        vorticity = dv_dx - du_dy
        
        return vorticity
    
    def plot_comparison(self, test_case: str):
        """Generate comparison plots for a test case"""
        if test_case not in self.results:
            print(f"No results found for test case: {test_case}")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Burgers Equation Comparison: {test_case}', fontsize=16)
        
        # Create coordinate arrays
        x = np.linspace(0, self.params.domain_size, self.params.nx)
        y = np.linspace(0, self.params.domain_size, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Plot initial condition (u-velocity)
        initial_state = self.results[test_case]['initial_condition']
        u_initial, v_initial = initial_state[0], initial_state[1]
        
        im0 = axes[0,0].contourf(X, Y, u_initial, levels=20, cmap='RdBu_r')
        axes[0,0].set_title('Initial u-velocity')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0,0])
        
        # Initial vorticity
        vorticity_initial = self.compute_vorticity(u_initial, v_initial)
        im1 = axes[0,1].contourf(X, Y, vorticity_initial, levels=20, cmap='RdBu_r')
        axes[0,1].set_title('Initial Vorticity')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,1])
        
        # u-velocity cross-section comparison
        y_mid_idx = self.params.ny // 2
        axes[0,2].plot(x, u_initial[:, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        # Plot all methods - u velocity
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_state = solution['conservative']
                u_final = final_state[0]
                
                axes[0,2].plot(x, u_final[:, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[0,2].set_title(f'u-velocity Cross-section (y = {self.params.domain_size/2:.1f})')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('u')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Energy dissipation comparison
        energy_metrics = self.compute_energy_dissipation(test_case)
        if energy_metrics:
            method_names = list(energy_metrics.keys())
            energy_dissipation = [energy_metrics[name]['energy_dissipation'] for name in method_names]
            
            bars = axes[1,0].bar(range(len(method_names)), energy_dissipation)
            axes[1,0].set_title('Energy Dissipation')
            axes[1,0].set_xlabel('Method')
            axes[1,0].set_ylabel('Relative Energy Loss')
            axes[1,0].set_xticks(range(len(method_names)))
            axes[1,0].set_xticklabels(method_names, rotation=45, ha='right')
            axes[1,0].grid(True, alpha=0.3)
            
            # Color bars according to method colors
            for i, bar in enumerate(bars):
                if i < len(self.params.spatial_methods):
                    bar.set_color(self.params.spatial_methods[i]['color'])
        
        # Enstrophy dissipation comparison
        if energy_metrics:
            enstrophy_dissipation = [energy_metrics[name]['enstrophy_dissipation'] for name in method_names]
            
            bars = axes[1,1].bar(range(len(method_names)), enstrophy_dissipation)
            axes[1,1].set_title('Enstrophy Dissipation')
            axes[1,1].set_xlabel('Method')
            axes[1,1].set_ylabel('Relative Enstrophy Loss')
            axes[1,1].set_xticks(range(len(method_names)))
            axes[1,1].set_xticklabels(method_names, rotation=45, ha='right')
            axes[1,1].grid(True, alpha=0.3)
            
            # Color bars
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
                bars = axes[1,2].bar(range(len(method_names)), compute_times)
                axes[1,2].set_title('Computation Time Comparison')
                axes[1,2].set_xlabel('Method')
                axes[1,2].set_ylabel('Time (seconds)')
                axes[1,2].set_xticks(range(len(method_names)))
                axes[1,2].set_xticklabels(method_names, rotation=45, ha='right')
                axes[1,2].grid(True, alpha=0.3)
                
                # Color bars
                for i, bar in enumerate(bars):
                    if i < len(self.params.spatial_methods):
                        bar.set_color(self.params.spatial_methods[i]['color'])
        
        plt.tight_layout()
        
        if self.params.save_plots:
            filename = os.path.join(self.params.output_dir, f'burgers_comparison_{test_case}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {filename}")
        
        if self.params.show_plots:
            plt.show()
        else:
            plt.close()
    
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
            self.plot_comparison(test_case)
        
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
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = BurgersComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()