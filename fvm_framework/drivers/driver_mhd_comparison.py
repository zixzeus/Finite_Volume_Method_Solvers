"""
MHD Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D magnetohydrodynamics equations,
testing plasma physics phenomena including magnetic field interactions and shock propagation.

Physics: ∂U/∂t + ∂F(U)/∂x + ∂G(U)/∂y = 0
Where U = [ρ, ρu, ρv, ρw, E, Bx, By, Bz] (8 variables)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from fvm_framework.core.solver import FVMSolver
from fvm_framework.testcases.testsuite_mhd_2D import get_mhd_test_case, list_mhd_test_cases


@dataclass
class MHDComparisonParameters:
    """Parameters for MHD equation comparison tests"""
    # Grid parameters
    nx: int = 64
    ny: int = 64
    domain_size: float = 1.0
    
    # Physics parameters
    gamma: float = 5.0/3.0  # Ideal gas ratio for plasma
    
    # Simulation parameters
    final_time: float = 0.1
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
            self.test_cases = ['orszag_tang_vortex', 'brio_wu_shock', 'magnetic_reconnection', 'current_sheet']
        
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
                    'name': 'Slope Limiter + HLL',
                    'reconstruction_type': 'slope_limiter',
                    'flux_type': 'hll',
                    'flux_params': {'riemann_solver': 'hll'},
                    'color': 'red',
                    'linestyle': '--'
                },
                {
                    'name': 'WENO5 + HLLC',
                    'reconstruction_type': 'weno5',
                    'flux_type': 'hllc',
                    'flux_params': {'riemann_solver': 'hllc'},
                    'color': 'green',
                    'linestyle': '-.'
                },
                {
                    'name': 'Slope Limiter + HLLD',
                    'reconstruction_type': 'slope_limiter',
                    'flux_type': 'hlld',
                    'flux_params': {'riemann_solver': 'hlld'},
                    'color': 'purple',
                    'linestyle': ':'
                }
            ]


class MHDComparison:
    """Main comparison test class for MHD equations"""
    
    def __init__(self, params: MHDComparisonParameters):
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
                'equation': 'mhd',
                'params': {
                    'gamma': self.params.gamma
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
                'monitor_interval': 50
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
            initial_state = get_mhd_test_case(test_case, self.params.nx, self.params.ny)
            
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
        print(f"\n=== Running MHD Comparison: {test_case} ===")
        
        test_results = {}
        test_timings = {}
        
        # Get initial condition for reference
        initial_state = get_mhd_test_case(test_case, self.params.nx, self.params.ny)
        
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
    
    def compute_magnetic_divB_error(self, magnetic_field: np.ndarray) -> float:
        """Compute RMS error in magnetic field divergence ∇·B = 0"""
        Bx, By, Bz = magnetic_field[5], magnetic_field[6], magnetic_field[7]
        
        dx = self.params.domain_size / self.params.nx
        dy = self.params.domain_size / self.params.ny
        
        dBx_dx = np.gradient(Bx, dx, axis=0)
        dBy_dy = np.gradient(By, dy, axis=1)
        
        div_B = dBx_dx + dBy_dy  # Note: ∂Bz/∂z = 0 for 2D
        rms_error = np.sqrt(np.mean(div_B**2))
        
        return rms_error
    
    def compute_mhd_energy_components(self, state: np.ndarray) -> Dict[str, float]:
        """Compute kinetic, magnetic, and total energy components"""
        rho = state[0]
        rho_u = state[1]
        rho_v = state[2]
        rho_w = state[3]
        E = state[4]
        Bx = state[5]
        By = state[6]
        Bz = state[7]
        
        # Kinetic energy
        kinetic_energy = 0.5 * np.sum((rho_u**2 + rho_v**2 + rho_w**2) / np.maximum(rho, 1e-15))
        
        # Magnetic energy
        magnetic_energy = 0.5 * np.sum(Bx**2 + By**2 + Bz**2)
        
        # Total energy
        total_energy = np.sum(E)
        
        # Internal energy
        internal_energy = total_energy - kinetic_energy - magnetic_energy
        
        return {
            'kinetic': kinetic_energy,
            'magnetic': magnetic_energy,
            'internal': internal_energy,
            'total': total_energy
        }
    
    def compute_conservation_errors(self, test_case: str) -> Dict[str, Dict[str, float]]:
        """Compute conservation errors for mass, momentum, energy, and magnetic flux"""
        if test_case not in self.results:
            return {}
        
        errors = {}
        initial_condition = self.results[test_case]['initial_condition']
        
        # Compute initial conserved quantities
        initial_energies = self.compute_mhd_energy_components(initial_condition)
        initial_mass = np.sum(initial_condition[0])
        initial_mom_x = np.sum(initial_condition[1])
        initial_mom_y = np.sum(initial_condition[2])
        initial_divB_error = self.compute_magnetic_divB_error(initial_condition)
        
        for method_name, solution in self.results[test_case]['solutions'].items():
            final_state = solution['conservative']
            
            # Compute final quantities
            final_energies = self.compute_mhd_energy_components(final_state)
            final_mass = np.sum(final_state[0])
            final_mom_x = np.sum(final_state[1])
            final_mom_y = np.sum(final_state[2])
            final_divB_error = self.compute_magnetic_divB_error(final_state)
            
            # Compute relative errors
            mass_error = abs(final_mass - initial_mass) / abs(initial_mass)
            mom_x_error = abs(final_mom_x - initial_mom_x) / abs(initial_mom_x + 1e-15)
            mom_y_error = abs(final_mom_y - initial_mom_y) / abs(initial_mom_y + 1e-15)
            energy_error = abs(final_energies['total'] - initial_energies['total']) / abs(initial_energies['total'])
            
            errors[method_name] = {
                'mass': mass_error,
                'momentum_x': mom_x_error,
                'momentum_y': mom_y_error,
                'energy': energy_error,
                'divB_error': final_divB_error,
                'magnetic_energy_ratio': final_energies['magnetic'] / (final_energies['total'] + 1e-15),
                'kinetic_energy_ratio': final_energies['kinetic'] / (final_energies['total'] + 1e-15)
            }
        
        return errors
    
    def compute_current_density(self, magnetic_field: np.ndarray) -> np.ndarray:
        """Compute current density J = ∇ × B"""
        Bx, By, Bz = magnetic_field[5], magnetic_field[6], magnetic_field[7]
        
        dx = self.params.domain_size / self.params.nx
        dy = self.params.domain_size / self.params.ny
        
        # For 2D, J has only z-component: Jz = ∂By/∂x - ∂Bx/∂y
        dBy_dx = np.gradient(By, dx, axis=0)
        dBx_dy = np.gradient(Bx, dy, axis=1)
        Jz = dBy_dx - dBx_dy
        
        return Jz
    
    def plot_comparison(self, test_case: str):
        """Generate comparison plots for a test case"""
        if test_case not in self.results:
            print(f"No results found for test case: {test_case}")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'MHD Equation Comparison: {test_case}', fontsize=16)
        
        # Create coordinate arrays
        x = np.linspace(0, self.params.domain_size, self.params.nx)
        y = np.linspace(0, self.params.domain_size, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Plot initial conditions
        initial_state = self.results[test_case]['initial_condition']
        
        # Density
        im0 = axes[0,0].contourf(X, Y, initial_state[0], levels=20, cmap='viridis')
        axes[0,0].set_title('Initial Density')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0,0])
        
        # Magnetic field magnitude
        B_mag_initial = np.sqrt(initial_state[5]**2 + initial_state[6]**2 + initial_state[7]**2)
        im1 = axes[0,1].contourf(X, Y, B_mag_initial, levels=20, cmap='plasma')
        axes[0,1].set_title('Initial |B|')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,1])
        
        # Current density
        J_initial = self.compute_current_density(initial_state)
        im2 = axes[0,2].contourf(X, Y, J_initial, levels=20, cmap='RdBu_r')
        axes[0,2].set_title('Initial Current Density Jz')
        axes[0,2].set_xlabel('x')
        axes[0,2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0,2])
        
        # Cross-section comparisons
        y_mid_idx = self.params.ny // 2
        
        # Density cross-section
        axes[1,0].plot(x, initial_state[0, :, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_state = solution['conservative']
                
                axes[1,0].plot(x, final_state[0, :, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[1,0].set_title(f'Density Cross-section (y = {self.params.domain_size/2:.1f})')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('ρ')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Magnetic field cross-section
        axes[1,1].plot(x, initial_state[5, :, y_mid_idx], 'k-', linewidth=2, label='Initial Bx')
        
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_state = solution['conservative']
                
                axes[1,1].plot(x, final_state[5, :, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[1,1].set_title(f'Bx Cross-section (y = {self.params.domain_size/2:.1f})')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('Bx')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Current density cross-section
        axes[1,2].plot(x, J_initial[:, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in self.results[test_case]['solutions']:
                solution = self.results[test_case]['solutions'][method_name]
                final_state = solution['conservative']
                J_final = self.compute_current_density(final_state)
                
                axes[1,2].plot(x, J_final[:, y_mid_idx], 
                             color=method['color'], 
                             linestyle=method['linestyle'],
                             linewidth=1.5,
                             label=method_name)
        
        axes[1,2].set_title(f'Current Density Cross-section (y = {self.params.domain_size/2:.1f})')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('Jz')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        # Error comparisons
        errors = self.compute_conservation_errors(test_case)
        if errors:
            method_names = list(errors.keys())
            
            # Total conservation error
            total_errors = [errors[name]['mass'] + errors[name]['energy'] for name in method_names]
            bars = axes[2,0].bar(range(len(method_names)), total_errors)
            axes[2,0].set_title('Conservation Error (Mass + Energy)')
            axes[2,0].set_xlabel('Method')
            axes[2,0].set_ylabel('Total Error')
            axes[2,0].set_xticks(range(len(method_names)))
            axes[2,0].set_xticklabels(method_names, rotation=45, ha='right')
            axes[2,0].set_yscale('log')
            axes[2,0].grid(True, alpha=0.3)
            
            for i, bar in enumerate(bars):
                if i < len(self.params.spatial_methods):
                    bar.set_color(self.params.spatial_methods[i]['color'])
            
            # ∇·B error
            divB_errors = [errors[name]['divB_error'] for name in method_names]
            bars = axes[2,1].bar(range(len(method_names)), divB_errors)
            axes[2,1].set_title('∇·B RMS Error')
            axes[2,1].set_xlabel('Method')
            axes[2,1].set_ylabel('RMS(∇·B)')
            axes[2,1].set_xticks(range(len(method_names)))
            axes[2,1].set_xticklabels(method_names, rotation=45, ha='right')
            axes[2,1].set_yscale('log')
            axes[2,1].grid(True, alpha=0.3)
            
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
                bars = axes[2,2].bar(range(len(method_names)), compute_times)
                axes[2,2].set_title('Computation Time Comparison')
                axes[2,2].set_xlabel('Method')
                axes[2,2].set_ylabel('Time (seconds)')
                axes[2,2].set_xticks(range(len(method_names)))
                axes[2,2].set_xticklabels(method_names, rotation=45, ha='right')
                axes[2,2].grid(True, alpha=0.3)
                
                for i, bar in enumerate(bars):
                    if i < len(self.params.spatial_methods):
                        bar.set_color(self.params.spatial_methods[i]['color'])
        
        plt.tight_layout()
        
        if self.params.save_plots:
            filename = os.path.join(self.params.output_dir, f'mhd_comparison_{test_case}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {filename}")
        
        if self.params.show_plots:
            plt.show()
        else:
            plt.close()
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*60)
        print("MHD EQUATION COMPARISON SUMMARY")
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
                print("Conservation & Physical Errors:")
                for method_name, error in errors.items():
                    print(f"  {method_name:25s}: Mass: {error['mass']:8.2e}, "
                          f"Energy: {error['energy']:8.2e}, ∇·B: {error['divB_error']:8.2e}")
                    print(f"  {'':<25s}  Mag/Total: {error['magnetic_energy_ratio']:6.3f}, "
                          f"Kin/Total: {error['kinetic_energy_ratio']:6.3f}")
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("Starting MHD Equation Comparison Tests")
        print(f"Grid: {self.params.nx} × {self.params.ny}")
        print(f"Gamma: {self.params.gamma}")
        print(f"Methods: {len(self.params.spatial_methods)}")
        print(f"Test cases: {len(self.params.test_cases)}")
        
        for test_case in self.params.test_cases:
            self.run_comparison_test(test_case)
            self.plot_comparison(test_case)
        
        self.print_summary()


def main():
    """Main function to run MHD comparison tests"""
    # Create comparison parameters
    params = MHDComparisonParameters(
        nx=64,
        ny=64,
        final_time=0.08,
        cfl_number=0.3,
        gamma=5.0/3.0,
        test_cases=['orszag_tang_vortex', 'current_sheet'],  # Start with stable cases
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = MHDComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()