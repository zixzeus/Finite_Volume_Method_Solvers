"""
MHD Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D magnetohydrodynamics equations,
testing plasma physics phenomena including magnetic field interactions and shock propagation.

Physics: ∂U/∂t + ∂F(U)/∂x + ∂G(U)/∂y = 0
Where U = [ρ, ρu, ρv, ρw, E, Bx, By, Bz] (8 variables)
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
            self.test_cases = ['orszag_tang_vortex', 'brio_wu_shock', 'magnetic_reconnection', 'current_sheet']
        
        if self.outputtimes is None:
            # Create 5 equally spaced time points from 0 to final_time
            self.outputtimes = [i * self.final_time / 4 for i in range(5)]
        
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
                'output_interval': self.params.final_time,
                'monitor_interval': 50,
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
            initial_state = get_mhd_test_case(test_case, self.params.nx, self.params.ny)
            
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
            if 'conservative' in solution:
                final_state = solution['conservative']
            else:
                final_state = solution['final_solution']['conservative']
            
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
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('mhd')
        
        # Use common multi-variable comparison plotting
        plotter.plot_multi_variable_comparison(
            test_case=test_case,
            results=self.results,
            variables=physics_config['variables'],
            title_suffix=physics_config['title_suffix']
        )
    
    def plot_time_series(self, test_case: str, method_name: str):
        """Generate time series plots for specified output times"""
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('mhd')
        
        # Multi-variable time series showing all 8 MHD variables
        plotter.plot_multi_variable_time_series(
            test_case=test_case,
            method_name=method_name,
            results=self.results,
            variables=physics_config['variables']
        )
    
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
    """Main function to run MHD comparison tests"""
    # Create comparison parameters
    params = MHDComparisonParameters(
        nx=64,
        ny=64,
        final_time=0.08,
        cfl_number=0.3,
        gamma=5.0/3.0,
        test_cases=['orszag_tang_vortex', 'current_sheet'],  # Start with stable cases
        outputtimes=[0.0, 0.02, 0.04, 0.06, 0.08],  # 5 equally spaced points from 0 to final_time
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = MHDComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()