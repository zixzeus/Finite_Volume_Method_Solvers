"""
Sound Wave Equation Comparison Test Driver

This test driver compares different spatial discretization methods for 2D linear sound wave equations,
testing acoustic wave propagation and dispersion properties.

Physics: ∂p/∂t + c²(∂u/∂x + ∂v/∂y) = 0
         ∂u/∂t + ∂p/∂x = 0  
         ∂v/∂t + ∂p/∂y = 0
Where [p, u, v] are pressure and velocity components
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
from fvm_framework.testcases.testsuite_sound_wave_2D import get_sound_wave_test_case, list_sound_wave_test_cases


@dataclass
class SoundWaveComparisonParameters:
    """Parameters for sound wave equation comparison tests"""
    # Grid parameters
    nx: int = 64
    ny: int = 64
    domain_size: float = 1.0
    
    # Physics parameters
    sound_speed: float = 1.0
    density: float = 1.0
    
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
            self.test_cases = ['gaussian_pulse', 'plane_wave', 'standing_wave', 'circular_wave', 'acoustic_dipole']
        
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


class SoundWaveComparison:
    """Main comparison test class for sound wave equations"""
    
    def __init__(self, params: SoundWaveComparisonParameters):
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
                'equation': 'sound',
                'params': {
                    'sound_speed': self.params.sound_speed,
                    'density': self.params.density
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
            initial_state = get_sound_wave_test_case(test_case, self.params.nx, self.params.ny)
            
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
        print(f"\n=== Running Sound Wave Comparison: {test_case} ===")
        
        test_results = {}
        test_timings = {}
        
        # Get initial condition for reference
        initial_state = get_sound_wave_test_case(test_case, self.params.nx, self.params.ny)
        
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
    
    def compute_acoustic_energy(self, state: np.ndarray) -> Dict[str, float]:
        """Compute acoustic energy components"""
        p = state[0]  # Pressure
        u = state[1]  # x-velocity
        v = state[2]  # y-velocity
        
        # Acoustic energy density = (1/2ρc²)p² + (ρ/2)(u² + v²)
        pressure_energy = 0.5 * np.sum(p**2) / (self.params.density * self.params.sound_speed**2)
        kinetic_energy = 0.5 * self.params.density * np.sum(u**2 + v**2)
        total_acoustic_energy = pressure_energy + kinetic_energy
        
        return {
            'pressure': pressure_energy,
            'kinetic': kinetic_energy,
            'total': total_acoustic_energy
        }
    
    def compute_wave_amplitude(self, test_case: str) -> Dict[str, Dict[str, float]]:
        """Compute wave amplitude preservation metrics"""
        if test_case not in self.results:
            return {}
        
        metrics = {}
        initial_condition = self.results[test_case]['initial_condition']
        
        # Initial metrics
        initial_energy = self.compute_acoustic_energy(initial_condition)
        initial_p_amplitude = np.max(np.abs(initial_condition[0]))
        initial_u_amplitude = np.max(np.abs(initial_condition[1]))
        initial_v_amplitude = np.max(np.abs(initial_condition[2]))
        
        for method_name, solution in self.results[test_case]['solutions'].items():
            if 'conservative' in solution:
                final_state = solution['conservative']
            else:
                final_state = solution['final_solution']['conservative']
            
            # Final metrics
            final_energy = self.compute_acoustic_energy(final_state)
            final_p_amplitude = np.max(np.abs(final_state[0]))
            final_u_amplitude = np.max(np.abs(final_state[1]))
            final_v_amplitude = np.max(np.abs(final_state[2]))
            
            # Relative changes
            energy_loss = 1.0 - (final_energy['total'] / (initial_energy['total'] + 1e-15))
            p_amplitude_loss = 1.0 - (final_p_amplitude / (initial_p_amplitude + 1e-15))
            u_amplitude_loss = 1.0 - (final_u_amplitude / (initial_u_amplitude + 1e-15))
            v_amplitude_loss = 1.0 - (final_v_amplitude / (initial_v_amplitude + 1e-15))
            
            metrics[method_name] = {
                'energy_loss': energy_loss,
                'pressure_amplitude_loss': p_amplitude_loss,
                'u_amplitude_loss': u_amplitude_loss,
                'v_amplitude_loss': v_amplitude_loss,
                'total_amplitude_loss': p_amplitude_loss + u_amplitude_loss + v_amplitude_loss
            }
        
        return metrics
    
    def compute_phase_velocity_error(self, test_case: str) -> Dict[str, float]:
        """Compute phase velocity errors for wave propagation tests"""
        if test_case not in self.results or test_case not in ['plane_wave', 'circular_wave']:
            return {}
        
        errors = {}
        # This is a simplified phase velocity check
        # For more accurate analysis, would need to track wave fronts
        
        for method_name, solution_data in self.results[test_case]['solutions'].items():
            # Placeholder for phase velocity error calculation
            # In practice, this would involve FFT analysis or wave front tracking
            errors[method_name] = 0.0  # Simplified for now
        
        return errors
    
    def plot_comparison(self, test_case: str):
        """Generate comparison plots for a test case"""
        plotter = FVMPlotter(self.params)
        physics_config = create_physics_specific_plotter('sound_wave')
        
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
        physics_config = create_physics_specific_plotter('sound_wave')
        
        # Multi-variable time series showing all 3 sound wave variables
        plotter.plot_multi_variable_time_series(
            test_case=test_case,
            method_name=method_name,
            results=self.results,
            variables=physics_config['variables']
        )
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*60)
        print("SOUND WAVE EQUATION COMPARISON SUMMARY")
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
            
            # Print amplitude preservation
            amplitude_metrics = self.compute_wave_amplitude(test_case)
            if amplitude_metrics:
                print("Amplitude Preservation:")
                for method_name, metrics in amplitude_metrics.items():
                    print(f"  {method_name:25s}: Energy Loss: {metrics['energy_loss']:6.3f}, "
                          f"P Loss: {metrics['pressure_amplitude_loss']:6.3f}, "
                          f"Total: {metrics['total_amplitude_loss']:6.3f}")
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("Starting Sound Wave Equation Comparison Tests")
        print(f"Grid: {self.params.nx} × {self.params.ny}")
        print(f"Sound speed: {self.params.sound_speed}")
        
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
    """Main function to run sound wave comparison tests"""
    # Create comparison parameters
    params = SoundWaveComparisonParameters(
        nx=64,
        ny=64,
        final_time=0.3,
        cfl_number=0.4,
        sound_speed=1.0,
        test_cases=['gaussian_pulse', 'plane_wave', 'standing_wave'],
        outputtimes=[0.0, 0.075, 0.15, 0.225, 0.3],  # 5 equally spaced points from 0 to final_time
        save_plots=True,
        show_plots=False
    )
    
    # Run comparison tests
    comparison = SoundWaveComparison(params)
    comparison.run_all_tests()


if __name__ == "__main__":
    main()