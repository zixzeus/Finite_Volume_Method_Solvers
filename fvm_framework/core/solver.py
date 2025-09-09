"""
Complete FVM Solver Interface

This module provides a high-level interface that integrates all components
of the finite volume method framework into a unified solver.
"""

import numpy as np
from typing import Optional, Dict, Any
import time

from .data_container import FVMDataContainer2D, GridGeometry
from .pipeline import FVMPipeline, PipelineMonitor


class FVMSolver:
    """
    Complete 2D Finite Volume Method Solver.
    
    This class provides a high-level interface for setting up and running
    FVM simulations with various numerical schemes and boundary conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FVM solver with configuration.
        
        Args:
            config: Configuration dictionary with solver parameters
        """
        # Default configuration
        self.config = {
            'grid': {'nx': 100, 'ny': 100, 'dx': 0.01, 'dy': 0.01},
            'physics': {
                'equation': 'euler',  # Physics equation type: 'euler', 'burgers', 'advection', 'mhd', 'sound'
                'params': {
                    # Common parameters
                    'gamma': 1.4,
                    'gas_constant': 287.0,
                    # Burgers equation parameters
                    'viscosity': 0.01,
                    # Advection equation parameters
                    'advection_x': 1.0,
                    'advection_y': 0.0,
                    # Sound wave equation parameters
                    'sound_speed': 1.0,
                    'density': 1.0
                }
            },
            'numerical': {
                'reconstruction_type': 'constant',  # Spatial reconstruction method
                'flux_type': 'lax_friedrichs',      # Flux calculator type
                'time_scheme': 'rk3',               # Time integration scheme
                'cfl_number': 0.5,
                'boundary_type': 'periodic',
                'flux_params': {}                   # Additional flux calculator parameters
            },
            'simulation': {
                'final_time': 1.0,
                'output_interval': 0.1,
                'monitor_interval': 100
            }
        }
        
        # Update with user configuration
        if config:
            self._update_config(self.config, config)
        
        # Initialize components
        self._initialize_geometry()
        self._initialize_data_container()
        self._initialize_boundary_conditions()
        self._initialize_solver_components()
        self._initialize_monitoring()
        
        # Simulation state
        self.current_time = 0.0
        self.time_step = 0
        self.is_initialized = False
        
    def _update_config(self, base_config: dict, update_config: dict):
        """Recursively update configuration"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _initialize_geometry(self):
        """Initialize grid geometry"""
        grid_config = self.config['grid']
        self.geometry = GridGeometry(
            nx=grid_config['nx'],
            ny=grid_config['ny'], 
            dx=grid_config['dx'],
            dy=grid_config['dy'],
            x_min=grid_config.get('x_min', 0.0),
            y_min=grid_config.get('y_min', 0.0)
        )
    
    def _initialize_data_container(self):
        """Initialize data container with ghost cells"""
        physics_type = self.config['physics']['equation']
        
        # Determine number of variables based on physics equation
        num_vars_map = {
            'euler': 5,     # [rho, rho*u, rho*v, rho*w, E]
            'burgers': 2,   # [u, v] 
            'advection': 1, # [u]
            'mhd': 8,       # [rho, rho*u, rho*v, rho*w, E, Bx, By, Bz]
            'sound': 3      # [p, u, v]
        }
        num_vars = num_vars_map.get(physics_type, 5)
        
        self.data = FVMDataContainer2D(self.geometry, num_vars=num_vars, num_ghost=2)
        
    def _initialize_boundary_conditions(self):
        """Initialize boundary condition manager"""
        from fvm_framework.boundary.boundary_conditions import BoundaryManager
        self.boundary_manager = BoundaryManager()
        
        # Set default boundary condition
        boundary_type = self.config['numerical']['boundary_type']
        if boundary_type == 'periodic':
            from fvm_framework.boundary.boundary_conditions import PeriodicBC
            self.boundary_manager.set_default_boundary(PeriodicBC())
        elif boundary_type == 'reflective':
            from fvm_framework.boundary.boundary_conditions import ReflectiveBC
            self.boundary_manager.set_default_boundary(ReflectiveBC())
        elif boundary_type == 'transmissive':
            from fvm_framework.boundary.boundary_conditions import TransmissiveBC
            self.boundary_manager.set_default_boundary(TransmissiveBC())
    
    def _initialize_solver_components(self):
        """Initialize spatial and temporal solvers"""
        numerical_config = self.config['numerical']
        
        # Initialize FVM Pipeline with modular architecture
        self.pipeline = FVMPipeline(
            boundary_type=numerical_config['boundary_type'],
            reconstruction_type=numerical_config.get('reconstruction_type', 'constant'),
            flux_type=numerical_config.get('flux_type', 'lax_friedrichs'),
            time_scheme=numerical_config.get('time_scheme', 'rk3'),
            source_type=numerical_config.get('source_type', None),
            **numerical_config.get('flux_params', {})
        )
        
        # Store CFL number for time step calculation
        self.cfl_number = numerical_config['cfl_number']
    
    def _initialize_monitoring(self):
        """Initialize monitoring and diagnostics"""
        monitor_interval = self.config['simulation']['monitor_interval']
        self.monitor = PipelineMonitor(monitor_interval)
        
        # Statistics storage
        self.statistics = {
            'conservation_errors': [],
            'max_wave_speeds': [],
            'time_steps': [],
            'computation_times': []
        }
    
    def set_initial_conditions(self, initial_state: np.ndarray):
        """
        Set initial conditions using state array.
        
        Args:
            initial_state: State array with shape (num_vars, nx, ny) containing
                          initial conditions for all grid points
        """
        # Validate input array dimensions
        expected_shape = (self.data.num_vars, self.geometry.nx, self.geometry.ny)
        if initial_state.shape != expected_shape:
            raise ValueError(f"Initial state array shape {initial_state.shape} does not match "
                           f"expected shape {expected_shape}")
        
        # Set initial conditions directly in the interior cells
        ng = self.data.ng  # Number of ghost cells
        self.data.state[:, ng:ng+self.geometry.nx, ng:ng+self.geometry.ny] = initial_state
        
        self.is_initialized = True
        
        print(f"Initial conditions set on {self.geometry.nx} × {self.geometry.ny} grid")
    
    def set_boundary_conditions(self, boundaries: Dict[str, Any]):
        """
        Set specific boundary conditions for different regions.
        
        Args:
            boundaries: Dictionary specifying boundary conditions
                       e.g., {'left': 'inflow', 'right': 'outflow', ...}
        """
        for region, bc_config in boundaries.items():
            if isinstance(bc_config, str):
                # Simple boundary type
                if bc_config == 'reflective':
                    from fvm_framework.boundary.boundary_conditions import ReflectiveBC
                    bc = ReflectiveBC()
                elif bc_config == 'transmissive':
                    from fvm_framework.boundary.boundary_conditions import TransmissiveBC
                    bc = TransmissiveBC()
                elif bc_config == 'periodic':
                    from fvm_framework.boundary.boundary_conditions import PeriodicBC
                    bc = PeriodicBC()
                else:
                    raise ValueError(f"Unknown boundary type: {bc_config}")
            elif isinstance(bc_config, dict):
                # Complex boundary configurations not supported in generic solver
                raise ValueError(f"Complex boundary configurations not supported: {bc_config}. Use boundary condition objects directly.")
            else:
                # Custom boundary condition object
                bc = bc_config
            
            self.boundary_manager.set_boundary(region, bc)
    
    def _create_physics_equation(self):
        """Create physics equation based on configuration"""
        equation_type = self.config['physics']['equation']
        params = self.config['physics']['params']
        
        if equation_type == 'euler':
            from fvm_framework.physics.euler_equations import EulerEquations2D
            return EulerEquations2D(params.get('gamma', 1.4))
        elif equation_type == 'burgers':
            from fvm_framework.physics.burgers_equation import BurgersEquation2D
            return BurgersEquation2D(params.get('viscosity', 0.01))
        elif equation_type == 'advection':
            from fvm_framework.physics.advection_equation import AdvectionEquation2D
            return AdvectionEquation2D(
                params.get('advection_x', 1.0), 
                params.get('advection_y', 0.0)
            )
        elif equation_type == 'mhd':
            from fvm_framework.physics.mhd_equations import MHDEquations2D
            return MHDEquations2D(params.get('gamma', 5.0/3.0))
        elif equation_type == 'sound':
            from fvm_framework.physics.sound_wave_equations import SoundWaveEquations2D
            return SoundWaveEquations2D(
                params.get('sound_speed', 1.0),
                params.get('density', 1.0)
            )
        else:
            raise ValueError(f"Unknown physics equation: {equation_type}")
    
    def solve(self, final_time: Optional[float] = None, max_steps: Optional[int] = None):
        """
        Run the simulation using the modular pipeline framework.
        
        Args:
            final_time: Final simulation time (overrides config)
            max_steps: Maximum number of time steps
        """
        if not self.is_initialized:
            raise RuntimeError("Initial conditions must be set before solving")
        
        # Create physics equation based on configuration
        physics_equation = self._create_physics_equation()
        
        # Determine stopping criteria
        if final_time is None:
            final_time = self.config['simulation']['final_time']
        
        start_time = time.perf_counter()
        last_output_time = 0.0
        output_interval = self.config['simulation']['output_interval']
        
        print(f"Starting simulation to time {final_time}")
        print(f"Grid: {self.geometry.nx} × {self.geometry.ny}")
        print(f"Pipeline stages: {len(self.pipeline.stages)}")
        print("-" * 50)
        
        # Main time loop using pipeline
        while self.current_time < final_time:
            if max_steps and self.time_step >= max_steps:
                break
                
            step_start_time = time.perf_counter()
            
            # Compute time step
            dt = self._compute_time_step(physics_equation)
            dt = min(dt, final_time - self.current_time)
            
            # Execute pipeline time step
            pipeline_kwargs = {
                'physics_equation': physics_equation,
                'boundary_manager': self.boundary_manager,
                **self.config['physics']['params']
            }
            
            self.pipeline.execute_time_step(self.data, dt, **pipeline_kwargs)
            
            # Update time and step counter
            self.current_time += dt
            self.time_step += 1
            
            step_end_time = time.perf_counter()
            step_time = step_end_time - step_start_time
            
            # Update monitoring
            self.monitor.update(self.data, dt, physics_equation)
            
            # Store statistics
            self.statistics['time_steps'].append(self.current_time)
            self.statistics['computation_times'].append(step_time)
            
            if self.time_step % self.config['simulation']['monitor_interval'] == 0:
                conservation_error = self.data.get_conservation_error()
                self.statistics['conservation_errors'].append(conservation_error)
                
                if physics_equation is not None:
                    max_wave_speed = physics_equation.compute_max_wave_speed(self.data)
                    self.statistics['max_wave_speeds'].append(max_wave_speed)
            
            # Output progress
            if self.current_time - last_output_time >= output_interval or \
               self.current_time >= final_time:
                
                self._print_progress(step_time, dt, physics_equation)
                last_output_time = self.current_time
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f"Simulation completed in {total_time:.2f} seconds")
        print(f"Total time steps: {self.time_step}")
        print(f"Average time per step: {total_time/max(self.time_step, 1):.6f} seconds")
        
        # Print pipeline performance report
        self.pipeline.print_performance_report()
        
        # Print final statistics
        self._print_final_statistics()
    
    def _compute_time_step(self, physics_equation) -> float:
        """Compute stable time step using CFL condition"""
        max_wave_speed = physics_equation.compute_max_wave_speed(self.data)
        
        if max_wave_speed > 1e-15:
            min_dx = min(self.geometry.dx, self.geometry.dy)
            dt = self.cfl_number * min_dx / max_wave_speed
        else:
            dt = 1e-6  # Fallback for very small wave speeds
            
        return dt
    
    def _print_progress(self, step_time: float, dt: float, physics_equation):
        """Print simulation progress"""
        max_speed = physics_equation.compute_max_wave_speed(self.data)
        
        print(f"Step: {self.time_step:8d}, Time: {self.current_time:.6f}, "
              f"dt: {dt:.2e}, Max Speed: {max_speed:.2e}, "
              f"Step Time: {step_time:.6f}s")
    
    def _print_final_statistics(self):
        """Print final simulation statistics"""
        if len(self.statistics['conservation_errors']) > 0:
            final_errors = self.statistics['conservation_errors'][-1]
            max_conservation_drift = np.max(self.monitor.get_conservation_drift())
            
            print(f"Final conservation errors: {np.max(final_errors):.2e}")
            print(f"Maximum conservation drift: {max_conservation_drift:.2e}")
        
        if len(self.statistics['computation_times']) > 0:
            avg_step_time = np.mean(self.statistics['computation_times'])
            print(f"Average computation time per step: {avg_step_time:.6f} seconds")
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Get current solution state.
        
        Returns:
            Dictionary containing solution arrays
        """
        # Create coordinate arrays
        x = np.linspace(self.geometry.x_min + 0.5 * self.geometry.dx,
                       self.geometry.x_max - 0.5 * self.geometry.dx,
                       self.geometry.nx)
        y = np.linspace(self.geometry.y_min + 0.5 * self.geometry.dy,
                       self.geometry.y_max - 0.5 * self.geometry.dy,
                       self.geometry.ny)
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Extract interior solution (excluding ghost cells)
        ng = self.data.ng
        interior_state = self.data.state[:, ng:ng+self.geometry.nx, ng:ng+self.geometry.ny].copy()
        
        return {
            'x': X,
            'y': Y,
            'conservative': interior_state,
            'current_time': self.current_time,
            'time_step': self.time_step
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return {
            'statistics': self.statistics,
            'monitor': {
                'conservation_history': self.monitor.conservation_history,
                'wave_speed_history': self.monitor.max_wave_speed_history,
                'conservation_drift': self.monitor.get_conservation_drift()
            },
            'pipeline_performance': self.pipeline.get_performance_summary(),
            'config': self.config
        }


# Convenience functions for common setups

def create_blast_wave_solver(nx: int = 200, ny: int = 200, 
                           domain_size: float = 2.0) -> FVMSolver:
    """Create solver configured for blast wave problem"""
    config = {
        'grid': {
            'nx': nx, 'ny': ny,
            'dx': domain_size / nx, 'dy': domain_size / ny,
            'x_min': -domain_size/2, 'y_min': -domain_size/2
        },
        'numerical': {
            'reconstruction_type': 'slope_limiter',
            'flux_type': 'hllc',
            'time_scheme': 'rk3',
            'cfl_number': 0.4,
            'boundary_type': 'transmissive',
            'flux_params': {'riemann_solver': 'hllc'}
        },
        'simulation': {
            'final_time': 0.2,
            'output_interval': 0.02,
            'monitor_interval': 50
        }
    }
    
    return FVMSolver(config)


def create_shock_tube_solver(nx: int = 400, ny: int = 4,
                           domain_length: float = 1.0) -> FVMSolver:
    """Create solver configured for 1D shock tube (2D with thin domain)"""
    config = {
        'grid': {
            'nx': nx, 'ny': ny,
            'dx': domain_length / nx, 'dy': domain_length / (ny * 10),
            'x_min': 0.0, 'y_min': 0.0
        },
        'numerical': {
            'reconstruction_type': 'slope_limiter',
            'flux_type': 'hllc',
            'time_scheme': 'rk3',
            'cfl_number': 0.9,
            'boundary_type': 'transmissive',
            'flux_params': {'riemann_solver': 'hllc'}
        },
        'simulation': {
            'final_time': 0.2,
            'output_interval': 0.05,
            'monitor_interval': 20
        }
    }
    
    return FVMSolver(config)