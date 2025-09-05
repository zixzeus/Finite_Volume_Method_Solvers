"""
Complete FVM Solver Interface

This module provides a high-level interface that integrates all components
of the finite volume method framework into a unified solver.
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
import time

from .core.data_container import FVMDataContainer2D, GridGeometry
from .core.pipeline import FVMPipeline, PipelineMonitor
from .boundary.boundary_conditions import BoundaryManager, EulerBoundaryConditions
from .spatial.riemann_solvers import RiemannSolverFactory, RiemannFluxComputation
from .temporal.time_integrators import TimeIntegratorFactory, ResidualFunction, TemporalSolver


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
            'physics': {'gamma': 1.4, 'gas_constant': 287.0},
            'numerical': {
                'spatial_scheme': 'lax_friedrichs',  # Changed from riemann_solver
                'time_integrator': 'rk3',
                'cfl_number': 0.5,
                'boundary_type': 'periodic',
                'spatial_params': {}  # Additional params for spatial scheme
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
        """Initialize data container"""
        self.data = FVMDataContainer2D(self.geometry, num_vars=5)
        
    def _initialize_boundary_conditions(self):
        """Initialize boundary condition manager"""
        self.boundary_manager = BoundaryManager()
        
        # Set default boundary condition
        boundary_type = self.config['numerical']['boundary_type']
        if boundary_type == 'periodic':
            from .boundary.boundary_conditions import PeriodicBC
            self.boundary_manager.set_default_boundary(PeriodicBC())
        elif boundary_type == 'reflective':
            from .boundary.boundary_conditions import ReflectiveBC
            self.boundary_manager.set_default_boundary(ReflectiveBC())
        elif boundary_type == 'transmissive':
            from boundary.boundary_conditions import TransmissiveBC
            self.boundary_manager.set_default_boundary(TransmissiveBC())
    
    def _initialize_solver_components(self):
        """Initialize spatial and temporal solvers"""
        numerical_config = self.config['numerical']
        
        # Riemann solver
        self.riemann_solver = RiemannSolverFactory.create(numerical_config['riemann_solver'])
        self.flux_computer = RiemannFluxComputation(self.riemann_solver)
        
        # Time integrator
        self.time_integrator = TimeIntegratorFactory.create(numerical_config['time_integrator'])
        
        # Residual function
        self.residual_function = ResidualFunction(self.flux_computer)
        
        # Temporal solver
        self.temporal_solver = TemporalSolver(
            self.time_integrator,
            self.residual_function,
            cfl_number=numerical_config['cfl_number'],
            adaptive_dt=True
        )
        
        # Pipeline for monitoring
        self.pipeline = FVMPipeline(
            boundary_type=numerical_config['boundary_type'],
            spatial_scheme=numerical_config['spatial_scheme'],
            time_scheme=numerical_config['time_integrator'],
            **numerical_config.get('spatial_params', {})
        )
    
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
    
    def set_initial_conditions(self, init_function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Set initial conditions using a function.
        
        Args:
            init_function: Function that takes (x, y) coordinates and returns
                          state vector [rho, rho*u, rho*v, rho*w, E]
        """
        # Create coordinate arrays
        x = np.linspace(self.geometry.x_min + 0.5 * self.geometry.dx,
                       self.geometry.x_max - 0.5 * self.geometry.dx,
                       self.geometry.nx)
        y = np.linspace(self.geometry.y_min + 0.5 * self.geometry.dy,
                       self.geometry.y_max - 0.5 * self.geometry.dy,
                       self.geometry.ny)
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Apply initial conditions
        for i in range(self.geometry.nx):
            for j in range(self.geometry.ny):
                state_ij = init_function(X[i, j], Y[i, j])
                self.data.state[:, i, j] = state_ij
        
        # Invalidate primitive cache
        self.data._primitives_valid = False
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
                    from .boundary.boundary_conditions import ReflectiveBC
                    bc = ReflectiveBC()
                elif bc_config == 'transmissive':
                    from .boundary.boundary_conditions import TransmissiveBC
                    bc = TransmissiveBC()
                elif bc_config == 'periodic':
                    from .boundary.boundary_conditions import PeriodicBC
                    bc = PeriodicBC()
                else:
                    raise ValueError(f"Unknown boundary type: {bc_config}")
            elif isinstance(bc_config, dict):
                # Detailed boundary configuration
                if bc_config['type'] == 'inflow':
                    bc = EulerBoundaryConditions.subsonic_inflow(**bc_config['params'])
                elif bc_config['type'] == 'outflow':
                    bc = EulerBoundaryConditions.subsonic_outflow(bc_config['params']['pressure'])
                else:
                    raise ValueError(f"Unknown boundary configuration: {bc_config}")
            else:
                # Custom boundary condition object
                bc = bc_config
            
            self.boundary_manager.set_boundary(region, bc)
    
    def solve(self, final_time: Optional[float] = None, max_steps: Optional[int] = None):
        """
        Run the simulation.
        
        Args:
            final_time: Final simulation time (overrides config)
            max_steps: Maximum number of time steps
        """
        if not self.is_initialized:
            raise RuntimeError("Initial conditions must be set before solving")
        
        # Determine stopping criteria
        if final_time is None:
            final_time = self.config['simulation']['final_time']
        
        start_time = time.perf_counter()
        last_output_time = 0.0
        output_interval = self.config['simulation']['output_interval']
        
        print(f"Starting simulation to time {final_time}")
        print(f"Grid: {self.geometry.nx} × {self.geometry.ny}")
        print(f"Riemann solver: {self.riemann_solver.name}")
        print(f"Time integrator: {self.time_integrator.name}")
        print("-" * 50)
        
        # Main time loop
        while self.current_time < final_time:
            if max_steps and self.time_step >= max_steps:
                break
                
            step_start_time = time.perf_counter()
            
            # Solve one time step
            old_time = self.temporal_solver.current_time
            old_step = self.temporal_solver.time_step
            
            # Update temporal solver state to match our state
            self.temporal_solver.current_time = self.current_time
            self.temporal_solver.time_step = self.time_step
            
            # Take one time step
            remaining_time = final_time - self.current_time
            if remaining_time < self.temporal_solver.current_dt:
                # Last step - adjust time step
                old_dt = self.temporal_solver.current_dt
                self.temporal_solver.current_dt = remaining_time
                
                self.temporal_solver.solve_n_steps(
                    self.data, 1,
                    gamma=self.config['physics']['gamma'],
                    boundary_type=self.config['numerical']['boundary_type']
                )
                
                self.temporal_solver.current_dt = old_dt
            else:
                self.temporal_solver.solve_n_steps(
                    self.data, 1,
                    gamma=self.config['physics']['gamma'],
                    boundary_type=self.config['numerical']['boundary_type']
                )
            
            # Update our time and step
            self.current_time = self.temporal_solver.current_time
            self.time_step = self.temporal_solver.time_step
            
            step_end_time = time.perf_counter()
            step_time = step_end_time - step_start_time
            
            # Apply boundary conditions
            self.boundary_manager.apply_all(self.data)
            
            # Update monitoring
            self.monitor.update(self.data, self.temporal_solver.current_dt,
                              self.config['physics']['gamma'])
            
            # Store statistics
            self.statistics['time_steps'].append(self.current_time)
            self.statistics['computation_times'].append(step_time)
            
            if self.time_step % self.config['simulation']['monitor_interval'] == 0:
                conservation_error = self.data.get_conservation_error()
                max_wave_speed = self.data.get_max_wave_speed(self.config['physics']['gamma'])
                self.statistics['conservation_errors'].append(conservation_error)
                self.statistics['max_wave_speeds'].append(max_wave_speed)
            
            # Output progress
            if self.current_time - last_output_time >= output_interval or \
               self.current_time >= final_time:
                
                self._print_progress(step_time)
                last_output_time = self.current_time
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f"Simulation completed in {total_time:.2f} seconds")
        print(f"Total time steps: {self.time_step}")
        print(f"Average time per step: {total_time/max(self.time_step, 1):.6f} seconds")
        
        # Print final statistics
        self._print_final_statistics()
    
    def _print_progress(self, step_time: float):
        """Print simulation progress"""
        max_speed = self.data.get_max_wave_speed(self.config['physics']['gamma'])
        dt = self.temporal_solver.current_dt
        
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
    
    def get_solution(self) -> Dict[str, np.ndarray]:
        """
        Get current solution state.
        
        Returns:
            Dictionary containing solution arrays
        """
        primitives = self.data.get_primitives(self.config['physics']['gamma'])
        
        # Create coordinate arrays
        x = np.linspace(self.geometry.x_min + 0.5 * self.geometry.dx,
                       self.geometry.x_max - 0.5 * self.geometry.dx,
                       self.geometry.nx)
        y = np.linspace(self.geometry.y_min + 0.5 * self.geometry.dy,
                       self.geometry.y_max - 0.5 * self.geometry.dy,
                       self.geometry.ny)
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        return {
            'x': X,
            'y': Y,
            'density': primitives[0],
            'velocity_x': primitives[1],
            'velocity_y': primitives[2],
            'velocity_z': primitives[3],
            'pressure': primitives[4],
            'conservative': self.data.state.copy(),
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
            'solver_status': self.temporal_solver.get_status(),
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
            'riemann_solver': 'hllc',
            'time_integrator': 'rk3',
            'cfl_number': 0.4,
            'boundary_type': 'transmissive'
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
            'riemann_solver': 'hllc',
            'time_integrator': 'rk3', 
            'cfl_number': 0.9,
            'boundary_type': 'transmissive'
        },
        'simulation': {
            'final_time': 0.2,
            'output_interval': 0.05,
            'monitor_interval': 20
        }
    }
    
    return FVMSolver(config)