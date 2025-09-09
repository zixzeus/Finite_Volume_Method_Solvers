"""
Pipeline Framework for Finite Volume Method Solvers

This module implements the computation pipeline architecture that orchestrates
the various stages of FVM computation in a data-driven manner.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import time
import numpy as np
from .data_container import FVMDataContainer2D

if TYPE_CHECKING:
    from fvm_framework.spatial.reconstruction.base_reconstruction import ReconstructionScheme
    from fvm_framework.spatial.flux_calculation.base_flux import FluxCalculator
    from fvm_framework.temporal.time_integrators import TimeIntegrator


class ComputationStage(ABC):
    """Abstract base class for all computation stages in the FVM pipeline"""
    
    def __init__(self, name: str):
        self.name = name
        self._execution_time = 0.0
        self._call_count = 0
        
    @abstractmethod
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Process the data through this computation stage.
        
        Args:
            data: FVM data container
            **kwargs: Additional parameters specific to each stage
        """
        pass
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this stage"""
        avg_time = self._execution_time / max(self._call_count, 1)
        return {
            'total_time': self._execution_time,
            'call_count': self._call_count,
            'avg_time': avg_time
        }
    
    def _time_execution(self, func, *args, **kwargs):
        """Internal method to time execution of stage processing"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        self._execution_time += (end_time - start_time)
        self._call_count += 1
        
        return result


class BoundaryStage(ComputationStage):
    """Stage for applying boundary conditions"""
    
    def __init__(self, boundary_type: str = 'periodic'):
        super().__init__("BoundaryConditions")
        self.boundary_type = boundary_type
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Apply boundary conditions to the data"""
        def _apply_boundaries():
            bc_type = kwargs.get('boundary_type', self.boundary_type)
            data.apply_boundary_conditions(bc_type)
            
        self._time_execution(_apply_boundaries)


class ReconstructionStage(ComputationStage):
    """
    Stage for spatial reconstruction of interface values.
    
    This stage performs actual reconstruction to compute left and right interface states
    from cell-centered values, storing results in the data container's interface_states arrays.
    """
    
    def __init__(self, reconstruction_type: str = 'constant'):
        super().__init__("SpatialReconstruction")
        self.reconstruction_type = reconstruction_type
        self._reconstruction_scheme: Optional['ReconstructionScheme'] = None
        
    def _initialize_reconstruction(self):
        """Initialize reconstruction scheme on first use"""
        if self._reconstruction_scheme is None:
            from fvm_framework.spatial.reconstruction.factory import create_reconstruction
            self._reconstruction_scheme = create_reconstruction(self.reconstruction_type)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Perform spatial reconstruction to compute interface states"""
        def _reconstruct():
            self._initialize_reconstruction()
            
            # Perform reconstruction using the spatial reconstruction module
            # Note: The reconstruction methods expect to work with ghost cells in the state array
            x_interfaces, y_interfaces = self._reconstruction_scheme.reconstruct_all_interfaces(data)
            x_left, x_right = x_interfaces
            y_left, y_right = y_interfaces
            
            # Store interface states in data container
            # These are interior interfaces only (nx+1, ny) and (nx, ny+1)
            data.interface_states_x = (x_left, x_right)
            data.interface_states_y = (y_left, y_right)
            
        self._time_execution(_reconstruct)


class FluxStage(ComputationStage):
    """Stage for computing numerical fluxes using flux calculators directly"""
    
    def __init__(self, flux_type: str = 'lax_friedrichs', **flux_params):
        super().__init__("FluxComputation")
        self.flux_type = flux_type
        self.flux_params = flux_params
        self._flux_calculator: Optional['FluxCalculator'] = None
        
    def _initialize_flux_calculator(self):
        """Initialize flux calculator on first use"""
        if self._flux_calculator is None:
            from fvm_framework.spatial.flux_calculation.factory import create_flux_calculator
            self._flux_calculator = create_flux_calculator(self.flux_type, **self.flux_params)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute numerical fluxes using flux calculator directly"""
        def _compute_fluxes():
            # Get physics equation from kwargs
            physics_equation = kwargs.get('physics_equation')
            if physics_equation is None:
                raise ValueError("FluxStage requires physics_equation in kwargs")
            
            self._initialize_flux_calculator()
            
            # Get interface states from ReconstructionStage
            if data.interface_states_x is None or data.interface_states_y is None:
                raise ValueError("FluxStage requires interface states from ReconstructionStage")
            
            x_left, x_right = data.interface_states_x
            y_left, y_right = data.interface_states_y
            
            # Compute fluxes directly using flux calculator
            # Store in interior portion of flux arrays using indexing helpers
            interior_flux_x = self._flux_calculator.compute_all_x_fluxes(
                x_left, x_right, physics_equation, **kwargs
            )
            interior_flux_y = self._flux_calculator.compute_all_y_fluxes(
                y_left, y_right, physics_equation, **kwargs
            )
            
            # Store in data container flux arrays (map to ghost cell flux arrays)
            ng = data.ng
            data.flux_x[:, ng:ng+data.nx+1, ng:ng+data.ny] = interior_flux_x
            data.flux_y[:, ng:ng+data.nx, ng:ng+data.ny+1] = interior_flux_y
            
        self._time_execution(_compute_fluxes)


class SourceStage(ComputationStage):
    """Stage for computing source terms"""
    
    def __init__(self, source_type: Optional[str] = None):
        super().__init__("SourceTerms")
        self.source_type = source_type
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute source terms"""
        def _compute_sources():
            if self.source_type is None:
                # No source terms - zero out the source array
                data.source.fill(0.0)
            else:
                # Placeholder for source term implementations
                pass
                
        self._time_execution(_compute_sources)


class TemporalStage(ComputationStage):
    """Stage for temporal integration using precomputed fluxes"""
    
    def __init__(self, scheme: str = 'rk3'):
        super().__init__("TemporalIntegration")
        self.scheme = scheme
        self._integrator: Optional['TimeIntegrator'] = None
        
    def _initialize_integrator(self):
        """Initialize time integrator on first use"""
        if self._integrator is None:
            from fvm_framework.temporal.time_integrators import TimeIntegratorFactory
            self._integrator = TimeIntegratorFactory.create(self.scheme)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Perform temporal integration using precomputed fluxes"""
        def _time_integrate():
            dt = kwargs.get('dt')
            if dt is None:
                raise ValueError("TemporalStage requires 'dt' in kwargs")
            
            self._initialize_integrator()
            
            # Simple residual function that uses precomputed fluxes
            def residual_function(data_container: FVMDataContainer2D, **res_kwargs) -> np.ndarray:
                # Flux divergence is computed using fluxes already stored in data_container
                # by previous stages (ReconstructionStage -> FluxStage)
                return data_container.compute_residual()
            
            # Apply time integrator
            self._integrator.integrate(data, dt, residual_function, **kwargs)
            
        self._time_execution(_time_integrate)


class FVMPipeline:
    """
    Main pipeline orchestrator for finite volume method computation.
    
    The pipeline executes stages in the following order:
    1. BoundaryStage: Apply boundary conditions → fill ghost cells
    2. ReconstructionStage: Spatial reconstruction → compute interface states
    3. FluxStage: Flux computation → compute numerical fluxes from interface states  
    4. SourceStage: Source term computation → compute source terms
    5. TemporalStage: Temporal integration → advance solution using residual = -∇·F + S
    """
    
    def __init__(self, 
                 boundary_type: str = 'periodic',
                 reconstruction_type: str = 'constant', 
                 flux_type: str = 'lax_friedrichs',
                 time_scheme: str = 'rk3',
                 source_type: Optional[str] = None,
                 **flux_params):
        """
        Initialize the FVM pipeline with modular components.
        
        Args:
            boundary_type: Type of boundary conditions
            reconstruction_type: Spatial reconstruction method ('constant', 'slope_limiter', 'weno5', etc.)
            flux_type: Flux calculator type ('lax_friedrichs', 'hll', 'hllc', etc.)
            time_scheme: Time integration scheme ('euler', 'rk2', 'rk3', 'rk4')
            source_type: Source term type (None for no sources)
            **flux_params: Additional parameters for flux calculator (e.g., riemann_solver='hllc')
        """
        self.stages: List[ComputationStage] = [
            BoundaryStage(boundary_type),
            ReconstructionStage(reconstruction_type),
            FluxStage(flux_type, **flux_params),
            SourceStage(source_type),
            TemporalStage(time_scheme)
        ]
        
        self.total_steps = 0
        self.total_time = 0.0
        
    def execute_time_step(self, data: FVMDataContainer2D, dt: float, **kwargs) -> None:
        """
        Execute a single time step through the pipeline.
        
        Args:
            data: FVM data container
            dt: Time step size
            **kwargs: Additional parameters passed to stages
        """
        start_time = time.perf_counter()
        
        # Execute each stage in sequence
        for stage in self.stages[:-1]:  # All stages except temporal
            stage.process(data, **kwargs)
        
        # Execute temporal stage with dt parameter
        self.stages[-1].process(data, dt=dt, **kwargs)
        
        end_time = time.perf_counter()
        self.total_time += (end_time - start_time)
        self.total_steps += 1
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        stage_stats = {}
        total_stage_time = 0.0
        
        for stage in self.stages:
            stats = stage.get_performance_stats()
            stage_stats[stage.name] = stats
            total_stage_time += stats['total_time']
            
        avg_step_time = self.total_time / max(self.total_steps, 1)
        
        return {
            'total_steps': self.total_steps,
            'total_time': self.total_time,
            'avg_step_time': avg_step_time,
            'stage_breakdown': stage_stats,
            'overhead_time': self.total_time - total_stage_time
        }
    
    def print_performance_report(self):
        """Print a detailed performance report"""
        summary = self.get_performance_summary()
        
        print("="*60)
        print("FVM PIPELINE PERFORMANCE REPORT")
        print("="*60)
        print(f"Total time steps executed: {summary['total_steps']}")
        print(f"Total execution time: {summary['total_time']:.6f} seconds")
        print(f"Average time per step: {summary['avg_step_time']:.6f} seconds")
        print(f"Pipeline overhead: {summary['overhead_time']:.6f} seconds")
        print()
        
        print("Stage-wise breakdown:")
        print("-" * 40)
        for stage_name, stats in summary['stage_breakdown'].items():
            percentage = (stats['total_time'] / summary['total_time']) * 100
            print(f"{stage_name:20s}: {stats['total_time']:8.6f}s ({percentage:5.1f}%)")
            print(f"{'':20s}  Calls: {stats['call_count']:6d}, "
                  f"Avg: {stats['avg_time']:8.6f}s")
        print("="*60)
        
    def reset_performance_counters(self):
        """Reset all performance counters"""
        self.total_steps = 0
        self.total_time = 0.0
        
        for stage in self.stages:
            stage._execution_time = 0.0
            stage._call_count = 0


class PipelineMonitor:
    """Monitor for tracking pipeline execution and diagnostics"""
    
    def __init__(self, monitor_interval: int = 100):
        self.monitor_interval = monitor_interval
        self.step_count = 0
        self.conservation_history = []
        self.max_wave_speed_history = []
        
    def update(self, data: FVMDataContainer2D, dt: float, physics_equation=None):
        """Update monitoring data"""
        self.step_count += 1
        
        if self.step_count % self.monitor_interval == 0:
            # Track conservation
            conservation = data.get_conservation_error()
            self.conservation_history.append(conservation.copy())
            
            # Track maximum wave speed for stability
            if physics_equation is not None:
                max_speed = physics_equation.compute_max_wave_speed(data)
                self.max_wave_speed_history.append(max_speed)
            
    def get_conservation_drift(self) -> np.ndarray:
        """Get conservation drift over time"""
        if len(self.conservation_history) < 2:
            return np.zeros(5)  # No drift if less than 2 measurements
            
        initial = self.conservation_history[0]
        current = self.conservation_history[-1]
        return np.abs(current - initial) / np.abs(initial + 1e-15)
    
    def print_status(self, current_time: float):
        """Print current status"""
        if len(self.conservation_history) > 0:
            drift = self.get_conservation_drift()
            max_drift = np.max(drift)
            
            print(f"Step: {self.step_count:8d}, Time: {current_time:.6f}, "
                  f"Max Conservation Drift: {max_drift:.2e}")