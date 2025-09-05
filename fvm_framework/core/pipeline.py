"""
Pipeline Framework for Finite Volume Method Solvers

This module implements the computation pipeline architecture that orchestrates
the various stages of FVM computation in a data-driven manner.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import numpy as np
from .data_container import FVMDataContainer2D


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
    """Stage for spatial reconstruction of interface values"""
    
    def __init__(self, reconstruction_type: str = 'linear'):
        super().__init__("SpatialReconstruction")
        self.reconstruction_type = reconstruction_type
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Perform spatial reconstruction"""
        def _reconstruct():
            # Placeholder for reconstruction implementation
            # This will be implemented in the spatial schemes module
            pass
            
        self._time_execution(_reconstruct)


class FluxStage(ComputationStage):
    """Stage for computing numerical fluxes using unified spatial discretization"""
    
    def __init__(self, spatial_scheme: str = 'lax_friedrichs', **scheme_params):
        super().__init__("FluxComputation")
        self.spatial_scheme_name = spatial_scheme
        self.scheme_params = scheme_params
        
        # Import here to avoid circular imports
        from spatial.factory import SpatialDiscretizationFactory
        self.spatial_scheme = SpatialDiscretizationFactory.create(spatial_scheme, **scheme_params)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute numerical fluxes using the selected spatial scheme"""
        def _compute_fluxes():
            # Get physics equation from kwargs
            physics_equation = kwargs.get('physics_equation')
            if physics_equation is None:
                raise ValueError("FluxStage requires physics_equation in kwargs")
            
            # Compute fluxes using unified interface
            self.spatial_scheme.compute_fluxes(data, physics_equation, **kwargs)
            
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
    """Stage for temporal integration using unified spatial discretization"""
    
    def __init__(self, scheme: str = 'rk3', spatial_scheme: str = 'lax_friedrichs', **spatial_params):
        super().__init__("TemporalIntegration")
        self.scheme = scheme
        self.spatial_scheme_name = spatial_scheme
        self.spatial_params = spatial_params
        
        # Import here to avoid circular imports
        from temporal.time_integrators import TimeIntegratorFactory
        from spatial.factory import SpatialDiscretizationFactory
        
        self.integrator = TimeIntegratorFactory.create(scheme)
        self.spatial_scheme = SpatialDiscretizationFactory.create(spatial_scheme, **spatial_params)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Perform temporal integration using unified spatial discretization"""
        def _time_integrate():
            dt = kwargs.get('dt')
            physics_equation = kwargs.get('physics_equation')
            
            if dt is None or physics_equation is None:
                raise ValueError("TemporalStage requires 'dt' and 'physics_equation' in kwargs")
            
            # Define residual function that uses our spatial scheme
            def residual_function(state: np.ndarray) -> np.ndarray:
                # Temporarily update data state
                old_state = data.state.copy()
                data.state = state
                
                # Compute fluxes using spatial scheme
                self.spatial_scheme.compute_fluxes(data, physics_equation, **kwargs)
                
                # Compute flux divergence (residual = -∇·F)
                if hasattr(self.spatial_scheme, 'compute_flux_divergence'):
                    residual = self.spatial_scheme.compute_flux_divergence(data)
                else:
                    # Default flux divergence computation
                    residual = self._compute_flux_divergence(data)
                
                # Restore original state
                data.state = old_state
                return residual
            
            # Apply time integrator
            new_state = self.integrator.step(data.state, dt, residual_function)
            data.state = new_state
            
        self._time_execution(_time_integrate)
    
    def _compute_flux_divergence(self, data: FVMDataContainer2D) -> np.ndarray:
        """Default flux divergence computation"""
        residual = np.zeros_like(data.state)
        
        # Interior points only
        for i in range(1, data.nx - 1):
            for j in range(1, data.ny - 1):
                # Flux divergence: ∇·F = (F_{i+1/2} - F_{i-1/2})/dx + (G_{j+1/2} - G_{j-1/2})/dy
                flux_div_x = (data.flux_x[:, i+1, j] - data.flux_x[:, i, j]) / data.geometry.dx
                flux_div_y = (data.flux_y[:, i, j+1] - data.flux_y[:, i, j]) / data.geometry.dy
                residual[:, i, j] = -(flux_div_x + flux_div_y)
        
        return residual


class FVMPipeline:
    """
    Main pipeline orchestrator for finite volume method computation.
    
    The pipeline executes stages in the following order:
    1. Boundary conditions
    2. Spatial reconstruction
    3. Flux computation
    4. Source term computation
    5. Temporal integration
    """
    
    def __init__(self, 
                 boundary_type: str = 'periodic',
                 reconstruction_type: str = 'linear', 
                 spatial_scheme: str = 'lax_friedrichs',
                 time_scheme: str = 'rk3',
                 source_type: Optional[str] = None,
                 **spatial_params):
        """
        Initialize the FVM pipeline with unified spatial discretization.
        
        Args:
            boundary_type: Type of boundary conditions
            reconstruction_type: Spatial reconstruction method  
            spatial_scheme: Spatial discretization scheme ('lax_friedrichs', 'tvdlf', 'hll', etc.)
            time_scheme: Time integration scheme
            source_type: Source term type (None for no sources)
            **spatial_params: Additional parameters for spatial scheme (e.g., limiter='minmod')
        """
        self.stages: List[ComputationStage] = [
            BoundaryStage(boundary_type),
            ReconstructionStage(reconstruction_type),
            FluxStage(spatial_scheme, **spatial_params),
            SourceStage(source_type),
            TemporalStage(time_scheme, spatial_scheme, **spatial_params)
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
        
    def update(self, data: FVMDataContainer2D, dt: float, gamma: float = 1.4):
        """Update monitoring data"""
        self.step_count += 1
        
        if self.step_count % self.monitor_interval == 0:
            # Track conservation
            conservation = data.get_conservation_error()
            self.conservation_history.append(conservation.copy())
            
            # Track maximum wave speed for stability
            max_speed = data.get_max_wave_speed(gamma)
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