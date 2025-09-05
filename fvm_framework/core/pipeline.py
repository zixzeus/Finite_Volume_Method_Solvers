"""
Pipeline Framework for Finite Volume Method Solvers

This module implements the computation pipeline architecture that orchestrates
the various stages of FVM computation in a data-driven manner.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import numpy as np
from core.data_container import FVMDataContainer2D


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
    """Stage for computing numerical fluxes"""
    
    def __init__(self, riemann_solver: str = 'hllc'):
        super().__init__("FluxComputation")
        self.riemann_solver = riemann_solver
        # Import here to avoid circular imports
        from spatial.riemann_solvers import RiemannSolverFactory, RiemannFluxComputation
        self.solver = RiemannSolverFactory.create(riemann_solver)
        self.flux_computer = RiemannFluxComputation(self.solver)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute numerical fluxes"""
        def _compute_fluxes():
            gamma = kwargs.get('gamma', 1.4)
            self.flux_computer.compute_fluxes(data, gamma)
            
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
    """Stage for temporal integration"""
    
    def __init__(self, scheme: str = 'rk3'):
        super().__init__("TemporalIntegration")
        self.scheme = scheme
        # Import here to avoid circular imports
        from temporal.time_integrators import TimeIntegratorFactory, ResidualFunction
        from spatial.riemann_solvers import RiemannSolverFactory, RiemannFluxComputation
        
        self.integrator = TimeIntegratorFactory.create(scheme)
        # Create a simple residual function for flux divergence
        solver = RiemannSolverFactory.create('hllc')
        flux_computer = RiemannFluxComputation(solver)
        self.residual_function = ResidualFunction(flux_computer)
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Perform temporal integration"""
        def _time_integrate():
            dt = kwargs['dt']
            # Use residual function that computes flux divergence
            residual = data.compute_residual()
            
            # Simple forward Euler update (can be replaced with integrator)
            data.state_new[:] = data.state + dt * residual
            data.swap_states()
            
        self._time_execution(_time_integrate)


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
                 riemann_solver: str = 'hllc',
                 time_scheme: str = 'rk3',
                 source_type: Optional[str] = None):
        """
        Initialize the FVM pipeline.
        
        Args:
            boundary_type: Type of boundary conditions
            reconstruction_type: Spatial reconstruction method
            riemann_solver: Riemann solver type
            time_scheme: Time integration scheme
            source_type: Source term type (None for no sources)
        """
        self.stages: List[ComputationStage] = [
            BoundaryStage(boundary_type),
            ReconstructionStage(reconstruction_type),
            FluxStage(riemann_solver),
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