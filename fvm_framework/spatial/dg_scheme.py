"""
Discontinuous Galerkin Scheme Wrapper

This module provides a unified interface for Discontinuous Galerkin methods
to work within the spatial discretization framework.
"""

import numpy as np
from fvm_framework.core.data_container import FVMDataContainer2D
from .base import DiscontinuousGalerkinScheme
from .discontinuous_galerkin import DGSolver2D, DGDataContainer2D, DGIntegrationInterface


class DGScheme(DiscontinuousGalerkinScheme):
    """Discontinuous Galerkin scheme wrapper"""
    
    def __init__(self, polynomial_order: int = 1, riemann_solver: str = 'hllc'):
        """
        Initialize DG scheme.
        
        Args:
            polynomial_order: Polynomial order (0, 1, or 2)
            riemann_solver: Riemann solver for numerical fluxes
        """
        name = f"DG_P{polynomial_order}"
        
        super().__init__(name, polynomial_order)
        
        self.polynomial_order = polynomial_order
        self.riemann_solver = riemann_solver
        self.dg_solver = None
        self.dg_data = None
        self.dg_interface = None
    
    def initialize_solver(self, data: FVMDataContainer2D, physics_equation, **kwargs):
        """Initialize the DG solver with problem data"""
        # Create DG solver
        self.dg_solver = DGSolver2D(
            polynomial_order=self.polynomial_order,
            riemann_solver=self.riemann_solver,
            flux_function=getattr(physics_equation, 'compute_flux', None)
        )
        
        # Create DG data container
        self.dg_data = DGDataContainer2D(
            data.geometry, 
            physics_equation.num_variables,
            self.polynomial_order
        )
        
        # Create integration interface
        self.dg_interface = DGIntegrationInterface(self.dg_solver, self.dg_data)
        
        # Project initial conditions
        if hasattr(data, 'state') and data.state is not None:
            # Convert FVM data to DG format
            self._convert_fvm_to_dg(data)
    
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """Compute fluxes using DG method"""
        if self.dg_solver is None:
            self.initialize_solver(data, physics_equation, **kwargs)
        
        # Update DG data from FVM data
        self._update_dg_data(data)
        
        # Compute DG residual
        if self.dg_interface is not None:
            residual = self.dg_interface.compute_spatial_residual(**kwargs)
        else:
            return  # Cannot compute without initialized interface
        
        # Convert residual back to flux format for compatibility
        self._convert_residual_to_fluxes(data, residual)
    
    def _convert_fvm_to_dg(self, data: FVMDataContainer2D):
        """Convert FVM data format to DG format"""
        # Use cell averages as P0 coefficients
        if self.dg_data is not None:
            self.dg_data.set_cell_averages(data.state)
    
    def _update_dg_data(self, data: FVMDataContainer2D):
        """Update DG data container with current FVM state"""
        if self.dg_data is not None:
            # Update P0 coefficients with cell averages
            self.dg_data.coefficients[:, :, :, 0] = data.state
    
    def _convert_residual_to_fluxes(self, data: FVMDataContainer2D, residual: np.ndarray):
        """Convert DG residual to flux format for time integration"""
        # For compatibility with FVM framework, we store the spatial residual
        # as a negative divergence (since residual = -∇·F)
        nx, ny = data.nx, data.ny
        
        # Extract cell averages from residual (P0 coefficients)
        cell_residual = residual[:, :, :, 0]
        
        # Convert to flux differences
        # This is a simplified conversion - full DG doesn't use face fluxes directly
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    data.flux_x[:, i, j] = 0.0
                elif i == nx:
                    data.flux_x[:, i, j] = 0.0
                else:
                    # Approximate flux from residual difference
                    data.flux_x[:, i, j] = 0.5 * (cell_residual[:, i-1, j] + cell_residual[:, i, j])
        
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    data.flux_y[:, i, j] = 0.0
                elif j == ny:
                    data.flux_y[:, i, j] = 0.0
                else:
                    # Approximate flux from residual difference
                    data.flux_y[:, i, j] = 0.5 * (cell_residual[:, i, j-1] + cell_residual[:, i, j])
    
    def compute_time_step(self, data: FVMDataContainer2D, physics_equation, cfl: float = 0.5) -> float:
        """Compute stable time step for DG method"""
        # DG methods typically need smaller time steps
        dg_cfl = cfl / (2 * self.polynomial_order + 1)
        
        # Use physics equation's time step computation
        if hasattr(physics_equation, 'compute_time_step'):
            return physics_equation.compute_time_step(data, dg_cfl)
        
        # Fallback: simple CFL condition
        max_wave_speed = self._estimate_max_wave_speed(data, physics_equation)
        dt_x = dg_cfl * data.geometry.dx / max_wave_speed
        dt_y = dg_cfl * data.geometry.dy / max_wave_speed
        
        return min(dt_x, dt_y)
    
    def _estimate_max_wave_speed(self, data: FVMDataContainer2D, physics_equation) -> float:
        """Estimate maximum wave speed for CFL condition"""
        if hasattr(physics_equation, 'compute_wave_speeds'):
            max_speed = 0.0
            for i in range(data.nx):
                for j in range(data.ny):
                    speeds = physics_equation.compute_wave_speeds(data.state[:, i, j])
                    max_speed = max(max_speed, np.max(np.abs(speeds)))
            return max_speed
        
        # Fallback for simple cases
        return 1.0
    
    def supports_physics(self, physics_type: str) -> bool:
        """DG methods support most physics equations"""
        return True  # DG is quite general
    
    def get_solution_for_output(self, data: FVMDataContainer2D) -> np.ndarray:
        """Get solution in format suitable for output"""
        if self.dg_interface is not None:
            return self.dg_interface.get_solution_for_output()
        return data.state