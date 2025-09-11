"""
Riemann Solver-based Flux Calculator

This module implements flux calculation using Riemann solvers.
It wraps the existing Riemann solver functionality to work with
the new modular reconstruction + flux calculation architecture.
"""

import numpy as np
from .base_flux import RiemannBasedFluxCalculator


class RiemannFlux(RiemannBasedFluxCalculator):
    """
    Riemann solver-based numerical flux calculator.
    
    Uses approximate Riemann solvers (HLL, HLLC, HLLD, etc.) to compute
    numerical fluxes from left and right interface states.
    """
    
    def __init__(self, riemann_solver_type: str = 'hllc'):
        """
        Initialize Riemann flux calculator.
        
        Args:
            riemann_solver_type: Type of Riemann solver ('hll', 'hllc', 'hlld', 'exact')
        """
        super().__init__(f"Riemann_{riemann_solver_type.upper()}", riemann_solver_type)
    
    def compute_numerical_flux(self, left_state: np.ndarray, right_state: np.ndarray,
                              physics_equation, direction: int, **kwargs) -> np.ndarray:
        """
        Compute numerical flux using Riemann solver.
        
        Args:
            left_state: Left interface state vector
            right_state: Right interface state vector
            physics_equation: Physics equation object
            direction: 0 for x-direction, 1 for y-direction
            **kwargs: Additional parameters (gamma, etc.)
            
        Returns:
            Numerical flux vector from Riemann solver
        """
        # Get Riemann solver
        riemann_solver = self._get_riemann_solver()
        
        # Check if this is a new generic solver

        return riemann_solver.solve(left_state, right_state, physics_equation, direction, **kwargs)
    
    def compute_all_x_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all Riemann fluxes in x-direction efficiently.
        
        Args:
            left_states: Left interface states, shape (num_vars, nx+1, ny)
            right_states: Right interface states, shape (num_vars, nx+1, ny)
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, nx+1, ny)
        """
        # Get Riemann solver once for efficiency
        riemann_solver = self._get_riemann_solver()
        
        num_vars, nx_plus_1, ny = left_states.shape
        fluxes = np.zeros((num_vars, nx_plus_1, ny))
        
        # Compute fluxes efficiently
        for j in range(ny):
            for i in range(nx_plus_1):
                # New generic solver - pass physics_equation
                flux_kwargs = {k: v for k, v in kwargs.items() if k != 'direction'}
                fluxes[:, i, j] = riemann_solver.solve(
                    left_states[:, i, j], right_states[:, i, j], 
                    physics_equation, 0, **flux_kwargs
                )

        
        return fluxes
    
    def compute_all_y_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all Riemann fluxes in y-direction efficiently.
        
        Args:
            left_states: Left interface states, shape (num_vars, nx, ny+1)
            right_states: Right interface states, shape (num_vars, nx, ny+1)
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, nx, ny+1)
        """
        # Get Riemann solver once for efficiency
        riemann_solver = self._get_riemann_solver()
        
        num_vars, nx, ny_plus_1 = left_states.shape
        fluxes = np.zeros((num_vars, nx, ny_plus_1))
        
        # Compute fluxes efficiently
        for i in range(nx):
            for j in range(ny_plus_1):

                # New generic solver - pass physics_equation
                flux_kwargs = {k: v for k, v in kwargs.items() if k != 'direction'}
                fluxes[:, i, j] = riemann_solver.solve(
                    left_states[:, i, j], right_states[:, i, j],
                    physics_equation, 1, **flux_kwargs
                )
        
        return fluxes
    
    def needs_wave_speed(self) -> bool:
        """Riemann solvers compute wave speeds internally"""
        return False
    
    def supports_physics(self, physics_type: str) -> bool:
        """Check physics compatibility"""
        physics_type = physics_type.lower()
        
        # HLLD is specialized for MHD
        if self.riemann_solver_type == 'hlld':
            return physics_type in ['mhd', 'magnetohydrodynamics']
        
        # Other solvers support most hyperbolic systems
        return physics_type in ['euler', 'compressible_flow', 'gas_dynamics', 'mhd']
    
    def get_riemann_solver_type(self) -> str:
        """Get the type of Riemann solver"""
        return self.riemann_solver_type
    
    def set_riemann_solver_type(self, solver_type: str):
        """Change the Riemann solver type"""
        self.riemann_solver_type = solver_type
        self.name = f"Riemann_{solver_type.upper()}"
        self.riemann_solver = None  # Force re-initialization


class HLLFlux(RiemannFlux):
    """HLL Riemann solver flux calculator"""
    
    def __init__(self):
        super().__init__('hll')


class HLLCFlux(RiemannFlux):
    """HLLC Riemann solver flux calculator"""
    
    def __init__(self):
        super().__init__('hllc')


class HLLDFlux(RiemannFlux):
    """HLLD Riemann solver flux calculator (for MHD)"""
    
    def __init__(self):
        super().__init__('hlld')
    
    def supports_physics(self, physics_type: str) -> bool:
        """HLLD is specialized for MHD equations"""
        return physics_type.lower() in ['mhd', 'magnetohydrodynamics']


class ExactRiemannFlux(RiemannFlux):
    """Exact Riemann solver flux calculator"""
    
    def __init__(self):
        super().__init__('exact')