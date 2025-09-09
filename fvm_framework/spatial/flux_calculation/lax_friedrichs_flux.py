"""
Lax-Friedrichs Flux Calculator

This module implements the Lax-Friedrichs numerical flux calculation.
It takes left and right interface states and computes the Lax-Friedrichs flux.
This extracts the flux calculation part from the original Lax-Friedrichs scheme.
"""

import numpy as np
from fvm_framework.core.data_container import FVMDataContainer2D
from .base_flux import DissipativeFluxCalculator


class LaxFriedrichsFlux(DissipativeFluxCalculator):
    """
    Lax-Friedrichs numerical flux calculator.
    
    Computes numerical flux using the Lax-Friedrichs formula:
    F_{i+1/2} = 0.5 * (F(u_L) + F(u_R)) - 0.5 * α * (u_R - u_L)
    
    where:
    - u_L, u_R are left and right interface states (from reconstruction)
    - F(u) is the physical flux
    - α is the maximum wave speed
    """
    
    def __init__(self):
        super().__init__("LaxFriedrichs")
        self.alpha = None  # Maximum wave speed (computed dynamically)
    
    def compute_numerical_flux(self, left_state: np.ndarray, right_state: np.ndarray,
                              physics_equation, direction: int, **kwargs) -> np.ndarray:
        """
        Compute Lax-Friedrichs numerical flux.
        
        Args:
            left_state: Left interface state vector
            right_state: Right interface state vector
            physics_equation: Physics equation object
            direction: 0 for x-direction, 1 for y-direction
            **kwargs: Additional parameters (must contain wave speed info)
            
        Returns:
            Numerical flux vector
        """
        # Get wave speed
        alpha = kwargs.get('alpha', None)
        if alpha is None:
            # Try to compute from physics equation
            data = kwargs.get('data', None)
            if data is not None and hasattr(physics_equation, 'compute_max_wave_speed'):
                alpha = physics_equation.compute_max_wave_speed(data)
            else:
                alpha = 1.0  # Fallback
        
        # Compute physical fluxes
        f_left = self.compute_physical_flux(left_state, physics_equation, direction, **kwargs)
        f_right = self.compute_physical_flux(right_state, physics_equation, direction, **kwargs)
        
        # Lax-Friedrichs flux formula
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (right_state - left_state)
    
    def compute_all_x_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all Lax-Friedrichs fluxes in x-direction efficiently.
        
        Args:
            left_states: Left interface states, shape (num_vars, nx+1, ny)
            right_states: Right interface states, shape (num_vars, nx+1, ny)
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, nx+1, ny)
        """
        # Get or compute wave speed once for efficiency
        alpha = kwargs.get('alpha', None)
        if alpha is None:
            data = kwargs.get('data', None)
            if data is not None and hasattr(physics_equation, 'compute_max_wave_speed'):
                alpha = physics_equation.compute_max_wave_speed(data)
                kwargs['alpha'] = alpha  # Cache for individual flux computations
            else:
                alpha = 1.0
                kwargs['alpha'] = alpha
        
        # Compute fluxes using vectorized approach
        nx_interfaces, ny = left_states.shape[1], left_states.shape[2]
        num_vars = left_states.shape[0]
        fluxes = np.zeros((num_vars, nx_interfaces, ny))
        
        for j in range(ny):
            for i in range(nx_interfaces):
                u_left = left_states[:, i, j]
                u_right = right_states[:, i, j]
                fluxes[:, i, j] = self.compute_numerical_flux(
                    u_left, u_right, physics_equation, direction=0, **kwargs
                )
        
        return fluxes
    
    def compute_all_y_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all Lax-Friedrichs fluxes in y-direction efficiently.
        
        Args:
            left_states: Left interface states, shape (num_vars, nx, ny+1)
            right_states: Right interface states, shape (num_vars, nx, ny+1)
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, nx, ny+1)
        """
        # Get or compute wave speed once for efficiency
        alpha = kwargs.get('alpha', None)
        if alpha is None:
            data = kwargs.get('data', None)
            if data is not None and hasattr(physics_equation, 'compute_max_wave_speed'):
                alpha = physics_equation.compute_max_wave_speed(data)
                kwargs['alpha'] = alpha  # Cache for individual flux computations
            else:
                alpha = 1.0
                kwargs['alpha'] = alpha
        
        # Compute fluxes using vectorized approach
        nx, ny_interfaces = left_states.shape[1], left_states.shape[2]
        num_vars = left_states.shape[0]
        fluxes = np.zeros((num_vars, nx, ny_interfaces))
        
        for i in range(nx):
            for j in range(ny_interfaces):
                u_left = left_states[:, i, j]
                u_right = right_states[:, i, j]
                fluxes[:, i, j] = self.compute_numerical_flux(
                    u_left, u_right, physics_equation, direction=1, **kwargs
                )
        
        return fluxes
    
    def needs_wave_speed(self) -> bool:
        """Lax-Friedrichs flux needs maximum wave speed"""
        return True
    
    def get_wave_speed(self) -> float:
        """Get current wave speed"""
        return self.alpha if self.alpha is not None else 0.0
    
    def set_wave_speed(self, alpha: float):
        """Set wave speed manually"""
        self.alpha = alpha