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
        alpha = kwargs.get("alpha",1)
        
        # Compute physical fluxes
        f_left = self.compute_physical_flux(left_state, physics_equation, direction, **kwargs)
        f_right = self.compute_physical_flux(right_state, physics_equation, direction, **kwargs)
        
        # Lax-Friedrichs flux formula
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (right_state - left_state)
    
    def compute_numerical_flux_vectorized(self, left_states: np.ndarray, right_states: np.ndarray,
                                        physics_equation, direction: int, **kwargs) -> np.ndarray:
        """
        Vectorized Lax-Friedrichs flux computation.
        
        Args:
            left_states: Left interface states, shape (num_vars, num_interfaces)
            right_states: Right interface states, shape (num_vars, num_interfaces)
            physics_equation: Physics equation object
            direction: 0 for x-direction, 1 for y-direction
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, num_interfaces)
        """
        # Get wave speed
        alpha = kwargs.get("alpha", 1)
        
        # Compute physical fluxes vectorized
        num_vars, num_interfaces = left_states.shape
        f_left = np.zeros((num_vars, num_interfaces))
        f_right = np.zeros((num_vars, num_interfaces))
        
        for i in range(num_interfaces):
            f_left[:, i] = self.compute_physical_flux(left_states[:, i], physics_equation, direction, **kwargs)
            f_right[:, i] = self.compute_physical_flux(right_states[:, i], physics_equation, direction, **kwargs)
        
        # Vectorized Lax-Friedrichs formula
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (right_states - left_states)
    
    def compute_all_x_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all Lax-Friedrichs fluxes in x-direction efficiently.
        
        Pre-computes wave speed once, then uses base class vectorized method.
        """
        # Pre-compute wave speed once for efficiency
        data = kwargs.get("data")
        alpha = physics_equation.compute_max_wave_speed(data)
        kwargs["alpha"] = alpha
        
        # Use base class vectorized implementation
        return super().compute_all_x_fluxes(left_states, right_states, physics_equation, **kwargs)
    
    def compute_all_y_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all Lax-Friedrichs fluxes in y-direction efficiently.
        
        Pre-computes wave speed once, then uses base class vectorized method.
        """
        # Pre-compute wave speed once for efficiency
        data = kwargs.get("data")
        alpha = physics_equation.compute_max_wave_speed(data)
        kwargs["alpha"] = alpha
        
        # Use base class vectorized implementation
        return super().compute_all_y_fluxes(left_states, right_states, physics_equation, **kwargs)
    
    def needs_wave_speed(self) -> bool:
        """Lax-Friedrichs flux needs maximum wave speed"""
        return True
    
    def get_wave_speed(self) -> float:
        """Get current wave speed"""
        return self.alpha if self.alpha is not None else 0.0
    
    def set_wave_speed(self, alpha: float):
        """Set wave speed manually"""
        self.alpha = alpha