"""
Lax-Friedrichs Spatial Discretization Scheme

This module implements the first-order Lax-Friedrichs scheme, which is 
stable but dissipative. It's a good baseline method for comparison.
"""

import numpy as np
from fvm_framework.core.data_container import FVMDataContainer2D
from .base import FiniteVolumeScheme


class LaxFriedrichsScheme(FiniteVolumeScheme):
    """
    Lax-Friedrichs scheme implementation.
    
    A first-order, stable but dissipative scheme.
    Formula: F_{i+1/2} = 0.5 * (F(U_i) + F(U_{i+1})) - 0.5 * α * (U_{i+1} - U_i)
    where α is the maximum wave speed.
    """
    
    def __init__(self):
        super().__init__("LaxFriedrichs", order=1)
        self.alpha = None  # Maximum wave speed (computed dynamically)
    
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """Compute Lax-Friedrichs fluxes"""
        # Use physics equation to compute wave speed
        self.alpha = physics_equation.compute_max_wave_speed(data)
        
        # Compute fluxes in both directions
        self._compute_x_fluxes(data, physics_equation, **kwargs)
        self._compute_y_fluxes(data, physics_equation, **kwargs)
    
    def _compute_x_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs):
        """Compute fluxes in x-direction"""
        nx, ny = data.nx, data.ny
        
        for j in range(ny):
            for i in range(nx + 1):
                # Get left and right states
                if i == 0:
                    u_left = u_right = data.state[:, 0, j]
                elif i == nx:
                    u_left = u_right = data.state[:, -1, j]
                else:
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                
                # Compute physical fluxes
                f_left = self._compute_physical_flux(u_left, physics_equation, direction=0, **kwargs)
                f_right = self._compute_physical_flux(u_right, physics_equation, direction=0, **kwargs)
                
                # Lax-Friedrichs flux
                data.flux_x[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * self.alpha * (u_right - u_left)
    
    def _compute_y_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs):
        """Compute fluxes in y-direction"""
        nx, ny = data.nx, data.ny
        
        for i in range(nx):
            for j in range(ny + 1):
                # Get left and right states (in y-direction)
                if j == 0:
                    u_left = u_right = data.state[:, i, 0]
                elif j == ny:
                    u_left = u_right = data.state[:, i, -1]
                else:
                    u_left = data.state[:, i, j-1]
                    u_right = data.state[:, i, j]
                
                # Compute physical fluxes
                f_left = self._compute_physical_flux(u_left, physics_equation, direction=1, **kwargs)
                f_right = self._compute_physical_flux(u_right, physics_equation, direction=1, **kwargs)
                
                # Lax-Friedrichs flux
                data.flux_y[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * self.alpha * (u_right - u_left)
    
    def _compute_physical_flux(self, state: np.ndarray, physics_equation, direction: int, **kwargs) -> np.ndarray:
        """Compute physical flux for given state"""
        if hasattr(physics_equation, 'compute_flux_x') and direction == 0:
            return physics_equation.compute_flux_x(state)
        elif hasattr(physics_equation, 'compute_flux_y') and direction == 1:
            return physics_equation.compute_flux_y(state)
        elif hasattr(physics_equation, 'compute_fluxes'):
            return physics_equation.compute_fluxes(state, direction)
        else:
            # Fallback for simple equations
            if direction == 0 and hasattr(physics_equation, 'velocity_x'):
                # Simple advection-like: F = u * velocity
                return np.array([state[0] * physics_equation.velocity_x])
            elif direction == 1 and hasattr(physics_equation, 'velocity_y'):
                return np.array([state[0] * physics_equation.velocity_y])
            else:
                return np.array([state[0]])  # Even simpler fallback