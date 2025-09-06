"""
TVD Lax-Friedrichs Spatial Discretization Scheme

This module implements the second-order TVD (Total Variation Diminishing) 
Lax-Friedrichs scheme with flux limiters to reduce numerical diffusion.
"""

import numpy as np
from fvm_framework.core.data_container import FVMDataContainer2D
from .base import FiniteVolumeScheme


class TVDLFScheme(FiniteVolumeScheme):
    """
    Total Variation Diminishing Lax-Friedrichs scheme.
    
    A second-order extension of Lax-Friedrichs with flux limiters
    to maintain TVD property and reduce numerical diffusion.
    """
    
    def __init__(self, limiter_type: str = 'minmod'):
        super().__init__("TVDLF", order=2)
        self.limiter_type = limiter_type
        self.limiter_function = self._get_limiter_function()
        self.alpha = None
    
    def _get_limiter_function(self):
        """Get the flux limiter function"""
        limiters = {
            'minmod': self._minmod_limiter,
            'superbee': self._superbee_limiter,
            'van_leer': self._van_leer_limiter,
            'mc': self._mc_limiter
        }
        return limiters.get(self.limiter_type, self._minmod_limiter)
    
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """Compute TVDLF fluxes with slope limiting"""
        # Get maximum wave speed
        self.alpha = self.get_max_wave_speed(data, physics_equation, **kwargs)
        
        # Compute limited slopes
        slopes_x = self._compute_limited_slopes_x(data)
        slopes_y = self._compute_limited_slopes_y(data)
        
        # Compute fluxes with reconstruction
        self._compute_x_fluxes_tvd(data, slopes_x, physics_equation, **kwargs)
        self._compute_y_fluxes_tvd(data, slopes_y, physics_equation, **kwargs)
    
    def _compute_limited_slopes_x(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute limited slopes in x-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for j in range(ny):
                for i in range(1, nx-1):
                    # Central differences
                    left_diff = data.state[var, i, j] - data.state[var, i-1, j]
                    right_diff = data.state[var, i+1, j] - data.state[var, i, j]
                    
                    # Apply limiter
                    slopes[var, i, j] = self.limiter_function(left_diff, right_diff)
        
        return slopes
    
    def _compute_limited_slopes_y(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute limited slopes in y-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for i in range(nx):
                for j in range(1, ny-1):
                    # Central differences
                    left_diff = data.state[var, i, j] - data.state[var, i, j-1]
                    right_diff = data.state[var, i, j+1] - data.state[var, i, j]
                    
                    # Apply limiter
                    slopes[var, i, j] = self.limiter_function(left_diff, right_diff)
        
        return slopes
    
    def _compute_x_fluxes_tvd(self, data: FVMDataContainer2D, slopes: np.ndarray, 
                             physics_equation, **kwargs):
        """Compute TVD fluxes in x-direction"""
        nx, ny = data.nx, data.ny
        
        for j in range(ny):
            for i in range(nx + 1):
                # Get interface states using reconstruction
                if i == 0 or i == nx:
                    # Boundary
                    if i == 0:
                        u_left = u_right = data.state[:, 0, j]
                    else:
                        u_left = u_right = data.state[:, -1, j]
                else:
                    # Reconstruct interface states
                    u_left = data.state[:, i-1, j] + 0.5 * slopes[:, i-1, j]
                    u_right = data.state[:, i, j] - 0.5 * slopes[:, i, j]
                
                # Compute fluxes at reconstructed states
                f_left = self._compute_physical_flux(u_left, physics_equation, direction=0, **kwargs)
                f_right = self._compute_physical_flux(u_right, physics_equation, direction=0, **kwargs)
                
                # TVDLF flux
                data.flux_x[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * self.alpha * (u_right - u_left)
    
    def _compute_y_fluxes_tvd(self, data: FVMDataContainer2D, slopes: np.ndarray, 
                             physics_equation, **kwargs):
        """Compute TVD fluxes in y-direction"""
        nx, ny = data.nx, data.ny
        
        for i in range(nx):
            for j in range(ny + 1):
                # Get interface states using reconstruction
                if j == 0 or j == ny:
                    # Boundary
                    if j == 0:
                        u_left = u_right = data.state[:, i, 0]
                    else:
                        u_left = u_right = data.state[:, i, -1]
                else:
                    # Reconstruct interface states
                    u_left = data.state[:, i, j-1] + 0.5 * slopes[:, i, j-1]
                    u_right = data.state[:, i, j] - 0.5 * slopes[:, i, j]
                
                # Compute fluxes at reconstructed states
                f_left = self._compute_physical_flux(u_left, physics_equation, direction=1, **kwargs)
                f_right = self._compute_physical_flux(u_right, physics_equation, direction=1, **kwargs)
                
                # TVDLF flux
                data.flux_y[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * self.alpha * (u_right - u_left)
    
    def _compute_physical_flux(self, state: np.ndarray, physics_equation, direction: int, **kwargs) -> np.ndarray:
        """Compute physical flux for given state (same as LaxFriedrichs)"""
        if hasattr(physics_equation, 'compute_flux_x') and direction == 0:
            return physics_equation.compute_flux_x(state)
        elif hasattr(physics_equation, 'compute_flux_y') and direction == 1:
            return physics_equation.compute_flux_y(state)
        elif hasattr(physics_equation, 'compute_fluxes'):
            return physics_equation.compute_fluxes(state, direction)
        else:
            # Fallback for simple equations
            if direction == 0 and hasattr(physics_equation, 'velocity_x'):
                return np.array([state[0] * physics_equation.velocity_x])
            elif direction == 1 and hasattr(physics_equation, 'velocity_y'):
                return np.array([state[0] * physics_equation.velocity_y])
            else:
                return np.array([state[0]])
    
    # Flux limiters
    def _minmod_limiter(self, a: float, b: float) -> float:
        """MinMod limiter - most dissipative"""
        if a * b <= 0:
            return 0.0
        elif abs(a) < abs(b):
            return a
        else:
            return b
    
    def _superbee_limiter(self, a: float, b: float) -> float:
        """Superbee limiter - least dissipative"""
        if a * b <= 0:
            return 0.0
        elif abs(a) > abs(b):
            return 2.0 * b if abs(b) < 0.5 * abs(a) else (a + b) if abs(a) < 2.0 * abs(b) else 2.0 * a
        else:
            return 2.0 * a if abs(a) < 0.5 * abs(b) else (a + b) if abs(b) < 2.0 * abs(a) else 2.0 * b
    
    def _van_leer_limiter(self, a: float, b: float) -> float:
        """Van Leer limiter - smooth"""
        if a * b <= 0:
            return 0.0
        else:
            return 2.0 * a * b / (a + b)
    
    def _mc_limiter(self, a: float, b: float) -> float:
        """Monotonized Central limiter"""
        if a * b <= 0:
            return 0.0
        else:
            return min(2.0 * abs(a), 2.0 * abs(b), 0.5 * abs(a + b)) * np.sign(a)