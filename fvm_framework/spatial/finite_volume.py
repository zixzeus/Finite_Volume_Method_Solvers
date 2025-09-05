"""
Finite Volume Spatial Discretization Methods

This module implements various finite volume spatial discretization schemes
including Lax-Friedrichs, TVDLF, and higher-order methods with flux limiters.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from abc import ABC, abstractmethod
from core.data_container import FVMDataContainer2D


class SpatialScheme(ABC):
    """Abstract base class for spatial discretization schemes"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute_fluxes(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute numerical fluxes at cell interfaces"""
        pass


class LaxFriedrichs(SpatialScheme):
    """
    Lax-Friedrichs scheme for finite volume method.
    
    This is a first-order, dissipative scheme that is very stable
    but has significant numerical diffusion.
    """
    
    def __init__(self):
        super().__init__("LaxFriedrichs")
    
    def compute_fluxes(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Compute Lax-Friedrichs fluxes.
        
        Formula: F_{i+1/2} = 0.5 * (F(U_i) + F(U_{i+1})) - 0.5 * α * (U_{i+1} - U_i)
        where α is the maximum wave speed.
        """
        gamma = kwargs.get('gamma', 1.4)
        
        # Compute maximum wave speed
        data.compute_primitives(gamma)
        max_speed = data.get_max_wave_speed(gamma)
        
        # X-direction fluxes
        self._compute_x_fluxes(data, max_speed, gamma)
        
        # Y-direction fluxes
        self._compute_y_fluxes(data, max_speed, gamma)
    
    def _compute_x_fluxes(self, data: FVMDataContainer2D, max_speed: float, gamma: float):
        """Compute fluxes in x-direction"""
        nx, ny = data.nx, data.ny
        
        for j in range(ny):
            for i in range(nx + 1):
                # Get left and right states
                if i == 0:
                    # Left boundary
                    u_left = data.state[:, 0, j]
                    u_right = data.state[:, 0, j]
                elif i == nx:
                    # Right boundary
                    u_left = data.state[:, -1, j]
                    u_right = data.state[:, -1, j]
                else:
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                
                # Compute physical fluxes
                f_left = self._euler_flux_x(u_left, gamma)
                f_right = self._euler_flux_x(u_right, gamma)
                
                # Lax-Friedrichs flux
                data.flux_x[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * max_speed * (u_right - u_left)
    
    def _compute_y_fluxes(self, data: FVMDataContainer2D, max_speed: float, gamma: float):
        """Compute fluxes in y-direction"""
        nx, ny = data.nx, data.ny
        
        for i in range(nx):
            for j in range(ny + 1):
                # Get left and right states (in y-direction)
                if j == 0:
                    # Bottom boundary
                    u_left = data.state[:, i, 0]
                    u_right = data.state[:, i, 0]
                elif j == ny:
                    # Top boundary
                    u_left = data.state[:, i, -1]
                    u_right = data.state[:, i, -1]
                else:
                    u_left = data.state[:, i, j-1]
                    u_right = data.state[:, i, j]
                
                # Compute physical fluxes
                f_left = self._euler_flux_y(u_left, gamma)
                f_right = self._euler_flux_y(u_right, gamma)
                
                # Lax-Friedrichs flux
                data.flux_y[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * max_speed * (u_right - u_left)
    
    def _euler_flux_x(self, u: np.ndarray, gamma: float) -> np.ndarray:
        """Compute Euler flux in x-direction"""
        rho, rho_u, rho_v, rho_w, E = u[0], u[1], u[2], u[3], u[4]
        
        # Avoid division by zero
        rho = max(rho, 1e-15)
        u_vel = rho_u / rho
        v_vel = rho_v / rho
        w_vel = rho_w / rho
        
        # Pressure
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        p = (gamma - 1.0) * (E - kinetic_energy)
        p = max(p, 1e-15)
        
        # Flux vector F = [ρu, ρu² + p, ρuv, ρuw, u(E + p)]
        flux = np.array([
            rho_u,
            rho_u * u_vel + p,
            rho_u * v_vel,
            rho_u * w_vel,
            u_vel * (E + p)
        ])
        
        return flux
    
    def _euler_flux_y(self, u: np.ndarray, gamma: float) -> np.ndarray:
        """Compute Euler flux in y-direction"""
        rho, rho_u, rho_v, rho_w, E = u[0], u[1], u[2], u[3], u[4]
        
        # Avoid division by zero
        rho = max(rho, 1e-15)
        u_vel = rho_u / rho
        v_vel = rho_v / rho
        w_vel = rho_w / rho
        
        # Pressure
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        p = (gamma - 1.0) * (E - kinetic_energy)
        p = max(p, 1e-15)
        
        # Flux vector G = [ρv, ρuv, ρv² + p, ρvw, v(E + p)]
        flux = np.array([
            rho_v,
            rho_v * u_vel,
            rho_v * v_vel + p,
            rho_v * w_vel,
            v_vel * (E + p)
        ])
        
        return flux


class TVDLF(SpatialScheme):
    """
    Total Variation Diminishing Lax-Friedrichs scheme.
    
    A second-order extension of Lax-Friedrichs with flux limiters
    to maintain TVD property and reduce numerical diffusion.
    """
    
    def __init__(self, limiter_type: str = 'minmod'):
        super().__init__("TVDLF")
        self.limiter_type = limiter_type
        
        # Select flux limiter
        limiters = {
            'minmod': self._minmod_limiter,
            'superbee': self._superbee_limiter,
            'van_leer': self._van_leer_limiter,
            'mc': self._mc_limiter
        }
        self.limiter = limiters.get(limiter_type, self._minmod_limiter)
    
    def compute_fluxes(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute TVDLF fluxes with slope limiting"""
        gamma = kwargs.get('gamma', 1.4)
        
        # First compute slopes with limiting
        slopes_x = self._compute_limited_slopes_x(data)
        slopes_y = self._compute_limited_slopes_y(data)
        
        # Compute interface states using slopes
        self._compute_x_fluxes_tvd(data, slopes_x, gamma)
        self._compute_y_fluxes_tvd(data, slopes_y, gamma)
    
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
                    slopes[var, i, j] = self.limiter(left_diff, right_diff)
        
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
                    slopes[var, i, j] = self.limiter(left_diff, right_diff)
        
        return slopes
    
    def _compute_x_fluxes_tvd(self, data: FVMDataContainer2D, slopes: np.ndarray, gamma: float):
        """Compute TVD fluxes in x-direction"""
        nx, ny = data.nx, data.ny
        max_speed = data.get_max_wave_speed(gamma)
        
        for j in range(ny):
            for i in range(nx + 1):
                # Get interface states using reconstruction
                if i == 0 or i == nx:
                    # Boundary - use cell center values
                    if i == 0:
                        u_left = u_right = data.state[:, 0, j]
                    else:
                        u_left = u_right = data.state[:, -1, j]
                else:
                    # Reconstruct interface states
                    u_left = data.state[:, i-1, j] + 0.5 * slopes[:, i-1, j]
                    u_right = data.state[:, i, j] - 0.5 * slopes[:, i, j]
                
                # Compute fluxes at reconstructed states
                lax_friedrichs = LaxFriedrichs()
                f_left = lax_friedrichs._euler_flux_x(u_left, gamma)
                f_right = lax_friedrichs._euler_flux_x(u_right, gamma)
                
                # TVDLF flux
                data.flux_x[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * max_speed * (u_right - u_left)
    
    def _compute_y_fluxes_tvd(self, data: FVMDataContainer2D, slopes: np.ndarray, gamma: float):
        """Compute TVD fluxes in y-direction"""
        nx, ny = data.nx, data.ny
        max_speed = data.get_max_wave_speed(gamma)
        
        for i in range(nx):
            for j in range(ny + 1):
                # Get interface states using reconstruction
                if j == 0 or j == ny:
                    # Boundary - use cell center values
                    if j == 0:
                        u_left = u_right = data.state[:, i, 0]
                    else:
                        u_left = u_right = data.state[:, i, -1]
                else:
                    # Reconstruct interface states
                    u_left = data.state[:, i, j-1] + 0.5 * slopes[:, i, j-1]
                    u_right = data.state[:, i, j] - 0.5 * slopes[:, i, j]
                
                # Compute fluxes at reconstructed states
                lax_friedrichs = LaxFriedrichs()
                f_left = lax_friedrichs._euler_flux_y(u_left, gamma)
                f_right = lax_friedrichs._euler_flux_y(u_right, gamma)
                
                # TVDLF flux
                data.flux_y[:, i, j] = 0.5 * (f_left + f_right) - 0.5 * max_speed * (u_right - u_left)
    
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


class FluxSplitting(SpatialScheme):
    """
    Base class for flux splitting methods.
    
    Implements dimensional splitting for multi-dimensional problems.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def compute_fluxes(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute fluxes using dimensional splitting"""
        # This will be implemented by subclasses
        pass
    
    def _split_flux_jacobian(self, u: np.ndarray, direction: int, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split flux Jacobian into positive and negative parts.
        
        Args:
            u: Conservative variable vector
            direction: 0 for x-direction, 1 for y-direction
            gamma: Heat capacity ratio
            
        Returns:
            Tuple of (positive_flux, negative_flux)
        """
        rho, rho_u, rho_v, rho_w, E = u[0], u[1], u[2], u[3], u[4]
        
        # Avoid division by zero
        rho = max(rho, 1e-15)
        u_vel = rho_u / rho
        v_vel = rho_v / rho
        w_vel = rho_w / rho
        
        # Pressure and sound speed
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        p = max((gamma - 1.0) * (E - kinetic_energy), 1e-15)
        c = np.sqrt(gamma * p / rho)
        
        if direction == 0:  # x-direction
            # Eigenvalues: u-c, u, u, u, u+c
            lambda_vals = np.array([u_vel - c, u_vel, u_vel, u_vel, u_vel + c])
        else:  # y-direction
            # Eigenvalues: v-c, v, v, v, v+c
            lambda_vals = np.array([v_vel - c, v_vel, v_vel, v_vel, v_vel + c])
        
        # Split eigenvalues
        lambda_plus = np.maximum(lambda_vals, 0)
        lambda_minus = np.minimum(lambda_vals, 0)
        
        # This is a simplified splitting - full implementation would require
        # eigenvector decomposition
        return lambda_plus, lambda_minus


class UpwindScheme(SpatialScheme):
    """
    Simple upwind scheme using wave speed information.
    
    This scheme selects the upwind flux based on the sign of characteristic speeds.
    """
    
    def __init__(self):
        super().__init__("Upwind")
    
    def compute_fluxes(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Compute upwind fluxes"""
        gamma = kwargs.get('gamma', 1.4)
        
        # Compute primitives for wave speed calculation
        data.compute_primitives(gamma)
        
        # X-direction fluxes
        self._compute_x_fluxes_upwind(data, gamma)
        
        # Y-direction fluxes  
        self._compute_y_fluxes_upwind(data, gamma)
    
    def _compute_x_fluxes_upwind(self, data: FVMDataContainer2D, gamma: float):
        """Compute upwind fluxes in x-direction"""
        nx, ny = data.nx, data.ny
        
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0 or i == nx:
                    # Boundary fluxes
                    if i == 0:
                        u_state = data.state[:, 0, j]
                    else:
                        u_state = data.state[:, -1, j]
                    data.flux_x[:, i, j] = self._euler_flux_x(u_state, gamma)
                else:
                    # Interior interface
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                    
                    # Get velocities for upwind decision
                    rho_left = max(u_left[0], 1e-15)
                    rho_right = max(u_right[0], 1e-15)
                    u_vel_left = u_left[1] / rho_left
                    u_vel_right = u_right[1] / rho_right
                    
                    # Simple upwind flux selection
                    if u_vel_left + u_vel_right > 0:
                        # Flow to the right - use left state
                        data.flux_x[:, i, j] = self._euler_flux_x(u_left, gamma)
                    else:
                        # Flow to the left - use right state
                        data.flux_x[:, i, j] = self._euler_flux_x(u_right, gamma)
    
    def _compute_y_fluxes_upwind(self, data: FVMDataContainer2D, gamma: float):
        """Compute upwind fluxes in y-direction"""
        nx, ny = data.nx, data.ny
        
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0 or j == ny:
                    # Boundary fluxes
                    if j == 0:
                        u_state = data.state[:, i, 0]
                    else:
                        u_state = data.state[:, i, -1]
                    data.flux_y[:, i, j] = self._euler_flux_y(u_state, gamma)
                else:
                    # Interior interface
                    u_left = data.state[:, i, j-1]
                    u_right = data.state[:, i, j]
                    
                    # Get velocities for upwind decision
                    rho_left = max(u_left[0], 1e-15)
                    rho_right = max(u_right[0], 1e-15)
                    v_vel_left = u_left[2] / rho_left
                    v_vel_right = u_right[2] / rho_right
                    
                    # Simple upwind flux selection
                    if v_vel_left + v_vel_right > 0:
                        # Flow upward - use left (bottom) state
                        data.flux_y[:, i, j] = self._euler_flux_y(u_left, gamma)
                    else:
                        # Flow downward - use right (top) state
                        data.flux_y[:, i, j] = self._euler_flux_y(u_right, gamma)
    
    def _euler_flux_x(self, u: np.ndarray, gamma: float) -> np.ndarray:
        """Compute Euler flux in x-direction (same as LaxFriedrichs)"""
        lax_friedrichs = LaxFriedrichs()
        return lax_friedrichs._euler_flux_x(u, gamma)
    
    def _euler_flux_y(self, u: np.ndarray, gamma: float) -> np.ndarray:
        """Compute Euler flux in y-direction (same as LaxFriedrichs)"""
        lax_friedrichs = LaxFriedrichs()
        return lax_friedrichs._euler_flux_y(u, gamma)