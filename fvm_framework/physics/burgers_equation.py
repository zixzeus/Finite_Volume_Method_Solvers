"""
2D Burgers' Equation

This module implements the 2D Burgers' equation, which is a fundamental
nonlinear PDE used for testing numerical methods.

Equation: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
          ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v

Conservative variables: [u, v] where u,v are velocity components
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
from core.data_container import FVMDataContainer2D


@dataclass
class BurgersState:
    """Structure for Burgers equation variables"""
    u_velocity: float
    v_velocity: float
    viscosity: Optional[float] = 0.01


class BurgersEquation2D:
    """2D Burgers' equation implementation"""
    
    def __init__(self, viscosity: float = 0.01):
        self.viscosity = viscosity
        self.name = "Burgers2D"
        self.num_vars = 2
    
    def conservative_to_primitive(self, conservative: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables (trivial for Burgers)"""
        return conservative.copy()
    
    def primitive_to_conservative(self, primitive: np.ndarray) -> np.ndarray:
        """Convert primitive to conservative variables (trivial for Burgers)"""
        return primitive.copy()
    
    def compute_fluxes(self, state: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute convective fluxes for Burgers equation
        
        Args:
            state: Conservative variables [u, v]
            direction: 0 for x-direction, 1 for y-direction
        
        Returns:
            Flux vector
        """
        u, v = state[0], state[1]
        
        if direction == 0:  # x-direction
            return np.array([
                0.5 * u * u,    # u²/2
                u * v           # uv
            ])
        else:  # y-direction
            return np.array([
                u * v,          # uv
                0.5 * v * v     # v²/2
            ])
    
    def max_wave_speed(self, state: np.ndarray, direction: int) -> float:
        """Compute maximum wave speed for CFL condition"""
        u, v = state[0], state[1]
        
        if direction == 0:  # x-direction
            return np.abs(u)
        else:  # y-direction
            return np.abs(v)
    
    def apply_source_terms(self, data: FVMDataContainer2D, dt: float, **kwargs) -> None:
        """Apply viscous terms (diffusion) as source terms"""
        if self.viscosity > 0:
            # Simple second-order diffusion approximation
            # This is a placeholder - proper viscous terms would use gradient computation
            dx = data.geometry.dx
            dy = data.geometry.dy
            
            # Second derivatives approximation
            d2u_dx2 = (np.roll(data.state[0], -1, axis=0) - 2*data.state[0] + np.roll(data.state[0], 1, axis=0)) / dx**2
            d2u_dy2 = (np.roll(data.state[0], -1, axis=1) - 2*data.state[0] + np.roll(data.state[0], 1, axis=1)) / dy**2
            
            d2v_dx2 = (np.roll(data.state[1], -1, axis=0) - 2*data.state[1] + np.roll(data.state[1], 1, axis=0)) / dx**2
            d2v_dy2 = (np.roll(data.state[1], -1, axis=1) - 2*data.state[1] + np.roll(data.state[1], 1, axis=1)) / dy**2
            
            # Apply viscous terms
            data.state[0] += dt * self.viscosity * (d2u_dx2 + d2u_dy2)
            data.state[1] += dt * self.viscosity * (d2v_dx2 + d2v_dy2)


class BurgersInitialConditions:
    """Initial condition generators for Burgers equation"""
    
    @staticmethod
    def sine_wave(data: FVMDataContainer2D, amplitude: float = 1.0, 
                  wave_number: float = 2.0) -> None:
        """Initialize with sine wave"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                data.state[0, i, j] = amplitude * np.sin(wave_number * np.pi * x)
                data.state[1, i, j] = amplitude * np.sin(wave_number * np.pi * y)
    
    @staticmethod
    def gaussian_pulse(data: FVMDataContainer2D, center_x: float = 0.5, 
                      center_y: float = 0.5, sigma: float = 0.1) -> None:
        """Initialize with Gaussian pulse"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                r_sq = (x - center_x)**2 + (y - center_y)**2
                gaussian = np.exp(-r_sq / (2 * sigma**2))
                
                data.state[0, i, j] = gaussian
                data.state[1, i, j] = gaussian
    
    @staticmethod
    def shock_interaction(data: FVMDataContainer2D) -> None:
        """Initialize for shock interaction test"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                if x < 0.5 and y < 0.5:
                    data.state[0, i, j] = 1.0
                    data.state[1, i, j] = 1.0
                elif x >= 0.5 and y < 0.5:
                    data.state[0, i, j] = -1.0
                    data.state[1, i, j] = 1.0
                elif x < 0.5 and y >= 0.5:
                    data.state[0, i, j] = 1.0
                    data.state[1, i, j] = -1.0
                else:
                    data.state[0, i, j] = -1.0
                    data.state[1, i, j] = -1.0