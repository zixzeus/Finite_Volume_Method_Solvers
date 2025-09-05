"""
2D Linear Advection Equation

This module implements the 2D linear advection equation, which is the simplest
hyperbolic PDE and serves as a benchmark for testing numerical schemes.

Equation: ∂u/∂t + a∂u/∂x + b∂u/∂y = 0

Conservative variable: [u] where u is the scalar quantity being advected
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
from core.data_container import FVMDataContainer2D


@dataclass
class AdvectionState:
    """Structure for advection equation variables"""
    scalar: float
    advection_x: float
    advection_y: float


class AdvectionEquation2D:
    """2D linear advection equation implementation"""
    
    def __init__(self, advection_x: float = 1.0, advection_y: float = 0.0):
        self.advection_x = advection_x  # Advection velocity in x-direction
        self.advection_y = advection_y  # Advection velocity in y-direction
        self.name = "Advection2D"
        self.num_vars = 1
    
    def conservative_to_primitive(self, conservative: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables (trivial for advection)"""
        return conservative.copy()
    
    def primitive_to_conservative(self, primitive: np.ndarray) -> np.ndarray:
        """Convert primitive to conservative variables (trivial for advection)"""
        return primitive.copy()
    
    def compute_fluxes(self, state: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute fluxes for linear advection equation
        
        Args:
            state: Conservative variables [u]
            direction: 0 for x-direction, 1 for y-direction
        
        Returns:
            Flux vector
        """
        u = state[0]
        
        if direction == 0:  # x-direction
            return np.array([self.advection_x * u])
        else:  # y-direction
            return np.array([self.advection_y * u])
    
    def max_wave_speed(self, state: np.ndarray, direction: int) -> float:
        """Compute maximum wave speed for CFL condition"""
        if direction == 0:  # x-direction
            return np.abs(self.advection_x)
        else:  # y-direction
            return np.abs(self.advection_y)
    
    def apply_source_terms(self, data: FVMDataContainer2D, dt: float, **kwargs) -> None:
        """Apply source terms (none for pure advection)"""
        pass
    
    def set_advection_velocity(self, vx: float, vy: float) -> None:
        """Set the advection velocity components"""
        self.advection_x = vx
        self.advection_y = vy


class AdvectionInitialConditions:
    """Initial condition generators for advection equation"""
    
    @staticmethod
    def gaussian_pulse(data: FVMDataContainer2D, center_x: float = 0.5, 
                      center_y: float = 0.5, sigma: float = 0.1, 
                      amplitude: float = 1.0) -> None:
        """Initialize with Gaussian pulse"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                r_sq = (x - center_x)**2 + (y - center_y)**2
                data.state[0, i, j] = amplitude * np.exp(-r_sq / (2 * sigma**2))
    
    @staticmethod
    def cosine_bell(data: FVMDataContainer2D, center_x: float = 0.5, 
                   center_y: float = 0.5, radius: float = 0.2) -> None:
        """Initialize with smooth cosine bell"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if r <= radius:
                    data.state[0, i, j] = 0.5 * (1 + np.cos(np.pi * r / radius))
                else:
                    data.state[0, i, j] = 0.0
    
    @staticmethod
    def step_function(data: FVMDataContainer2D, center_x: float = 0.5, 
                     center_y: float = 0.5, radius: float = 0.2, 
                     inner_value: float = 1.0, outer_value: float = 0.0) -> None:
        """Initialize with step function (discontinuous)"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if r <= radius:
                    data.state[0, i, j] = inner_value
                else:
                    data.state[0, i, j] = outer_value
    
    @staticmethod
    def sine_wave(data: FVMDataContainer2D, wave_number_x: float = 2.0, 
                  wave_number_y: float = 2.0, amplitude: float = 1.0) -> None:
        """Initialize with 2D sine wave"""
        geometry = data.geometry
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                data.state[0, i, j] = amplitude * np.sin(wave_number_x * np.pi * x) * \
                                     np.sin(wave_number_y * np.pi * y)
    
    @staticmethod
    def rotating_gaussian(data: FVMDataContainer2D, center_x: float = 0.5, 
                         center_y: float = 0.5, sigma: float = 0.1, 
                         amplitude: float = 1.0, rotation_angle: float = 0.0) -> None:
        """Initialize with rotated Gaussian for rotating flow tests"""
        geometry = data.geometry
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        for i in range(geometry.nx):
            for j in range(geometry.ny):
                x = geometry.x_min + (i + 0.5) * geometry.dx
                y = geometry.y_min + (j + 0.5) * geometry.dy
                
                # Rotate coordinates
                x_rot = cos_theta * (x - center_x) - sin_theta * (y - center_y)
                y_rot = sin_theta * (x - center_x) + cos_theta * (y - center_y)
                
                r_sq = x_rot**2 + y_rot**2
                data.state[0, i, j] = amplitude * np.exp(-r_sq / (2 * sigma**2))


class AdvectionTestCases:
    """Standard test cases for advection equation"""
    
    @staticmethod
    def solid_body_rotation(data: FVMDataContainer2D, omega: float = 1.0) -> Tuple[float, float]:
        """
        Set up solid body rotation velocity field
        
        Args:
            data: Data container
            omega: Angular velocity
            
        Returns:
            (vx, vy): Velocity field components as functions of position
        """
        def velocity_field(x, y):
            center_x = 0.5
            center_y = 0.5
            vx = -omega * (y - center_y)
            vy = omega * (x - center_x)
            return vx, vy
        
        return velocity_field
    
    @staticmethod
    def diagonal_advection() -> Tuple[float, float]:
        """Diagonal advection with unit velocity"""
        return 1.0, 1.0
    
    @staticmethod
    def deformation_flow(data: FVMDataContainer2D, time: float = 0.0) -> Tuple[float, float]:
        """
        Deformation flow field for testing scheme accuracy
        
        Args:
            data: Data container  
            time: Current time
            
        Returns:
            (vx, vy): Velocity components
        """
        def velocity_field(x, y):
            vx = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * time)
            vy = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * time)
            return vx, vy
            
        return velocity_field