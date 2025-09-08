"""
2D Burgers' Equation

This module implements the 2D Burgers' equation, which is a fundamental
nonlinear PDE used for testing numerical methods.

Equation: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
          ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v

Conservative variables: [u, v] where u,v are velocity components
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from fvm_framework.core.data_container import FVMDataContainer2D
from .physics_base import PhysicsState, ConservationLaw


@dataclass
class BurgersState(PhysicsState):
    """Structure for Burgers equation variables"""
    u_velocity: float
    v_velocity: float
    viscosity: Optional[float] = 0.01
    
    def to_array(self) -> np.ndarray:
        """Convert to array: [u, v]"""
        return np.array([self.u_velocity, self.v_velocity])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'BurgersState':
        """Create BurgersState from array [u, v]"""
        return cls(u_velocity=array[0], v_velocity=array[1])
    
    def copy(self) -> 'BurgersState':
        """Create a copy of the state"""
        return BurgersState(
            u_velocity=self.u_velocity,
            v_velocity=self.v_velocity,
            viscosity=self.viscosity
        )
    
    def validate(self) -> bool:
        """Validate physical consistency"""
        return (not np.isnan(self.u_velocity) and 
                not np.isnan(self.v_velocity))


class BurgersEquation2D(ConservationLaw):
    """2D Burgers' equation implementation"""
    
    def __init__(self, viscosity: float = 0.01):
        super().__init__("2D Burgers Equation", num_variables=2, num_dimensions=2)
        self.viscosity = viscosity
    
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
    
    def get_variable_names(self) -> list:
        """Get names of conservative variables"""
        return ['u_velocity', 'v_velocity']
    
    def get_primitive_names(self) -> list:
        """Get names of primitive variables"""
        return ['u_velocity', 'v_velocity']


