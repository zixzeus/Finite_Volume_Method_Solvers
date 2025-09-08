"""
2D Linear Advection Equation

This module implements the 2D linear advection equation, which is the simplest
hyperbolic PDE and serves as a benchmark for testing numerical schemes.

Equation: ∂u/∂t + a∂u/∂x + b∂u/∂y = 0

Conservative variable: [u] where u is the scalar quantity being advected
"""

import numpy as np
from dataclasses import dataclass
from fvm_framework.core.data_container import FVMDataContainer2D
from .physics_base import PhysicsState, ConservationLaw


@dataclass
class AdvectionState(PhysicsState):
    """Structure for advection equation variables"""
    scalar: float
    advection_x: float = 1.0
    advection_y: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to array: [u]"""
        return np.array([self.scalar])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'AdvectionState':
        """Create AdvectionState from array [u]"""
        return cls(scalar=array[0])
    
    def copy(self) -> 'AdvectionState':
        """Create a copy of the state"""
        return AdvectionState(
            scalar=self.scalar,
            advection_x=self.advection_x,
            advection_y=self.advection_y
        )
    
    def validate(self) -> bool:
        """Validate physical consistency"""
        return not np.isnan(self.scalar)


class AdvectionEquation2D(ConservationLaw):
    """2D linear advection equation implementation"""
    
    def __init__(self, advection_x: float = 1.0, advection_y: float = 0.0):
        super().__init__("2D Linear Advection Equation", num_variables=1, num_dimensions=2)
        self.advection_x = advection_x  # Advection velocity in x-direction
        self.advection_y = advection_y  # Advection velocity in y-direction
    
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
    
    def get_variable_names(self) -> list:
        """Get names of conservative variables"""
        return ['scalar']
    
    def get_primitive_names(self) -> list:
        """Get names of primitive variables"""
        return ['scalar']


