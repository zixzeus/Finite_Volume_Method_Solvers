"""
2D Linear Sound Wave Equations

This module implements the linearized 2D sound wave equations for acoustic wave propagation.
These equations are derived from the linearized Euler equations around a uniform state.

System of equations:
∂p/∂t + ρ₀c₀²(∂u/∂x + ∂v/∂y) = 0     (pressure perturbation)
∂u/∂t + (1/ρ₀)(∂p/∂x) = 0              (x-momentum)
∂v/∂t + (1/ρ₀)(∂p/∂y) = 0              (y-momentum)

Conservative variables: [p, u, v] where:
- p is pressure perturbation
- u, v are velocity perturbations
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from .physics_base import PhysicsState, ConservationLaw


@dataclass
class SoundWaveState(PhysicsState):
    """Structure for sound wave primitive variables"""
    pressure: float
    velocity_x: float
    velocity_y: float
    sound_speed: Optional[float] = None
    
    def to_array(self) -> np.ndarray:
        """Convert to array: [p, u, v]"""
        return np.array([self.pressure, self.velocity_x, self.velocity_y])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'SoundWaveState':
        """Create SoundWaveState from array [p, u, v]"""
        return cls(
            pressure=array[0],
            velocity_x=array[1],
            velocity_y=array[2]
        )
    
    def copy(self) -> 'SoundWaveState':
        """Create a copy of the state"""
        return SoundWaveState(
            pressure=self.pressure,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            sound_speed=self.sound_speed
        )
    
    def validate(self) -> bool:
        """Validate physical consistency"""
        return (not np.isnan(self.pressure) and
                not np.isnan(self.velocity_x) and
                not np.isnan(self.velocity_y))


class SoundWaveEquations2D(ConservationLaw):
    """
    2D Linear Sound Wave Equations implementation.
    
    This system describes small-amplitude acoustic waves propagating
    in a uniform medium with density ρ₀ and sound speed c₀.
    """
    
    def __init__(self, sound_speed: float = 1.0, density: float = 1.0):
        """
        Initialize sound wave equations.
        
        Args:
            sound_speed: Reference sound speed c₀
            density: Reference density ρ₀
        """
        super().__init__("2D Linear Sound Wave Equations", num_variables=3, num_dimensions=2)
        self.sound_speed = sound_speed
        self.density = density
    
    def conservative_to_primitive(self, u: np.ndarray) -> SoundWaveState:
        """
        Convert conservative to primitive variables.
        For linear sound waves, they are the same.
        
        Args:
            u: Conservative variables [p, u, v]
            
        Returns:
            SoundWaveState with primitive variables
        """
        return SoundWaveState(
            pressure=u[0],
            velocity_x=u[1], 
            velocity_y=u[2],
            sound_speed=self.sound_speed
        )
    
    def primitive_to_conservative(self, state: SoundWaveState) -> np.ndarray:
        """
        Convert primitive to conservative variables.
        For linear sound waves, they are the same.
        
        Args:
            state: SoundWaveState with primitive variables
            
        Returns:
            Conservative variables [p, u, v]
        """
        return np.array([state.pressure, state.velocity_x, state.velocity_y])
    
    def compute_flux_x(self, u: np.ndarray) -> np.ndarray:
        """
        Compute flux vector in x-direction.
        
        Args:
            u: Conservative variables [p, u, v]
            
        Returns:
            Flux vector in x-direction
        """
        p, u_vel, v_vel = u[0], u[1], u[2]
        
        flux_x = np.array([
            self.density * self.sound_speed**2 * u_vel,  # Pressure flux: ρ₀c₀²u
            p / self.density,                            # x-momentum flux: p/ρ₀
            0.0                                          # y-momentum flux: 0
        ])
        
        return flux_x
    
    def compute_flux_y(self, u: np.ndarray) -> np.ndarray:
        """
        Compute flux vector in y-direction.
        
        Args:
            u: Conservative variables [p, u, v]
            
        Returns:
            Flux vector in y-direction
        """
        p, u_vel, v_vel = u[0], u[1], u[2]
        
        flux_y = np.array([
            self.density * self.sound_speed**2 * v_vel,  # Pressure flux: ρ₀c₀²v
            0.0,                                         # x-momentum flux: 0
            p / self.density                             # y-momentum flux: p/ρ₀
        ])
        
        return flux_y
    
    def max_wave_speed(self, state: np.ndarray, direction: int) -> float:
        """Compute maximum wave speed (required by base class)"""
        # For linear sound waves, max speed is just the sound speed
        # plus any convective velocity
        p, u, v = state
        if direction == 0:
            return self.sound_speed + abs(u)
        else:
            return self.sound_speed + abs(v)
    
    def compute_eigenvalues(self, u: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute eigenvalues of flux Jacobian matrix.
        
        Args:
            u: Conservative variables [p, u, v]
            direction: 0 for x-direction, 1 for y-direction
            
        Returns:
            Array of eigenvalues (3 values for sound waves)
        """
        state = self.conservative_to_primitive(u)
        c = self.sound_speed
        
        if direction == 0:  # x-direction
            u_vel = state.velocity_x
            eigenvals = np.array([
                u_vel - c,  # Left-going sound wave
                u_vel,      # Entropy/vorticity wave
                u_vel + c   # Right-going sound wave
            ])
        else:  # y-direction
            v_vel = state.velocity_y
            eigenvals = np.array([
                v_vel - c,  # Left-going sound wave
                v_vel,      # Entropy/vorticity wave
                v_vel + c   # Right-going sound wave
            ])
        
        return eigenvals
    
    def compute_max_wave_speed(self, data) -> float:
        """
        Compute maximum wave speed for CFL condition.
        
        For linear sound waves, this is simply the sound speed.
        """
        if hasattr(data, 'state') and data.state.shape[0] >= 3:
            # Account for convective velocity
            max_u = np.max(np.abs(data.state[1]))  # max |u|
            max_v = np.max(np.abs(data.state[2]))  # max |v|
            max_convective = max(max_u, max_v)
            return self.sound_speed + max_convective
        
        return self.sound_speed
    
    def compute_fluxes(self, state: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute fluxes for given direction (unified interface).
        
        Args:
            state: Conservative variables
            direction: 0 for x-direction, 1 for y-direction
            
        Returns:
            Flux vector
        """
        if direction == 0:
            return self.compute_flux_x(state)
        else:
            return self.compute_flux_y(state)
    
    def apply_source_terms(self, data, dt: float, **kwargs) -> None:
        """Apply source terms (none for linear sound waves)"""
        pass
    
    def get_variable_names(self) -> list:
        """Get names of conservative variables"""
        return ['pressure', 'velocity_x', 'velocity_y']
    
    def get_primitive_names(self) -> list:
        """Get names of primitive variables"""
        return ['pressure', 'velocity_x', 'velocity_y']


