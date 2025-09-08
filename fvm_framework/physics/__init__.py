"""
Physics modules for 2D finite volume method framework.

This package contains implementations of various physical systems
including Euler equations, magnetohydrodynamics, and other conservation laws.
All physics equations inherit from common base classes for consistency.
"""

# Base classes for unified interface
from .physics_base import PhysicsState, PhysicsEquations, ConservationLaw

# Specific physics implementations
from .euler_equations import EulerEquations2D, EulerState
from .mhd_equations import MHDEquations2D, MHDState
from .burgers_equation import BurgersEquation2D, BurgersState
from .advection_equation import AdvectionEquation2D, AdvectionState
from .sound_wave_equations import SoundWaveEquations2D, SoundWaveState

__all__ = [
    # Base classes
    'PhysicsState',
    'PhysicsEquations',
    'ConservationLaw',
    
    # Specific implementations
    'EulerEquations2D',
    'EulerState',
    'MHDEquations2D',
    'MHDState', 
    'BurgersEquation2D',
    'BurgersState',
    'AdvectionEquation2D',
    'AdvectionState',
    'SoundWaveEquations2D',
    'SoundWaveState'
]