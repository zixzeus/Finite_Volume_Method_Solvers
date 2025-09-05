"""
Physics modules for 2D finite volume method framework.

This package contains implementations of various physical systems
including Euler equations, magnetohydrodynamics, and other conservation laws.
"""

from .euler_equations import EulerEquations2D
from .mhd_equations import MHDEquations2D
from .burgers_equation import BurgersEquation2D
from .advection_equation import AdvectionEquation2D

__all__ = [
    'EulerEquations2D',
    'MHDEquations2D', 
    'BurgersEquation2D',
    'AdvectionEquation2D'
]