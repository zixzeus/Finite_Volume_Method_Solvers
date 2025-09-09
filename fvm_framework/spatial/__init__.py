"""
Spatial Discretization Module

This module provides a modular framework for spatial discretization methods
using separate reconstruction and flux calculation components.
"""

# Import modular components
from . import reconstruction
from . import flux_calculation

__all__ = [
    # Component modules
    'reconstruction',
    'flux_calculation',
]