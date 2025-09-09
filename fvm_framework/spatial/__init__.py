"""
Spatial Discretization Module

This module provides a modular framework for spatial discretization methods
using separate reconstruction and flux calculation components.
"""

from .factory import SpatialDiscretizationFactory, create_spatial_scheme
from .base import SpatialDiscretization
from .modular_spatial_scheme import ModularSpatialScheme

# Import modular components
from . import reconstruction
from . import flux_calculation

__all__ = [
    # Factory
    'SpatialDiscretizationFactory',
    'create_spatial_scheme',
    
    # Base classes
    'SpatialDiscretization',
    
    # Modular scheme
    'ModularSpatialScheme',
    
    # Component modules
    'reconstruction',
    'flux_calculation',
]