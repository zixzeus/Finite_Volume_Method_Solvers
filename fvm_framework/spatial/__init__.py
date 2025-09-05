"""
Spatial Discretization Module

This module provides a unified framework for all spatial discretization methods
including finite volume, Riemann solver-based, and discontinuous Galerkin methods.
"""

from .factory import SpatialDiscretizationFactory, create_spatial_scheme
from .base import (
    SpatialDiscretization,
    FiniteVolumeScheme,
    RiemannBasedScheme,
    HighOrderScheme,
    DiscontinuousGalerkinScheme
)

# Import specific schemes
from .lax_friedrichs import LaxFriedrichsScheme
from .tvd_lax_friedrichs import TVDLFScheme
from .riemann_schemes import HLLRiemannScheme, HLLCRiemannScheme, HLLDRiemannScheme
from .dg_scheme import DGScheme

__all__ = [
    # Factory
    'SpatialDiscretizationFactory',
    'create_spatial_scheme',
    
    # Base classes
    'SpatialDiscretization',
    'FiniteVolumeScheme', 
    'RiemannBasedScheme',
    'HighOrderScheme',
    'DiscontinuousGalerkinScheme',
    
    # Finite volume schemes
    'LaxFriedrichsScheme',
    'TVDLFScheme',
    
    # Riemann solver schemes
    'HLLRiemannScheme',
    'HLLCRiemannScheme', 
    'HLLDRiemannScheme',
    
    # High-order schemes
    'DGScheme',
]