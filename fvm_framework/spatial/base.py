"""
Base Classes for Spatial Discretization

This module defines the abstract base class for all spatial discretization methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from fvm_framework.core.data_container import FVMDataContainer2D


class SpatialDiscretization(ABC):
    """
    Abstract base class for all spatial discretization methods.
    
    This class defines the interface that all spatial schemes must implement,
    whether they are finite volume, discontinuous Galerkin, or other methods.
    """
    
    def __init__(self, name: str, order: int = 1, scheme_type: str = "unknown"):
        """
        Initialize spatial discretization method.
        
        Args:
            name: Name of the discretization method
            order: Spatial accuracy order
            scheme_type: Type of scheme (e.g., "finite_volume", "discontinuous_galerkin")
        """
        self.name = name
        self.order = order
        self.scheme_type = scheme_type
    
    @abstractmethod
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """
        Compute numerical fluxes for the spatial discretization.
        
        Args:
            data: FVM data container with state and geometry
            physics_equation: Physics equation object (Euler, MHD, etc.)
            **kwargs: Additional parameters (dt, gamma, etc.)
        """
        pass
    
    def get_max_wave_speed(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> float:
        """
        Get maximum wave speed for CFL condition.
        
        Args:
            data: FVM data container
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Maximum wave speed
        """
        return physics_equation.compute_max_wave_speed(data)
    
    def supports_physics(self, physics_type: str) -> bool:
        """
        Check if this scheme supports a given physics type.
        
        Args:
            physics_type: Physics equation type (e.g., "euler", "mhd")
            
        Returns:
            True if physics type is supported
        """
        return True  # Default: support all physics types
    
    def get_stencil_size(self) -> int:
        """
        Get stencil size required by this scheme.
        
        Returns:
            Number of cells required on each side
        """
        return max(1, (self.order + 1) // 2)
    
    def needs_boundary_treatment(self) -> bool:
        """
        Check if this scheme needs special boundary treatment.
        
        Returns:
            True if special boundary handling is required
        """
        return self.order > 1
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.name}', order={self.order})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"{self.__class__.__name__}(name='{self.name}', "
               f"order={self.order}, scheme_type='{self.scheme_type}')")