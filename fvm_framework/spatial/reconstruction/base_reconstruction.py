"""
Base Classes for Spatial Reconstruction Methods

This module defines the abstract base class that all spatial reconstruction 
methods must inherit from, providing a unified interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from fvm_framework.core.data_container import FVMDataContainer2D


class ReconstructionScheme(ABC):
    """
    Abstract base class for all spatial reconstruction methods.
    
    Reconstruction methods compute left and right interface states from 
    cell-centered values. These interface states are then used by flux
    calculators to compute numerical fluxes.
    """
    
    def __init__(self, name: str, order: int = 1):
        self.name = name
        self.order = order  # Spatial accuracy order
    
    @abstractmethod
    def reconstruct_x_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct left and right states at x-interfaces.
        
        Args:
            data: FVM data container with state and geometry
            
        Returns:
            Tuple of (left_states, right_states) at x-interfaces
            left_states: shape (num_vars, nx+1, ny) - left states at x-interfaces
            right_states: shape (num_vars, nx+1, ny) - right states at x-interfaces
        """
        pass
    
    @abstractmethod
    def reconstruct_y_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct left and right states at y-interfaces.
        
        Args:
            data: FVM data container with state and geometry
            
        Returns:
            Tuple of (left_states, right_states) at y-interfaces
            left_states: shape (num_vars, nx, ny+1) - left states at y-interfaces  
            right_states: shape (num_vars, nx, ny+1) - right states at y-interfaces
        """
        pass
    
    def reconstruct_all_interfaces(self, data: FVMDataContainer2D) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Reconstruct states at all interfaces.
        
        Args:
            data: FVM data container
            
        Returns:
            Tuple of ((x_left, x_right), (y_left, y_right))
        """
        x_interfaces = self.reconstruct_x_interfaces(data)
        y_interfaces = self.reconstruct_y_interfaces(data)
        return x_interfaces, y_interfaces
    
    def get_stencil_size(self) -> int:
        """
        Get the stencil size required for this reconstruction method.
        
        Returns:
            Number of cells required on each side for reconstruction
        """
        # Default: first-order needs only immediate neighbors
        return max(1, (self.order + 1) // 2)
    
    def needs_boundary_treatment(self) -> bool:
        """
        Check if this reconstruction method needs special boundary treatment.
        
        Returns:
            True if special boundary handling is required
        """
        return self.order > 1
    
    def supports_variable(self, var_index: int) -> bool:
        """
        Check if reconstruction method supports specific variable.
        
        Args:
            var_index: Index of variable to check
            
        Returns:
            True if variable is supported (default: all variables supported)
        """
        return True


class FirstOrderReconstruction(ReconstructionScheme):
    """Base class for first-order reconstruction methods"""
    
    def __init__(self, name: str):
        super().__init__(name, order=1)


class SecondOrderReconstruction(ReconstructionScheme):
    """Base class for second-order reconstruction methods"""
    
    def __init__(self, name: str):
        super().__init__(name, order=2)
        
    def needs_boundary_treatment(self) -> bool:
        return True


class HighOrderReconstruction(ReconstructionScheme):
    """Base class for high-order reconstruction methods"""
    
    def __init__(self, name: str, order: int):
        super().__init__(name, order=order)
        
    def needs_boundary_treatment(self) -> bool:
        return True
    
    def get_stencil_size(self) -> int:
        return (self.order + 1) // 2


class LimiterMixin:
    """
    Mixin class for reconstruction methods that use slope/flux limiters.
    """
    
    def __init__(self, limiter_type: str = 'minmod'):
        self.limiter_type = limiter_type
        self.limiter_function = self._get_limiter_function()
    
    def _get_limiter_function(self):
        """Get the limiter function"""
        limiters = {
            'minmod': self._minmod_limiter,
            'superbee': self._superbee_limiter,
            'van_leer': self._van_leer_limiter,
            'mc': self._mc_limiter,
            'none': lambda a, b: 0.5 * (a + b)  # No limiting (central difference)
        }
        return limiters.get(self.limiter_type, self._minmod_limiter)
    
    def _minmod_limiter(self, a: float, b: float) -> float:
        """MinMod limiter - most dissipative"""
        if a * b <= 0:
            return 0.0
        elif abs(a) < abs(b):
            return a
        else:
            return b
    
    def _superbee_limiter(self, a: float, b: float) -> float:
        """Superbee limiter - least dissipative"""
        if a * b <= 0:
            return 0.0
        elif abs(a) > abs(b):
            return 2.0 * b if abs(b) < 0.5 * abs(a) else (a + b) if abs(a) < 2.0 * abs(b) else 2.0 * a
        else:
            return 2.0 * a if abs(a) < 0.5 * abs(b) else (a + b) if abs(b) < 2.0 * abs(a) else 2.0 * b
    
    def _van_leer_limiter(self, a: float, b: float) -> float:
        """Van Leer limiter - smooth"""
        if a * b <= 0:
            return 0.0
        else:
            return 2.0 * a * b / (a + b)
    
    def _mc_limiter(self, a: float, b: float) -> float:
        """Monotonized Central limiter"""
        if a * b <= 0:
            return 0.0
        else:
            return min(2.0 * abs(a), 2.0 * abs(b), 0.5 * abs(a + b)) * np.sign(a)