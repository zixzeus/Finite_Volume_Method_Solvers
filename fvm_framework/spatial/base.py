"""
Base Classes for Spatial Discretization Methods

This module defines the abstract base classes that all spatial discretization 
methods must inherit from, providing a unified interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from core.data_container import FVMDataContainer2D


class SpatialDiscretization(ABC):
    """
    Abstract base class for all spatial discretization methods.
    
    This provides a unified interface for:
    - Finite volume schemes (Lax-Friedrichs, TVDLF, etc.)
    - Riemann solver-based methods (HLL, HLLC, HLLD)
    - High-order methods (DG, WENO, etc.)
    """
    
    def __init__(self, name: str, order: int = 1, scheme_type: str = "generic"):
        self.name = name
        self.order = order  # Spatial accuracy order
        self.scheme_type = scheme_type  # 'finite_volume', 'riemann_based', 'discontinuous_galerkin', etc.
    
    @abstractmethod
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """
        Compute numerical fluxes at cell interfaces.
        
        Args:
            data: FVM data container with state and geometry
            physics_equation: Physics equation object (Euler, MHD, etc.)
            **kwargs: Additional parameters (dt, gamma, etc.)
        """
        pass
    
    def get_max_wave_speed(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> float:
        """
        Get maximum wave speed for CFL condition.
        Default implementation - can be overridden by subclasses.
        """
        if hasattr(physics_equation, 'compute_max_wave_speed'):
            return physics_equation.compute_max_wave_speed(data)
        else:
            # Fallback for simple cases
            return 1.0
    
    def needs_reconstruction(self) -> bool:
        """Return True if scheme needs state reconstruction"""
        return self.order > 1
    
    def supports_physics(self, physics_type: str) -> bool:
        """Check if this scheme supports the given physics type"""
        # Default: support all physics types
        return True
    
    def compute_flux_divergence(self, data: FVMDataContainer2D) -> np.ndarray:
        """
        Compute flux divergence for time integration.
        Default implementation for finite volume methods.
        
        Returns residual = -∇·F for explicit time stepping.
        """
        residual = np.zeros_like(data.state)
        
        # Interior points only
        for i in range(1, data.nx - 1):
            for j in range(1, data.ny - 1):
                # Flux divergence: ∇·F = (F_{i+1/2} - F_{i-1/2})/dx + (G_{j+1/2} - G_{j-1/2})/dy
                flux_div_x = (data.flux_x[:, i+1, j] - data.flux_x[:, i, j]) / data.geometry.dx
                flux_div_y = (data.flux_y[:, i, j+1] - data.flux_y[:, i, j]) / data.geometry.dy
                residual[:, i, j] = -(flux_div_x + flux_div_y)
        
        return residual


class FiniteVolumeScheme(SpatialDiscretization):
    """Base class for finite volume schemes"""
    
    def __init__(self, name: str, order: int = 1):
        super().__init__(name, order, "finite_volume")


class RiemannBasedScheme(SpatialDiscretization):
    """Base class for Riemann solver-based schemes"""
    
    def __init__(self, name: str, riemann_solver_type: str, order: int = 1):
        super().__init__(name, order, "riemann_based")
        self.riemann_solver_type = riemann_solver_type
        self.riemann_solver = None
        
    def _get_riemann_solver(self):
        """Lazy initialization of Riemann solver"""
        if self.riemann_solver is None:
            from .riemann_solvers import RiemannSolverFactory
            self.riemann_solver = RiemannSolverFactory.create(self.riemann_solver_type)
        return self.riemann_solver


class HighOrderScheme(SpatialDiscretization):
    """Base class for high-order schemes like DG, WENO, etc."""
    
    def __init__(self, name: str, order: int = 2):
        super().__init__(name, order, "high_order")
        
    def needs_reconstruction(self) -> bool:
        """High-order schemes always need reconstruction"""
        return True


class DiscontinuousGalerkinScheme(HighOrderScheme):
    """Base class for Discontinuous Galerkin schemes"""
    
    def __init__(self, name: str, polynomial_order: int = 1):
        super().__init__(name, polynomial_order + 1)  # Formal order of accuracy
        self.scheme_type = "discontinuous_galerkin"
        self.polynomial_order = polynomial_order