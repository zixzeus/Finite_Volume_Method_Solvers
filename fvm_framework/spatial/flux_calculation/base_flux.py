"""
Base Classes for Flux Calculation Methods

This module defines the abstract base class that all numerical flux
calculation methods must inherit from, providing a unified interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from fvm_framework.core.data_container import FVMDataContainer2D


class FluxCalculator(ABC):
    """
    Abstract base class for all numerical flux calculation methods.
    
    Flux calculators compute numerical fluxes from left and right interface states
    that have been reconstructed by spatial reconstruction methods.
    """
    
    def __init__(self, name: str, flux_type: str = "generic"):
        self.name = name
        self.flux_type = flux_type  # 'lax_friedrichs', 'riemann_based', 'central', etc.
    
    @abstractmethod
    def compute_numerical_flux(self, left_state: np.ndarray, right_state: np.ndarray, 
                              physics_equation, direction: int, **kwargs) -> np.ndarray:
        """
        Compute numerical flux from left and right interface states.
        
        Args:
            left_state: Left interface state vector
            right_state: Right interface state vector  
            physics_equation: Physics equation object (Euler, MHD, etc.)
            direction: 0 for x-direction, 1 for y-direction
            **kwargs: Additional parameters (dt, gamma, etc.)
            
        Returns:
            Numerical flux vector at the interface
        """
        pass
    
    def compute_all_x_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all numerical fluxes in x-direction.
        
        Args:
            left_states: Left interface states, shape (num_vars, nx+1, ny)
            right_states: Right interface states, shape (num_vars, nx+1, ny)
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, nx+1, ny)
        """
        num_vars, nx_plus_1, ny = left_states.shape
        fluxes = np.zeros((num_vars, nx_plus_1, ny))
        
        for j in range(ny):
            for i in range(nx_plus_1):
                fluxes[:, i, j] = self.compute_numerical_flux(
                    left_states[:, i, j], right_states[:, i, j], 
                    physics_equation, direction=0, **kwargs
                )
        
        return fluxes
    
    def compute_all_y_fluxes(self, left_states: np.ndarray, right_states: np.ndarray,
                           physics_equation, **kwargs) -> np.ndarray:
        """
        Compute all numerical fluxes in y-direction.
        
        Args:
            left_states: Left interface states, shape (num_vars, nx, ny+1)
            right_states: Right interface states, shape (num_vars, nx, ny+1)
            physics_equation: Physics equation object
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, nx, ny+1)
        """
        num_vars, nx, ny_plus_1 = left_states.shape
        fluxes = np.zeros((num_vars, nx, ny_plus_1))
        
        for i in range(nx):
            for j in range(ny_plus_1):
                fluxes[:, i, j] = self.compute_numerical_flux(
                    left_states[:, i, j], right_states[:, i, j],
                    physics_equation, direction=1, **kwargs
                )
        
        return fluxes
    
    def compute_physical_flux(self, state: np.ndarray, physics_equation, 
                            direction: int, **kwargs) -> np.ndarray:
        """
        Compute physical flux for given state.
        
        Args:
            state: State vector
            physics_equation: Physics equation object
            direction: 0 for x-direction, 1 for y-direction
            **kwargs: Additional parameters
            
        Returns:
            Physical flux vector
        """
        if hasattr(physics_equation, 'compute_flux_x') and direction == 0:
            return physics_equation.compute_flux_x(state)
        elif hasattr(physics_equation, 'compute_flux_y') and direction == 1:
            return physics_equation.compute_flux_y(state)
        elif hasattr(physics_equation, 'compute_fluxes'):
            return physics_equation.compute_fluxes(state, direction)
        else:
            # Fallback for simple equations
            if direction == 0 and hasattr(physics_equation, 'velocity_x'):
                return np.array([state[0] * physics_equation.velocity_x])
            elif direction == 1 and hasattr(physics_equation, 'velocity_y'):
                return np.array([state[0] * physics_equation.velocity_y])
            else:
                return np.array([state[0]])
    
    def get_max_wave_speed(self, physics_equation, data: Optional[FVMDataContainer2D] = None) -> float:
        """
        Get maximum wave speed for this flux calculator.
        
        Args:
            physics_equation: Physics equation object
            data: Optional data container for wave speed computation
            
        Returns:
            Maximum wave speed
        """
        if hasattr(physics_equation, 'compute_max_wave_speed') and data is not None:
            return physics_equation.compute_max_wave_speed(data)
        else:
            # Fallback
            return 1.0
    
    def supports_physics(self, physics_type: str) -> bool:
        """
        Check if this flux calculator supports the given physics type.
        
        Args:
            physics_type: Type of physics equation
            
        Returns:
            True if physics type is supported
        """
        return True  # Default: support all physics types
    
    def needs_wave_speed(self) -> bool:
        """
        Check if this flux calculator needs wave speed information.
        
        Returns:
            True if wave speed is required
        """
        return False  # Override in subclasses if needed


class DissipativeFluxCalculator(FluxCalculator):
    """Base class for dissipative flux calculators (like Lax-Friedrichs)"""
    
    def __init__(self, name: str):
        super().__init__(name, "dissipative")
        
    def needs_wave_speed(self) -> bool:
        """Dissipative methods typically need wave speed"""
        return True


class RiemannBasedFluxCalculator(FluxCalculator):
    """Base class for Riemann solver-based flux calculators"""
    
    def __init__(self, name: str, riemann_solver_type: str):
        super().__init__(name, "riemann_based")
        self.riemann_solver_type = riemann_solver_type
        self.riemann_solver = None
        
    def _get_riemann_solver(self):
        """Lazy initialization of Riemann solver"""
        if self.riemann_solver is None:
            from ..riemann_solvers import RiemannSolverFactory
            self.riemann_solver = RiemannSolverFactory.create(self.riemann_solver_type)
        return self.riemann_solver


class CentralFluxCalculator(FluxCalculator):
    """Base class for central flux calculators"""
    
    def __init__(self, name: str):
        super().__init__(name, "central")
        
    def needs_wave_speed(self) -> bool:
        """Central methods usually don't need explicit wave speed"""
        return False