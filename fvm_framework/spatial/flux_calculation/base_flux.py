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
    
    def compute_numerical_flux_vectorized(self, left_states: np.ndarray, right_states: np.ndarray,
                                        physics_equation, direction: int, **kwargs) -> np.ndarray:
        """
        Vectorized computation of numerical fluxes.
        
        Default implementation falls back to loop. Subclasses should override
        this method for better performance.
        
        Args:
            left_states: Left interface states, shape (num_vars, num_interfaces)
            right_states: Right interface states, shape (num_vars, num_interfaces)
            physics_equation: Physics equation object
            direction: 0 for x-direction, 1 for y-direction
            **kwargs: Additional parameters
            
        Returns:
            Numerical fluxes, shape (num_vars, num_interfaces)
        """
        num_vars, num_interfaces = left_states.shape
        fluxes = np.zeros((num_vars, num_interfaces))
        
        for i in range(num_interfaces):
            fluxes[:, i] = self.compute_numerical_flux(
                left_states[:, i], right_states[:, i], 
                physics_equation, direction, **kwargs
            )
        
        return fluxes
    
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
        
        # Try to use vectorized computation
        for j in range(ny):
            # Reshape for vectorized computation: (num_vars, nx+1)
            left_j = left_states[:, :, j]
            right_j = right_states[:, :, j]
            
            # Use vectorized method
            fluxes[:, :, j] = self.compute_numerical_flux_vectorized(
                left_j, right_j, physics_equation, direction=0, **kwargs
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
        
        # Try to use vectorized computation
        for i in range(nx):
            # Reshape for vectorized computation: (num_vars, ny+1)
            left_i = left_states[:, i, :]
            right_i = right_states[:, i, :]
            
            # Use vectorized method
            fluxes[:, i, :] = self.compute_numerical_flux_vectorized(
                left_i, right_i, physics_equation, direction=1, **kwargs
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
    
    def compute_fluxes(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Unified interface to compute fluxes using FVMDataContainer2D.
        
        This method provides a unified interface that takes the data container
        and computes fluxes, storing them directly in the data container.
        
        Args:
            data: FVMDataContainer2D containing interface states and flux arrays
            **kwargs: Additional parameters including physics_equation
        """
        # Get physics equation from kwargs
        physics_equation = kwargs.get('physics_equation')
        if physics_equation is None:
            raise ValueError("FluxCalculator requires physics_equation in kwargs")
        
        # Check for interface states
        if data.interface_states_x is None or data.interface_states_y is None:
            raise ValueError("FluxCalculator requires interface states from ReconstructionStage")
        
        x_left, x_right = data.interface_states_x
        y_left, y_right = data.interface_states_y
        
        # Remove physics_equation from kwargs to avoid duplicate arguments
        flux_kwargs = {k: v for k, v in kwargs.items() if k != 'physics_equation'}
        # Add data parameter for wave speed computation
        flux_kwargs['data'] = data
        
        # Compute fluxes using existing methods
        interior_flux_x = self.compute_all_x_fluxes(
            x_left, x_right, physics_equation, **flux_kwargs
        )
        interior_flux_y = self.compute_all_y_fluxes(
            y_left, y_right, physics_equation, **flux_kwargs
        )
        
        # Store in data container flux arrays (map to ghost cell flux arrays)
        ng = data.ng
        data.flux_x[:, ng:ng+data.nx+1, ng:ng+data.ny] = interior_flux_x
        data.flux_y[:, ng:ng+data.nx, ng:ng+data.ny+1] = interior_flux_y


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
            # Try to use new generic Riemann solver first
            from ..riemann_solvers import RiemannSolverFactory
            self.riemann_solver = RiemannSolverFactory.create(self.riemann_solver_type)
        return self.riemann_solver

