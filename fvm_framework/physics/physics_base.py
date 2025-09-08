"""
Base Classes for Physics Equations

This module defines abstract base classes for physics states and equations,
providing a unified interface for all physics implementations in the framework.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class PhysicsState(ABC):
    """
    Abstract base class for physics state variables.
    
    Defines the interface for converting between primitive and conservative
    variables, and provides common functionality for all physics states.
    """
    
    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array"""
        pass
    
    @classmethod
    @abstractmethod
    def from_array(cls, array: np.ndarray) -> 'PhysicsState':
        """Create state from numpy array"""
        pass
    
    @abstractmethod
    def copy(self) -> 'PhysicsState':
        """Create a copy of the state"""
        pass
    
    def validate(self) -> bool:
        """Validate physical consistency of state variables"""
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_array()})"


class PhysicsEquations(ABC):
    """
    Abstract base class for physics equations.
    
    Defines the standard interface that all physics equation implementations
    must follow for integration with the FVM framework.
    """
    
    def __init__(self, name: str, num_variables: int):
        """
        Initialize physics equations.
        
        Args:
            name: Human-readable name of the equations
            num_variables: Number of conservative variables
        """
        self.name = name
        self.num_variables = num_variables
    
    # Core physics methods (required)
    
    @abstractmethod
    def conservative_to_primitive(self, conservative: np.ndarray) -> np.ndarray:
        """
        Convert conservative to primitive variables.
        
        Args:
            conservative: Conservative variables array
            
        Returns:
            Primitive variables array
        """
        pass
    
    @abstractmethod
    def primitive_to_conservative(self, primitive: np.ndarray) -> np.ndarray:
        """
        Convert primitive to conservative variables.
        
        Args:
            primitive: Primitive variables array
            
        Returns:
            Conservative variables array
        """
        pass
    
    @abstractmethod
    def compute_fluxes(self, state: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute numerical fluxes.
        
        Args:
            state: Conservative variables
            direction: Spatial direction (0=x, 1=y, 2=z)
            
        Returns:
            Flux vector
        """
        pass
    
    @abstractmethod
    def max_wave_speed(self, state: np.ndarray, direction: int) -> float:
        """
        Compute maximum characteristic wave speed.
        
        Args:
            state: Conservative variables
            direction: Spatial direction (0=x, 1=y, 2=z)
            
        Returns:
            Maximum wave speed
        """
        pass
    
    # Optional physics methods (with default implementations)
    
    def compute_eigenvalues(self, state: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute characteristic eigenvalues (wave speeds).
        
        Args:
            state: Conservative variables
            direction: Spatial direction (0=x, 1=y, 2=z)
            
        Returns:
            Array of eigenvalues
        """
        # Default: return max wave speed repeated
        max_speed = self.max_wave_speed(state, direction)
        return np.full(self.num_variables, max_speed)
    
    def apply_source_terms(self, data, dt: float, **kwargs) -> None:
        """
        Apply source terms to the solution.
        
        Args:
            data: FVM data container
            dt: Time step size
            **kwargs: Additional parameters
        """
        # Default: no source terms
        pass
    
    def compute_max_wave_speed(self, data) -> float:
        """
        Compute maximum wave speed over entire domain.
        
        Args:
            data: FVM data container
            
        Returns:
            Maximum wave speed
        """
        if hasattr(data, 'get_interior_state'):
            interior_state = data.get_interior_state()
        elif hasattr(data, 'state'):
            interior_state = data.state
        else:
            return 1.0
            
        max_speed = 0.0
        
        # Check both x and y directions
        for direction in [0, 1]:
            if interior_state.ndim == 3:  # (num_vars, nx, ny)
                for i in range(interior_state.shape[1]):
                    for j in range(interior_state.shape[2]):
                        state_ij = interior_state[:, i, j]
                        speed = self.max_wave_speed(state_ij, direction)
                        max_speed = max(max_speed, abs(speed))
            else:
                # Flatten and check each state
                states = interior_state.reshape(self.num_variables, -1)
                for k in range(states.shape[1]):
                    state_k = states[:, k]
                    speed = self.max_wave_speed(state_k, direction)
                    max_speed = max(max_speed, abs(speed))
        
        return max_speed if max_speed > 1e-15 else 1.0
    
    # Utility methods
    
    def get_variable_names(self) -> list:
        """Get names of conservative variables"""
        return [f"var_{i}" for i in range(self.num_variables)]
    
    def get_primitive_names(self) -> list:
        """Get names of primitive variables"""
        return [f"prim_{i}" for i in range(self.num_variables)]
    
    def validate_state(self, state: np.ndarray) -> bool:
        """
        Validate physical consistency of state.
        
        Args:
            state: Conservative variables
            
        Returns:
            True if state is physically valid
        """
        # Basic checks
        if state.size != self.num_variables:
            return False
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return False
            
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this physics implementation"""
        return {
            'name': self.name,
            'num_variables': self.num_variables,
            'variable_names': self.get_variable_names(),
            'primitive_names': self.get_primitive_names(),
            'has_pressure': hasattr(self, 'compute_pressure'),
            'has_temperature': hasattr(self, 'compute_temperature')
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', num_vars={self.num_variables})"


class ConservationLaw(PhysicsEquations):
    """
    Base class for hyperbolic conservation laws.
    
    Specialized base class for physics that follow the conservation law form:
    ∂U/∂t + ∇·F(U) = S(U)
    """
    
    def __init__(self, name: str, num_variables: int, num_dimensions: int = 2):
        """
        Initialize conservation law.
        
        Args:
            name: Name of the conservation law
            num_variables: Number of conservative variables
            num_dimensions: Spatial dimensions (default: 2)
        """
        super().__init__(name, num_variables)
        self.num_dimensions = num_dimensions
    
    def compute_flux_jacobian(self, state: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute flux Jacobian matrix ∂F/∂U.
        
        Args:
            state: Conservative variables
            direction: Spatial direction
            
        Returns:
            Jacobian matrix (num_vars × num_vars)
        """
        # Default: finite difference approximation
        eps = 1e-8
        jacobian = np.zeros((self.num_variables, self.num_variables))
        
        f0 = self.compute_fluxes(state, direction)
        
        for i in range(self.num_variables):
            state_pert = state.copy()
            state_pert[i] += eps
            f_pert = self.compute_fluxes(state_pert, direction)
            jacobian[:, i] = (f_pert - f0) / eps
            
        return jacobian
    
    def is_hyperbolic(self, state: np.ndarray, direction: int) -> bool:
        """
        Check if system is hyperbolic at given state.
        
        Args:
            state: Conservative variables
            direction: Spatial direction
            
        Returns:
            True if system has real eigenvalues
        """
        try:
            jacobian = self.compute_flux_jacobian(state, direction)
            eigenvals = np.linalg.eigvals(jacobian)
            return bool(np.all(np.isreal(eigenvals)))
        except:
            return False