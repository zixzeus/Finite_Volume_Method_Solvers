"""
Boundary Conditions for 2D Finite Volume Method

This module implements various boundary condition types for FVM simulations,
including periodic, reflective, transmissive, and custom boundary conditions.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import numpy as np
from core.data_container import FVMDataContainer2D

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Apply boundary condition to the data"""
        pass


class PeriodicBC(BoundaryCondition):
    """Periodic boundary conditions"""
    
    def __init__(self):
        super().__init__("Periodic")
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Apply periodic boundary conditions.
        For periodic BCs, ghost cells are filled with values from opposite boundary.
        """
        # Note: For interior-only computations, periodic BCs are handled
        # automatically by the flux computation through modular arithmetic
        # This implementation is for explicit ghost cell filling if needed
        pass


class ReflectiveBC(BoundaryCondition):
    """Reflective (wall) boundary conditions"""
    
    def __init__(self, wall_velocity: Optional[np.ndarray] = None):
        super().__init__("Reflective")
        if wall_velocity is None:
            self.wall_velocity = np.zeros(3)
        else:
            self.wall_velocity = wall_velocity
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Apply reflective boundary conditions.
        Normal velocity component is reflected, tangential components preserved.
        """
        # Left and right boundaries (x-normal)
        self._apply_x_wall(data, 0, 1, -1)      # Left wall
        self._apply_x_wall(data, -1, -2, -1)    # Right wall
        
        # Bottom and top boundaries (y-normal)
        self._apply_y_wall(data, 0, 1, -1)      # Bottom wall
        self._apply_y_wall(data, -1, -2, -1)    # Top wall
    
    def _apply_x_wall(self, data: FVMDataContainer2D, boundary_idx: int, 
                      interior_idx: int, normal_sign: int):
        """Apply reflective BC for x-normal wall"""
        # Copy interior values
        data.state[:, boundary_idx, :] = data.state[:, interior_idx, :]
        
        # Reflect x-momentum (normal component)
        data.state[1, boundary_idx, :] *= normal_sign
        
        # Add wall velocity contribution if moving wall
        if np.any(self.wall_velocity != 0):
            rho = data.state[0, boundary_idx, :]
            data.state[1, boundary_idx, :] = rho * self.wall_velocity[0]
            data.state[2, boundary_idx, :] = rho * self.wall_velocity[1]
            data.state[3, boundary_idx, :] = rho * self.wall_velocity[2]
    
    def _apply_y_wall(self, data: FVMDataContainer2D, boundary_idx: int,
                      interior_idx: int, normal_sign: int):
        """Apply reflective BC for y-normal wall"""
        # Copy interior values
        data.state[:, :, boundary_idx] = data.state[:, :, interior_idx]
        
        # Reflect y-momentum (normal component)
        data.state[2, :, boundary_idx] *= normal_sign
        
        # Add wall velocity contribution if moving wall
        if np.any(self.wall_velocity != 0):
            rho = data.state[0, :, boundary_idx]
            data.state[1, :, boundary_idx] = rho * self.wall_velocity[0]
            data.state[2, :, boundary_idx] = rho * self.wall_velocity[1]
            data.state[3, :, boundary_idx] = rho * self.wall_velocity[2]


class TransmissiveBC(BoundaryCondition):
    """Transmissive (outflow) boundary conditions"""
    
    def __init__(self):
        super().__init__("Transmissive")
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Apply transmissive boundary conditions.
        Boundary values are extrapolated from interior (zero-gradient).
        """
        # Left and right boundaries
        data.state[:, 0, :] = data.state[:, 1, :]     # Left
        data.state[:, -1, :] = data.state[:, -2, :]   # Right
        
        # Bottom and top boundaries  
        data.state[:, :, 0] = data.state[:, :, 1]     # Bottom
        data.state[:, :, -1] = data.state[:, :, -2]   # Top


class InflowBC(BoundaryCondition):
    """Inflow boundary condition with prescribed state"""
    
    def __init__(self, inflow_state: np.ndarray, boundaries: list):
        super().__init__("Inflow")
        self.inflow_state = inflow_state  # Shape: (num_vars,)
        self.boundaries = boundaries      # List of boundary names: ['left', 'right', 'bottom', 'top']
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Apply inflow boundary condition"""
        for boundary in self.boundaries:
            if boundary == 'left':
                data.state[:, 0, :] = self.inflow_state.reshape(-1, 1)
            elif boundary == 'right':
                data.state[:, -1, :] = self.inflow_state.reshape(-1, 1)
            elif boundary == 'bottom':
                data.state[:, :, 0] = self.inflow_state.reshape(-1, 1)
            elif boundary == 'top':
                data.state[:, :, -1] = self.inflow_state.reshape(-1, 1)


class CustomBC(BoundaryCondition):
    """Custom boundary condition with user-defined function"""
    
    def __init__(self, name: str, apply_function: Callable[[FVMDataContainer2D], None]):
        super().__init__(name)
        self.apply_function = apply_function
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Apply custom boundary condition"""
        self.apply_function(data, **kwargs)


class BoundaryManager:
    """
    Manager for handling multiple boundary condition types and regions.
    
    This class allows for mixed boundary conditions on different boundaries
    and provides a unified interface for applying all boundary conditions.
    """
    
    def __init__(self):
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        self.default_bc: Optional[BoundaryCondition] = None
        
    def set_default_boundary(self, bc: BoundaryCondition):
        """Set default boundary condition for all boundaries"""
        self.default_bc = bc
        
    def set_boundary(self, region: str, bc: BoundaryCondition):
        """
        Set boundary condition for specific region.
        
        Args:
            region: Boundary region ('left', 'right', 'bottom', 'top', or 'all')
            bc: Boundary condition to apply
        """
        self.boundary_conditions[region] = bc
        
    def apply_all(self, data: FVMDataContainer2D, **kwargs):
        """Apply all boundary conditions"""
        # Apply default boundary condition first if set
        if self.default_bc is not None:
            self.default_bc.apply(data, **kwargs)
        
        # Apply specific regional boundary conditions
        for region, bc in self.boundary_conditions.items():
            if region == 'all':
                bc.apply(data, **kwargs)
            else:
                self._apply_regional_bc(data, region, bc, **kwargs)
    
    def _apply_regional_bc(self, data: FVMDataContainer2D, region: str, 
                          bc: BoundaryCondition, **kwargs):
        """Apply boundary condition to specific region"""
        # Create a temporary data container for regional application
        # This is a simplified implementation - full regional support would
        # require more sophisticated ghost cell management
        bc.apply(data, **kwargs)


class EulerBoundaryConditions:
    """
    Specialized boundary conditions for Euler equations.
    
    This class provides physically meaningful boundary conditions
    specifically for compressible flow (Euler equations).
    """
    
    @staticmethod
    def subsonic_inflow(rho_inf: float, u_inf: float, v_inf: float, 
                       w_inf: float, p_inf: float) -> InflowBC:
        """
        Create subsonic inflow boundary condition.
        
        Args:
            rho_inf: Inflow density
            u_inf, v_inf, w_inf: Inflow velocity components
            p_inf: Inflow pressure
            
        Returns:
            InflowBC with prescribed conservative variables
        """
        # Convert primitive to conservative variables
        E_inf = p_inf / 0.4 + 0.5 * rho_inf * (u_inf**2 + v_inf**2 + w_inf**2)
        
        inflow_state = np.array([
            rho_inf,                # density
            rho_inf * u_inf,        # x-momentum
            rho_inf * v_inf,        # y-momentum  
            rho_inf * w_inf,        # z-momentum
            E_inf                   # total energy
        ])
        
        return InflowBC(inflow_state, ['left'])  # Typically inflow from left
    
    @staticmethod
    def subsonic_outflow(p_back: float) -> CustomBC:
        """
        Create subsonic outflow boundary condition with prescribed back pressure.
        
        Args:
            p_back: Back pressure at outflow
            
        Returns:
            CustomBC that maintains back pressure while extrapolating other variables
        """
        def apply_outflow(data: FVMDataContainer2D, **kwargs):
            gamma = kwargs.get('gamma', 1.4)
            
            # Extrapolate from interior
            data.state[:, -1, :] = data.state[:, -2, :]
            
            # Fix pressure at boundary
            rho = data.state[0, -1, :]
            u = data.state[1, -1, :] / rho
            v = data.state[2, -1, :] / rho
            w = data.state[3, -1, :] / rho
            
            # Recompute energy with prescribed pressure
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            data.state[4, -1, :] = p_back / (gamma - 1.0) + kinetic_energy
        
        return CustomBC("SubsonicOutflow", apply_outflow)
    
    @staticmethod
    def wall_with_heat_transfer(wall_temp: float, heat_transfer_coeff: float = 0.0) -> CustomBC:
        """
        Create wall boundary condition with heat transfer.
        
        Args:
            wall_temp: Wall temperature
            heat_transfer_coeff: Heat transfer coefficient
            
        Returns:
            CustomBC implementing wall with heat transfer
        """
        def apply_wall_heat(data: FVMDataContainer2D, **kwargs):
            gamma = kwargs.get('gamma', 1.4)
            R = kwargs.get('gas_constant', 287.0)
            
            # Apply standard reflective BC first
            reflective = ReflectiveBC()
            reflective.apply(data)
            
            # Modify energy based on wall temperature
            if wall_temp > 0:
                rho = data.state[0, :, 0]  # Bottom wall example
                # Set temperature-based energy (simplified)
                u = data.state[1, :, 0] / rho
                v = data.state[2, :, 0] / rho
                w = data.state[3, :, 0] / rho
                
                p = rho * R * wall_temp
                kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
                data.state[4, :, 0] = p / (gamma - 1.0) + kinetic_energy
        
        return CustomBC("WallHeatTransfer", apply_wall_heat)