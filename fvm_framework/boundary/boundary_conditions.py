"""
Boundary Conditions for 2D Finite Volume Method

This module implements various boundary condition types for FVM simulations,
including periodic, reflective, transmissive, and custom boundary conditions.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import numpy as np
from fvm_framework.core.data_container import FVMDataContainer2D

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Apply boundary condition to the data (deprecated - use fill_ghost_cells)"""
        pass
        
    @abstractmethod  
    def fill_ghost_cells(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Fill ghost cells according to boundary condition"""
        pass


class PeriodicBC(BoundaryCondition):
    """Periodic boundary conditions"""
    
    def __init__(self):
        super().__init__("Periodic")
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Legacy method - use fill_ghost_cells instead"""
        self.fill_ghost_cells(data, **kwargs)
        
    def fill_ghost_cells(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Fill ghost cells with periodic boundary conditions.
        Ghost cells are filled with values from opposite boundary.
        """
        ng = data.ng
        
        # Left-Right periodicity
        # Left ghost cells = rightmost interior cells
        data.state[:, :ng, :] = data.state[:, -(2*ng):-ng, :]
        # Right ghost cells = leftmost interior cells  
        data.state[:, -ng:, :] = data.state[:, ng:2*ng, :]
        
        # Bottom-Top periodicity
        # Bottom ghost cells = topmost interior cells
        data.state[:, :, :ng] = data.state[:, :, -(2*ng):-ng]
        # Top ghost cells = bottommost interior cells
        data.state[:, :, -ng:] = data.state[:, :, ng:2*ng]


class ReflectiveBC(BoundaryCondition):
    """Reflective (wall) boundary conditions"""
    
    def __init__(self, wall_velocity: Optional[np.ndarray] = None):
        super().__init__("Reflective")
        if wall_velocity is None:
            self.wall_velocity = np.zeros(3)
        else:
            self.wall_velocity = wall_velocity
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Legacy method - use fill_ghost_cells instead"""
        self.fill_ghost_cells(data, **kwargs)
        
    def fill_ghost_cells(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Fill ghost cells with reflective boundary conditions.
        Normal velocity component is reflected, tangential components preserved.
        """
        ng = data.ng
        
        # Left wall (x-normal): reflect x-momentum
        for g in range(ng):
            ghost_i = ng - 1 - g  # ghost indices: ng-1, ng-2, ..., 0
            mirror_i = ng + g     # mirror indices: ng, ng+1, ..., 2*ng-1
            
            # Copy all variables
            data.state[:, ghost_i, :] = data.state[:, mirror_i, :]
            # Reflect x-momentum
            data.state[1, ghost_i, :] *= -1
            # Apply wall velocity if specified
            if np.any(self.wall_velocity != 0):
                rho = data.state[0, ghost_i, :]
                data.state[1, ghost_i, :] = rho * (-self.wall_velocity[0])  # Reflect wall velocity
                data.state[2, ghost_i, :] = rho * self.wall_velocity[1]
                data.state[3, ghost_i, :] = rho * self.wall_velocity[2]
        
        # Right wall (x-normal): reflect x-momentum
        for g in range(ng):
            ghost_i = -(g + 1)      # ghost indices: -1, -2, ..., -ng
            mirror_i = -(ng + g + 1) # mirror indices: -(ng+1), -(ng+2), ..., -(2*ng)
            
            data.state[:, ghost_i, :] = data.state[:, mirror_i, :]
            data.state[1, ghost_i, :] *= -1
            if np.any(self.wall_velocity != 0):
                rho = data.state[0, ghost_i, :]
                data.state[1, ghost_i, :] = rho * (-self.wall_velocity[0])
                data.state[2, ghost_i, :] = rho * self.wall_velocity[1] 
                data.state[3, ghost_i, :] = rho * self.wall_velocity[2]
        
        # Bottom wall (y-normal): reflect y-momentum
        for g in range(ng):
            ghost_j = ng - 1 - g
            mirror_j = ng + g
            
            data.state[:, :, ghost_j] = data.state[:, :, mirror_j]
            data.state[2, :, ghost_j] *= -1
            if np.any(self.wall_velocity != 0):
                rho = data.state[0, :, ghost_j]
                data.state[1, :, ghost_j] = rho * self.wall_velocity[0]
                data.state[2, :, ghost_j] = rho * (-self.wall_velocity[1])
                data.state[3, :, ghost_j] = rho * self.wall_velocity[2]
        
        # Top wall (y-normal): reflect y-momentum  
        for g in range(ng):
            ghost_j = -(g + 1)
            mirror_j = -(ng + g + 1)
            
            data.state[:, :, ghost_j] = data.state[:, :, mirror_j]
            data.state[2, :, ghost_j] *= -1
            if np.any(self.wall_velocity != 0):
                rho = data.state[0, :, ghost_j]
                data.state[1, :, ghost_j] = rho * self.wall_velocity[0]
                data.state[2, :, ghost_j] = rho * (-self.wall_velocity[1])
                data.state[3, :, ghost_j] = rho * self.wall_velocity[2]
    


class TransmissiveBC(BoundaryCondition):
    """Transmissive (outflow) boundary conditions"""
    
    def __init__(self):
        super().__init__("Transmissive")
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Legacy method - use fill_ghost_cells instead"""
        self.fill_ghost_cells(data, **kwargs)
        
    def fill_ghost_cells(self, data: FVMDataContainer2D, **kwargs) -> None:
        """
        Fill ghost cells with transmissive boundary conditions.
        Ghost cell values are extrapolated from interior (zero-gradient).
        """
        ng = data.ng
        
        # Left boundary: zero-gradient extrapolation
        for g in range(ng):
            data.state[:, g, :] = data.state[:, ng, :]  # All left ghost cells = first interior
            
        # Right boundary: zero-gradient extrapolation
        for g in range(ng):
            data.state[:, -(g+1), :] = data.state[:, -(ng+1), :]  # All right ghost cells = last interior
            
        # Bottom boundary: zero-gradient extrapolation
        for g in range(ng):
            data.state[:, :, g] = data.state[:, :, ng]  # All bottom ghost cells = first interior
            
        # Top boundary: zero-gradient extrapolation
        for g in range(ng):
            data.state[:, :, -(g+1)] = data.state[:, :, -(ng+1)]  # All top ghost cells = last interior


class InflowBC(BoundaryCondition):
    """Inflow boundary condition with prescribed state"""
    
    def __init__(self, inflow_state: np.ndarray, boundaries: list):
        super().__init__("Inflow")
        self.inflow_state = inflow_state  # Shape: (num_vars,)
        self.boundaries = boundaries      # List of boundary names: ['left', 'right', 'bottom', 'top']
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Legacy method - use fill_ghost_cells instead"""
        self.fill_ghost_cells(data, **kwargs)
        
    def fill_ghost_cells(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Fill ghost cells with prescribed inflow state"""
        ng = data.ng
        
        for boundary in self.boundaries:
            if boundary == 'left':
                for g in range(ng):
                    data.state[:, g, :] = self.inflow_state.reshape(-1, 1)
            elif boundary == 'right':
                for g in range(ng):
                    data.state[:, -(g+1), :] = self.inflow_state.reshape(-1, 1)
            elif boundary == 'bottom':
                for g in range(ng):
                    data.state[:, :, g] = self.inflow_state.reshape(-1, 1)
            elif boundary == 'top':
                for g in range(ng):
                    data.state[:, :, -(g+1)] = self.inflow_state.reshape(-1, 1)


class CustomBC(BoundaryCondition):
    """Custom boundary condition with user-defined function"""
    
    def __init__(self, name: str, apply_function: Callable[[FVMDataContainer2D], None]):
        super().__init__(name)
        self.apply_function = apply_function
    
    def apply(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Legacy method - use fill_ghost_cells instead"""
        self.fill_ghost_cells(data, **kwargs)
        
    def fill_ghost_cells(self, data: FVMDataContainer2D, **kwargs) -> None:
        """Fill ghost cells using custom function"""
        # For backward compatibility, call the custom function
        # Users should update their functions to work with ghost cells
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
        """Apply all boundary conditions (legacy method)"""
        self.fill_all_ghost_cells(data, **kwargs)
        
    def fill_all_ghost_cells(self, data: FVMDataContainer2D, **kwargs):
        """Fill all ghost cells according to boundary conditions"""
        # Apply default boundary condition first if set
        if self.default_bc is not None:
            self.default_bc.fill_ghost_cells(data, **kwargs)
        
        # Apply specific regional boundary conditions
        for region, bc in self.boundary_conditions.items():
            if region == 'all':
                bc.fill_ghost_cells(data, **kwargs)
            else:
                self._apply_regional_bc(data, region, bc, **kwargs)
    
    def _apply_regional_bc(self, data: FVMDataContainer2D, region: str, 
                          bc: BoundaryCondition, **kwargs):
        """Apply boundary condition to specific region"""
        # For regional boundary conditions, we need to selectively fill
        # only the ghost cells corresponding to that region
        # This is a simplified implementation - more sophisticated regional
        # support would require masking specific ghost cell regions
        
        if region in ['left', 'right', 'bottom', 'top']:
            # For now, apply to all ghost cells and let the BC implementation
            # handle region-specific logic
            bc.fill_ghost_cells(data, region=region, **kwargs)
        else:
            bc.fill_ghost_cells(data, **kwargs)


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
            ng = data.ng
            
            # Fill right boundary ghost cells
            for g in range(ng):
                ghost_i = -(g + 1)
                interior_i = -(ng + 1)  # Last interior cell
                
                # Extrapolate from interior
                data.state[:, ghost_i, :] = data.state[:, interior_i, :]
                
                # Fix pressure at ghost cells
                rho = data.state[0, ghost_i, :]
                u = data.state[1, ghost_i, :] / rho
                v = data.state[2, ghost_i, :] / rho
                w = data.state[3, ghost_i, :] / rho
                
                # Recompute energy with prescribed pressure
                kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
                data.state[4, ghost_i, :] = p_back / (gamma - 1.0) + kinetic_energy
        
        return CustomBC("SubsonicOutflow", apply_outflow)
    
    @staticmethod
    def wall_with_heat_transfer(wall_temp: float, heat_transfer_coeff: float = 0.0) -> CustomBC:
        """
        Create wall boundary condition with heat transfer.
        
        Args:
            wall_temp: Wall temperature
            heat_transfer_coeff: Heat transfer coefficient (currently unused - reserved for future)
            
        Returns:
            CustomBC implementing wall with heat transfer
        """
        def apply_wall_heat(data: FVMDataContainer2D, **kwargs):
            gamma = kwargs.get('gamma', 1.4)
            R = kwargs.get('gas_constant', 287.0)
            ng = data.ng
            
            # Apply standard reflective BC first
            reflective = ReflectiveBC()
            reflective.fill_ghost_cells(data)
            
            # Modify energy in bottom wall ghost cells based on wall temperature
            if wall_temp > 0:
                for g in range(ng):
                    ghost_j = g  # Bottom ghost cells
                    
                    rho = data.state[0, :, ghost_j]
                    # Set temperature-based energy (simplified)
                    u = data.state[1, :, ghost_j] / rho
                    v = data.state[2, :, ghost_j] / rho
                    w = data.state[3, :, ghost_j] / rho
                    
                    p = rho * R * wall_temp
                    kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
                    data.state[4, :, ghost_j] = p / (gamma - 1.0) + kinetic_energy
        
        return CustomBC("WallHeatTransfer", apply_wall_heat)