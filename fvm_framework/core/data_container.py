"""
2D Finite Volume Method Data Container with Structure of Arrays Layout

This module implements a high-performance data container optimized for 
vectorization and cache-friendly memory access patterns.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridGeometry:
    """Grid geometry parameters"""
    nx: int
    ny: int
    dx: float
    dy: float
    x_min: float = 0.0
    y_min: float = 0.0
    
    @property
    def x_max(self) -> float:
        return self.x_min + self.nx * self.dx
    
    @property
    def y_max(self) -> float:
        return self.y_min + self.ny * self.dy
    
    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy


class FVMDataContainer2D:
    """
    Structure of Arrays (SoA) data container for 2D finite volume method with ghost cells.
    
    Optimized for vectorization and cache performance:
    - Contiguous memory layout for each physical variable
    - SIMD-friendly data alignment
    - Efficient block-wise access patterns
    - Ghost cell support for proper boundary condition handling
    """
    
    def __init__(self, geometry: GridGeometry, num_vars: int = 5, num_ghost: int = 2):
        """
        Initialize 2D FVM data container with ghost cells.
        
        Args:
            geometry: Grid geometry parameters (interior domain)
            num_vars: Number of conservation variables
            num_ghost: Number of ghost cell layers (default: 2)
        """
        self.geometry = geometry
        self.num_vars = num_vars
        self.nx, self.ny = geometry.nx, geometry.ny  # Interior grid size
        self.ng = num_ghost  # Number of ghost layers
        self.total_cells = self.nx * self.ny
        
        # Total grid size including ghost cells
        self.nx_total = self.nx + 2 * self.ng
        self.ny_total = self.ny + 2 * self.ng
        
        # Interior grid slice indices
        self.interior_slice = (slice(self.ng, self.ng + self.nx), 
                              slice(self.ng, self.ng + self.ny))
        
        # Conservative variables using SoA layout (including ghost cells)
        self.state = np.zeros((self.num_vars, self.nx_total, self.ny_total), dtype=np.float64, order='C')
        self.state_new = np.zeros((self.num_vars, self.nx_total, self.ny_total), dtype=np.float64, order='C')
        
        # Flux arrays for x and y directions (including ghost cell interfaces)
        # x-direction: need nx_total+1 interfaces for nx_total cells (including ghosts)
        # y-direction: need ny_total+1 interfaces for ny_total cells (including ghosts)
        self.flux_x = np.zeros((self.num_vars, self.nx_total + 1, self.ny_total), dtype=np.float64, order='C')
        self.flux_y = np.zeros((self.num_vars, self.nx_total, self.ny_total + 1), dtype=np.float64, order='C')
        
        # Source terms (interior only - ghost cells don't need source terms)
        # Shape: (num_vars, nx, ny) - matches interior domain size only
        self.source = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        
        # Temporary arrays for intermediate calculations (including ghost cells)
        self.temp_state = np.zeros((self.num_vars, self.nx_total, self.ny_total), dtype=np.float64, order='C')
        
        # Interface state arrays for reconstruction (interior interfaces only)
        # These store reconstructed states at interior interfaces for flux calculation
        # X 方向界面状态 (interior interfaces: nx+1 interfaces for nx interior cells)
        self.interface_states_x = (
            np.zeros((self.num_vars, self.nx + 1, self.ny)),  # left states at x-interfaces
            np.zeros((self.num_vars, self.nx + 1, self.ny))   # right states at x-interfaces
        )

        # Y 方向界面状态 (interior interfaces: ny+1 interfaces for ny interior cells)
        self.interface_states_y = (
            np.zeros((self.num_vars, self.nx, self.ny + 1)),  # left states at y-interfaces  
            np.zeros((self.num_vars, self.nx, self.ny + 1))   # right states at y-interfaces
        )
        
        # Primitive variables removed - use physics equation classes instead
        
    def reset_state(self):
        """Reset all state arrays to zero"""
        self.state.fill(0.0)
        self.state_new.fill(0.0)
        
    def swap_states(self):
        """Swap current and new state arrays for time stepping"""
        self.state, self.state_new = self.state_new, self.state
        
    def copy_state(self, source_state: Optional[np.ndarray] = None):
        """Copy state array"""
        if source_state is None:
            np.copyto(self.state_new, self.state)
        else:
            np.copyto(self.state, source_state)
    
    def get_interior_state(self) -> np.ndarray:
        """
        Get interior state data (excluding ghost cells).
        
        Returns:
            Interior state with shape (num_vars, nx, ny)
        """
        return self.state[:, self.interior_slice[0], self.interior_slice[1]]
    
    def set_interior_state(self, data: np.ndarray):
        """Set interior state data"""
        self.state[:, self.interior_slice[0], self.interior_slice[1]] = data
        
    def get_ghost_region(self, region: str) -> np.ndarray:
        """
        Get ghost cell region.
        
        Args:
            region: 'left', 'right', 'bottom', 'top', 
                   'bottom_left', 'bottom_right', 'top_left', 'top_right'
        
        Returns:
            Ghost cell data for specified region
        """
        ng = self.ng
        if region == 'left':
            return self.state[:, :ng, ng:-ng]
        elif region == 'right':
            return self.state[:, -ng:, ng:-ng]
        elif region == 'bottom':
            return self.state[:, ng:-ng, :ng]
        elif region == 'top':
            return self.state[:, ng:-ng, -ng:]
        elif region == 'bottom_left':
            return self.state[:, :ng, :ng]
        elif region == 'bottom_right':
            return self.state[:, -ng:, :ng]
        elif region == 'top_left':
            return self.state[:, :ng, -ng:]
        elif region == 'top_right':
            return self.state[:, -ng:, -ng:]
        else:
            raise ValueError(f"Unknown ghost region: {region}")
    
    def get_block(self, i_start: int, i_end: int, j_start: int, j_end: int) -> np.ndarray:
        """
        Get a block of data for cache-efficient processing (including ghost cells).
        
        Args:
            i_start, i_end: x-direction indices (total grid coordinates)
            j_start, j_end: y-direction indices (total grid coordinates)
            
        Returns:
            Block of state data with shape (num_vars, i_end-i_start, j_end-j_start)
        """
        return self.state[:, i_start:i_end, j_start:j_end]
    
    def set_block(self, i_start: int, i_end: int, j_start: int, j_end: int, data: np.ndarray):
        """Set a block of data"""
        self.state[:, i_start:i_end, j_start:j_end] = data
        
    def get_density(self) -> np.ndarray:
        """Get interior density field (first conservative variable)"""
        return self.get_interior_state()[0]
    
    def get_momentum(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get interior momentum components"""
        interior = self.get_interior_state()
        return interior[1], interior[2], interior[3]
    
    def get_energy(self) -> np.ndarray:
        """Get interior total energy field"""
        return self.get_interior_state()[4]
    
    def get_interior_flux_x(self) -> np.ndarray:
        """
        Get interior x-direction flux interfaces (excluding ghost interfaces).
        
        Returns:
            Interior x-flux with shape (num_vars, nx+1, ny)
        """
        ng = self.ng
        return self.flux_x[:, ng:ng+self.nx+1, ng:ng+self.ny]
    
    def get_interior_flux_y(self) -> np.ndarray:
        """
        Get interior y-direction flux interfaces (excluding ghost interfaces).
        
        Returns:
            Interior y-flux with shape (num_vars, nx, ny+1)
        """
        ng = self.ng
        return self.flux_y[:, ng:ng+self.nx, ng:ng+self.ny+1]
    
    def flux_x_index(self, i: int, j: int) -> tuple:
        """
        Convert interior domain indices to full flux_x array indices.
        
        Args:
            i: x-interface index in interior domain (0 to nx)
            j: y-cell index in interior domain (0 to ny-1)
            
        Returns:
            (flux_i, flux_j) indices for flux_x array
        """
        return (i + self.ng, j + self.ng)
    
    def flux_y_index(self, i: int, j: int) -> tuple:
        """
        Convert interior domain indices to full flux_y array indices.
        
        Args:
            i: x-cell index in interior domain (0 to nx-1)
            j: y-interface index in interior domain (0 to ny)
            
        Returns:
            (flux_i, flux_j) indices for flux_y array
        """
        return (i + self.ng, j + self.ng)
    
    # Physics-related methods moved to physics equation classes
    # Use physics_equation.conservative_to_primitive(data.state) instead
    
    def apply_boundary_conditions(self, boundary_manager):
        """
        Apply boundary conditions using ghost cells.
        
        Args:
            boundary_manager: BoundaryManager instance to fill ghost cells
        """
        boundary_manager.fill_all_ghost_cells(self)
    
    def compute_residual(self) -> np.ndarray:
        """
        Compute residual for time integration (interior cells only).
        
        Returns:
            Residual array for interior cells with shape (num_vars, nx, ny)
        """
        residual = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64)
        
        # Get interior flux slices
        interior_flux_x = self.get_interior_flux_x()  # Shape: (num_vars, nx+1, ny)
        interior_flux_y = self.get_interior_flux_y()  # Shape: (num_vars, nx, ny+1)
        
        # Flux differences in x-direction for interior cells
        # For cell i: flux_x[i+1] - flux_x[i]
        residual -= (interior_flux_x[:, 1:, :] - interior_flux_x[:, :-1, :]) / self.geometry.dx
        
        # Flux differences in y-direction for interior cells  
        # For cell j: flux_y[j+1] - flux_y[j]
        residual -= (interior_flux_y[:, :, 1:] - interior_flux_y[:, :, :-1]) / self.geometry.dy
        
        # Add source terms (already interior-only)
        residual += self.source
        
        return residual
    
    def get_conservation_error(self) -> np.ndarray:
        """
        Check conservation properties (interior cells only).
        
        Returns:
            Conservation error for each variable
        """
        interior_state = self.get_interior_state()
        total = np.sum(interior_state, axis=(1, 2)) * self.geometry.cell_volume
        return total
    
    def __repr__(self) -> str:
        return (f"FVMDataContainer2D(nx={self.nx}, ny={self.ny}, "
                f"num_vars={self.num_vars}, num_ghost={self.ng}, "
                f"total_shape={self.state.shape}, interior_shape=({self.num_vars}, {self.nx}, {self.ny}))")