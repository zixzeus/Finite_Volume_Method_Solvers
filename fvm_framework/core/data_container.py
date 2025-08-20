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
    Structure of Arrays (SoA) data container for 2D finite volume method.
    
    Optimized for vectorization and cache performance:
    - Contiguous memory layout for each physical variable
    - SIMD-friendly data alignment
    - Efficient block-wise access patterns
    """
    
    def __init__(self, geometry: GridGeometry, num_vars: int = 5):
        """
        Initialize 2D FVM data container.
        
        Args:
            geometry: Grid geometry parameters
            num_vars: Number of conservation variables (default: 5 for Euler equations)
        """
        self.geometry = geometry
        self.num_vars = num_vars
        self.nx, self.ny = geometry.nx, geometry.ny
        self.total_cells = self.nx * self.ny
        
        # Conservative variables using SoA layout
        # For Euler equations: [density, momentum_x, momentum_y, momentum_z, energy]
        self.state = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        self.state_new = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        
        # Flux arrays for x and y directions
        self.flux_x = np.zeros((self.num_vars, self.nx + 1, self.ny), dtype=np.float64, order='C')
        self.flux_y = np.zeros((self.num_vars, self.nx, self.ny + 1), dtype=np.float64, order='C')
        
        # Source terms
        self.source = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        
        # Temporary arrays for intermediate calculations
        self.temp_state = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        
        # Primitive variables cache (density, velocity_x, velocity_y, velocity_z, pressure)
        self.primitives = np.zeros((self.num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        self._primitives_valid = False
        
    def reset_state(self):
        """Reset all state arrays to zero"""
        self.state.fill(0.0)
        self.state_new.fill(0.0)
        self._primitives_valid = False
        
    def swap_states(self):
        """Swap current and new state arrays for time stepping"""
        self.state, self.state_new = self.state_new, self.state
        self._primitives_valid = False
        
    def copy_state(self, source_state: Optional[np.ndarray] = None):
        """Copy state array"""
        if source_state is None:
            np.copyto(self.state_new, self.state)
        else:
            np.copyto(self.state, source_state)
        self._primitives_valid = False
    
    def get_block(self, i_start: int, i_end: int, j_start: int, j_end: int) -> np.ndarray:
        """
        Get a block of data for cache-efficient processing.
        
        Args:
            i_start, i_end: x-direction indices
            j_start, j_end: y-direction indices
            
        Returns:
            Block of state data with shape (num_vars, i_end-i_start, j_end-j_start)
        """
        return self.state[:, i_start:i_end, j_start:j_end]
    
    def set_block(self, i_start: int, i_end: int, j_start: int, j_end: int, data: np.ndarray):
        """Set a block of data"""
        self.state[:, i_start:i_end, j_start:j_end] = data
        self._primitives_valid = False
        
    def get_density(self) -> np.ndarray:
        """Get density field (first conservative variable)"""
        return self.state[0]
    
    def get_momentum(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get momentum components"""
        return self.state[1], self.state[2], self.state[3]
    
    def get_energy(self) -> np.ndarray:
        """Get total energy field"""
        return self.state[4]
    
    def compute_primitives(self, gamma: float = 1.4):
        """
        Compute primitive variables from conservative variables.
        
        Args:
            gamma: Heat capacity ratio
        """
        if self._primitives_valid:
            return
            
        rho = self.state[0]
        mom_x = self.state[1]
        mom_y = self.state[2]
        mom_z = self.state[3]
        energy = self.state[4]
        
        # Avoid division by zero
        rho_safe = np.maximum(rho, 1e-15)
        
        # Compute primitive variables
        self.primitives[0] = rho  # density
        self.primitives[1] = mom_x / rho_safe  # velocity_x
        self.primitives[2] = mom_y / rho_safe  # velocity_y
        self.primitives[3] = mom_z / rho_safe  # velocity_z
        
        # Pressure from ideal gas law
        kinetic_energy = 0.5 * (mom_x**2 + mom_y**2 + mom_z**2) / rho_safe
        self.primitives[4] = (gamma - 1.0) * (energy - kinetic_energy)
        
        # Ensure positive pressure
        self.primitives[4] = np.maximum(self.primitives[4], 1e-15)
        
        self._primitives_valid = True
    
    def get_primitives(self, gamma: float = 1.4) -> np.ndarray:
        """
        Get primitive variables, computing if necessary.
        
        Returns:
            Primitive variables [density, velocity_x, velocity_y, velocity_z, pressure]
        """
        self.compute_primitives(gamma)
        return self.primitives
    
    def get_sound_speed(self, gamma: float = 1.4) -> np.ndarray:
        """Compute local sound speed"""
        self.compute_primitives(gamma)
        rho = self.primitives[0]
        pressure = self.primitives[4]
        return np.sqrt(gamma * pressure / rho)
    
    def get_max_wave_speed(self, gamma: float = 1.4) -> float:
        """Get maximum wave speed for CFL condition"""
        self.compute_primitives(gamma)
        
        u = np.abs(self.primitives[1])
        v = np.abs(self.primitives[2])
        c = self.get_sound_speed(gamma)
        
        max_speed_x = np.max(u + c)
        max_speed_y = np.max(v + c)
        
        return max(max_speed_x, max_speed_y)
    
    def apply_boundary_conditions(self, bc_type: str = 'periodic'):
        """
        Apply boundary conditions.
        
        Args:
            bc_type: Type of boundary condition ('periodic', 'reflective', 'transmissive')
        """
        if bc_type == 'periodic':
            # Periodic boundary conditions
            pass  # Already handled by flux computation
        elif bc_type == 'reflective':
            # Reflective (wall) boundary conditions
            # Zero normal velocity at boundaries
            self.state[1, 0, :] = -self.state[1, 1, :]      # Left wall: -u
            self.state[1, -1, :] = -self.state[1, -2, :]    # Right wall: -u
            self.state[2, :, 0] = -self.state[2, :, 1]      # Bottom wall: -v
            self.state[2, :, -1] = -self.state[2, :, -2]    # Top wall: -v
        elif bc_type == 'transmissive':
            # Transmissive (outflow) boundary conditions
            self.state[:, 0, :] = self.state[:, 1, :]       # Left
            self.state[:, -1, :] = self.state[:, -2, :]     # Right
            self.state[:, :, 0] = self.state[:, :, 1]       # Bottom
            self.state[:, :, -1] = self.state[:, :, -2]     # Top
        
        self._primitives_valid = False
    
    def compute_residual(self) -> np.ndarray:
        """
        Compute residual for time integration.
        
        Returns:
            Residual array with same shape as state
        """
        residual = np.zeros_like(self.state)
        
        # Flux differences in x-direction
        residual[:, :-1, :] -= (self.flux_x[:, 1:-1, :] - self.flux_x[:, :-2, :]) / self.geometry.dx
        
        # Flux differences in y-direction
        residual[:, :, :-1] -= (self.flux_y[:, :, 1:-1] - self.flux_y[:, :, :-2]) / self.geometry.dy
        
        # Add source terms
        residual += self.source
        
        return residual
    
    def get_conservation_error(self) -> np.ndarray:
        """
        Check conservation properties.
        
        Returns:
            Conservation error for each variable
        """
        total = np.sum(self.state, axis=(1, 2)) * self.geometry.cell_volume
        return total
    
    def __repr__(self) -> str:
        return (f"FVMDataContainer2D(nx={self.nx}, ny={self.ny}, "
                f"num_vars={self.num_vars}, shape={self.state.shape})")