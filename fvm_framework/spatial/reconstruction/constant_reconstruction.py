"""
Constant (First-Order) Reconstruction

This module implements first-order constant reconstruction where interface
states are simply taken as the cell-centered values. This replaces the 
reconstruction part of the original Lax-Friedrichs scheme.
"""

import numpy as np
from typing import Tuple
from fvm_framework.core.data_container import FVMDataContainer2D
from .base_reconstruction import FirstOrderReconstruction


class ConstantReconstruction(FirstOrderReconstruction):
    """
    First-order constant reconstruction.
    
    Interface states are taken directly from neighboring cell centers.
    This is the simplest possible reconstruction and is equivalent to 
    the reconstruction used in the original Lax-Friedrichs scheme.
    
    For interface i+1/2:
    - Left state: u_{i+1/2}^L = u_i
    - Right state: u_{i+1/2}^R = u_{i+1}
    """
    
    def __init__(self):
        super().__init__("Constant")
    
    def reconstruct_x_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at x-direction interfaces using constant values.
        
        Args:
            data: FVM data container with ghost cells
            
        Returns:
            Tuple of (left_states, right_states) at x-interfaces
            Shape: (num_vars, nx+1, ny) for interior interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        
        # Initialize interface states (interior interfaces only)
        left_states = np.zeros((num_vars, nx + 1, ny))
        right_states = np.zeros((num_vars, nx + 1, ny))
        
        for j in range(ny):
            # Map interior j to state array index
            j_state = j + ng
            
            for i in range(nx + 1):
                if i == 0:
                    # Left boundary interface: use left ghost and first interior cell
                    left_states[:, i, j] = data.state[:, ng - 1, j_state]  # Left ghost cell
                    right_states[:, i, j] = data.state[:, ng, j_state]      # First interior cell
                elif i == nx:
                    # Right boundary interface: use last interior and right ghost cell
                    left_states[:, i, j] = data.state[:, ng + nx - 1, j_state]  # Last interior cell
                    right_states[:, i, j] = data.state[:, ng + nx, j_state]     # Right ghost cell
                else:
                    # Interior interface i: between cells (i-1) and i
                    left_states[:, i, j] = data.state[:, ng + i - 1, j_state]   # Left interior cell
                    right_states[:, i, j] = data.state[:, ng + i, j_state]      # Right interior cell
        
        return left_states, right_states
    
    def reconstruct_y_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at y-direction interfaces using constant values.
        
        Args:
            data: FVM data container with ghost cells
            
        Returns:
            Tuple of (left_states, right_states) at y-interfaces
            Shape: (num_vars, nx, ny+1) for interior interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        
        # Initialize interface states (interior interfaces only)
        left_states = np.zeros((num_vars, nx, ny + 1))
        right_states = np.zeros((num_vars, nx, ny + 1))
        
        for i in range(nx):
            # Map interior i to state array index
            i_state = i + ng
            
            for j in range(ny + 1):
                if j == 0:
                    # Bottom boundary interface: use bottom ghost and first interior cell
                    left_states[:, i, j] = data.state[:, i_state, ng - 1]  # Bottom ghost cell
                    right_states[:, i, j] = data.state[:, i_state, ng]      # First interior cell
                elif j == ny:
                    # Top boundary interface: use last interior and top ghost cell
                    left_states[:, i, j] = data.state[:, i_state, ng + ny - 1]  # Last interior cell
                    right_states[:, i, j] = data.state[:, i_state, ng + ny]     # Top ghost cell
                else:
                    # Interior interface j: between cells (j-1) and j
                    left_states[:, i, j] = data.state[:, i_state, ng + j - 1]   # Bottom interior cell
                    right_states[:, i, j] = data.state[:, i_state, ng + j]      # Top interior cell
        
        return left_states, right_states
    
    def get_stencil_size(self) -> int:
        """Constant reconstruction needs only immediate neighbors"""
        return 1
    
    def needs_boundary_treatment(self) -> bool:
        """Constant reconstruction handles boundaries naturally"""
        return False