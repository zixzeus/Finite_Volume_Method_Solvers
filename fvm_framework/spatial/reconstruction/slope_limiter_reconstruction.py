"""
Slope Limiter Reconstruction (MUSCL/TVD)

This module implements second-order reconstruction using slope limiters
to maintain Total Variation Diminishing (TVD) property. This replaces
the reconstruction part of the original TVD Lax-Friedrichs scheme.
"""

import numpy as np
from typing import Tuple
from fvm_framework.core.data_container import FVMDataContainer2D
from .base_reconstruction import SecondOrderReconstruction, LimiterMixin


class SlopeLimiterReconstruction(SecondOrderReconstruction, LimiterMixin):
    """
    Second-order MUSCL/TVD reconstruction with slope limiters.
    
    This method computes limited slopes at each cell center and uses them
    to reconstruct interface states, maintaining TVD property to avoid
    spurious oscillations near discontinuities.
    
    For interface i+1/2:
    - Left state: u_{i+1/2}^L = u_i + 0.5 * slope_i
    - Right state: u_{i+1/2}^R = u_{i+1} - 0.5 * slope_{i+1}
    
    where slopes are computed using flux limiters.
    """
    
    def __init__(self, limiter_type: str = 'minmod'):
        SecondOrderReconstruction.__init__(self, "SlopeLimiter")
        LimiterMixin.__init__(self, limiter_type)
    
    def reconstruct_x_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at x-direction interfaces using slope limiters.
        
        Args:
            data: FVM data container with ghost cells
            
        Returns:
            Tuple of (left_states, right_states) at x-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        
        # Compute limited slopes in x-direction
        slopes = self._compute_limited_slopes_x(data)
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx + 1, ny))
        right_states = np.zeros((num_vars, nx + 1, ny))
        
        for j in range(ny):
            # Map interior j to state array index
            j_state = j + ng
            
            for i in range(nx + 1):
                if i == 0:
                    # Left boundary interface: use left ghost and first interior cell
                    left_states[:, i, j] = data.state[:, ng - 1, j_state]  # Left ghost cell
                    right_states[:, i, j] = data.state[:, ng, j_state] + 0.5 * slopes[:, 0, j]  # First interior cell with slope
                elif i == nx:
                    # Right boundary interface: use last interior and right ghost cell
                    left_states[:, i, j] = data.state[:, ng + nx - 1, j_state] + 0.5 * slopes[:, nx-1, j]  # Last interior cell with slope
                    right_states[:, i, j] = data.state[:, ng + nx, j_state]  # Right ghost cell
                else:
                    # Interior interface: reconstruct using slopes
                    left_states[:, i, j] = data.state[:, ng + i - 1, j_state] + 0.5 * slopes[:, i-1, j]
                    right_states[:, i, j] = data.state[:, ng + i, j_state] - 0.5 * slopes[:, i, j]
        
        return left_states, right_states
    
    def reconstruct_y_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at y-direction interfaces using slope limiters.
        
        Args:
            data: FVM data container with ghost cells
            
        Returns:
            Tuple of (left_states, right_states) at y-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        
        # Compute limited slopes in y-direction
        slopes = self._compute_limited_slopes_y(data)
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx, ny + 1))
        right_states = np.zeros((num_vars, nx, ny + 1))
        
        for i in range(nx):
            # Map interior i to state array index
            i_state = i + ng
            
            for j in range(ny + 1):
                if j == 0:
                    # Bottom boundary interface: use bottom ghost and first interior cell
                    left_states[:, i, j] = data.state[:, i_state, ng - 1]  # Bottom ghost cell
                    right_states[:, i, j] = data.state[:, i_state, ng] + 0.5 * slopes[:, i, 0]  # First interior cell with slope
                elif j == ny:
                    # Top boundary interface: use last interior and top ghost cell
                    left_states[:, i, j] = data.state[:, i_state, ng + ny - 1] + 0.5 * slopes[:, i, ny-1]  # Last interior cell with slope
                    right_states[:, i, j] = data.state[:, i_state, ng + ny]  # Top ghost cell
                else:
                    # Interior interface: reconstruct using slopes
                    left_states[:, i, j] = data.state[:, i_state, ng + j - 1] + 0.5 * slopes[:, i, j-1]
                    right_states[:, i, j] = data.state[:, i_state, ng + j] - 0.5 * slopes[:, i, j]
        
        return left_states, right_states
    
    def _compute_limited_slopes_x(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute limited slopes in x-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for j in range(ny):
                # Map interior j to state array index
                j_state = j + ng
                
                for i in range(1, nx-1):
                    # Map interior i to state array index
                    i_left = ng + i - 1
                    i_center = ng + i
                    i_right = ng + i + 1
                    
                    # Central differences
                    left_diff = data.state[var, i_center, j_state] - data.state[var, i_left, j_state]
                    right_diff = data.state[var, i_right, j_state] - data.state[var, i_center, j_state]
                    
                    # Apply limiter
                    slopes[var, i, j] = self.limiter_function(left_diff, right_diff)
                
                # Boundary cells: use one-sided differences with proper limiting
                if nx > 1:
                    # Left boundary (i=0): use forward difference but avoid zero limiting
                    i_center = ng
                    i_right = ng + 1
                    forward_diff = data.state[var, i_right, j_state] - data.state[var, i_center, j_state]
                    # Use ghost cell if available for better boundary treatment
                    if ng > 0:
                        i_left_ghost = ng - 1
                        backward_diff = data.state[var, i_center, j_state] - data.state[var, i_left_ghost, j_state]
                        slopes[var, 0, j] = self.limiter_function(backward_diff, forward_diff)
                    else:
                        # Fallback: use reduced slope to avoid zero
                        slopes[var, 0, j] = 0.5 * forward_diff
                    
                    # Right boundary (i=nx-1): use backward difference but avoid zero limiting
                    i_left = ng + nx - 2
                    i_center = ng + nx - 1
                    backward_diff = data.state[var, i_center, j_state] - data.state[var, i_left, j_state]
                    # Use ghost cell if available for better boundary treatment
                    if ng > 0:
                        i_right_ghost = ng + nx
                        forward_diff = data.state[var, i_right_ghost, j_state] - data.state[var, i_center, j_state]
                        slopes[var, nx-1, j] = self.limiter_function(backward_diff, forward_diff)
                    else:
                        # Fallback: use reduced slope to avoid zero
                        slopes[var, nx-1, j] = 0.5 * backward_diff
        
        return slopes
    
    def _compute_limited_slopes_y(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute limited slopes in y-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for i in range(nx):
                # Map interior i to state array index
                i_state = i + ng
                
                for j in range(1, ny-1):
                    # Map interior j to state array index
                    j_bottom = ng + j - 1
                    j_center = ng + j
                    j_top = ng + j + 1
                    
                    # Central differences
                    left_diff = data.state[var, i_state, j_center] - data.state[var, i_state, j_bottom]
                    right_diff = data.state[var, i_state, j_top] - data.state[var, i_state, j_center]
                    
                    # Apply limiter
                    slopes[var, i, j] = self.limiter_function(left_diff, right_diff)
                
                # Boundary cells: use one-sided differences with proper limiting
                if ny > 1:
                    # Bottom boundary (j=0): use forward difference but avoid zero limiting
                    j_center = ng
                    j_top = ng + 1
                    forward_diff = data.state[var, i_state, j_top] - data.state[var, i_state, j_center]
                    # Use ghost cell if available for better boundary treatment
                    if ng > 0:
                        j_bottom_ghost = ng - 1
                        backward_diff = data.state[var, i_state, j_center] - data.state[var, i_state, j_bottom_ghost]
                        slopes[var, i, 0] = self.limiter_function(backward_diff, forward_diff)
                    else:
                        # Fallback: use reduced slope to avoid zero
                        slopes[var, i, 0] = 0.5 * forward_diff
                    
                    # Top boundary (j=ny-1): use backward difference but avoid zero limiting
                    j_bottom = ng + ny - 2
                    j_center = ng + ny - 1
                    backward_diff = data.state[var, i_state, j_center] - data.state[var, i_state, j_bottom]
                    # Use ghost cell if available for better boundary treatment
                    if ng > 0:
                        j_top_ghost = ng + ny
                        forward_diff = data.state[var, i_state, j_top_ghost] - data.state[var, i_state, j_center]
                        slopes[var, i, ny-1] = self.limiter_function(backward_diff, forward_diff)
                    else:
                        # Fallback: use reduced slope to avoid zero
                        slopes[var, i, ny-1] = 0.5 * backward_diff
        
        return slopes
    
    def get_stencil_size(self) -> int:
        """Slope limiter reconstruction needs 3-point stencil"""
        return 2
    
    def needs_boundary_treatment(self) -> bool:
        """Second-order reconstruction needs special boundary treatment"""
        return True
    
    def get_limiter_type(self) -> str:
        """Get current limiter type"""
        return self.limiter_type
    
    def set_limiter_type(self, limiter_type: str):
        """Change limiter type"""
        self.limiter_type = limiter_type
        self.limiter_function = self._get_limiter_function()