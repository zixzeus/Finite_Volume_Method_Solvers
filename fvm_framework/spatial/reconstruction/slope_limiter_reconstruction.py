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
            data: FVM data container
            
        Returns:
            Tuple of (left_states, right_states) at x-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        
        # Compute limited slopes in x-direction
        slopes = self._compute_limited_slopes_x(data)
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx + 1, ny))
        right_states = np.zeros((num_vars, nx + 1, ny))
        
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    # Left boundary: both states from boundary cell
                    left_states[:, i, j] = data.state[:, 0, j]
                    right_states[:, i, j] = data.state[:, 0, j]
                elif i == nx:
                    # Right boundary: both states from boundary cell
                    left_states[:, i, j] = data.state[:, -1, j]
                    right_states[:, i, j] = data.state[:, -1, j]
                else:
                    # Interior interface: reconstruct using slopes
                    left_states[:, i, j] = data.state[:, i-1, j] + 0.5 * slopes[:, i-1, j]
                    right_states[:, i, j] = data.state[:, i, j] - 0.5 * slopes[:, i, j]
        
        return left_states, right_states
    
    def reconstruct_y_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at y-direction interfaces using slope limiters.
        
        Args:
            data: FVM data container
            
        Returns:
            Tuple of (left_states, right_states) at y-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        
        # Compute limited slopes in y-direction
        slopes = self._compute_limited_slopes_y(data)
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx, ny + 1))
        right_states = np.zeros((num_vars, nx, ny + 1))
        
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    # Bottom boundary: both states from boundary cell
                    left_states[:, i, j] = data.state[:, i, 0]
                    right_states[:, i, j] = data.state[:, i, 0]
                elif j == ny:
                    # Top boundary: both states from boundary cell
                    left_states[:, i, j] = data.state[:, i, -1]
                    right_states[:, i, j] = data.state[:, i, -1]
                else:
                    # Interior interface: reconstruct using slopes
                    left_states[:, i, j] = data.state[:, i, j-1] + 0.5 * slopes[:, i, j-1]
                    right_states[:, i, j] = data.state[:, i, j] - 0.5 * slopes[:, i, j]
        
        return left_states, right_states
    
    def _compute_limited_slopes_x(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute limited slopes in x-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for j in range(ny):
                for i in range(1, nx-1):
                    # Central differences
                    left_diff = data.state[var, i, j] - data.state[var, i-1, j]
                    right_diff = data.state[var, i+1, j] - data.state[var, i, j]
                    
                    # Apply limiter
                    slopes[var, i, j] = self.limiter_function(left_diff, right_diff)
                
                # Boundary cells: use one-sided differences or zero slope
                if nx > 1:
                    # Left boundary
                    forward_diff = data.state[var, 1, j] - data.state[var, 0, j]
                    slopes[var, 0, j] = self.limiter_function(0.0, forward_diff)
                    
                    # Right boundary  
                    backward_diff = data.state[var, -1, j] - data.state[var, -2, j]
                    slopes[var, -1, j] = self.limiter_function(backward_diff, 0.0)
        
        return slopes
    
    def _compute_limited_slopes_y(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute limited slopes in y-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for i in range(nx):
                for j in range(1, ny-1):
                    # Central differences
                    left_diff = data.state[var, i, j] - data.state[var, i, j-1]
                    right_diff = data.state[var, i, j+1] - data.state[var, i, j]
                    
                    # Apply limiter
                    slopes[var, i, j] = self.limiter_function(left_diff, right_diff)
                
                # Boundary cells: use one-sided differences or zero slope
                if ny > 1:
                    # Bottom boundary
                    forward_diff = data.state[var, i, 1] - data.state[var, i, 0]
                    slopes[var, i, 0] = self.limiter_function(0.0, forward_diff)
                    
                    # Top boundary
                    backward_diff = data.state[var, i, -1] - data.state[var, i, -2]
                    slopes[var, i, -1] = self.limiter_function(backward_diff, 0.0)
        
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