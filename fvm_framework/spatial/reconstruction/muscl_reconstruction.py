"""
MUSCL (Monotonic Upstream-Centered Scheme for Conservation Laws) Reconstruction

This module implements MUSCL reconstruction with various limiter options.
MUSCL provides second-order accuracy while maintaining monotonicity through
slope limiting.
"""

import numpy as np
from typing import Tuple
from fvm_framework.core.data_container import FVMDataContainer2D
from .base_reconstruction import SecondOrderReconstruction, LimiterMixin


class MUSCLReconstruction(SecondOrderReconstruction, LimiterMixin):
    """
    MUSCL (Monotonic Upstream-Centered Scheme for Conservation Laws) reconstruction.
    
    MUSCL reconstruction provides second-order accuracy by using linear reconstruction
    within each cell, with slope limiters to maintain monotonicity and avoid spurious
    oscillations near discontinuities.
    
    The reconstruction is based on:
    u(x) = u_i + σ_i * (x - x_i) / Δx
    
    where σ_i is the limited slope in cell i.
    """
    
    def __init__(self, limiter_type: str = 'van_leer', kappa: float = 1.0/3.0):
        """
        Initialize MUSCL reconstruction.
        
        Args:
            limiter_type: Type of slope limiter ('minmod', 'superbee', 'van_leer', 'mc')
            kappa: MUSCL parameter (-1 ≤ κ ≤ 1)
                   κ = -1: fully upwind
                   κ = 0:  second-order upwind
                   κ = 1/3: third-order upwind-biased (optimal)
                   κ = 1:  central difference
        """
        SecondOrderReconstruction.__init__(self, "MUSCL")
        LimiterMixin.__init__(self, limiter_type)
        
        if not -1.0 <= kappa <= 1.0:
            raise ValueError("MUSCL parameter kappa must be in [-1, 1]")
        
        self.kappa = kappa
    
    def reconstruct_x_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at x-direction interfaces using MUSCL.
        
        Args:
            data: FVM data container
            
        Returns:
            Tuple of (left_states, right_states) at x-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        
        # Compute MUSCL slopes in x-direction
        slopes = self._compute_muscl_slopes_x(data)
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx + 1, ny))
        right_states = np.zeros((num_vars, nx + 1, ny))
        
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    # Left boundary
                    left_states[:, i, j] = data.state[:, 0, j]
                    right_states[:, i, j] = data.state[:, 0, j]
                elif i == nx:
                    # Right boundary
                    left_states[:, i, j] = data.state[:, -1, j]
                    right_states[:, i, j] = data.state[:, -1, j]
                else:
                    # Interior interface: MUSCL reconstruction
                    # Left state: extrapolate from left cell
                    left_states[:, i, j] = data.state[:, i-1, j] + 0.5 * slopes[:, i-1, j]
                    # Right state: extrapolate from right cell
                    right_states[:, i, j] = data.state[:, i, j] - 0.5 * slopes[:, i, j]
        
        return left_states, right_states
    
    def reconstruct_y_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at y-direction interfaces using MUSCL.
        
        Args:
            data: FVM data container
            
        Returns:
            Tuple of (left_states, right_states) at y-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        
        # Compute MUSCL slopes in y-direction
        slopes = self._compute_muscl_slopes_y(data)
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx, ny + 1))
        right_states = np.zeros((num_vars, nx, ny + 1))
        
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    # Bottom boundary
                    left_states[:, i, j] = data.state[:, i, 0]
                    right_states[:, i, j] = data.state[:, i, 0]
                elif j == ny:
                    # Top boundary
                    left_states[:, i, j] = data.state[:, i, -1]
                    right_states[:, i, j] = data.state[:, i, -1]
                else:
                    # Interior interface: MUSCL reconstruction
                    # Left state: extrapolate from bottom cell
                    left_states[:, i, j] = data.state[:, i, j-1] + 0.5 * slopes[:, i, j-1]
                    # Right state: extrapolate from top cell
                    right_states[:, i, j] = data.state[:, i, j] - 0.5 * slopes[:, i, j]
        
        return left_states, right_states
    
    def _compute_muscl_slopes_x(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute MUSCL slopes in x-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for j in range(ny):
                for i in range(nx):
                    # Compute forward and backward differences
                    if i == 0:
                        # Left boundary: forward difference
                        if nx > 1:
                            forward_diff = data.state[var, i+1, j] - data.state[var, i, j]
                            backward_diff = forward_diff  # Use forward as backward
                        else:
                            forward_diff = backward_diff = 0.0
                    elif i == nx - 1:
                        # Right boundary: backward difference
                        backward_diff = data.state[var, i, j] - data.state[var, i-1, j]
                        forward_diff = backward_diff  # Use backward as forward
                    else:
                        # Interior: both differences available
                        forward_diff = data.state[var, i+1, j] - data.state[var, i, j]
                        backward_diff = data.state[var, i, j] - data.state[var, i-1, j]
                    
                    # MUSCL slope computation
                    slopes[var, i, j] = self._compute_muscl_slope(forward_diff, backward_diff)
        
        return slopes
    
    def _compute_muscl_slopes_y(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute MUSCL slopes in y-direction"""
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        slopes = np.zeros((num_vars, nx, ny))
        
        for var in range(num_vars):
            for i in range(nx):
                for j in range(ny):
                    # Compute forward and backward differences
                    if j == 0:
                        # Bottom boundary: forward difference
                        if ny > 1:
                            forward_diff = data.state[var, i, j+1] - data.state[var, i, j]
                            backward_diff = forward_diff  # Use forward as backward
                        else:
                            forward_diff = backward_diff = 0.0
                    elif j == ny - 1:
                        # Top boundary: backward difference
                        backward_diff = data.state[var, i, j] - data.state[var, i, j-1]
                        forward_diff = backward_diff  # Use backward as forward
                    else:
                        # Interior: both differences available
                        forward_diff = data.state[var, i, j+1] - data.state[var, i, j]
                        backward_diff = data.state[var, i, j] - data.state[var, i, j-1]
                    
                    # MUSCL slope computation
                    slopes[var, i, j] = self._compute_muscl_slope(forward_diff, backward_diff)
        
        return slopes
    
    def _compute_muscl_slope(self, forward_diff: float, backward_diff: float) -> float:
        """
        Compute MUSCL slope with limiting.
        
        Args:
            forward_diff: Forward difference (u_{i+1} - u_i)
            backward_diff: Backward difference (u_i - u_{i-1})
            
        Returns:
            Limited slope
        """
        # MUSCL interpolation
        # σ_i = limiter(r) * Δu_i
        # where r = (u_i - u_{i-1}) / (u_{i+1} - u_i) for forward bias
        
        if abs(forward_diff) < 1e-12:
            # Avoid division by zero
            return 0.0
        
        # Compute slope ratio
        r = backward_diff / forward_diff if abs(forward_diff) > 1e-12 else 0.0
        
        # Apply MUSCL limiter function
        phi = self._muscl_limiter_function(r)
        
        # MUSCL slope: combine with kappa parameter
        # Standard MUSCL: σ_i = φ(r) * Δu_{i+1/2}
        # where Δu_{i+1/2} = (1-κ)/2 * Δu_{i-1/2} + (1+κ)/2 * Δu_{i+1/2}
        
        upwind_slope = backward_diff
        central_slope = 0.5 * (forward_diff + backward_diff)
        
        # Combine based on kappa parameter
        base_slope = (1 - self.kappa) * upwind_slope + self.kappa * central_slope
        
        # Apply limiter
        return phi * base_slope
    
    def _muscl_limiter_function(self, r: float) -> float:
        """
        MUSCL limiter function φ(r).
        
        Args:
            r: Slope ratio
            
        Returns:
            Limiter value
        """
        if r <= 0:
            return 0.0
        
        # Apply selected limiter
        if self.limiter_type == 'minmod':
            return min(1.0, r)
        elif self.limiter_type == 'superbee':
            return max(0.0, min(2.0*r, 1.0), min(r, 2.0))
        elif self.limiter_type == 'van_leer':
            return (r + abs(r)) / (1.0 + abs(r))
        elif self.limiter_type == 'mc':
            return max(0.0, min(2.0*r, 0.5*(1.0+r), 2.0))
        else:
            # Default to minmod
            return min(1.0, r)
    
    def get_stencil_size(self) -> int:
        """MUSCL reconstruction needs 3-point stencil"""
        return 2
    
    def needs_boundary_treatment(self) -> bool:
        """MUSCL needs boundary treatment for high accuracy"""
        return True
    
    def get_kappa_parameter(self) -> float:
        """Get MUSCL kappa parameter"""
        return self.kappa
    
    def set_kappa_parameter(self, kappa: float):
        """Set MUSCL kappa parameter"""
        if not -1.0 <= kappa <= 1.0:
            raise ValueError("MUSCL parameter kappa must be in [-1, 1]")
        self.kappa = kappa