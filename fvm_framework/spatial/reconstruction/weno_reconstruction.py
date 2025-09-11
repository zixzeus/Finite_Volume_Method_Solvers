"""
WENO (Weighted Essentially Non-Oscillatory) Reconstruction

This module implements high-order WENO reconstruction schemes that provide
high accuracy in smooth regions while avoiding spurious oscillations near
discontinuities through adaptive stencil weighting.
"""

import numpy as np
from typing import Tuple
from fvm_framework.core.data_container import FVMDataContainer2D
from .base_reconstruction import HighOrderReconstruction


class WENOReconstruction(HighOrderReconstruction):
    """
    WENO (Weighted Essentially Non-Oscillatory) reconstruction.
    
    Implements WENO schemes that use multiple stencils with adaptive weights
    to achieve high-order accuracy in smooth regions while maintaining
    essentially non-oscillatory behavior near discontinuities.
    
    Currently supports WENO3 (3rd order) and WENO5 (5th order).
    """
    
    def __init__(self, order: int = 3, epsilon: float = 1e-6):
        """
        Initialize WENO reconstruction.
        
        Args:
            order: WENO order (3 or 5 supported)
            epsilon: Small parameter to avoid division by zero
        """
        if order not in [3, 5]:
            raise ValueError("WENO reconstruction currently supports only orders 3 and 5")
        
        super().__init__("WENO", order)
        self.epsilon = epsilon
        
        # WENO coefficients
        if order == 3:
            self._init_weno3_coefficients()
        elif order == 5:
            self._init_weno5_coefficients()
    
    def _init_weno3_coefficients(self):
        """Initialize WENO3 coefficients"""
        # Ideal weights (for linear reconstruction)
        self.gamma = np.array([1.0/3.0, 2.0/3.0])
        
        # Reconstruction coefficients for each stencil
        # Stencil 0: cells (i-1, i)
        # Stencil 1: cells (i, i+1)
        self.c = np.array([
            [-0.5, 1.5],    # Stencil 0 coefficients
            [0.5, 0.5]      # Stencil 1 coefficients
        ])
    
    def _init_weno5_coefficients(self):
        """Initialize WENO5 coefficients"""
        # Ideal weights
        self.gamma = np.array([1.0/10.0, 6.0/10.0, 3.0/10.0])
        
        # Reconstruction coefficients for each stencil
        # Stencil 0: cells (i-2, i-1, i)
        # Stencil 1: cells (i-1, i, i+1)  
        # Stencil 2: cells (i, i+1, i+2)
        self.c = np.array([
            [1.0/3.0, -7.0/6.0, 11.0/6.0],      # Stencil 0 coefficients
            [-1.0/6.0, 5.0/6.0, 1.0/3.0],       # Stencil 1 coefficients
            [1.0/3.0, 5.0/6.0, -1.0/6.0]        # Stencil 2 coefficients
        ])
    
    def reconstruct_x_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at x-direction interfaces using WENO.
        
        Args:
            data: FVM data container with ghost cells
            
        Returns:
            Tuple of (left_states, right_states) at x-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx + 1, ny))
        right_states = np.zeros((num_vars, nx + 1, ny))
        
        for var in range(num_vars):
            for j in range(ny):
                # Map interior j to state array index
                j_state = j + ng
                
                # Extract 1D slice INCLUDING ghost cells for WENO stencils
                u_slice = data.state[var, :, j_state]
                
                # Reconstruct left and right states at interior interfaces only
                u_left, u_right = self._weno_reconstruct_1d(u_slice, nx, ng, direction='x')
                
                left_states[var, :, j] = u_left
                right_states[var, :, j] = u_right
        
        return left_states, right_states
    
    def reconstruct_y_interfaces(self, data: FVMDataContainer2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct states at y-direction interfaces using WENO.
        
        Args:
            data: FVM data container with ghost cells
            
        Returns:
            Tuple of (left_states, right_states) at y-interfaces
        """
        nx, ny, num_vars = data.nx, data.ny, data.num_vars
        ng = data.ng
        
        # Initialize interface states
        left_states = np.zeros((num_vars, nx, ny + 1))
        right_states = np.zeros((num_vars, nx, ny + 1))
        
        for var in range(num_vars):
            for i in range(nx):
                # Map interior i to state array index
                i_state = i + ng
                
                # Extract 1D slice INCLUDING ghost cells for WENO stencils
                u_slice = data.state[var, i_state, :]
                
                # Reconstruct left and right states at interior interfaces only
                u_left, u_right = self._weno_reconstruct_1d(u_slice, ny, ng, direction='y')
                
                left_states[var, i, :] = u_left
                right_states[var, i, :] = u_right
        
        return left_states, right_states
    
    def _weno_reconstruct_1d(self, u: np.ndarray, n_interior: int, ng: int, direction: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform WENO reconstruction on 1D array with ghost cells.
        
        Args:
            u: 1D array of cell-centered values including ghost cells
            n_interior: Number of interior cells
            ng: Number of ghost cells
            direction: 'x' or 'y' for boundary handling
            
        Returns:
            Tuple of (left_interface_states, right_interface_states) for interior interfaces
        """
        # Initialize interface states for interior interfaces only
        u_left = np.zeros(n_interior + 1)
        u_right = np.zeros(n_interior + 1)
        
        if self.order == 3:
            # WENO3 reconstruction
            for i in range(n_interior + 1):
                if i == 0:
                    # Left boundary interface: use ghost cell and first interior
                    u_left[i] = u[ng - 1]  # Left ghost cell
                    u_right[i] = u[ng]     # First interior cell
                elif i == n_interior:
                    # Right boundary interface: use last interior and ghost cell
                    u_left[i] = u[ng + n_interior - 1]  # Last interior cell
                    u_right[i] = u[ng + n_interior]     # Right ghost cell
                else:
                    # Interior interface i (between interior cells i-1 and i)
                    cell_idx = ng + i  # Map to full array index
                    u_left[i] = self._weno3_reconstruct_left(u, cell_idx)
                    u_right[i] = self._weno3_reconstruct_right(u, cell_idx)
                    
        elif self.order == 5:
            # WENO5 reconstruction
            for i in range(n_interior + 1):
                if i == 0:
                    # Left boundary interface
                    u_left[i] = u[ng - 1]  # Left ghost cell
                    u_right[i] = u[ng]     # First interior cell
                elif i == n_interior:
                    # Right boundary interface
                    u_left[i] = u[ng + n_interior - 1]  # Last interior cell
                    u_right[i] = u[ng + n_interior]     # Right ghost cell
                elif i <= 1 or i >= n_interior - 1:
                    # Near boundaries: fall back to simple linear reconstruction
                    cell_idx = ng + i
                    if i == 1 and n_interior > 1:
                        # Simple linear reconstruction for near-boundary
                        u_left[i] = 0.5 * (u[cell_idx - 1] + u[cell_idx])
                        u_right[i] = 0.5 * (u[cell_idx] + u[cell_idx + 1])
                    elif i == n_interior - 1 and n_interior > 1:
                        u_left[i] = 0.5 * (u[cell_idx - 1] + u[cell_idx])
                        u_right[i] = 0.5 * (u[cell_idx] + u[cell_idx + 1])
                    else:
                        # Fallback to cell-centered values
                        u_left[i] = u[ng + i - 1]
                        u_right[i] = u[ng + i]
                else:
                    # Interior interface with full WENO5 stencil
                    cell_idx = ng + i
                    u_left[i] = self._weno5_reconstruct_left(u, cell_idx)
                    u_right[i] = self._weno5_reconstruct_right(u, cell_idx)
        
        return u_left, u_right
    
    def _weno3_reconstruct_left(self, u: np.ndarray, i: int) -> float:
        """WENO3 reconstruction for left state at interface i"""
        # Two candidate stencils
        # S0: (i-2, i-1) -> reconstruct to i-1/2
        # S1: (i-1, i) -> reconstruct to i-1/2
        
        # Handle boundaries
        if i == 1:
            # Only one stencil available
            return 0.5 * (u[0] + u[1])
        
        # Stencil values
        v0 = np.array([u[i-2], u[i-1]]) if i >= 2 else np.array([u[0], u[0]])
        v1 = np.array([u[i-1], u[i]])
        
        # Candidate reconstructions
        u0 = np.dot(self.c[0], v0)
        u1 = np.dot(self.c[1], v1)
        
        # Smoothness indicators (simple difference-based)
        beta0 = (v0[1] - v0[0])**2 if i >= 2 else 0
        beta1 = (v1[1] - v1[0])**2
        
        # Weights
        alpha0 = self.gamma[0] / (self.epsilon + beta0)**2
        alpha1 = self.gamma[1] / (self.epsilon + beta1)**2
        
        w0 = alpha0 / (alpha0 + alpha1)
        w1 = alpha1 / (alpha0 + alpha1)
        
        return w0 * u0 + w1 * u1
    
    def _weno3_reconstruct_right(self, u: np.ndarray, i: int) -> float:
        """WENO3 reconstruction for right state at interface i"""
        # Two candidate stencils for right state
        # S0: (i-1, i) -> reconstruct to i+1/2  
        # S1: (i, i+1) -> reconstruct to i+1/2
        
        n = len(u)
        if i >= n - 1:
            return 0.5 * (u[-2] + u[-1]) if n > 1 else u[-1]
        
        # Stencil values
        v0 = np.array([u[i-1], u[i]]) if i >= 1 else np.array([u[0], u[1]])
        v1 = np.array([u[i], u[i+1]]) if i < n - 1 else np.array([u[-1], u[-1]])
        
        # Candidate reconstructions (mirror coefficients for right reconstruction)
        u0 = np.dot(self.c[1][::-1], v0)  # Reversed coefficients
        u1 = np.dot(self.c[0][::-1], v1)  # Reversed coefficients
        
        # Smoothness indicators
        beta0 = (v0[1] - v0[0])**2
        beta1 = (v1[1] - v1[0])**2 if i < n - 1 else 0
        
        # Weights (reversed gamma)
        alpha0 = self.gamma[1] / (self.epsilon + beta0)**2
        alpha1 = self.gamma[0] / (self.epsilon + beta1)**2
        
        w0 = alpha0 / (alpha0 + alpha1)
        w1 = alpha1 / (alpha0 + alpha1)
        
        return w0 * u0 + w1 * u1
    
    def _weno5_reconstruct_left(self, u: np.ndarray, i: int) -> float:
        """WENO5 reconstruction for left state at interface i"""
        # Three candidate stencils
        # S0: (i-3, i-2, i-1) -> reconstruct to i-1/2
        # S1: (i-2, i-1, i) -> reconstruct to i-1/2
        # S2: (i-1, i, i+1) -> reconstruct to i-1/2
        
        # Stencil values (with boundary handling)
        v0 = self._get_stencil_values(u, [i-3, i-2, i-1])
        v1 = self._get_stencil_values(u, [i-2, i-1, i])
        v2 = self._get_stencil_values(u, [i-1, i, i+1])
        
        # Candidate reconstructions
        u0 = np.dot(self.c[0], v0)
        u1 = np.dot(self.c[1], v1)
        u2 = np.dot(self.c[2], v2)
        
        # Smoothness indicators (Jiang-Shu)
        beta0 = self._smoothness_indicator_js(v0)
        beta1 = self._smoothness_indicator_js(v1)
        beta2 = self._smoothness_indicator_js(v2)
        
        # Weights
        alpha0 = self.gamma[0] / (self.epsilon + beta0)**2
        alpha1 = self.gamma[1] / (self.epsilon + beta1)**2
        alpha2 = self.gamma[2] / (self.epsilon + beta2)**2
        
        alpha_sum = alpha0 + alpha1 + alpha2
        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum
        
        return w0 * u0 + w1 * u1 + w2 * u2
    
    def _weno5_reconstruct_right(self, u: np.ndarray, i: int) -> float:
        """WENO5 reconstruction for right state at interface i"""
        # Mirror operation for right reconstruction
        # Use reversed stencils and coefficients
        
        # Stencil values (with boundary handling)
        v0 = self._get_stencil_values(u, [i+2, i+1, i])
        v1 = self._get_stencil_values(u, [i+1, i, i-1])
        v2 = self._get_stencil_values(u, [i, i-1, i-2])
        
        # Candidate reconstructions (reversed coefficients)
        u0 = np.dot(self.c[0][::-1], v0)
        u1 = np.dot(self.c[1][::-1], v1)
        u2 = np.dot(self.c[2][::-1], v2)
        
        # Smoothness indicators
        beta0 = self._smoothness_indicator_js(v0)
        beta1 = self._smoothness_indicator_js(v1)
        beta2 = self._smoothness_indicator_js(v2)
        
        # Weights (reversed gamma)
        alpha0 = self.gamma[2] / (self.epsilon + beta0)**2
        alpha1 = self.gamma[1] / (self.epsilon + beta1)**2
        alpha2 = self.gamma[0] / (self.epsilon + beta2)**2
        
        alpha_sum = alpha0 + alpha1 + alpha2
        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum
        
        return w0 * u0 + w1 * u1 + w2 * u2
    
    def _get_stencil_values(self, u: np.ndarray, indices: list) -> np.ndarray:
        """Get stencil values with robust boundary handling"""
        n = len(u)
        values = np.zeros(len(indices))
        
        for k, idx in enumerate(indices):
            if idx < 0:
                # Linear extrapolation for left boundary
                if n >= 2:
                    # Use linear extrapolation: u[0] + (u[0] - u[1]) * |idx|
                    extrap = u[0] + (u[0] - u[1]) * abs(idx)
                    # Clamp to reasonable bounds to prevent extreme values
                    values[k] = max(min(extrap, 10.0 * u[0]), 0.1 * u[0])
                else:
                    values[k] = u[0]
            elif idx >= n:
                # Linear extrapolation for right boundary  
                if n >= 2:
                    # Use linear extrapolation: u[-1] + (u[-1] - u[-2]) * (idx - (n-1))
                    extrap = u[-1] + (u[-1] - u[-2]) * (idx - (n - 1))
                    # Clamp to reasonable bounds to prevent extreme values
                    values[k] = max(min(extrap, 10.0 * u[-1]), 0.1 * u[-1])
                else:
                    values[k] = u[-1]
            else:
                values[k] = u[idx]
        
        return values
    
    def _smoothness_indicator_js(self, v: np.ndarray) -> float:
        """
        Jiang-Shu smoothness indicator for 3-point stencil.
        
        Args:
            v: Stencil values [v0, v1, v2]
            
        Returns:
            Smoothness indicator value
        """
        if len(v) == 2:
            # For WENO3
            return (v[1] - v[0])**2
        elif len(v) == 3:
            # For WENO5
            d0 = v[1] - v[0]
            d1 = v[2] - v[1]
            return 13.0/12.0 * (d0 - d1)**2 + 0.25 * (d0 - 3*d1)**2
        else:
            return 0.0
    
    def get_stencil_size(self) -> int:
        """Get stencil size for WENO"""
        return (self.order + 1) // 2
    
    def needs_boundary_treatment(self) -> bool:
        """WENO needs special boundary treatment"""
        return True