"""
2D Discontinuous Galerkin (DG) Finite Element Method

This module implements DG spatial discretization for 2D conservation laws
with polynomial orders P0, P1, and P2 (0th to 2nd order accuracy).
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict
from abc import ABC, abstractmethod
import scipy.special as sp

from fvm_framework.core.data_container import FVMDataContainer2D, GridGeometry
from .riemann_solvers import RiemannSolver, RiemannSolverFactory


class DGDataContainer2D:
    """
    Data container for 2D Discontinuous Galerkin method.
    
    Stores solution coefficients for polynomial basis functions.
    Data layout: [num_variables, nx, ny, num_basis_functions]
    """
    
    def __init__(self, geometry: GridGeometry, num_vars: int, polynomial_order: int):
        """
        Initialize DG data container.
        
        Args:
            geometry: Grid geometry
            num_vars: Number of conservation variables
            polynomial_order: Polynomial order (0, 1, or 2)
        """
        self.geometry = geometry
        self.num_vars = num_vars
        self.polynomial_order = polynomial_order
        self.nx, self.ny = geometry.nx, geometry.ny
        
        # Number of basis functions for tensor product basis
        self.num_basis = (polynomial_order + 1) ** 2
        
        # Solution coefficients [num_vars, nx, ny, num_basis]
        self.coefficients = np.zeros((num_vars, self.nx, self.ny, self.num_basis), dtype=np.float64)
        self.coefficients_new = np.zeros((num_vars, self.nx, self.ny, self.num_basis), dtype=np.float64)
        
        # Fluxes at interfaces
        self.flux_x = np.zeros((num_vars, self.nx + 1, self.ny, self.num_basis), dtype=np.float64)
        self.flux_y = np.zeros((num_vars, self.nx, self.ny + 1, self.num_basis), dtype=np.float64)
        
        # Mass matrix and its inverse
        self.mass_matrix = self._compute_mass_matrix()
        self.mass_matrix_inv = np.linalg.inv(self.mass_matrix)
        
        # Quadrature points and weights
        self.quad_points, self.quad_weights = self._setup_quadrature()
        
        # Basis functions evaluated at quadrature points
        self.basis_at_quad = self._evaluate_basis_at_quadrature()
        
    def _compute_mass_matrix(self) -> np.ndarray:
        """Compute mass matrix for the reference element"""
        mass = np.zeros((self.num_basis, self.num_basis))
        
        # Use higher-order quadrature for accurate mass matrix
        quad_points, quad_weights = self._gauss_legendre_2d(self.polynomial_order + 1)
        
        for i in range(self.num_basis):
            for j in range(self.num_basis):
                integral = 0.0
                for q, (xi, eta) in enumerate(quad_points):
                    phi_i = self._evaluate_basis_function(i, xi, eta)
                    phi_j = self._evaluate_basis_function(j, xi, eta)
                    integral += phi_i * phi_j * quad_weights[q]
                mass[i, j] = integral
                
        return mass
    
    def _setup_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Setup quadrature points and weights"""
        # Use sufficient quadrature points for polynomial order
        quad_order = max(2, self.polynomial_order + 1)
        return self._gauss_legendre_2d(quad_order)
    
    def _gauss_legendre_2d(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D Gauss-Legendre quadrature points and weights.
        
        Args:
            order: Number of points in each direction
            
        Returns:
            Tuple of (points, weights) where points is Nx2 array
        """
        # 1D Gauss-Legendre points and weights
        xi_1d, w_1d = sp.legendre.leggauss(order)
        
        # Tensor product for 2D
        points = []
        weights = []
        
        for i in range(order):
            for j in range(order):
                points.append([xi_1d[i], xi_1d[j]])
                weights.append(w_1d[i] * w_1d[j])
        
        return np.array(points), np.array(weights)
    
    def _evaluate_basis_at_quadrature(self) -> np.ndarray:
        """Evaluate all basis functions at quadrature points"""
        num_quad = len(self.quad_points)
        basis_vals = np.zeros((self.num_basis, num_quad))
        
        for i in range(self.num_basis):
            for q, (xi, eta) in enumerate(self.quad_points):
                basis_vals[i, q] = self._evaluate_basis_function(i, xi, eta)
        
        return basis_vals
    
    def _evaluate_basis_function(self, basis_index: int, xi: float, eta: float) -> float:
        """
        Evaluate basis function at reference coordinates (xi, eta) ∈ [-1,1]²
        
        Uses tensor product Legendre polynomials.
        """
        # Convert linear index to 2D indices
        order = self.polynomial_order
        i = basis_index // (order + 1)
        j = basis_index % (order + 1)
        
        # Legendre polynomials
        if i == 0:
            phi_xi = 1.0
        elif i == 1:
            phi_xi = xi
        elif i == 2:
            phi_xi = 0.5 * (3 * xi**2 - 1)
        else:
            phi_xi = 0.0  # Higher orders not implemented
            
        if j == 0:
            phi_eta = 1.0
        elif j == 1:
            phi_eta = eta
        elif j == 2:
            phi_eta = 0.5 * (3 * eta**2 - 1)
        else:
            phi_eta = 0.0  # Higher orders not implemented
            
        return phi_xi * phi_eta
    
    def get_cell_averages(self) -> np.ndarray:
        """
        Extract cell averages (first coefficient for each cell).
        
        Returns:
            Array with shape [num_vars, nx, ny]
        """
        # First coefficient corresponds to the constant (P0) mode
        return self.coefficients[:, :, :, 0].copy()
    
    def set_cell_averages(self, averages: np.ndarray):
        """Set cell averages and zero higher-order modes"""
        self.coefficients.fill(0.0)
        self.coefficients[:, :, :, 0] = averages
    
    def swap_coefficients(self):
        """Swap current and new coefficient arrays"""
        self.coefficients, self.coefficients_new = self.coefficients_new, self.coefficients


class DGSolver2D:
    """
    2D Discontinuous Galerkin solver for conservation laws.
    
    Implements spatial discretization using DG method with various polynomial orders.
    """
    
    def __init__(self, polynomial_order: int = 1, riemann_solver: str = 'hllc',
                 flux_function: Optional[Callable] = None):
        """
        Initialize DG solver.
        
        Args:
            polynomial_order: Polynomial order (0, 1, or 2)
            riemann_solver: Type of Riemann solver
            flux_function: Physical flux function
        """
        self.polynomial_order = polynomial_order
        self.riemann_solver = RiemannSolverFactory.create(riemann_solver)
        self.flux_function = flux_function or self._euler_flux
        
        # Slope limiter for stability
        self.use_limiter = polynomial_order > 0
        self.limiter_type = 'minmod'
        
    def compute_spatial_residual(self, dg_data: DGDataContainer2D, **kwargs) -> np.ndarray:
        """
        Compute spatial residual for DG discretization.
        
        Args:
            dg_data: DG data container
            **kwargs: Additional parameters
            
        Returns:
            Residual array with same shape as coefficients
        """
        gamma = kwargs.get('gamma', 1.4)
        
        # Initialize residual
        residual = np.zeros_like(dg_data.coefficients)
        
        # Volume integral contribution
        self._compute_volume_integral(dg_data, residual, gamma)
        
        # Surface integral contribution (numerical fluxes)
        self._compute_surface_integral(dg_data, residual, gamma)
        
        # Apply inverse mass matrix
        self._apply_inverse_mass_matrix(dg_data, residual)
        
        # Apply slope limiter if needed
        if self.use_limiter:
            self._apply_slope_limiter(dg_data, residual)
        
        return residual
    
    def _compute_volume_integral(self, dg_data: DGDataContainer2D, residual: np.ndarray, gamma: float):
        """Compute volume integral contribution"""
        nx, ny = dg_data.nx, dg_data.ny
        num_vars = dg_data.num_vars
        
        for i in range(nx):
            for j in range(ny):
                for var in range(num_vars):
                    # Get coefficients for this cell
                    cell_coeffs = dg_data.coefficients[var, i, j, :]
                    
                    # Compute flux at quadrature points
                    for q, (xi, eta) in enumerate(dg_data.quad_points):
                        # Evaluate solution at quadrature point
                        u_quad = np.zeros(num_vars)
                        for v in range(num_vars):
                            u_quad[v] = np.dot(dg_data.coefficients[v, i, j, :], 
                                             dg_data.basis_at_quad[:, q])
                        
                        # Compute physical flux
                        flux_x, flux_y = self._compute_physical_flux(u_quad, gamma)
                        
                        # Add to volume integral
                        weight = dg_data.quad_weights[q]
                        for k in range(dg_data.num_basis):
                            # Derivative of basis function (simplified for tensor product)
                            dphi_dxi, dphi_deta = self._compute_basis_derivatives(k, xi, eta)
                            
                            # Chain rule: ∂φ/∂x = (2/dx) * ∂φ/∂ξ
                            dphi_dx = (2.0 / dg_data.geometry.dx) * dphi_dxi
                            dphi_dy = (2.0 / dg_data.geometry.dy) * dphi_deta
                            
                            # Add flux divergence
                            residual[var, i, j, k] -= weight * (
                                flux_x[var] * dphi_dx + flux_y[var] * dphi_dy
                            )
    
    def _compute_surface_integral(self, dg_data: DGDataContainer2D, residual: np.ndarray, gamma: float):
        """Compute surface integral (numerical flux) contribution"""
        nx, ny = dg_data.nx, dg_data.ny
        
        # X-direction interfaces
        for i in range(nx + 1):
            for j in range(ny):
                self._compute_x_interface_flux(dg_data, residual, i, j, gamma)
        
        # Y-direction interfaces
        for i in range(nx):
            for j in range(ny + 1):
                self._compute_y_interface_flux(dg_data, residual, i, j, gamma)
    
    def _compute_x_interface_flux(self, dg_data: DGDataContainer2D, residual: np.ndarray, 
                                 i: int, j: int, gamma: float):
        """Compute numerical flux at x-direction interface"""
        nx, ny = dg_data.nx, dg_data.ny
        num_vars = dg_data.num_vars
        
        # Get left and right states
        if i == 0:
            # Left boundary
            u_left = u_right = self._evaluate_solution_at_interface(dg_data, 0, j, 'right')
        elif i == nx:
            # Right boundary
            u_left = u_right = self._evaluate_solution_at_interface(dg_data, nx-1, j, 'left')
        else:
            # Interior interface
            u_left = self._evaluate_solution_at_interface(dg_data, i-1, j, 'right')
            u_right = self._evaluate_solution_at_interface(dg_data, i, j, 'left')
        
        # Compute numerical flux
        numerical_flux = self.riemann_solver.solve(u_left, u_right, 0, gamma)
        
        # Add to surface integral
        for var in range(num_vars):
            # Left cell contribution (if exists)
            if i > 0:
                for k in range(dg_data.num_basis):
                    # Basis function at right edge (xi = 1)
                    phi_val = self._evaluate_basis_at_edge(k, 1, 0)  # Right edge
                    residual[var, i-1, j, k] += numerical_flux[var] * phi_val / dg_data.geometry.dx
            
            # Right cell contribution (if exists)
            if i < nx:
                for k in range(dg_data.num_basis):
                    # Basis function at left edge (xi = -1)
                    phi_val = self._evaluate_basis_at_edge(k, -1, 0)  # Left edge
                    residual[var, i, j, k] -= numerical_flux[var] * phi_val / dg_data.geometry.dx
    
    def _compute_y_interface_flux(self, dg_data: DGDataContainer2D, residual: np.ndarray,
                                 i: int, j: int, gamma: float):
        """Compute numerical flux at y-direction interface"""
        nx, ny = dg_data.nx, dg_data.ny
        num_vars = dg_data.num_vars
        
        # Get bottom and top states
        if j == 0:
            # Bottom boundary
            u_bottom = u_top = self._evaluate_solution_at_interface(dg_data, i, 0, 'top')
        elif j == ny:
            # Top boundary
            u_bottom = u_top = self._evaluate_solution_at_interface(dg_data, i, ny-1, 'bottom')
        else:
            # Interior interface
            u_bottom = self._evaluate_solution_at_interface(dg_data, i, j-1, 'top')
            u_top = self._evaluate_solution_at_interface(dg_data, i, j, 'bottom')
        
        # Compute numerical flux
        numerical_flux = self.riemann_solver.solve(u_bottom, u_top, 1, gamma)
        
        # Add to surface integral
        for var in range(num_vars):
            # Bottom cell contribution (if exists)
            if j > 0:
                for k in range(dg_data.num_basis):
                    # Basis function at top edge (eta = 1)
                    phi_val = self._evaluate_basis_at_edge(k, 0, 1)  # Top edge
                    residual[var, i, j-1, k] += numerical_flux[var] * phi_val / dg_data.geometry.dy
            
            # Top cell contribution (if exists)
            if j < ny:
                for k in range(dg_data.num_basis):
                    # Basis function at bottom edge (eta = -1)
                    phi_val = self._evaluate_basis_at_edge(k, 0, -1)  # Bottom edge
                    residual[var, i, j, k] -= numerical_flux[var] * phi_val / dg_data.geometry.dy
    
    def _evaluate_solution_at_interface(self, dg_data: DGDataContainer2D, i: int, j: int, 
                                       side: str) -> np.ndarray:
        """
        Evaluate solution at cell interface.
        
        Args:
            dg_data: DG data container
            i, j: Cell indices
            side: 'left', 'right', 'bottom', or 'top'
            
        Returns:
            Solution vector at interface
        """
        num_vars = dg_data.num_vars
        u_interface = np.zeros(num_vars)
        
        # Map side to reference coordinates
        if side == 'left':
            xi, eta = -1.0, 0.0
        elif side == 'right':
            xi, eta = 1.0, 0.0
        elif side == 'bottom':
            xi, eta = 0.0, -1.0
        elif side == 'top':
            xi, eta = 0.0, 1.0
        else:
            xi, eta = 0.0, 0.0  # Cell center
        
        # Evaluate solution
        for var in range(num_vars):
            for k in range(dg_data.num_basis):
                phi_val = dg_data._evaluate_basis_function(k, xi, eta)
                u_interface[var] += dg_data.coefficients[var, i, j, k] * phi_val
        
        return u_interface
    
    def _evaluate_basis_at_edge(self, basis_index: int, xi: float, eta: float) -> float:
        """Evaluate basis function at edge coordinates"""
        order = self.polynomial_order
        i = basis_index // (order + 1)
        j = basis_index % (order + 1)
        
        # Legendre polynomials
        if i == 0:
            phi_xi = 1.0
        elif i == 1:
            phi_xi = xi
        elif i == 2:
            phi_xi = 0.5 * (3 * xi**2 - 1)
        else:
            phi_xi = 0.0
            
        if j == 0:
            phi_eta = 1.0
        elif j == 1:
            phi_eta = eta
        elif j == 2:
            phi_eta = 0.5 * (3 * eta**2 - 1)
        else:
            phi_eta = 0.0
            
        return phi_xi * phi_eta
    
    def _compute_basis_derivatives(self, basis_index: int, xi: float, eta: float) -> Tuple[float, float]:
        """Compute derivatives of basis function with respect to reference coordinates"""
        order = self.polynomial_order
        i = basis_index // (order + 1)
        j = basis_index % (order + 1)
        
        # Derivatives of Legendre polynomials
        if i == 0:
            phi_xi = 1.0
            dphi_xi = 0.0
        elif i == 1:
            phi_xi = xi
            dphi_xi = 1.0
        elif i == 2:
            phi_xi = 0.5 * (3 * xi**2 - 1)
            dphi_xi = 3.0 * xi
        else:
            phi_xi = dphi_xi = 0.0
            
        if j == 0:
            phi_eta = 1.0
            dphi_eta = 0.0
        elif j == 1:
            phi_eta = eta
            dphi_eta = 1.0
        elif j == 2:
            phi_eta = 0.5 * (3 * eta**2 - 1)
            dphi_eta = 3.0 * eta
        else:
            phi_eta = dphi_eta = 0.0
        
        # Product rule for tensor product
        dphi_dxi = dphi_xi * phi_eta
        dphi_deta = phi_xi * dphi_eta
        
        return dphi_dxi, dphi_deta
    
    def _apply_inverse_mass_matrix(self, dg_data: DGDataContainer2D, residual: np.ndarray):
        """Apply inverse mass matrix to residual"""
        nx, ny, num_vars = dg_data.nx, dg_data.ny, dg_data.num_vars
        
        for i in range(nx):
            for j in range(ny):
                for var in range(num_vars):
                    # Apply M^{-1} to residual vector
                    residual[var, i, j, :] = np.dot(dg_data.mass_matrix_inv, 
                                                   residual[var, i, j, :])
    
    def _apply_slope_limiter(self, dg_data: DGDataContainer2D, residual: np.ndarray):
        """Apply slope limiter for stability (simplified implementation)"""
        if self.polynomial_order == 0:
            return  # No limiting needed for P0
        
        # This is a simplified limiter - a full implementation would
        # require more sophisticated TVB limiting
        nx, ny, num_vars = dg_data.nx, dg_data.ny, dg_data.num_vars
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for var in range(num_vars):
                    # Get cell averages of neighbors
                    u_center = dg_data.coefficients[var, i, j, 0]
                    u_left = dg_data.coefficients[var, i-1, j, 0]
                    u_right = dg_data.coefficients[var, i+1, j, 0]
                    u_bottom = dg_data.coefficients[var, i, j-1, 0]
                    u_top = dg_data.coefficients[var, i, j+1, 0]
                    
                    # Apply minmod limiter to slopes
                    if self.polynomial_order >= 1:
                        # X-direction slope (coefficient 1)
                        slope_x = dg_data.coefficients[var, i, j, 1]
                        left_diff = u_center - u_left
                        right_diff = u_right - u_center
                        limited_slope_x = self._minmod_limiter(slope_x, left_diff, right_diff)
                        dg_data.coefficients[var, i, j, 1] = limited_slope_x
                        
                        # Y-direction slope (coefficient 2 if num_basis >= 3)
                        if dg_data.num_basis >= 3:
                            slope_y = dg_data.coefficients[var, i, j, 2]
                            bottom_diff = u_center - u_bottom
                            top_diff = u_top - u_center
                            limited_slope_y = self._minmod_limiter(slope_y, bottom_diff, top_diff)
                            dg_data.coefficients[var, i, j, 2] = limited_slope_y
    
    def _minmod_limiter(self, slope: float, left_diff: float, right_diff: float) -> float:
        """Minmod slope limiter"""
        if left_diff * right_diff <= 0:
            return 0.0
        elif abs(slope) <= abs(left_diff) and abs(slope) <= abs(right_diff):
            return slope
        elif abs(left_diff) <= abs(right_diff):
            return left_diff
        else:
            return right_diff
    
    def _compute_physical_flux(self, u: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute physical flux vectors F(u) and G(u).
        
        Args:
            u: Conservative variable vector
            gamma: Heat capacity ratio
            
        Returns:
            Tuple of (flux_x, flux_y)
        """
        return self.flux_function(u, gamma)
    
    def _euler_flux(self, u: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Default Euler flux function"""
        rho, rho_u, rho_v, rho_w, E = u[0], u[1], u[2], u[3], u[4]
        
        # Avoid division by zero
        rho = max(rho, 1e-15)
        u_vel = rho_u / rho
        v_vel = rho_v / rho
        w_vel = rho_w / rho
        
        # Pressure
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        p = max((gamma - 1.0) * (E - kinetic_energy), 1e-15)
        
        # X-direction flux
        flux_x = np.array([
            rho_u,
            rho_u * u_vel + p,
            rho_u * v_vel,
            rho_u * w_vel,
            u_vel * (E + p)
        ])
        
        # Y-direction flux
        flux_y = np.array([
            rho_v,
            rho_v * u_vel,
            rho_v * v_vel + p,
            rho_v * w_vel,
            v_vel * (E + p)
        ])
        
        return flux_x, flux_y
    
    def project_initial_condition(self, dg_data: DGDataContainer2D, 
                                 initial_condition: Callable, **kwargs):
        """
        Project initial condition onto DG space using L2 projection.
        
        Args:
            dg_data: DG data container
            initial_condition: Function that returns initial state at (x, y)
            **kwargs: Additional parameters
        """
        nx, ny = dg_data.nx, dg_data.ny
        num_vars = dg_data.num_vars
        
        for i in range(nx):
            for j in range(ny):
                # Cell center coordinates
                x_center = dg_data.geometry.x_min + (i + 0.5) * dg_data.geometry.dx
                y_center = dg_data.geometry.y_min + (j + 0.5) * dg_data.geometry.dy
                
                # L2 projection
                for k in range(dg_data.num_basis):
                    projection_integral = np.zeros(num_vars)
                    
                    for q, (xi, eta) in enumerate(dg_data.quad_points):
                        # Map to physical coordinates
                        x_phys = x_center + 0.5 * dg_data.geometry.dx * xi
                        y_phys = y_center + 0.5 * dg_data.geometry.dy * eta
                        
                        # Evaluate initial condition
                        u_init = initial_condition(x_phys, y_phys, **kwargs)
                        
                        # Add to integral
                        phi_val = dg_data._evaluate_basis_function(k, xi, eta)
                        weight = dg_data.quad_weights[q]
                        projection_integral += u_init * phi_val * weight
                    
                    # Apply inverse mass matrix element
                    for var in range(num_vars):
                        dg_data.coefficients[var, i, j, k] = (projection_integral[var] / 
                                                            dg_data.mass_matrix[k, k])


class DGIntegrationInterface:
    """
    Interface to integrate DG solver with existing time integration framework.
    """
    
    def __init__(self, dg_solver: DGSolver2D, dg_data: DGDataContainer2D):
        self.dg_solver = dg_solver
        self.dg_data = dg_data
        
    def compute_spatial_residual(self, **kwargs) -> np.ndarray:
        """Compute spatial residual for time integration"""
        return self.dg_solver.compute_spatial_residual(self.dg_data, **kwargs)
    
    def get_solution_for_output(self) -> np.ndarray:
        """Get solution in format compatible with visualization"""
        return self.dg_data.get_cell_averages()
    
    def apply_boundary_conditions(self, bc_type: str = 'periodic'):
        """Apply boundary conditions (simplified for DG)"""
        # DG boundary conditions are handled through numerical fluxes
        # This is a placeholder for more sophisticated BC implementations
        pass