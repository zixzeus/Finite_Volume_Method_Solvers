"""
Riemann Solvers for 2D Finite Volume Method

This module implements various approximate Riemann solvers including
HLL, HLLC, and HLLD for compressible flow simulations.
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from core.data_container import FVMDataContainer2D


class RiemannSolver(ABC):
    """Abstract base class for Riemann solvers"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def solve(self, u_left: np.ndarray, u_right: np.ndarray, 
              direction: int, gamma: float = 1.4) -> np.ndarray:
        """
        Solve Riemann problem at interface.
        
        Args:
            u_left: Left state vector [rho, rho*u, rho*v, rho*w, E]
            u_right: Right state vector
            direction: 0 for x-direction, 1 for y-direction
            gamma: Heat capacity ratio
            
        Returns:
            Numerical flux at the interface
        """
        pass
    
    def compute_primitive_variables(self, u: np.ndarray, gamma: float) -> Tuple[float, ...]:
        """Convert conservative to primitive variables"""
        rho = max(u[0], 1e-15)
        u_vel = u[1] / rho
        v_vel = u[2] / rho
        w_vel = u[3] / rho
        E = u[4]
        
        # Pressure
        kinetic_energy = 0.5 * rho * (u_vel**2 + v_vel**2 + w_vel**2)
        p = max((gamma - 1.0) * (E - kinetic_energy), 1e-15)
        
        # Sound speed
        c = np.sqrt(gamma * p / rho)
        
        return rho, u_vel, v_vel, w_vel, p, c
    
    def compute_physical_flux(self, u: np.ndarray, direction: int, gamma: float) -> np.ndarray:
        """Compute physical flux vector"""
        rho, u_vel, v_vel, w_vel, p, c = self.compute_primitive_variables(u, gamma)
        E = u[4]
        
        if direction == 0:  # x-direction flux
            return np.array([
                rho * u_vel,                    # Mass flux
                rho * u_vel**2 + p,            # x-momentum flux
                rho * u_vel * v_vel,           # y-momentum flux
                rho * u_vel * w_vel,           # z-momentum flux
                u_vel * (E + p)                # Energy flux
            ])
        else:  # y-direction flux
            return np.array([
                rho * v_vel,                    # Mass flux
                rho * v_vel * u_vel,           # x-momentum flux
                rho * v_vel**2 + p,            # y-momentum flux
                rho * v_vel * w_vel,           # z-momentum flux
                v_vel * (E + p)                # Energy flux
            ])


class HLLSolver(RiemannSolver):
    """
    HLL (Harten-Lax-van Leer) Riemann solver.
    
    This is a two-wave approximate solver that estimates the left and right
    going wave speeds and assumes a constant intermediate state.
    """
    
    def __init__(self):
        super().__init__("HLL")
    
    def solve(self, u_left: np.ndarray, u_right: np.ndarray,
              direction: int, gamma: float = 1.4) -> np.ndarray:
        """Solve Riemann problem using HLL approximation"""
        
        # Get primitive variables
        rho_l, u_l, v_l, w_l, p_l, c_l = self.compute_primitive_variables(u_left, gamma)
        rho_r, u_r, v_r, w_r, p_r, c_r = self.compute_primitive_variables(u_right, gamma)
        
        # Select normal velocity based on direction
        if direction == 0:  # x-direction
            vel_l, vel_r = u_l, u_r
        else:  # y-direction
            vel_l, vel_r = v_l, v_r
        
        # Estimate wave speeds using simple approach
        # Left-going wave speed
        s_l = min(vel_l - c_l, vel_r - c_r)
        
        # Right-going wave speed
        s_r = max(vel_l + c_l, vel_r + c_r)
        
        # Compute physical fluxes
        f_left = self.compute_physical_flux(u_left, direction, gamma)
        f_right = self.compute_physical_flux(u_right, direction, gamma)
        
        # HLL flux formula
        if s_l >= 0:
            # Supersonic to the right
            return f_left
        elif s_r <= 0:
            # Supersonic to the left
            return f_right
        else:
            # Subsonic - use HLL average
            return (s_r * f_left - s_l * f_right + s_l * s_r * (u_right - u_left)) / (s_r - s_l)


class HLLCSolver(RiemannSolver):
    """
    HLLC (Harten-Lax-van Leer-Contact) Riemann solver.
    
    Extension of HLL that restores the missing contact discontinuity
    by considering three waves: left acoustic, contact, and right acoustic.
    """
    
    def __init__(self):
        super().__init__("HLLC")
    
    def solve(self, u_left: np.ndarray, u_right: np.ndarray,
              direction: int, gamma: float = 1.4) -> np.ndarray:
        """Solve Riemann problem using HLLC approximation"""
        
        # Get primitive variables
        rho_l, u_l, v_l, w_l, p_l, c_l = self.compute_primitive_variables(u_left, gamma)
        rho_r, u_r, v_r, w_r, p_r, c_r = self.compute_primitive_variables(u_right, gamma)
        
        # Select normal velocity based on direction
        if direction == 0:  # x-direction
            vel_l, vel_r = u_l, u_r
            tang1_l, tang1_r = v_l, v_r
            tang2_l, tang2_r = w_l, w_r
        else:  # y-direction
            vel_l, vel_r = v_l, v_r
            tang1_l, tang1_r = u_l, u_r
            tang2_l, tang2_r = w_l, w_r
        
        # Estimate wave speeds
        # Roe averages for better wave speed estimates
        sqrt_rho_l = np.sqrt(rho_l)
        sqrt_rho_r = np.sqrt(rho_r)
        sqrt_rho_sum = sqrt_rho_l + sqrt_rho_r
        
        vel_roe = (sqrt_rho_l * vel_l + sqrt_rho_r * vel_r) / sqrt_rho_sum
        h_l = (u_left[4] + p_l) / rho_l
        h_r = (u_right[4] + p_r) / rho_r
        h_roe = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / sqrt_rho_sum
        c_roe = np.sqrt((gamma - 1.0) * (h_roe - 0.5 * vel_roe**2))
        
        # Wave speeds
        s_l = min(vel_l - c_l, vel_roe - c_roe)
        s_r = max(vel_r + c_r, vel_roe + c_roe)
        
        # Contact wave speed
        s_star = (p_r - p_l + rho_l * vel_l * (s_l - vel_l) - rho_r * vel_r * (s_r - vel_r)) / \
                 (rho_l * (s_l - vel_l) - rho_r * (s_r - vel_r))
        
        # Compute physical fluxes
        f_left = self.compute_physical_flux(u_left, direction, gamma)
        f_right = self.compute_physical_flux(u_right, direction, gamma)
        
        # HLLC flux
        if s_l >= 0:
            return f_left
        elif s_r <= 0:
            return f_right
        elif s_star >= 0:
            # Left star region
            u_star_l = self._compute_star_state(u_left, s_l, s_star, direction, gamma)
            return f_left + s_l * (u_star_l - u_left)
        else:
            # Right star region
            u_star_r = self._compute_star_state(u_right, s_r, s_star, direction, gamma)
            return f_right + s_r * (u_star_r - u_right)
    
    def _compute_star_state(self, u: np.ndarray, s: float, s_star: float,
                           direction: int, gamma: float) -> np.ndarray:
        """Compute star region state"""
        rho, u_vel, v_vel, w_vel, p, c = self.compute_primitive_variables(u, gamma)
        
        if direction == 0:  # x-direction
            vel_normal = u_vel
            vel_tang1 = v_vel
            vel_tang2 = w_vel
        else:  # y-direction
            vel_normal = v_vel
            vel_tang1 = u_vel
            vel_tang2 = w_vel
        
        # Density in star region
        rho_star = rho * (s - vel_normal) / (s - s_star)
        
        # Energy in star region
        E_star = u[4] / rho * rho_star + rho_star * (s_star - vel_normal) * \
                (s_star + p / (rho * (s - vel_normal)))
        
        # Construct star state
        if direction == 0:  # x-direction
            return np.array([
                rho_star,
                rho_star * s_star,      # rho * u*
                rho_star * vel_tang1,   # rho * v
                rho_star * vel_tang2,   # rho * w
                E_star
            ])
        else:  # y-direction
            return np.array([
                rho_star,
                rho_star * vel_tang1,   # rho * u
                rho_star * s_star,      # rho * v*
                rho_star * vel_tang2,   # rho * w
                E_star
            ])


class HLLDSolver(RiemannSolver):
    """
    HLLD (Harten-Lax-van Leer-Discontinuities) Riemann solver.
    
    Extension of HLLC for magnetohydrodynamics (MHD) that accounts for
    Alfvén waves. Can also be used for pure hydrodynamics.
    """
    
    def __init__(self):
        super().__init__("HLLD")
    
    def solve(self, u_left: np.ndarray, u_right: np.ndarray,
              direction: int, gamma: float = 1.4) -> np.ndarray:
        """
        Solve Riemann problem using HLLD approximation.
        
        For pure hydrodynamics (no magnetic field), this reduces to HLLC.
        """
        # For pure hydrodynamics, HLLD is essentially HLLC
        # This is a simplified implementation
        hllc_solver = HLLCSolver()
        return hllc_solver.solve(u_left, u_right, direction, gamma)


class ExactRiemannSolver(RiemannSolver):
    """
    Exact Riemann solver for Euler equations.
    
    This solver finds the exact solution to the Riemann problem
    but is computationally expensive. Useful for validation.
    """
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-10):
        super().__init__("Exact")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def solve(self, u_left: np.ndarray, u_right: np.ndarray,
              direction: int, gamma: float = 1.4) -> np.ndarray:
        """
        Solve Riemann problem exactly using iterative method.
        
        This is a simplified implementation. A full exact solver
        would require more sophisticated pressure iteration.
        """
        # For now, fall back to HLLC as exact solver is complex
        # Full implementation would involve:
        # 1. Iterative solution for pressure in star region
        # 2. Determination of wave pattern (shock/rarefaction)
        # 3. Sampling of exact solution at x/t = 0
        
        hllc_solver = HLLCSolver()
        return hllc_solver.solve(u_left, u_right, direction, gamma)


class RiemannSolverFactory:
    """Factory for creating Riemann solvers"""
    
    _solvers = {
        'hll': HLLSolver,
        'hllc': HLLCSolver,
        'hlld': HLLDSolver,
        'exact': ExactRiemannSolver
    }
    
    @classmethod
    def create(cls, solver_type: str, **kwargs) -> RiemannSolver:
        """
        Create a Riemann solver of specified type.
        
        Args:
            solver_type: Type of solver ('hll', 'hllc', 'hlld', 'exact')
            **kwargs: Additional parameters for solver
            
        Returns:
            Riemann solver instance
        """
        solver_type = solver_type.lower()
        if solver_type not in cls._solvers:
            raise ValueError(f"Unknown Riemann solver type: {solver_type}")
        
        return cls._solvers[solver_type](**kwargs)
    
    @classmethod
    def available_solvers(cls) -> list:
        """Get list of available solver types"""
        return list(cls._solvers.keys())


class RiemannFluxComputation:
    """
    High-level interface for computing fluxes using Riemann solvers.
    
    This class provides methods to compute fluxes for the entire grid
    using a specified Riemann solver.
    """
    
    def __init__(self, solver: RiemannSolver):
        self.solver = solver
    
    def compute_fluxes(self, data: FVMDataContainer2D, gamma: float = 1.4, **kwargs):
        """Compute all fluxes using the Riemann solver"""
        self._compute_x_fluxes(data, gamma)
        self._compute_y_fluxes(data, gamma)
    
    def _compute_x_fluxes(self, data: FVMDataContainer2D, gamma: float):
        """Compute fluxes in x-direction"""
        nx, ny = data.nx, data.ny
        
        for j in range(ny):
            for i in range(nx + 1):
                # Get left and right states
                if i == 0:
                    # Left boundary - use boundary condition
                    u_left = u_right = data.state[:, 0, j]
                elif i == nx:
                    # Right boundary - use boundary condition
                    u_left = u_right = data.state[:, -1, j]
                else:
                    # Interior interface
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                
                # Solve Riemann problem
                data.flux_x[:, i, j] = self.solver.solve(u_left, u_right, 0, gamma)
    
    def _compute_y_fluxes(self, data: FVMDataContainer2D, gamma: float):
        """Compute fluxes in y-direction"""
        nx, ny = data.nx, data.ny
        
        for i in range(nx):
            for j in range(ny + 1):
                # Get left and right states (bottom and top)
                if j == 0:
                    # Bottom boundary
                    u_left = u_right = data.state[:, i, 0]
                elif j == ny:
                    # Top boundary
                    u_left = u_right = data.state[:, i, -1]
                else:
                    # Interior interface
                    u_left = data.state[:, i, j-1]  # Bottom cell
                    u_right = data.state[:, i, j]    # Top cell
                
                # Solve Riemann problem
                data.flux_y[:, i, j] = self.solver.solve(u_left, u_right, 1, gamma)


class AdaptiveRiemannSolver(RiemannSolver):
    """
    Adaptive Riemann solver that automatically selects the best solver
    based on local flow conditions.
    """
    
    def __init__(self, default_solver: str = 'hllc'):
        super().__init__("Adaptive")
        self.default_solver = RiemannSolverFactory.create(default_solver)
        self.hll_solver = RiemannSolverFactory.create('hll')
        self.exact_solver = RiemannSolverFactory.create('exact')
    
    def solve(self, u_left: np.ndarray, u_right: np.ndarray,
              direction: int, gamma: float = 1.4) -> np.ndarray:
        """
        Adaptively choose solver based on flow conditions.
        
        Uses different solvers based on:
        - Shock strength
        - Mach number
        - Pressure ratio
        """
        # Get primitive variables
        rho_l, u_l, v_l, w_l, p_l, c_l = self.compute_primitive_variables(u_left, gamma)
        rho_r, u_r, v_r, w_r, p_r, c_r = self.compute_primitive_variables(u_right, gamma)
        
        # Calculate pressure ratio and density ratio
        p_ratio = max(p_l, p_r) / min(p_l, p_r)
        rho_ratio = max(rho_l, rho_r) / min(rho_l, rho_r)
        
        # Calculate Mach numbers
        if direction == 0:
            mach_l = abs(u_l) / c_l
            mach_r = abs(u_r) / c_r
        else:
            mach_l = abs(v_l) / c_l
            mach_r = abs(v_r) / c_r
        
        max_mach = max(mach_l, mach_r)
        
        # Adaptive selection criteria
        if p_ratio > 10.0 or rho_ratio > 10.0 or max_mach > 2.0:
            # Strong discontinuity - use robust HLL
            return self.hll_solver.solve(u_left, u_right, direction, gamma)
        elif p_ratio < 1.1 and rho_ratio < 1.1 and max_mach < 0.1:
            # Weak discontinuity - could use exact solver
            # (but exact solver is expensive, so use HLLC)
            return self.default_solver.solve(u_left, u_right, direction, gamma)
        else:
            # Moderate discontinuity - use default HLLC
            return self.default_solver.solve(u_left, u_right, direction, gamma)