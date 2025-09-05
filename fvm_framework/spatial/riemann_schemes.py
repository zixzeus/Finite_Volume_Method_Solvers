"""
Riemann Solver-based Spatial Discretization Schemes

This module provides wrappers for existing Riemann solvers to work within
the unified spatial discretization framework.
"""

from core.data_container import FVMDataContainer2D
from .base import RiemannBasedScheme


class HLLRiemannScheme(RiemannBasedScheme):
    """HLL Riemann solver-based scheme"""
    
    def __init__(self):
        super().__init__("HLL_Riemann", "hll", order=1)
    
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """Compute fluxes using HLL Riemann solver"""
        riemann_solver = self._get_riemann_solver()
        
        # Use Riemann solver to compute fluxes
        nx, ny = data.nx, data.ny
        
        # X-direction fluxes
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    u_left = u_right = data.state[:, 0, j]
                elif i == nx:
                    u_left = u_right = data.state[:, -1, j]
                else:
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                
                # Solve Riemann problem
                gamma = kwargs.get('gamma', 1.4)
                flux = riemann_solver.solve(u_left, u_right, direction=0, gamma=gamma)
                data.flux_x[:, i, j] = flux
        
        # Y-direction fluxes
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    u_left = u_right = data.state[:, i, 0]
                elif j == ny:
                    u_left = u_right = data.state[:, i, -1]
                else:
                    u_left = data.state[:, i, j-1]
                    u_right = data.state[:, i, j]
                
                # Solve Riemann problem
                gamma = kwargs.get('gamma', 1.4)
                flux = riemann_solver.solve(u_left, u_right, direction=1, gamma=gamma)
                data.flux_y[:, i, j] = flux


class HLLCRiemannScheme(RiemannBasedScheme):
    """HLLC Riemann solver-based scheme"""
    
    def __init__(self):
        super().__init__("HLLC_Riemann", "hllc", order=1)
    
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """Compute fluxes using HLLC Riemann solver"""
        riemann_solver = self._get_riemann_solver()
        
        nx, ny = data.nx, data.ny
        gamma = kwargs.get('gamma', 1.4)
        
        # X-direction fluxes
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    u_left = u_right = data.state[:, 0, j]
                elif i == nx:
                    u_left = u_right = data.state[:, -1, j]
                else:
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                
                flux = riemann_solver.solve(u_left, u_right, direction=0, gamma=gamma)
                data.flux_x[:, i, j] = flux
        
        # Y-direction fluxes
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    u_left = u_right = data.state[:, i, 0]
                elif j == ny:
                    u_left = u_right = data.state[:, i, -1]
                else:
                    u_left = data.state[:, i, j-1]
                    u_right = data.state[:, i, j]
                
                flux = riemann_solver.solve(u_left, u_right, direction=1, gamma=gamma)
                data.flux_y[:, i, j] = flux


class HLLDRiemannScheme(RiemannBasedScheme):
    """HLLD Riemann solver-based scheme (for MHD)"""
    
    def __init__(self):
        super().__init__("HLLD_Riemann", "hlld", order=1)
    
    def compute_fluxes(self, data: FVMDataContainer2D, physics_equation, **kwargs) -> None:
        """Compute fluxes using HLLD Riemann solver"""
        riemann_solver = self._get_riemann_solver()
        
        nx, ny = data.nx, data.ny
        gamma = kwargs.get('gamma', 1.4)
        
        # X-direction fluxes
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    u_left = u_right = data.state[:, 0, j]
                elif i == nx:
                    u_left = u_right = data.state[:, -1, j]
                else:
                    u_left = data.state[:, i-1, j]
                    u_right = data.state[:, i, j]
                
                flux = riemann_solver.solve(u_left, u_right, direction=0, gamma=gamma)
                data.flux_x[:, i, j] = flux
        
        # Y-direction fluxes
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    u_left = u_right = data.state[:, i, 0]
                elif j == ny:
                    u_left = u_right = data.state[:, i, -1]
                else:
                    u_left = data.state[:, i, j-1]
                    u_right = data.state[:, i, j]
                
                flux = riemann_solver.solve(u_left, u_right, direction=1, gamma=gamma)
                data.flux_y[:, i, j] = flux
    
    def supports_physics(self, physics_type: str) -> bool:
        """HLLD is specialized for MHD equations"""
        return physics_type.lower() in ['mhd', 'magnetohydrodynamics']