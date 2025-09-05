"""
2D Euler Equations for Compressible Gas Dynamics

This module implements the 2D Euler equations for inviscid, compressible flow.
Conservative variables: [ρ, ρu, ρv, ρw, E]
where ρ is density, u,v,w are velocity components, and E is total energy.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

from ..core.data_container import FVMDataContainer2D


@dataclass
class EulerState:
    """Structure for Euler equation primitive variables"""
    density: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    pressure: float
    temperature: Optional[float] = None
    sound_speed: Optional[float] = None


class EulerEquations2D:
    """
    2D Euler equations for compressible gas dynamics.
    
    Governing equations:
    ∂ρ/∂t + ∇·(ρ𝐮) = 0                    (continuity)
    ∂(ρ𝐮)/∂t + ∇·(ρ𝐮⊗𝐮 + p𝐈) = 0         (momentum)
    ∂E/∂t + ∇·((E + p)𝐮) = 0              (energy)
    
    where E = ρe + ½ρ|𝐮|² is the total energy per unit volume.
    """
    
    def __init__(self, gamma: float = 1.4, gas_constant: float = 287.0):
        """
        Initialize Euler equations.
        
        Args:
            gamma: Heat capacity ratio (Cp/Cv)
            gas_constant: Specific gas constant (J/kg/K)
        """
        self.gamma = gamma
        self.gas_constant = gas_constant
        self.name = "2D Euler Equations"
        self.num_variables = 5  # [ρ, ρu, ρv, ρw, E]
    
    def conservative_to_primitive(self, u: np.ndarray) -> EulerState:
        """
        Convert conservative variables to primitive variables.
        
        Args:
            u: Conservative variables [ρ, ρu, ρv, ρw, E]
            
        Returns:
            EulerState with primitive variables
        """
        rho = max(u[0], 1e-15)  # Avoid division by zero
        
        # Velocities
        u_vel = u[1] / rho
        v_vel = u[2] / rho
        w_vel = u[3] / rho
        
        # Total energy
        E = u[4]
        
        # Kinetic energy
        kinetic_energy = 0.5 * rho * (u_vel**2 + v_vel**2 + w_vel**2)
        
        # Pressure from ideal gas law
        pressure = max((self.gamma - 1.0) * (E - kinetic_energy), 1e-15)
        
        # Sound speed
        sound_speed = np.sqrt(self.gamma * pressure / rho)
        
        # Temperature (if gas constant is provided)
        temperature = pressure / (rho * self.gas_constant) if self.gas_constant > 0 else None
        
        return EulerState(
            density=rho,
            velocity_x=u_vel,
            velocity_y=v_vel,
            velocity_z=w_vel,
            pressure=pressure,
            temperature=temperature,
            sound_speed=sound_speed
        )
    
    def primitive_to_conservative(self, state: EulerState) -> np.ndarray:
        """
        Convert primitive variables to conservative variables.
        
        Args:
            state: EulerState with primitive variables
            
        Returns:
            Conservative variables [ρ, ρu, ρv, ρw, E]
        """
        rho = state.density
        u, v, w = state.velocity_x, state.velocity_y, state.velocity_z
        p = state.pressure
        
        # Conservative variables
        rho_u = rho * u
        rho_v = rho * v
        rho_w = rho * w
        
        # Total energy
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        internal_energy = p / (self.gamma - 1.0)
        E = internal_energy + kinetic_energy
        
        return np.array([rho, rho_u, rho_v, rho_w, E])
    
    def compute_flux_x(self, u: np.ndarray) -> np.ndarray:
        """
        Compute flux vector in x-direction.
        
        F = [ρu, ρu² + p, ρuv, ρuw, u(E + p)]
        
        Args:
            u: Conservative variables
            
        Returns:
            Flux vector in x-direction
        """
        state = self.conservative_to_primitive(u)
        rho = state.density
        u_vel = state.velocity_x
        v_vel = state.velocity_y
        w_vel = state.velocity_z
        p = state.pressure
        E = u[4]
        
        flux_x = np.array([
            rho * u_vel,                    # Mass flux
            rho * u_vel**2 + p,            # x-momentum flux
            rho * u_vel * v_vel,           # y-momentum flux
            rho * u_vel * w_vel,           # z-momentum flux
            u_vel * (E + p)                # Energy flux
        ])
        
        return flux_x
    
    def compute_flux_y(self, u: np.ndarray) -> np.ndarray:
        """
        Compute flux vector in y-direction.
        
        G = [ρv, ρuv, ρv² + p, ρvw, v(E + p)]
        
        Args:
            u: Conservative variables
            
        Returns:
            Flux vector in y-direction
        """
        state = self.conservative_to_primitive(u)
        rho = state.density
        u_vel = state.velocity_x
        v_vel = state.velocity_y
        w_vel = state.velocity_z
        p = state.pressure
        E = u[4]
        
        flux_y = np.array([
            rho * v_vel,                    # Mass flux
            rho * v_vel * u_vel,           # x-momentum flux
            rho * v_vel**2 + p,            # y-momentum flux
            rho * v_vel * w_vel,           # z-momentum flux
            v_vel * (E + p)                # Energy flux
        ])
        
        return flux_y
    
    def compute_eigenvalues(self, u: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute eigenvalues of flux Jacobian matrix.
        
        Args:
            u: Conservative variables
            direction: 0 for x-direction, 1 for y-direction
            
        Returns:
            Array of eigenvalues
        """
        state = self.conservative_to_primitive(u)
        c = state.sound_speed
        
        if direction == 0:  # x-direction
            u_vel = state.velocity_x
            eigenvals = np.array([u_vel - c, u_vel, u_vel, u_vel, u_vel + c])
        else:  # y-direction
            v_vel = state.velocity_y
            eigenvals = np.array([v_vel - c, v_vel, v_vel, v_vel, v_vel + c])
        
        return eigenvals
    
    def compute_max_wave_speed(self, data: FVMDataContainer2D) -> float:
        """
        Compute maximum wave speed for CFL condition.
        
        Args:
            data: FVM data container
            
        Returns:
            Maximum wave speed in the domain
        """
        max_speed = 0.0
        nx, ny = data.nx, data.ny
        
        for i in range(nx):
            for j in range(ny):
                u = data.state[:, i, j]
                state = self.conservative_to_primitive(u)
                
                # Maximum wave speed is |u| + c or |v| + c
                speed_x = abs(state.velocity_x) + state.sound_speed
                speed_y = abs(state.velocity_y) + state.sound_speed
                local_max_speed = max(speed_x, speed_y)
                
                max_speed = max(max_speed, local_max_speed)
        
        return max_speed
    
    def compute_total_energy(self, data: FVMDataContainer2D) -> float:
        """Compute total energy in the domain"""
        return np.sum(data.state[4]) * data.geometry.cell_volume
    
    def compute_total_mass(self, data: FVMDataContainer2D) -> float:
        """Compute total mass in the domain"""
        return np.sum(data.state[0]) * data.geometry.cell_volume
    
    def compute_total_momentum(self, data: FVMDataContainer2D) -> Tuple[float, float]:
        """Compute total momentum in x and y directions"""
        momentum_x = np.sum(data.state[1]) * data.geometry.cell_volume
        momentum_y = np.sum(data.state[2]) * data.geometry.cell_volume
        return momentum_x, momentum_y
    
    def apply_boundary_conditions(self, data: FVMDataContainer2D, bc_type: str = 'periodic'):
        """
        Apply boundary conditions for Euler equations.
        
        Args:
            data: FVM data container
            bc_type: Boundary condition type
        """
        if bc_type == 'periodic':
            # Periodic boundaries - handled by flux computation
            pass
        elif bc_type == 'reflective':
            # Reflective (solid wall) boundaries
            self._apply_reflective_bc(data)
        elif bc_type == 'transmissive':
            # Transmissive (outflow) boundaries
            self._apply_transmissive_bc(data)
        elif bc_type == 'inflow':
            # Inflow boundary (would need reference state)
            self._apply_inflow_bc(data)
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    def _apply_reflective_bc(self, data: FVMDataContainer2D):
        """Apply reflective (wall) boundary conditions"""
        nx, ny = data.nx, data.ny
        
        # Left and right boundaries (reflect x-velocity)
        data.state[0, 0, :] = data.state[0, 1, :]      # Density
        data.state[1, 0, :] = -data.state[1, 1, :]     # x-momentum (reflect)
        data.state[2, 0, :] = data.state[2, 1, :]      # y-momentum
        data.state[3, 0, :] = data.state[3, 1, :]      # z-momentum
        data.state[4, 0, :] = data.state[4, 1, :]      # Energy
        
        data.state[0, -1, :] = data.state[0, -2, :]    # Density
        data.state[1, -1, :] = -data.state[1, -2, :]   # x-momentum (reflect)
        data.state[2, -1, :] = data.state[2, -2, :]    # y-momentum
        data.state[3, -1, :] = data.state[3, -2, :]    # z-momentum
        data.state[4, -1, :] = data.state[4, -2, :]    # Energy
        
        # Bottom and top boundaries (reflect y-velocity)
        data.state[0, :, 0] = data.state[0, :, 1]      # Density
        data.state[1, :, 0] = data.state[1, :, 1]      # x-momentum
        data.state[2, :, 0] = -data.state[2, :, 1]     # y-momentum (reflect)
        data.state[3, :, 0] = data.state[3, :, 1]      # z-momentum
        data.state[4, :, 0] = data.state[4, :, 1]      # Energy
        
        data.state[0, :, -1] = data.state[0, :, -2]    # Density
        data.state[1, :, -1] = data.state[1, :, -2]    # x-momentum
        data.state[2, :, -1] = -data.state[2, :, -2]   # y-momentum (reflect)
        data.state[3, :, -1] = data.state[3, :, -2]    # z-momentum
        data.state[4, :, -1] = data.state[4, :, -2]    # Energy
    
    def _apply_transmissive_bc(self, data: FVMDataContainer2D):
        """Apply transmissive (outflow) boundary conditions"""
        # Zero gradient boundary conditions
        data.state[:, 0, :] = data.state[:, 1, :]      # Left boundary
        data.state[:, -1, :] = data.state[:, -2, :]    # Right boundary
        data.state[:, :, 0] = data.state[:, :, 1]      # Bottom boundary
        data.state[:, :, -1] = data.state[:, :, -2]    # Top boundary
    
    def _apply_inflow_bc(self, data: FVMDataContainer2D):
        """Apply inflow boundary conditions (placeholder)"""
        # This would require specification of inflow state
        # For now, apply transmissive BC
        self._apply_transmissive_bc(data)
    
    def compute_source_terms(self, data: FVMDataContainer2D, **kwargs) -> np.ndarray:
        """
        Compute source terms for Euler equations.
        
        For inviscid Euler equations, source terms are typically zero
        unless there are body forces or other external effects.
        
        Args:
            data: FVM data container
            **kwargs: Additional parameters
            
        Returns:
            Source term array with same shape as state
        """
        # No source terms for standard Euler equations
        return np.zeros_like(data.state)
    
    def compute_time_step(self, data: FVMDataContainer2D, cfl_number: float = 0.5) -> float:
        """
        Compute maximum stable time step based on CFL condition.
        
        Args:
            data: FVM data container
            cfl_number: CFL number (should be < 1 for stability)
            
        Returns:
            Maximum stable time step
        """
        max_wave_speed = self.compute_max_wave_speed(data)
        min_dx = min(data.geometry.dx, data.geometry.dy)
        
        if max_wave_speed > 1e-15:
            dt_max = cfl_number * min_dx / max_wave_speed
        else:
            dt_max = 1e-6  # Fallback for very low speeds
        
        return dt_max


class EulerInitialConditions:
    """Common initial conditions for Euler equations"""
    
    @staticmethod
    def uniform_flow(density: float, velocity_x: float, velocity_y: float, 
                    pressure: float, velocity_z: float = 0.0, gamma: float = 1.4) -> Callable:
        """
        Uniform flow initial condition.
        
        Args:
            density: Uniform density
            velocity_x: Uniform x-velocity
            velocity_y: Uniform y-velocity
            pressure: Uniform pressure
            velocity_z: Uniform z-velocity
            gamma: Heat capacity ratio
            
        Returns:
            Function that returns initial state at any (x, y)
        """
        def initial_condition(x: float, y: float, **kwargs) -> np.ndarray:
            state = EulerState(density, velocity_x, velocity_y, velocity_z, pressure)
            euler_eq = EulerEquations2D(gamma)
            return euler_eq.primitive_to_conservative(state)
        
        return initial_condition
    
    @staticmethod
    def riemann_problem_2d(left_state: EulerState, right_state: EulerState, 
                          interface_x: float = 0.5, interface_y: float = 0.5,
                          gamma: float = 1.4) -> Callable:
        """
        2D Riemann problem initial condition.
        
        Args:
            left_state: Left state (primitive variables)
            right_state: Right state (primitive variables)
            interface_x: x-coordinate of interface
            interface_y: y-coordinate of interface
            gamma: Heat capacity ratio
            
        Returns:
            Function that returns initial state at any (x, y)
        """
        def initial_condition(x: float, y: float, **kwargs) -> np.ndarray:
            euler_eq = EulerEquations2D(gamma)
            
            # Simple 2D extension - use x-coordinate for interface
            if x < interface_x:
                return euler_eq.primitive_to_conservative(left_state)
            else:
                return euler_eq.primitive_to_conservative(right_state)
        
        return initial_condition
    
    @staticmethod
    def gaussian_pulse(center_x: float, center_y: float, width: float,
                      amplitude: float, background_density: float = 1.0,
                      background_pressure: float = 1.0, gamma: float = 1.4) -> Callable:
        """
        Gaussian density pulse initial condition.
        
        Args:
            center_x, center_y: Pulse center
            width: Pulse width
            amplitude: Pulse amplitude
            background_density: Background density
            background_pressure: Background pressure
            gamma: Heat capacity ratio
            
        Returns:
            Function that returns initial state at any (x, y)
        """
        def initial_condition(x: float, y: float, **kwargs) -> np.ndarray:
            # Distance from center
            r_squared = (x - center_x)**2 + (y - center_y)**2
            
            # Gaussian pulse
            density = background_density + amplitude * np.exp(-r_squared / (2 * width**2))
            
            state = EulerState(
                density=density,
                velocity_x=0.0,
                velocity_y=0.0,
                velocity_z=0.0,
                pressure=background_pressure
            )
            
            euler_eq = EulerEquations2D(gamma)
            return euler_eq.primitive_to_conservative(state)
        
        return initial_condition