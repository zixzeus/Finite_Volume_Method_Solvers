"""
2D Magnetohydrodynamics (MHD) Equations

This module implements the 2D ideal MHD equations for conducting fluids
in electromagnetic fields.

Conservative variables: [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
where ρ is density, u,v,w are velocity components, E is total energy,
and Bx,By,Bz are magnetic field components.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
from fvm_framework.core.data_container import FVMDataContainer2D


@dataclass
class MHDState:
    """Structure for MHD primitive variables"""
    density: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    pressure: float
    magnetic_x: float
    magnetic_y: float
    magnetic_z: float
    temperature: Optional[float] = None
    sound_speed: Optional[float] = None
    alfven_speed: Optional[float] = None


class MHDEquations2D:
    """
    2D Ideal Magnetohydrodynamics equations.
    
    Governing equations:
    ∂ρ/∂t + ∇·(ρ𝐮) = 0                              (continuity)
    ∂(ρ𝐮)/∂t + ∇·(ρ𝐮⊗𝐮 + (p + B²/2)𝐈 - 𝐁⊗𝐁) = 0   (momentum)
    ∂E/∂t + ∇·((E + p + B²/2)𝐮 - 𝐁(𝐁·𝐮)) = 0       (energy)
    ∂𝐁/∂t + ∇·(𝐮⊗𝐁 - 𝐁⊗𝐮) = 0                     (induction)
    
    with the divergence-free constraint ∇·𝐁 = 0
    """
    
    def __init__(self, gamma: float = 5.0/3.0, gas_constant: float = 1.0):
        """
        Initialize MHD equations.
        
        Args:
            gamma: Heat capacity ratio (typically 5/3 for monatomic gas)
            gas_constant: Specific gas constant
        """
        self.gamma = gamma
        self.gas_constant = gas_constant
        self.name = "2D Ideal MHD Equations"
        self.num_variables = 8  # [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    
    def conservative_to_primitive(self, u: np.ndarray) -> MHDState:
        """
        Convert conservative variables to primitive variables.
        
        Args:
            u: Conservative variables [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
            
        Returns:
            MHDState with primitive variables
        """
        rho = max(u[0], 1e-15)  # Avoid division by zero
        
        # Velocities
        u_vel = u[1] / rho
        v_vel = u[2] / rho
        w_vel = u[3] / rho
        
        # Total energy and magnetic field
        E = u[4]
        Bx, By, Bz = u[5], u[6], u[7]
        
        # Magnetic pressure
        B_squared = Bx**2 + By**2 + Bz**2
        magnetic_pressure = 0.5 * B_squared
        
        # Kinetic energy
        kinetic_energy = 0.5 * rho * (u_vel**2 + v_vel**2 + w_vel**2)
        
        # Gas pressure from ideal gas law
        pressure = max((self.gamma - 1.0) * (E - kinetic_energy - magnetic_pressure), 1e-15)
        
        # Sound speed and Alfvén speed
        sound_speed = np.sqrt(self.gamma * pressure / rho)
        alfven_speed = np.sqrt(B_squared / rho) if rho > 1e-15 else 0.0
        
        # Temperature (if gas constant is provided)
        temperature = pressure / (rho * self.gas_constant) if self.gas_constant > 0 else None
        
        return MHDState(
            density=rho,
            velocity_x=u_vel,
            velocity_y=v_vel,
            velocity_z=w_vel,
            pressure=pressure,
            magnetic_x=Bx,
            magnetic_y=By,
            magnetic_z=Bz,
            temperature=temperature,
            sound_speed=sound_speed,
            alfven_speed=alfven_speed
        )
    
    def primitive_to_conservative(self, state: MHDState) -> np.ndarray:
        """
        Convert primitive variables to conservative variables.
        
        Args:
            state: MHDState with primitive variables
            
        Returns:
            Conservative variables [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
        """
        rho = state.density
        u, v, w = state.velocity_x, state.velocity_y, state.velocity_z
        p = state.pressure
        Bx, By, Bz = state.magnetic_x, state.magnetic_y, state.magnetic_z
        
        # Conservative variables
        rho_u = rho * u
        rho_v = rho * v
        rho_w = rho * w
        
        # Total energy
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        magnetic_energy = 0.5 * (Bx**2 + By**2 + Bz**2)
        internal_energy = p / (self.gamma - 1.0)
        E = internal_energy + kinetic_energy + magnetic_energy
        
        return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])
    
    def compute_flux_x(self, u: np.ndarray) -> np.ndarray:
        """
        Compute flux vector in x-direction for MHD equations.
        
        Args:
            u: Conservative variables
            
        Returns:
            Flux vector in x-direction
        """
        state = self.conservative_to_primitive(u)
        rho = state.density
        u_vel, v_vel, w_vel = state.velocity_x, state.velocity_y, state.velocity_z
        p = state.pressure
        Bx, By, Bz = state.magnetic_x, state.magnetic_y, state.magnetic_z
        E = u[4]
        
        # Magnetic pressure
        B_squared = Bx**2 + By**2 + Bz**2
        magnetic_pressure = 0.5 * B_squared
        
        # Total pressure
        p_total = p + magnetic_pressure
        
        flux_x = np.array([
            rho * u_vel,                                    # Mass flux
            rho * u_vel**2 + p_total - Bx**2,             # x-momentum flux
            rho * u_vel * v_vel - Bx * By,                 # y-momentum flux
            rho * u_vel * w_vel - Bx * Bz,                 # z-momentum flux
            u_vel * (E + p_total) - Bx * (Bx*u_vel + By*v_vel + Bz*w_vel),  # Energy flux
            0.0,                                            # Bx flux (∂Bx/∂t + ∂(0)/∂x = 0)
            By * u_vel - Bx * v_vel,                       # By flux
            Bz * u_vel - Bx * w_vel                        # Bz flux
        ])
        
        return flux_x
    
    def compute_flux_y(self, u: np.ndarray) -> np.ndarray:
        """
        Compute flux vector in y-direction for MHD equations.
        
        Args:
            u: Conservative variables
            
        Returns:
            Flux vector in y-direction
        """
        state = self.conservative_to_primitive(u)
        rho = state.density
        u_vel, v_vel, w_vel = state.velocity_x, state.velocity_y, state.velocity_z
        p = state.pressure
        Bx, By, Bz = state.magnetic_x, state.magnetic_y, state.magnetic_z
        E = u[4]
        
        # Magnetic pressure
        B_squared = Bx**2 + By**2 + Bz**2
        magnetic_pressure = 0.5 * B_squared
        
        # Total pressure
        p_total = p + magnetic_pressure
        
        flux_y = np.array([
            rho * v_vel,                                    # Mass flux
            rho * v_vel * u_vel - By * Bx,                 # x-momentum flux
            rho * v_vel**2 + p_total - By**2,             # y-momentum flux
            rho * v_vel * w_vel - By * Bz,                 # z-momentum flux
            v_vel * (E + p_total) - By * (Bx*u_vel + By*v_vel + Bz*w_vel),  # Energy flux
            Bx * v_vel - By * u_vel,                       # Bx flux
            0.0,                                            # By flux (∂By/∂t + ∂(0)/∂y = 0)
            Bz * v_vel - By * w_vel                        # Bz flux
        ])
        
        return flux_y
    
    def compute_eigenvalues(self, u: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute eigenvalues of flux Jacobian matrix for MHD equations.
        
        Args:
            u: Conservative variables
            direction: 0 for x-direction, 1 for y-direction
            
        Returns:
            Array of eigenvalues (8 values for MHD)
        """
        state = self.conservative_to_primitive(u)
        rho = state.density
        cs = state.sound_speed  # Sound speed
        Bx, By, Bz = state.magnetic_x, state.magnetic_y, state.magnetic_z
        
        # Alfvén speeds
        ca_x = abs(Bx) / np.sqrt(rho) if rho > 1e-15 else 0.0
        ca_y = abs(By) / np.sqrt(rho) if rho > 1e-15 else 0.0
        
        # Fast and slow magnetosonic speeds
        B_squared = Bx**2 + By**2 + Bz**2
        va_squared = B_squared / rho if rho > 1e-15 else 0.0
        cs_squared = cs**2
        
        # Discriminant for magnetosonic speeds
        discriminant = np.sqrt((cs_squared + va_squared)**2 - 4*cs_squared*va_squared)
        
        cf = np.sqrt(0.5 * (cs_squared + va_squared + discriminant))  # Fast speed
        slow_speed_squared = 0.5 * (cs_squared + va_squared - discriminant)
        cs_slow = np.sqrt(max(slow_speed_squared, 0.0))  # Slow speed
        
        if direction == 0:  # x-direction
            u_vel = state.velocity_x
            eigenvals = np.array([
                u_vel - cf,      # Fast wave (left)
                u_vel - ca_x,    # Alfvén wave (left)
                u_vel - cs_slow, # Slow wave (left)
                u_vel,           # Entropy wave
                u_vel + cs_slow, # Slow wave (right)
                u_vel + ca_x,    # Alfvén wave (right)
                u_vel + cf,      # Fast wave (right)
                u_vel            # Divergence wave
            ])
        else:  # y-direction
            v_vel = state.velocity_y
            eigenvals = np.array([
                v_vel - cf,      # Fast wave (left)
                v_vel - ca_y,    # Alfvén wave (left)
                v_vel - cs_slow, # Slow wave (left)
                v_vel,           # Entropy wave
                v_vel + cs_slow, # Slow wave (right)
                v_vel + ca_y,    # Alfvén wave (right)
                v_vel + cf,      # Fast wave (right)
                v_vel            # Divergence wave
            ])
        
        return eigenvals
    
    def compute_max_wave_speed(self, data) -> float:
        """
        Compute maximum wave speed for CFL condition.
        
        For MHD, this is the fast magnetosonic speed.
        """
        max_speed = 0.0
        
        # Assume data has MHD state format with 8 variables
        if hasattr(data, 'state') and data.state.shape[0] == 8:
            nx, ny = data.state.shape[1], data.state.shape[2]
            
            for i in range(nx):
                for j in range(ny):
                    u = data.state[:, i, j]
                    eigenvals_x = self.compute_eigenvalues(u, 0)
                    eigenvals_y = self.compute_eigenvalues(u, 1)
                    
                    local_max_speed = max(np.max(np.abs(eigenvals_x)), np.max(np.abs(eigenvals_y)))
                    max_speed = max(max_speed, local_max_speed)
        
        return max_speed
    
    def compute_divergence_b(self, data) -> np.ndarray:
        """
        Compute divergence of magnetic field ∇·B.
        
        This should be zero for physically consistent solutions.
        """
        if not (hasattr(data, 'state') and data.state.shape[0] == 8):
            return np.array([0.0])
        
        Bx = data.state[5]  # Magnetic field x-component
        By = data.state[6]  # Magnetic field y-component
        
        # Compute divergence using finite differences
        dBx_dx = np.gradient(Bx, data.geometry.dx, axis=0)
        dBy_dy = np.gradient(By, data.geometry.dy, axis=1)
        
        div_B = dBx_dx + dBy_dy
        
        return div_B
    
    def apply_divergence_cleaning(self, data, cleaning_speed: float = 1.0):
        """
        Apply divergence cleaning to maintain ∇·B = 0.
        
        This is a simplified implementation - full cleaning would require
        additional wave equations or projection methods.
        """
        div_B = self.compute_divergence_b(data)
        
        # Simple approach: adjust Bx and By to reduce divergence
        # This is not a complete implementation of divergence cleaning
        correction_factor = 0.01 * cleaning_speed
        
        if hasattr(data, 'state') and data.state.shape[0] == 8:
            # Adjust magnetic field components
            data.state[5] -= correction_factor * np.gradient(div_B, data.geometry.dx, axis=0)
            data.state[6] -= correction_factor * np.gradient(div_B, data.geometry.dy, axis=1)


class MHDInitialConditions:
    """Common initial conditions for MHD equations"""
    
    @staticmethod
    def harris_current_sheet(thickness: float = 0.1, B0: float = 1.0, 
                           density_ratio: float = 0.1, gamma: float = 5.0/3.0) -> Callable:
        """
        Harris current sheet initial condition for magnetic reconnection.
        
        This creates a current sheet with antiparallel magnetic fields
        that can undergo magnetic reconnection.
        
        Args:
            thickness: Current sheet thickness
            B0: Magnetic field strength
            density_ratio: Density contrast across sheet
            gamma: Heat capacity ratio
            
        Returns:
            Function that returns initial state at any (x, y)
        """
        def initial_condition(x: float, y: float, **kwargs) -> np.ndarray:
            # Magnetic field configuration (Harris sheet)
            Bx = B0 * np.tanh(y / thickness)
            By = 0.0
            Bz = 0.0
            
            # Density profile
            density = 1.0 + density_ratio / np.cosh(y / thickness)**2
            
            # Pressure balance (total pressure constant)
            magnetic_pressure = 0.5 * (Bx**2 + By**2 + Bz**2)
            pressure = 0.5 * B0**2 - magnetic_pressure + 0.1  # Background pressure
            pressure = max(pressure, 0.01)  # Ensure positive pressure
            
            # Velocities (initially at rest)
            velocity_x = 0.0
            velocity_y = 0.0
            velocity_z = 0.0
            
            state = MHDState(
                density=density,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                velocity_z=velocity_z,
                pressure=pressure,
                magnetic_x=Bx,
                magnetic_y=By,
                magnetic_z=Bz
            )
            
            mhd_eq = MHDEquations2D(gamma)
            return mhd_eq.primitive_to_conservative(state)
        
        return initial_condition
    
    @staticmethod
    def orszag_tang_vortex(gamma: float = 5.0/3.0) -> Callable:
        """
        Orszag-Tang vortex initial condition.
        
        This is a classic MHD test problem featuring the interaction
        of magnetic and kinetic energy leading to turbulence.
        
        Args:
            gamma: Heat capacity ratio
            
        Returns:
            Function that returns initial state at any (x, y)
        """
        def initial_condition(x: float, y: float, **kwargs) -> np.ndarray:
            # Normalize coordinates to [0, 2π]
            x_norm = 2 * np.pi * x
            y_norm = 2 * np.pi * y
            
            # Initial conditions
            density = gamma**2
            pressure = gamma
            
            # Velocity field
            velocity_x = -np.sin(y_norm)
            velocity_y = np.sin(x_norm)
            velocity_z = 0.0
            
            # Magnetic field
            B0 = 1.0 / np.sqrt(4 * np.pi)
            magnetic_x = -B0 * np.sin(y_norm)
            magnetic_y = B0 * np.sin(2 * x_norm)
            magnetic_z = 0.0
            
            state = MHDState(
                density=density,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                velocity_z=velocity_z,
                pressure=pressure,
                magnetic_x=magnetic_x,
                magnetic_y=magnetic_y,
                magnetic_z=magnetic_z
            )
            
            mhd_eq = MHDEquations2D(gamma)
            return mhd_eq.primitive_to_conservative(state)
        
        return initial_condition