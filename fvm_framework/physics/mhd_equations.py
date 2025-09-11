"""
2D Magnetohydrodynamics (MHD) Equations

This module implements the 2D ideal MHD equations for conducting fluids
in electromagnetic fields.

Conservative variables: [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
where ρ is density, u,v,w are velocity components, E is total energy,
and Bx,By,Bz are magnetic field components.

Note: In 2D MHD simulations, often Bz (out-of-plane component) is evolved
while Bx,By are computed to maintain ∇·B = 0.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from fvm_framework.core.data_container import FVMDataContainer2D
from .physics_base import PhysicsState, ConservationLaw


@dataclass
class MHDState(PhysicsState):
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
    
    def to_array(self) -> np.ndarray:
        """Convert to array: [ρ, u, v, w, p, Bx, By, Bz]"""
        return np.array([
            self.density,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            self.pressure,
            self.magnetic_x,
            self.magnetic_y,
            self.magnetic_z
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'MHDState':
        """Create MHDState from array [ρ, u, v, w, p, Bx, By, Bz]"""
        return cls(
            density=array[0],
            velocity_x=array[1],
            velocity_y=array[2],
            velocity_z=array[3],
            pressure=array[4],
            magnetic_x=array[5],
            magnetic_y=array[6],
            magnetic_z=array[7]
        )
    
    def copy(self) -> 'MHDState':
        """Create a copy of the state"""
        return MHDState(
            density=self.density,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            velocity_z=self.velocity_z,
            pressure=self.pressure,
            magnetic_x=self.magnetic_x,
            magnetic_y=self.magnetic_y,
            magnetic_z=self.magnetic_z,
            temperature=self.temperature,
            sound_speed=self.sound_speed,
            alfven_speed=self.alfven_speed
        )
    
    def validate(self) -> bool:
        """Validate physical consistency"""
        return (self.density > 0 and 
                self.pressure > 0 and
                not np.isnan(self.density) and
                not np.isnan(self.pressure))


class MHDEquations2D(ConservationLaw):
    """
    2D Ideal Magnetohydrodynamics equations.
    
    Governing equations:
    ∂ρ/∂t + ∇·(ρ𝐮) = 0                              (continuity)
    ∂(ρ𝐮)/∂t + ∇·(ρ𝐮⊗𝐮 + (p + B²/2)𝐈 - 𝐁⊗𝐁) = 0   (momentum)
    ∂E/∂t + ∇·((E + p + B²/2)𝐮 - 𝐁(𝐁·𝐮)) = 0       (energy)
    ∂𝐁/∂t + ∇·(𝐮⊗𝐁 - 𝐁⊗𝐮) = 0                     (induction)
    
    with the divergence-free constraint ∇·𝐁 = 0
    """
    
    def __init__(self, gamma: float = 5.0/3.0):
        """
        Initialize MHD equations.
        
        Args:
            gamma: Heat capacity ratio (typically 5/3 for monatomic gas)
        """
        super().__init__("2D Ideal MHD Equations", num_variables=8, num_dimensions=2)
        self.gamma = gamma
    
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
        
        # Magnetic pressure (using normalized units: B²/2)
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
        temperature = None  # Temperature calculation not needed in ideal MHD
        
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
    
    def compute_fluxes(self, state: np.ndarray, direction: int) -> np.ndarray:
        """Compute fluxes for given direction (required by base class)"""
        if direction == 0:
            return self.compute_flux_x(state)
        else:
            return self.compute_flux_y(state)
    
    def max_wave_speed(self, state: np.ndarray, direction: int) -> float:
        """Compute maximum wave speed (required by base class)"""
        rho, rho_u, rho_v, rho_w, E, Bx, By, Bz = state
        
        u, v = rho_u/rho, rho_v/rho
        p = self.compute_pressure(state)
        
        # Sound speed and Alfven speeds (with numerical safety)
        rho_safe = max(rho, 1e-15)
        p_safe = max(p, 1e-15)
        cs = np.sqrt(self.gamma * p_safe / rho_safe)
        ca_x = abs(Bx) / np.sqrt(rho_safe)
        ca_y = abs(By) / np.sqrt(rho_safe)
        
        # Fast magnetosonic speed (with numerical safety)
        if direction == 0:
            discriminant = max((cs**2 + ca_x**2 + ca_y**2)**2 - 4*cs**2*ca_x**2, 0.0)
            cf = np.sqrt(0.5 * (cs**2 + ca_x**2 + ca_y**2 + np.sqrt(discriminant)))
            return abs(u) + cf
        else:
            discriminant = max((cs**2 + ca_x**2 + ca_y**2)**2 - 4*cs**2*ca_y**2, 0.0)
            cf = np.sqrt(0.5 * (cs**2 + ca_x**2 + ca_y**2 + np.sqrt(discriminant)))
            return abs(v) + cf
    
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
        
        # Magnetic pressure (using normalized units: B²/2)
        B_squared = Bx**2 + By**2 + Bz**2
        magnetic_pressure = 0.5 * B_squared
        
        # Total pressure
        p_total = p + magnetic_pressure
        
        flux_x = np.array([
            rho * u_vel,                                    # Mass flux
            rho * u_vel**2 + p_total - Bx**2,             # x-momentum flux
            rho * u_vel * v_vel - Bx * By,                 # y-momentum flux
            rho * u_vel * w_vel - Bx * Bz,                 # z-momentum flux
            (E + p_total) * u_vel - Bx * (Bx*u_vel + By*v_vel + Bz*w_vel),  # Energy flux
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
        
        # Magnetic pressure (using normalized units: B²/2)
        B_squared = Bx**2 + By**2 + Bz**2
        magnetic_pressure = 0.5 * B_squared
        
        # Total pressure
        p_total = p + magnetic_pressure
        
        flux_y = np.array([
            rho * v_vel,                                    # Mass flux
            rho * v_vel * u_vel - By * Bx,                 # x-momentum flux
            rho * v_vel**2 + p_total - By**2,             # y-momentum flux
            rho * v_vel * w_vel - By * Bz,                 # z-momentum flux
            (E + p_total) * v_vel - By * (Bx*u_vel + By*v_vel + Bz*w_vel),  # Energy flux
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
        cs = state.sound_speed if state.sound_speed is not None else np.sqrt(self.gamma * state.pressure / rho)
        Bx, By, Bz = state.magnetic_x, state.magnetic_y, state.magnetic_z
        
        # Magnetic field components in computational direction
        if direction == 0:  # x-direction
            B_perp_squared = By**2 + Bz**2
            B_parallel = abs(Bx)
        else:  # y-direction
            B_perp_squared = Bx**2 + Bz**2
            B_parallel = abs(By)
        
        # Alfvén speed in direction of interest
        ca = B_parallel / np.sqrt(rho) if rho > 1e-15 else 0.0
        
        # Total magnetic field
        B_squared = Bx**2 + By**2 + Bz**2
        va_squared = B_squared / rho if rho > 1e-15 else 0.0
        cs_squared = cs**2
        
        # Correct discriminant for magnetosonic speeds
        # cf² = 0.5 * [(cs² + va²) + √((cs² + va²)² - 4cs²(Bₙ²/ρ))]
        # cs² = 0.5 * [(cs² + va²) - √((cs² + va²)² - 4cs²(Bₙ²/ρ))]
        sum_speeds = cs_squared + va_squared
        B_normal_squared = B_parallel**2 / rho if rho > 1e-15 else 0.0
        discriminant_squared = sum_speeds**2 - 4*cs_squared*B_normal_squared
        discriminant = np.sqrt(max(discriminant_squared, 0.0))
        
        cf = np.sqrt(0.5 * (sum_speeds + discriminant))  # Fast magnetosonic speed
        cs_slow = np.sqrt(max(0.5 * (sum_speeds - discriminant), 0.0))  # Slow magnetosonic speed
        
        if direction == 0:  # x-direction
            u_vel = state.velocity_x
            eigenvals = np.array([
                u_vel - cf,      # Fast magnetosonic wave (backward)
                u_vel - ca,      # Alfvén wave (backward)
                u_vel - cs_slow, # Slow magnetosonic wave (backward)
                u_vel,           # Entropy wave
                u_vel,           # Divergence wave (∇·B = 0)
                u_vel + cs_slow, # Slow magnetosonic wave (forward)
                u_vel + ca,      # Alfvén wave (forward)
                u_vel + cf       # Fast magnetosonic wave (forward)
            ])
        else:  # y-direction
            v_vel = state.velocity_y
            eigenvals = np.array([
                v_vel - cf,      # Fast magnetosonic wave (backward)
                v_vel - ca,      # Alfvén wave (backward)
                v_vel - cs_slow, # Slow magnetosonic wave (backward)
                v_vel,           # Entropy wave
                v_vel,           # Divergence wave (∇·B = 0)
                v_vel + cs_slow, # Slow magnetosonic wave (forward)
                v_vel + ca,      # Alfvén wave (forward)
                v_vel + cf       # Fast magnetosonic wave (forward)
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

    # Additional methods required by base class
    
    def compute_pressure(self, state: np.ndarray) -> float:
        """Compute gas pressure from conservative variables"""
        rho, rho_u, rho_v, rho_w, E, Bx, By, Bz = state
        
        # Kinetic energy
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        
        # Magnetic energy
        magnetic_energy = 0.5 * (Bx**2 + By**2 + Bz**2)
        
        # Gas pressure (internal energy)
        internal_energy = E - kinetic_energy - magnetic_energy
        pressure = (self.gamma - 1.0) * internal_energy
        
        return max(pressure, 1e-10)  # Ensure positive pressure
    
    def get_variable_names(self) -> list:
        """Get names of conservative variables"""
        return ['density', 'momentum_x', 'momentum_y', 'momentum_z', 
                'energy', 'magnetic_x', 'magnetic_y', 'magnetic_z']
    
    def get_primitive_names(self) -> list:
        """Get names of primitive variables"""
        return ['density', 'velocity_x', 'velocity_y', 'velocity_z',
                'pressure', 'magnetic_x', 'magnetic_y', 'magnetic_z']
    
    def validate_state(self, state: np.ndarray) -> bool:
        """Validate physical consistency of MHD state"""
        if not super().validate_state(state):
            return False
        
        rho, rho_u, rho_v, rho_w, E, Bx, By, Bz = state
        
        # Check positive density
        if rho <= 0:
            return False
        
        # Check positive gas pressure
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        magnetic_energy = 0.5 * (Bx**2 + By**2 + Bz**2)
        internal_energy = E - kinetic_energy - magnetic_energy
        gas_pressure = (self.gamma - 1.0) * internal_energy
        
        if gas_pressure <= 0:
            return False
        
        return True


