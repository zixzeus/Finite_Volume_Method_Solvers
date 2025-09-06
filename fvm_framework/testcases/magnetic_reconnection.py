"""
Magnetic Reconnection Test Case

This module implements magnetic reconnection problems, which are fundamental
in plasma physics and space physics. The test case features current sheets
with antiparallel magnetic fields that can undergo reconnection.

The Harris current sheet is used as the primary configuration.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from fvm_framework.core.data_container import GridGeometry
from fvm_framework.physics.mhd_equations import MHDEquations2D, MHDState, MHDInitialConditions


@dataclass
class MHDDataContainer2D:
    """
    Simplified MHD data container for magnetic reconnection.
    This extends the basic data container for 8-variable MHD system.
    """
    def __init__(self, geometry: GridGeometry, num_vars: int = 8):
        self.geometry = geometry
        self.num_vars = num_vars
        self.nx, self.ny = geometry.nx, geometry.ny
        
        # State array for MHD variables [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
        self.state = np.zeros((num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        self.state_new = np.zeros((num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        
        # Flux arrays
        self.flux_x = np.zeros((num_vars, self.nx + 1, self.ny), dtype=np.float64, order='C')
        self.flux_y = np.zeros((num_vars, self.nx, self.ny + 1), dtype=np.float64, order='C')
        
        # Primitive variables cache
        self.primitives = np.zeros((num_vars, self.nx, self.ny), dtype=np.float64, order='C')
        self._primitives_valid = False
    
    def swap_states(self):
        """Swap current and new state arrays"""
        self.state, self.state_new = self.state_new, self.state
        self._primitives_valid = False


@dataclass
class MagneticReconnectionParameters:
    """Parameters for magnetic reconnection test case"""
    # Domain
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    
    # Grid resolution
    nx: int = 100
    ny: int = 100
    
    # Current sheet parameters
    sheet_thickness: float = 0.1
    magnetic_field_strength: float = 1.0
    density_ratio: float = 0.1
    background_pressure: float = 0.1
    
    # Perturbation parameters (to trigger reconnection)
    perturbation_amplitude: float = 0.01
    perturbation_wavelength: float = 1.0
    
    # Physics
    gamma: float = 5.0/3.0
    
    # Simulation parameters
    final_time: float = 5.0
    cfl_number: float = 0.3
    boundary_condition: str = 'periodic'
    
    # Divergence cleaning
    use_divergence_cleaning: bool = True
    cleaning_frequency: int = 10
    
    # Output parameters
    output_interval: float = 0.5
    save_snapshots: bool = True


class MagneticReconnection:
    """
    Magnetic reconnection test case implementation.
    
    This test features:
    - Harris current sheet configuration
    - Antiparallel magnetic fields
    - Magnetic reconnection and island formation
    - Energy conversion from magnetic to kinetic
    - Complex plasma dynamics
    """
    
    def __init__(self, parameters: Optional[MagneticReconnectionParameters] = None):
        """
        Initialize magnetic reconnection test case.
        
        Args:
            parameters: Test case parameters
        """
        self.params = parameters or MagneticReconnectionParameters()
        self.mhd_eq = MHDEquations2D(gamma=self.params.gamma)
        
        # Create grid geometry
        self.geometry = GridGeometry(
            nx=self.params.nx,
            ny=self.params.ny,
            dx=(self.params.x_max - self.params.x_min) / self.params.nx,
            dy=(self.params.y_max - self.params.y_min) / self.params.ny,
            x_min=self.params.x_min,
            y_min=self.params.y_min
        )
        
        # Initialize MHD data container
        self.data = MHDDataContainer2D(self.geometry)
        
        # Output data
        self.snapshots = []
        self.output_times = []
        
        # Reconnection diagnostics
        self.reconnection_rate_history = []
        self.magnetic_energy_history = []
        self.kinetic_energy_history = []
        self.div_b_history = []
        
    def setup_initial_conditions(self):
        """Setup initial conditions for magnetic reconnection"""
        nx, ny = self.params.nx, self.params.ny
        
        for i in range(nx):
            for j in range(ny):
                # Cell center coordinates
                x = self.params.x_min + (i + 0.5) * self.geometry.dx
                y = self.params.y_min + (j + 0.5) * self.geometry.dy
                
                # Harris current sheet
                sheet_thickness = self.params.sheet_thickness
                B0 = self.params.magnetic_field_strength
                
                # Magnetic field (Harris sheet)
                Bx = B0 * np.tanh(y / sheet_thickness)
                By = 0.0
                Bz = 0.0
                
                # Add perturbation to trigger reconnection
                if abs(y) < 2 * sheet_thickness:
                    perturbation = (self.params.perturbation_amplitude * 
                                  np.cos(2 * np.pi * x / self.params.perturbation_wavelength) *
                                  np.exp(-y**2 / (2 * sheet_thickness**2)))
                    By += perturbation
                
                # Density profile
                density = 1.0 + self.params.density_ratio / np.cosh(y / sheet_thickness)**2
                
                # Pressure balance
                magnetic_pressure = 0.5 * (Bx**2 + By**2 + Bz**2)
                pressure = 0.5 * B0**2 - magnetic_pressure + self.params.background_pressure
                pressure = max(pressure, 0.01)  # Ensure positive pressure
                
                # Initial velocities (at rest)
                velocity_x = 0.0
                velocity_y = 0.0
                velocity_z = 0.0
                
                # Create MHD state
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
                
                # Convert to conservative variables
                self.data.state[:, i, j] = self.mhd_eq.primitive_to_conservative(state)
        
        # Mark primitives as invalid
        self.data._primitives_valid = False
        
        print(f"Magnetic reconnection initial conditions set:")
        print(f"  Domain: [{self.params.x_min}, {self.params.x_max}] × [{self.params.y_min}, {self.params.y_max}]")
        print(f"  Grid: {nx} × {ny}")
        print(f"  Current sheet thickness: {self.params.sheet_thickness}")
        print(f"  Magnetic field strength: {self.params.magnetic_field_strength}")
        print(f"  Perturbation amplitude: {self.params.perturbation_amplitude}")
    
    def compute_magnetic_energy(self) -> float:
        """Compute total magnetic energy in the domain"""
        Bx = self.data.state[5]  # Magnetic field x-component
        By = self.data.state[6]  # Magnetic field y-component  
        Bz = self.data.state[7]  # Magnetic field z-component
        
        magnetic_energy_density = 0.5 * (Bx**2 + By**2 + Bz**2)
        total_magnetic_energy = np.sum(magnetic_energy_density) * self.geometry.cell_volume
        
        return total_magnetic_energy
    
    def compute_kinetic_energy(self) -> float:
        """Compute total kinetic energy in the domain"""
        rho = self.data.state[0]
        rho_u = self.data.state[1]
        rho_v = self.data.state[2]
        rho_w = self.data.state[3]
        
        kinetic_energy_density = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / np.maximum(rho, 1e-15)
        total_kinetic_energy = np.sum(kinetic_energy_density) * self.geometry.cell_volume
        
        return total_kinetic_energy
    
    def compute_reconnection_rate(self) -> float:
        """
        Compute magnetic reconnection rate.
        
        This is typically measured as the rate of magnetic flux change
        through the current sheet or the maximum electric field.
        """
        # Simplified calculation using maximum current density
        Bx = self.data.state[5]
        By = self.data.state[6]
        
        # Current density J = ∇ × B (z-component only for 2D)
        dBx_dy = np.gradient(Bx, self.geometry.dy, axis=1)
        dBy_dx = np.gradient(By, self.geometry.dx, axis=0)
        current_z = dBy_dx - dBx_dy
        
        # Reconnection rate proportional to maximum current
        reconnection_rate = np.max(np.abs(current_z))
        
        return reconnection_rate
    
    def compute_divergence_b_error(self) -> float:
        """Compute RMS error in magnetic field divergence"""
        div_B = self.mhd_eq.compute_divergence_b(self.data)
        rms_error = np.sqrt(np.mean(div_B**2))
        return rms_error
    
    def detect_magnetic_islands(self) -> Dict[str, float]:
        """
        Detect magnetic islands using flux function analysis.
        
        Returns:
            Dictionary with island detection results
        """
        # Compute vector potential A_z (flux function)
        # ∂A_z/∂x = -By, ∂A_z/∂y = Bx
        By = self.data.state[6]
        
        # Integrate to get flux function (simplified)
        # This is an approximation - full calculation would need proper boundary conditions
        flux_function = np.cumsum(-By, axis=0) * self.geometry.dx
        
        # Find extrema in flux function (island centers)
        grad_flux_x = np.gradient(flux_function, axis=0)
        grad_flux_y = np.gradient(flux_function, axis=1)
        grad_magnitude = np.sqrt(grad_flux_x**2 + grad_flux_y**2)
        
        # Simple island detection based on flux function topology
        num_extrema = np.sum(grad_magnitude < 0.1 * np.max(grad_magnitude))
        max_flux_variation = np.max(flux_function) - np.min(flux_function)
        
        return {
            'num_potential_islands': num_extrema,
            'flux_variation': max_flux_variation,
            'max_flux_gradient': np.max(grad_magnitude)
        }
    
    def save_snapshot(self, time: float, additional_data: Optional[Dict] = None):
        """Save current state as snapshot with reconnection diagnostics"""
        snapshot = {
            'time': time,
            'state': self.data.state.copy(),
            'magnetic_energy': self.compute_magnetic_energy(),
            'kinetic_energy': self.compute_kinetic_energy(),
            'reconnection_rate': self.compute_reconnection_rate(),
            'divergence_b_error': self.compute_divergence_b_error(),
            'magnetic_islands': self.detect_magnetic_islands()
        }
        
        if additional_data:
            snapshot.update(additional_data)
        
        self.snapshots.append(snapshot)
        self.output_times.append(time)
        
        # Update history arrays
        self.magnetic_energy_history.append(snapshot['magnetic_energy'])
        self.kinetic_energy_history.append(snapshot['kinetic_energy'])
        self.reconnection_rate_history.append(snapshot['reconnection_rate'])
        self.div_b_history.append(snapshot['divergence_b_error'])
    
    def get_visualization_data(self, variable: str = 'magnetic_field_lines') -> Dict[str, np.ndarray]:
        """
        Get data for visualization.
        
        Args:
            variable: Variable to visualize
            
        Returns:
            Dictionary with visualization data
        """
        # Create coordinate arrays
        x = np.linspace(self.params.x_min + 0.5*self.geometry.dx,
                       self.params.x_max - 0.5*self.geometry.dx, self.params.nx)
        y = np.linspace(self.params.y_min + 0.5*self.geometry.dy,
                       self.params.y_max - 0.5*self.geometry.dy, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if variable == 'magnetic_field_lines':
            # Vector potential for field lines
            By = self.data.state[6]
            flux_function = np.cumsum(-By, axis=0) * self.geometry.dx
            data = flux_function
            label = 'Magnetic Field Lines (Flux Function)'
            
        elif variable == 'current_density':
            Bx = self.data.state[5]
            By = self.data.state[6]
            dBx_dy = np.gradient(Bx, self.geometry.dy, axis=1)
            dBy_dx = np.gradient(By, self.geometry.dx, axis=0)
            data = dBy_dx - dBx_dy
            label = 'Current Density (J_z)'
            
        elif variable == 'density':
            data = self.data.state[0]
            label = 'Density'
            
        elif variable == 'pressure':
            # Compute pressure from conservative variables
            pressure_data = np.zeros((self.params.nx, self.params.ny))
            for i in range(self.params.nx):
                for j in range(self.params.ny):
                    state = self.mhd_eq.conservative_to_primitive(self.data.state[:, i, j])
                    pressure_data[i, j] = state.pressure
            data = pressure_data
            label = 'Pressure'
            
        elif variable == 'magnetic_energy':
            Bx = self.data.state[5]
            By = self.data.state[6] 
            Bz = self.data.state[7]
            data = 0.5 * (Bx**2 + By**2 + Bz**2)
            label = 'Magnetic Energy Density'
            
        elif variable == 'velocity_magnitude':
            rho = np.maximum(self.data.state[0], 1e-15)
            u = self.data.state[1] / rho
            v = self.data.state[2] / rho
            w = self.data.state[3] / rho
            data = np.sqrt(u**2 + v**2 + w**2)
            label = 'Velocity Magnitude'
            
        else:
            data = self.data.state[0]  # Default to density
            label = 'Density'
        
        return {
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'data': data,
            'label': label,
            'title': f'{label} - Magnetic Reconnection',
            'magnetic_field_x': self.data.state[5],
            'magnetic_field_y': self.data.state[6]
        }
    
    def run_simulation(self, solver, time_integrator, output_callback: Optional[Callable] = None):
        """
        Run the magnetic reconnection simulation.
        
        Args:
            solver: Spatial solver (must handle MHD equations)
            time_integrator: Time integration scheme
            output_callback: Optional callback for output processing
        """
        print("Running magnetic reconnection simulation...")
        
        # Setup initial conditions
        self.setup_initial_conditions()
        
        # Save initial snapshot
        if self.params.save_snapshots:
            self.save_snapshot(0.0)
        
        # Simulation loop
        current_time = 0.0
        next_output_time = self.params.output_interval
        step = 0
        
        while current_time < self.params.final_time:
            # Compute time step using MHD wave speeds
            max_wave_speed = self.mhd_eq.compute_max_wave_speed(self.data)
            min_dx = min(self.geometry.dx, self.geometry.dy)
            
            if max_wave_speed > 1e-15:
                dt = self.params.cfl_number * min_dx / max_wave_speed
            else:
                dt = 1e-6
            
            dt = min(dt, self.params.final_time - current_time)
            
            # Take time step
            time_integrator.integrate(self.data, dt, solver.compute_spatial_residual,
                                    gamma=self.params.gamma, equations='mhd')
            
            # Apply boundary conditions
            self.apply_boundary_conditions()
            
            # Apply divergence cleaning
            if (self.params.use_divergence_cleaning and 
                step % self.params.cleaning_frequency == 0):
                self.mhd_eq.apply_divergence_cleaning(self.data)
            
            # Update time
            current_time += dt
            step += 1
            
            # Output handling
            if current_time >= next_output_time or current_time >= self.params.final_time:
                if self.params.save_snapshots:
                    self.save_snapshot(current_time)
                
                if output_callback:
                    output_callback(self, current_time, step)
                
                print(f"Step {step}: t = {current_time:.4f}, dt = {dt:.2e}")
                
                # Print diagnostics
                mag_energy = self.compute_magnetic_energy()
                kin_energy = self.compute_kinetic_energy()
                reconnection_rate = self.compute_reconnection_rate()
                div_b_error = self.compute_divergence_b_error()
                
                print(f"  Magnetic energy: {mag_energy:.4f}")
                print(f"  Kinetic energy: {kin_energy:.4f}")
                print(f"  Reconnection rate: {reconnection_rate:.4f}")
                print(f"  ∇·B RMS error: {div_b_error:.2e}")
                
                next_output_time += self.params.output_interval
        
        print(f"Magnetic reconnection simulation completed!")
        print(f"  Final time: {current_time:.4f}")
        print(f"  Total steps: {step}")
        
        return {
            'final_time': current_time,
            'total_steps': step,
            'snapshots': self.snapshots,
            'output_times': self.output_times,
            'magnetic_energy_history': self.magnetic_energy_history,
            'kinetic_energy_history': self.kinetic_energy_history,
            'reconnection_rate_history': self.reconnection_rate_history,
            'divergence_b_history': self.div_b_history
        }
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions for MHD equations"""
        if self.params.boundary_condition == 'periodic':
            # Periodic boundaries for all variables
            # This is handled by the spatial discretization
            pass
        elif self.params.boundary_condition == 'conducting_wall':
            # Conducting wall: normal B = 0, tangential E = 0
            # This is a simplified implementation
            pass
        else:
            # Default to periodic
            pass