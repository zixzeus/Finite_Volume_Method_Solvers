"""
Kelvin-Helmholtz Instability Test Case

This module implements the Kelvin-Helmholtz instability, which occurs
at the interface between two fluid layers moving at different velocities.
This is a classic fluid dynamics instability test case.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from ..core.data_container import FVMDataContainer2D, GridGeometry
from ..physics.euler_equations import EulerEquations2D, EulerState


@dataclass
class KHInstabilityParameters:
    """Parameters for Kelvin-Helmholtz instability"""
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    
    # Grid resolution
    nx: int = 256
    ny: int = 256
    
    # Flow parameters
    velocity_upper: float = 0.5
    velocity_lower: float = -0.5
    interface_y: float = 0.5
    shear_thickness: float = 0.05
    
    # Fluid properties
    density_upper: float = 1.0
    density_lower: float = 2.0
    pressure_uniform: float = 1.0
    
    # Perturbation parameters
    perturbation_amplitude: float = 0.01
    perturbation_wavelength: float = 0.2
    perturbation_modes: int = 5
    
    # Physics
    gamma: float = 1.4
    
    # Simulation parameters
    final_time: float = 2.0
    cfl_number: float = 0.3
    boundary_condition: str = 'periodic'
    
    # Output parameters
    output_interval: float = 0.1
    save_snapshots: bool = True


class KelvinHelmholtzInstability:
    """
    Kelvin-Helmholtz instability test case.
    
    Features:
    - Shear flow interface
    - Density stratification
    - Growth of instability modes
    - Vortex formation and rollup
    - Mixing layer development
    """
    
    def __init__(self, parameters: Optional[KHInstabilityParameters] = None):
        self.params = parameters or KHInstabilityParameters()
        self.euler_eq = EulerEquations2D(gamma=self.params.gamma)
        
        # Create grid geometry
        self.geometry = GridGeometry(
            nx=self.params.nx,
            ny=self.params.ny,
            dx=(self.params.x_max - self.params.x_min) / self.params.nx,
            dy=(self.params.y_max - self.params.y_min) / self.params.ny,
            x_min=self.params.x_min,
            y_min=self.params.y_min
        )
        
        # Initialize data container
        self.data = FVMDataContainer2D(self.geometry, num_vars=5)
        
        # Output data
        self.snapshots = []
        self.output_times = []
        
    def setup_initial_conditions(self):
        """Setup initial conditions for KH instability"""
        nx, ny = self.params.nx, self.params.ny
        
        for i in range(nx):
            for j in range(ny):
                # Cell center coordinates
                x = self.params.x_min + (i + 0.5) * self.geometry.dx
                y = self.params.y_min + (j + 0.5) * self.geometry.dy
                
                # Distance from interface
                y_dist = y - self.params.interface_y
                
                # Smooth transition using tanh profile
                thickness = self.params.shear_thickness
                transition = 0.5 * (1.0 + np.tanh(y_dist / thickness))
                
                # Velocity profile
                velocity_x = (self.params.velocity_lower + 
                             (self.params.velocity_upper - self.params.velocity_lower) * transition)
                
                # Density profile  
                density = (self.params.density_lower + 
                          (self.params.density_upper - self.params.density_lower) * transition)
                
                # Add velocity perturbation to trigger instability
                perturbation = 0.0
                for mode in range(1, self.params.perturbation_modes + 1):
                    wavelength = self.params.perturbation_wavelength / mode
                    perturbation += (self.params.perturbation_amplitude / mode *
                                   np.sin(2 * np.pi * x / wavelength) *
                                   np.exp(-abs(y_dist) / thickness))
                
                velocity_y = perturbation
                
                # Create initial state
                state = EulerState(
                    density=density,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    velocity_z=0.0,
                    pressure=self.params.pressure_uniform
                )
                
                # Convert to conservative variables
                self.data.state[:, i, j] = self.euler_eq.primitive_to_conservative(state)
        
        self.data._primitives_valid = False
        
        print(f"Kelvin-Helmholtz instability initial conditions set:")
        print(f"  Domain: [{self.params.x_min}, {self.params.x_max}] × [{self.params.y_min}, {self.params.y_max}]")
        print(f"  Grid: {nx} × {ny}")
        print(f"  Velocity jump: {self.params.velocity_upper - self.params.velocity_lower}")
        print(f"  Density ratio: {self.params.density_lower / self.params.density_upper}")
        print(f"  Perturbation amplitude: {self.params.perturbation_amplitude}")
    
    def compute_vorticity(self) -> np.ndarray:
        """Compute vorticity field ω = ∇ × v"""
        self.data.compute_primitives(self.params.gamma)
        u = self.data.primitives[1]  # x-velocity
        v = self.data.primitives[2]  # y-velocity
        
        # Compute vorticity (z-component)
        du_dy = np.gradient(u, self.geometry.dy, axis=1)
        dv_dx = np.gradient(v, self.geometry.dx, axis=0)
        vorticity = dv_dx - du_dy
        
        return vorticity
    
    def compute_enstrophy(self) -> float:
        """Compute total enstrophy (integrated vorticity squared)"""
        vorticity = self.compute_vorticity()
        enstrophy = 0.5 * np.sum(vorticity**2) * self.geometry.cell_volume
        return enstrophy
    
    def compute_mixing_measure(self) -> float:
        """Compute mixing measure based on density gradients"""
        density = self.data.state[0]
        
        # Compute density gradients
        grad_rho_x = np.gradient(density, self.geometry.dx, axis=0)
        grad_rho_y = np.gradient(density, self.geometry.dy, axis=1)
        grad_magnitude = np.sqrt(grad_rho_x**2 + grad_rho_y**2)
        
        # Mixing measure
        mixing = np.sum(grad_magnitude) * self.geometry.cell_volume
        return mixing
    
    def save_snapshot(self, time: float, additional_data: Optional[Dict] = None):
        """Save current state as snapshot"""
        snapshot = {
            'time': time,
            'state': self.data.state.copy(),
            'primitives': self.data.get_primitives(self.params.gamma).copy(),
            'vorticity': self.compute_vorticity(),
            'enstrophy': self.compute_enstrophy(),
            'mixing_measure': self.compute_mixing_measure(),
            'conservation': {
                'mass': self.euler_eq.compute_total_mass(self.data),
                'energy': self.euler_eq.compute_total_energy(self.data),
                'momentum': self.euler_eq.compute_total_momentum(self.data)
            }
        }
        
        if additional_data:
            snapshot.update(additional_data)
        
        self.snapshots.append(snapshot)
        self.output_times.append(time)
    
    def get_visualization_data(self, variable: str = 'vorticity') -> Dict[str, np.ndarray]:
        """Get data for visualization"""
        # Create coordinate arrays
        x = np.linspace(self.params.x_min + 0.5*self.geometry.dx,
                       self.params.x_max - 0.5*self.geometry.dx, self.params.nx)
        y = np.linspace(self.params.y_min + 0.5*self.geometry.dy,
                       self.params.y_max - 0.5*self.geometry.dy, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        self.data.compute_primitives(self.params.gamma)
        
        if variable == 'vorticity':
            data = self.compute_vorticity()
            label = 'Vorticity'
        elif variable == 'density':
            data = self.data.primitives[0]
            label = 'Density'
        elif variable == 'pressure':
            data = self.data.primitives[4]
            label = 'Pressure'
        elif variable == 'velocity_x':
            data = self.data.primitives[1]
            label = 'X-Velocity'
        elif variable == 'velocity_y':
            data = self.data.primitives[2]
            label = 'Y-Velocity'
        elif variable == 'velocity_magnitude':
            u = self.data.primitives[1]
            v = self.data.primitives[2]
            data = np.sqrt(u**2 + v**2)
            label = 'Velocity Magnitude'
        else:
            data = self.compute_vorticity()
            label = 'Vorticity'
        
        return {
            'x': x, 'y': y, 'X': X, 'Y': Y,
            'data': data, 'label': label,
            'title': f'{label} - Kelvin-Helmholtz Instability',
            'velocity_x': self.data.primitives[1],
            'velocity_y': self.data.primitives[2]
        }
    
    def run_simulation(self, solver, time_integrator, output_callback: Optional[Callable] = None):
        """Run KH instability simulation"""
        print("Running Kelvin-Helmholtz instability simulation...")
        
        self.setup_initial_conditions()
        
        if self.params.save_snapshots:
            self.save_snapshot(0.0)
        
        current_time = 0.0
        next_output_time = self.params.output_interval
        step = 0
        
        while current_time < self.params.final_time:
            dt = self.euler_eq.compute_time_step(self.data, self.params.cfl_number)
            dt = min(dt, self.params.final_time - current_time)
            
            time_integrator.integrate(self.data, dt, solver.compute_spatial_residual,
                                    gamma=self.params.gamma)
            
            self.euler_eq.apply_boundary_conditions(self.data, self.params.boundary_condition)
            
            current_time += dt
            step += 1
            
            if current_time >= next_output_time or current_time >= self.params.final_time:
                if self.params.save_snapshots:
                    self.save_snapshot(current_time)
                
                if output_callback:
                    output_callback(self, current_time, step)
                
                enstrophy = self.compute_enstrophy()
                mixing = self.compute_mixing_measure()
                
                print(f"Step {step}: t = {current_time:.4f}, dt = {dt:.2e}")
                print(f"  Enstrophy: {enstrophy:.4f}")
                print(f"  Mixing measure: {mixing:.4f}")
                
                next_output_time += self.params.output_interval
        
        print(f"Kelvin-Helmholtz simulation completed!")
        return {
            'final_time': current_time,
            'total_steps': step,
            'snapshots': self.snapshots,
            'output_times': self.output_times
        }