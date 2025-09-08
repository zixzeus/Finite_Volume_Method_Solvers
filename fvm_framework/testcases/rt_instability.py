"""
Rayleigh-Taylor Instability Test Case

This module implements the Rayleigh-Taylor instability, which occurs when
a denser fluid is supported by a lighter fluid under gravity or acceleration.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from fvm_framework.core.data_container import FVMDataContainer2D, GridGeometry
from fvm_framework.physics.euler_equations import EulerEquations2D, EulerState


@dataclass
class RTInstabilityParameters:
    """Parameters for Rayleigh-Taylor instability"""
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 2.0
    
    # Grid resolution
    nx: int = 128
    ny: int = 256
    
    # Fluid properties
    density_heavy: float = 2.0
    density_light: float = 1.0
    interface_y: float = 1.0
    interface_thickness: float = 0.05
    
    # Pressure and gravity
    pressure_top: float = 1.0
    gravity: float = 1.0  # Acceleration
    
    # Perturbation parameters
    perturbation_amplitude: float = 0.01
    perturbation_wavelength: float = 0.2
    perturbation_modes: int = 3
    
    # Physics
    gamma: float = 1.4
    
    # Simulation parameters
    final_time: float = 3.0
    cfl_number: float = 0.3
    boundary_condition: str = 'reflective'
    
    # Output parameters
    output_interval: float = 0.2
    save_snapshots: bool = True


class RayleighTaylorInstability:
    """
    Rayleigh-Taylor instability test case.
    
    Features:
    - Density stratification with gravity
    - Interface perturbations
    - Bubble and spike formation
    - Mixing layer growth
    """
    
    def __init__(self, parameters: Optional[RTInstabilityParameters] = None):
        self.params = parameters or RTInstabilityParameters()
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
        """Setup initial conditions for RT instability"""
        nx, ny = self.params.nx, self.params.ny
        
        for i in range(nx):
            for j in range(ny):
                # Cell center coordinates
                x = self.params.x_min + (i + 0.5) * self.geometry.dx
                y = self.params.y_min + (j + 0.5) * self.geometry.dy
                
                # Distance from interface
                y_dist = y - self.params.interface_y
                
                # Smooth density transition
                thickness = self.params.interface_thickness
                transition = 0.5 * (1.0 + np.tanh(y_dist / thickness))
                
                density = (self.params.density_light + 
                          (self.params.density_heavy - self.params.density_light) * (1.0 - transition))
                
                # Hydrostatic pressure profile
                # p = p_top + ∫ρg dy from top to current position
                y_top = self.params.y_max
                pressure = self.params.pressure_top + self.params.gravity * density * (y_top - y)
                
                # Add interface perturbation
                interface_perturbation = 0.0
                for mode in range(1, self.params.perturbation_modes + 1):
                    wavelength = self.params.perturbation_wavelength * mode
                    interface_perturbation += (self.params.perturbation_amplitude / mode *
                                             np.sin(2 * np.pi * x / wavelength) *
                                             np.exp(-abs(y_dist) / thickness))
                
                # Initial velocities (small perturbation)
                velocity_x = 0.0
                velocity_y = interface_perturbation * 0.1  # Small initial velocity
                
                # Create initial state
                state = EulerState(
                    density=density,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    velocity_z=0.0,
                    pressure=pressure
                )
                
                # Convert to conservative variables
                self.data.state[:, i, j] = self.euler_eq.primitive_to_conservative(state)
        
        print(f"Rayleigh-Taylor instability initial conditions set:")
        print(f"  Domain: [{self.params.x_min}, {self.params.x_max}] × [{self.params.y_min}, {self.params.y_max}]")
        print(f"  Grid: {nx} × {ny}")
        print(f"  Density ratio: {self.params.density_heavy / self.params.density_light}")
        print(f"  Gravity: {self.params.gravity}")
        print(f"  Atwood number: {(self.params.density_heavy - self.params.density_light) / (self.params.density_heavy + self.params.density_light)}")
    
    def compute_gravity_source(self) -> np.ndarray:
        """Compute gravitational source terms"""
        source = np.zeros_like(self.data.state)
        
        rho = self.data.state[0]  # Density
        
        # Gravitational source terms
        source[2] = -rho * self.params.gravity  # y-momentum equation
        source[4] = -self.data.state[2] * self.params.gravity  # Energy equation
        
        return source
    
    def compute_mixing_width(self) -> float:
        """Compute mixing layer width"""
        density = self.data.state[0]
        
        # Find the extent of mixed region
        # This is simplified - a full calculation would use density variance
        rho_mean = 0.5 * (self.params.density_heavy + self.params.density_light)
        mixing_region = np.abs(density - rho_mean) < 0.5 * abs(self.params.density_heavy - self.params.density_light)
        
        # Find y-extent of mixing region
        y_coords = np.arange(self.params.ny) * self.geometry.dy + self.params.y_min
        y_mix_indices = np.any(mixing_region, axis=0)
        
        if np.any(y_mix_indices):
            y_mix_coords = y_coords[y_mix_indices]
            mixing_width = np.max(y_mix_coords) - np.min(y_mix_coords)
        else:
            mixing_width = 0.0
        
        return mixing_width
    
    def compute_bubble_spike_positions(self) -> Dict[str, float]:
        """Compute positions of bubbles (light fluid rising) and spikes (heavy fluid falling)"""
        density = self.data.state[0]
        
        # Find interface positions
        interface_mask = np.abs(density - 0.5 * (self.params.density_heavy + self.params.density_light)) < 0.25 * abs(self.params.density_heavy - self.params.density_light)
        
        bubble_height = 0.0
        spike_depth = 0.0
        
        if np.any(interface_mask):
            y_coords = np.arange(self.params.ny) * self.geometry.dy + self.params.y_min
            
            # For each x-position, find interface y-position
            for i in range(self.params.nx):
                interface_y_indices = np.where(interface_mask[i, :])[0]
                if len(interface_y_indices) > 0:
                    y_interface = y_coords[interface_y_indices]
                    
                    # Bubble: maximum height above initial interface
                    bubble_height = max(bubble_height, np.max(y_interface) - self.params.interface_y)
                    
                    # Spike: maximum depth below initial interface
                    spike_depth = max(spike_depth, self.params.interface_y - np.min(y_interface))
        
        return {
            'bubble_height': bubble_height,
            'spike_depth': spike_depth,
            'asymmetry': bubble_height / spike_depth if spike_depth > 1e-10 else 1.0
        }
    
    def save_snapshot(self, time: float, additional_data: Optional[Dict] = None):
        """Save current state as snapshot"""
        bubble_spike = self.compute_bubble_spike_positions()
        
        snapshot = {
            'time': time,
            'state': self.data.state.copy(),
            'primitives': self.data.get_primitives(self.params.gamma).copy(),
            'mixing_width': self.compute_mixing_width(),
            'bubble_height': bubble_spike['bubble_height'],
            'spike_depth': bubble_spike['spike_depth'],
            'asymmetry': bubble_spike['asymmetry'],
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
    
    def get_visualization_data(self, variable: str = 'density') -> Dict[str, np.ndarray]:
        """Get data for visualization"""
        # Create coordinate arrays
        x = np.linspace(self.params.x_min + 0.5*self.geometry.dx,
                       self.params.x_max - 0.5*self.geometry.dx, self.params.nx)
        y = np.linspace(self.params.y_min + 0.5*self.geometry.dy,
                       self.params.y_max - 0.5*self.geometry.dy, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        self.data.compute_primitives(self.params.gamma)
        
        if variable == 'density':
            data = self.data.primitives[0]
            label = 'Density'
        elif variable == 'pressure':
            data = self.data.primitives[4]
            label = 'Pressure'
        elif variable == 'velocity_y':
            data = self.data.primitives[2]
            label = 'Y-Velocity'
        elif variable == 'velocity_magnitude':
            u = self.data.primitives[1]
            v = self.data.primitives[2]
            data = np.sqrt(u**2 + v**2)
            label = 'Velocity Magnitude'
        else:
            data = self.data.primitives[0]
            label = 'Density'
        
        return {
            'x': x, 'y': y, 'X': X, 'Y': Y,
            'data': data, 'label': label,
            'title': f'{label} - Rayleigh-Taylor Instability'
        }
    
    def run_simulation(self, solver, time_integrator, output_callback: Optional[Callable] = None):
        """Run RT instability simulation"""
        print("Running Rayleigh-Taylor instability simulation...")
        
        self.setup_initial_conditions()
        
        if self.params.save_snapshots:
            self.save_snapshot(0.0)
        
        current_time = 0.0
        next_output_time = self.params.output_interval
        step = 0
        
        # Modified solver to include gravity
        original_compute_residual = solver.compute_spatial_residual if hasattr(solver, 'compute_spatial_residual') else None
        
        def compute_residual_with_gravity(data, **kwargs):
            if original_compute_residual:
                residual = original_compute_residual(data, **kwargs)
            else:
                # Fallback - assume we have a working residual computation
                residual = np.zeros_like(data.state)
            
            # Add gravitational source terms
            gravity_source = self.compute_gravity_source()
            return residual + gravity_source
        
        while current_time < self.params.final_time:
            dt = self.euler_eq.compute_time_step(self.data, self.params.cfl_number)
            dt = min(dt, self.params.final_time - current_time)
            
            time_integrator.integrate(self.data, dt, compute_residual_with_gravity,
                                    gamma=self.params.gamma)
            
            self.euler_eq.apply_boundary_conditions(self.data, self.params.boundary_condition)
            
            current_time += dt
            step += 1
            
            if current_time >= next_output_time or current_time >= self.params.final_time:
                if self.params.save_snapshots:
                    self.save_snapshot(current_time)
                
                if output_callback:
                    output_callback(self, current_time, step)
                
                mixing_width = self.compute_mixing_width()
                bubble_spike = self.compute_bubble_spike_positions()
                
                print(f"Step {step}: t = {current_time:.4f}, dt = {dt:.2e}")
                print(f"  Mixing width: {mixing_width:.4f}")
                print(f"  Bubble height: {bubble_spike['bubble_height']:.4f}")
                print(f"  Spike depth: {bubble_spike['spike_depth']:.4f}")
                
                next_output_time += self.params.output_interval
        
        print(f"Rayleigh-Taylor simulation completed!")
        return {
            'final_time': current_time,
            'total_steps': step,
            'snapshots': self.snapshots,
            'output_times': self.output_times
        }