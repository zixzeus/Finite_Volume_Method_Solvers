"""
Blast Wave Test Case

This module implements the classic 2D blast wave problem, which is a standard
benchmark for testing shock-capturing schemes and high-order methods.

The problem consists of a high-pressure region in the center of the domain
that expands outward, creating a strong shock wave.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from ..core.data_container import FVMDataContainer2D, GridGeometry
from ..physics.euler_equations import EulerEquations2D, EulerState, EulerInitialConditions


@dataclass
class BlastWaveParameters:
    """Parameters for blast wave test case"""
    # Domain
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    
    # Grid resolution
    nx: int = 100
    ny: int = 100
    
    # Blast parameters
    blast_center_x: float = 0.5
    blast_center_y: float = 0.5
    blast_radius: float = 0.1
    
    # Initial conditions
    high_pressure: float = 100.0      # Inside blast
    low_pressure: float = 1.0         # Outside blast
    density: float = 1.0              # Uniform density
    velocity_x: float = 0.0           # Initially at rest
    velocity_y: float = 0.0           # Initially at rest
    
    # Physics
    gamma: float = 1.4                # Heat capacity ratio
    
    # Simulation parameters
    final_time: float = 0.2
    cfl_number: float = 0.3
    boundary_condition: str = 'transmissive'
    
    # Output parameters
    output_interval: float = 0.02
    save_snapshots: bool = True


class BlastWave:
    """
    2D Blast Wave test case implementation.
    
    This classic test problem features:
    - Strong shock formation and propagation
    - Circular symmetry (in theory)
    - High gradients requiring shock-capturing
    - Known analytical properties for validation
    """
    
    def __init__(self, parameters: Optional[BlastWaveParameters] = None):
        """
        Initialize blast wave test case.
        
        Args:
            parameters: Test case parameters (uses defaults if None)
        """
        self.params = parameters or BlastWaveParameters()
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
        
        # Analytical solution data (if available)
        self.analytical_solution = None
        
        # Output data
        self.snapshots = []
        self.output_times = []
        
    def setup_initial_conditions(self):
        """Setup initial conditions for blast wave problem"""
        nx, ny = self.params.nx, self.params.ny
        
        for i in range(nx):
            for j in range(ny):
                # Cell center coordinates
                x = self.params.x_min + (i + 0.5) * self.geometry.dx
                y = self.params.y_min + (j + 0.5) * self.geometry.dy
                
                # Distance from blast center
                r = np.sqrt((x - self.params.blast_center_x)**2 + 
                          (y - self.params.blast_center_y)**2)
                
                # Set pressure based on radius
                if r <= self.params.blast_radius:
                    pressure = self.params.high_pressure
                else:
                    pressure = self.params.low_pressure
                
                # Create initial state
                state = EulerState(
                    density=self.params.density,
                    velocity_x=self.params.velocity_x,
                    velocity_y=self.params.velocity_y,
                    velocity_z=0.0,
                    pressure=pressure
                )
                
                # Convert to conservative variables
                self.data.state[:, i, j] = self.euler_eq.primitive_to_conservative(state)
        
        # Mark primitives as invalid
        self.data._primitives_valid = False
        
        print(f"Blast wave initial conditions set:")
        print(f"  Domain: [{self.params.x_min}, {self.params.x_max}] × [{self.params.y_min}, {self.params.y_max}]")
        print(f"  Grid: {nx} × {ny}")
        print(f"  Blast center: ({self.params.blast_center_x}, {self.params.blast_center_y})")
        print(f"  Blast radius: {self.params.blast_radius}")
        print(f"  Pressure ratio: {self.params.high_pressure / self.params.low_pressure}")
    
    def get_analytical_solution(self, t: float) -> Optional[Dict[str, np.ndarray]]:
        """
        Get analytical solution at time t (if available).
        
        For the blast wave problem, exact solutions are not generally available,
        but we can provide some analytical estimates for the shock radius and
        other properties based on similarity solutions.
        
        Args:
            t: Time
            
        Returns:
            Dictionary with analytical solution data or None
        """
        if t <= 0:
            return None
        
        # Sedov-Taylor blast wave similarity solution (approximate)
        # This assumes strong shock and gamma-law gas
        gamma = self.params.gamma
        
        # Total energy of blast (approximate)
        E0 = self.params.high_pressure * np.pi * self.params.blast_radius**2 / (gamma - 1)
        
        # Sedov-Taylor shock radius
        # R(t) = ξ * (E₀*t²/(ρ₀))^(1/(2+ν)) where ν=2 for 2D, ξ ≈ 1.0
        xi = 1.033  # Similarity constant for 2D
        rho0 = self.params.density
        
        shock_radius = xi * (E0 * t**2 / rho0)**(1.0 / (2 + 2))  # 2D case
        
        return {
            'shock_radius': shock_radius,
            'total_energy': E0,
            'time': t
        }
    
    def compute_error_norms(self, reference_solution: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute error norms against reference solution.
        
        Args:
            reference_solution: Reference solution array
            
        Returns:
            Dictionary with error norms
        """
        if reference_solution is None:
            return {'error': 'No reference solution provided'}
        
        # Compute L1, L2, and L∞ norms
        diff = self.data.state - reference_solution
        
        l1_norm = np.mean(np.abs(diff))
        l2_norm = np.sqrt(np.mean(diff**2))
        linf_norm = np.max(np.abs(diff))
        
        return {
            'L1_norm': l1_norm,
            'L2_norm': l2_norm,
            'Linf_norm': linf_norm
        }
    
    def compute_conservation_properties(self) -> Dict[str, float]:
        """Compute conservation properties of the current solution"""
        total_mass = self.euler_eq.compute_total_mass(self.data)
        total_energy = self.euler_eq.compute_total_energy(self.data)
        momentum_x, momentum_y = self.euler_eq.compute_total_momentum(self.data)
        
        return {
            'total_mass': total_mass,
            'total_energy': total_energy,
            'total_momentum_x': momentum_x,
            'total_momentum_y': momentum_y
        }
    
    def detect_shock_position(self) -> Dict[str, float]:
        """
        Detect shock position using pressure gradients.
        
        Returns:
            Dictionary with shock detection results
        """
        # Compute primitives
        self.data.compute_primitives(self.params.gamma)
        pressure = self.data.primitives[4]  # Pressure field
        
        # Compute pressure gradients
        grad_p_x = np.gradient(pressure, axis=0)
        grad_p_y = np.gradient(pressure, axis=1)
        grad_p_magnitude = np.sqrt(grad_p_x**2 + grad_p_y**2)
        
        # Find maximum gradient location
        max_grad_idx = np.unravel_index(np.argmax(grad_p_magnitude), grad_p_magnitude.shape)
        max_grad_x = self.params.x_min + (max_grad_idx[0] + 0.5) * self.geometry.dx
        max_grad_y = self.params.y_min + (max_grad_idx[1] + 0.5) * self.geometry.dy
        
        # Estimate shock radius
        shock_radius = np.sqrt((max_grad_x - self.params.blast_center_x)**2 + 
                              (max_grad_y - self.params.blast_center_y)**2)
        
        return {
            'shock_radius': shock_radius,
            'shock_center_x': max_grad_x,
            'shock_center_y': max_grad_y,
            'max_pressure_gradient': np.max(grad_p_magnitude)
        }
    
    def save_snapshot(self, time: float, additional_data: Optional[Dict] = None):
        """Save current state as snapshot"""
        snapshot = {
            'time': time,
            'state': self.data.state.copy(),
            'primitives': self.data.get_primitives(self.params.gamma).copy(),
            'conservation': self.compute_conservation_properties(),
            'shock_position': self.detect_shock_position()
        }
        
        if additional_data:
            snapshot.update(additional_data)
        
        self.snapshots.append(snapshot)
        self.output_times.append(time)
    
    def get_visualization_data(self, variable: str = 'density') -> Dict[str, np.ndarray]:
        """
        Get data for visualization.
        
        Args:
            variable: Variable to visualize ('density', 'pressure', 'velocity_magnitude', etc.)
            
        Returns:
            Dictionary with visualization data
        """
        # Compute primitives if needed
        self.data.compute_primitives(self.params.gamma)
        
        # Create coordinate arrays
        x = np.linspace(self.params.x_min + 0.5*self.geometry.dx,
                       self.params.x_max - 0.5*self.geometry.dx, self.params.nx)
        y = np.linspace(self.params.y_min + 0.5*self.geometry.dy,
                       self.params.y_max - 0.5*self.geometry.dy, self.params.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Select variable
        if variable == 'density':
            data = self.data.primitives[0]
            label = 'Density'
        elif variable == 'pressure':
            data = self.data.primitives[4]
            label = 'Pressure'
        elif variable == 'velocity_magnitude':
            u = self.data.primitives[1]
            v = self.data.primitives[2]
            data = np.sqrt(u**2 + v**2)
            label = 'Velocity Magnitude'
        elif variable == 'velocity_x':
            data = self.data.primitives[1]
            label = 'X-Velocity'
        elif variable == 'velocity_y':
            data = self.data.primitives[2]
            label = 'Y-Velocity'
        elif variable == 'energy':
            data = self.data.state[4]
            label = 'Total Energy'
        else:
            data = self.data.primitives[0]  # Default to density
            label = 'Density'
        
        return {
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'data': data,
            'label': label,
            'title': f'{label} - Blast Wave'
        }
    
    def run_simulation(self, solver, time_integrator, output_callback: Optional[Callable] = None):
        """
        Run the blast wave simulation.
        
        Args:
            solver: Spatial solver (e.g., finite volume scheme)
            time_integrator: Time integration scheme
            output_callback: Optional callback for output processing
        """
        print("Running blast wave simulation...")
        
        # Setup initial conditions
        self.setup_initial_conditions()
        
        # Save initial snapshot
        if self.params.save_snapshots:
            self.save_snapshot(0.0)
        
        # Initialize conservation tracking
        initial_conservation = self.compute_conservation_properties()
        
        # Simulation loop
        current_time = 0.0
        next_output_time = self.params.output_interval
        step = 0
        
        while current_time < self.params.final_time:
            # Compute time step
            dt = self.euler_eq.compute_time_step(self.data, self.params.cfl_number)
            dt = min(dt, self.params.final_time - current_time)
            
            # Take time step
            time_integrator.integrate(self.data, dt, solver.compute_spatial_residual, 
                                    gamma=self.params.gamma)
            
            # Apply boundary conditions
            self.euler_eq.apply_boundary_conditions(self.data, self.params.boundary_condition)
            
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
                
                # Print conservation properties
                current_conservation = self.compute_conservation_properties()
                mass_error = abs(current_conservation['total_mass'] - 
                               initial_conservation['total_mass']) / initial_conservation['total_mass']
                energy_error = abs(current_conservation['total_energy'] - 
                                 initial_conservation['total_energy']) / initial_conservation['total_energy']
                print(f"  Conservation errors - Mass: {mass_error:.2e}, Energy: {energy_error:.2e}")
                
                # Print shock position
                shock_info = self.detect_shock_position()
                print(f"  Shock radius: {shock_info['shock_radius']:.4f}")
                
                next_output_time += self.params.output_interval
        
        print(f"Blast wave simulation completed!")
        print(f"  Final time: {current_time:.4f}")
        print(f"  Total steps: {step}")
        
        # Final conservation check
        final_conservation = self.compute_conservation_properties()
        mass_error = abs(final_conservation['total_mass'] - 
                        initial_conservation['total_mass']) / initial_conservation['total_mass']
        energy_error = abs(final_conservation['total_energy'] - 
                          initial_conservation['total_energy']) / initial_conservation['total_energy']
        print(f"  Final conservation errors - Mass: {mass_error:.2e}, Energy: {energy_error:.2e}")
        
        return {
            'final_time': current_time,
            'total_steps': step,
            'conservation_errors': {
                'mass': mass_error,
                'energy': energy_error
            },
            'snapshots': self.snapshots,
            'output_times': self.output_times
        }