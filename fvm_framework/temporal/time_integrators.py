"""
Time Integration Schemes for Finite Volume Method

This module implements various explicit time integration methods including
forward Euler, Runge-Kutta schemes (RK2, RK3, RK4), and adaptive time stepping.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod
from core.data_container import FVMDataContainer2D


class TimeIntegrator(ABC):
    """Abstract base class for time integration schemes"""
    
    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order
    
    @abstractmethod
    def integrate(self, data: FVMDataContainer2D, dt: float, 
                 residual_function: Callable, **kwargs) -> None:
        """
        Advance solution by one time step.
        
        Args:
            data: FVM data container
            dt: Time step size
            residual_function: Function that computes spatial residual
            **kwargs: Additional parameters
        """
        pass
    
    def compute_cfl_timestep(self, data: FVMDataContainer2D, cfl_number: float,
                           gamma: float = 1.4) -> float:
        """
        Compute time step based on CFL condition.
        
        Args:
            data: FVM data container
            cfl_number: CFL number (typically 0.1-0.9)
            gamma: Heat capacity ratio
            
        Returns:
            Maximum stable time step
        """
        max_wave_speed = data.get_max_wave_speed(gamma)
        
        # Minimum grid spacing
        min_dx = min(data.geometry.dx, data.geometry.dy)
        
        # CFL condition: dt ≤ CFL * min(dx, dy) / max_wave_speed
        if max_wave_speed > 1e-15:
            dt_cfl = cfl_number * min_dx / max_wave_speed
        else:
            dt_cfl = 1e-6  # Fallback for very low speeds
        
        return dt_cfl


class ForwardEuler(TimeIntegrator):
    """
    Forward Euler time integration scheme.
    
    First-order accurate explicit method:
    U^{n+1} = U^n + dt * R(U^n)
    """
    
    def __init__(self):
        super().__init__("ForwardEuler", order=1)
    
    def integrate(self, data: FVMDataContainer2D, dt: float,
                 residual_function: Callable, **kwargs) -> None:
        """Forward Euler integration step"""
        
        # Compute spatial residual
        residual = residual_function(data, **kwargs)
        
        # Update solution: U^{n+1} = U^n + dt * R(U^n)
        data.state_new[:] = data.state + dt * residual
        
        # Swap state arrays
        data.swap_states()


class RungeKutta2(TimeIntegrator):
    """
    Second-order Runge-Kutta scheme (RK2).
    
    Also known as the midpoint method:
    k1 = R(U^n)
    k2 = R(U^n + dt/2 * k1)
    U^{n+1} = U^n + dt * k2
    """
    
    def __init__(self):
        super().__init__("RungeKutta2", order=2)
    
    def integrate(self, data: FVMDataContainer2D, dt: float,
                 residual_function: Callable, **kwargs) -> None:
        """RK2 integration step"""
        
        # Store initial state
        u0 = data.state.copy()
        
        # Stage 1: k1 = R(U^n)
        k1 = residual_function(data, **kwargs)
        
        # Intermediate update: U_temp = U^n + dt/2 * k1
        data.state[:] = u0 + 0.5 * dt * k1
        data._primitives_valid = False
        
        # Apply boundary conditions for intermediate state
        data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
        
        # Stage 2: k2 = R(U_temp)
        k2 = residual_function(data, **kwargs)
        
        # Final update: U^{n+1} = U^n + dt * k2
        data.state[:] = u0 + dt * k2
        data._primitives_valid = False


class RungeKutta3(TimeIntegrator):
    """
    Third-order Runge-Kutta scheme (RK3).
    
    TVD RK3 scheme (Shu-Osher form):
    U^{(1)} = U^n + dt * R(U^n)
    U^{(2)} = 3/4 * U^n + 1/4 * (U^{(1)} + dt * R(U^{(1)}))
    U^{n+1} = 1/3 * U^n + 2/3 * (U^{(2)} + dt * R(U^{(2)}))
    """
    
    def __init__(self):
        super().__init__("RungeKutta3", order=3)
    
    def integrate(self, data: FVMDataContainer2D, dt: float,
                 residual_function: Callable, **kwargs) -> None:
        """RK3 integration step"""
        
        # Store initial state
        u0 = data.state.copy()
        
        # Stage 1: U^{(1)} = U^n + dt * R(U^n)
        k1 = residual_function(data, **kwargs)
        data.state[:] = u0 + dt * k1
        data._primitives_valid = False
        data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
        
        # Stage 2: U^{(2)} = 3/4 * U^n + 1/4 * (U^{(1)} + dt * R(U^{(1)}))
        k2 = residual_function(data, **kwargs)
        data.state[:] = 0.75 * u0 + 0.25 * (data.state + dt * k2)
        data._primitives_valid = False
        data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
        
        # Stage 3: U^{n+1} = 1/3 * U^n + 2/3 * (U^{(2)} + dt * R(U^{(2)}))
        k3 = residual_function(data, **kwargs)
        data.state[:] = (1.0/3.0) * u0 + (2.0/3.0) * (data.state + dt * k3)
        data._primitives_valid = False


class RungeKutta4(TimeIntegrator):
    """
    Fourth-order Runge-Kutta scheme (RK4).
    
    Classical RK4 method:
    k1 = R(U^n)
    k2 = R(U^n + dt/2 * k1)
    k3 = R(U^n + dt/2 * k2)
    k4 = R(U^n + dt * k3)
    U^{n+1} = U^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def __init__(self):
        super().__init__("RungeKutta4", order=4)
    
    def integrate(self, data: FVMDataContainer2D, dt: float,
                 residual_function: Callable, **kwargs) -> None:
        """RK4 integration step"""
        
        # Store initial state
        u0 = data.state.copy()
        
        # Stage 1: k1 = R(U^n)
        k1 = residual_function(data, **kwargs)
        
        # Stage 2: k2 = R(U^n + dt/2 * k1)
        data.state[:] = u0 + 0.5 * dt * k1
        data._primitives_valid = False
        data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
        k2 = residual_function(data, **kwargs)
        
        # Stage 3: k3 = R(U^n + dt/2 * k2)
        data.state[:] = u0 + 0.5 * dt * k2
        data._primitives_valid = False
        data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
        k3 = residual_function(data, **kwargs)
        
        # Stage 4: k4 = R(U^n + dt * k3)
        data.state[:] = u0 + dt * k3
        data._primitives_valid = False
        data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
        k4 = residual_function(data, **kwargs)
        
        # Final update: U^{n+1} = U^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        data.state[:] = u0 + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        data._primitives_valid = False


class AdaptiveTimestepper(TimeIntegrator):
    """
    Adaptive time stepping with embedded Runge-Kutta methods.
    
    Uses embedded RK methods to estimate error and adapt time step.
    Currently implements RK45 (Dormand-Prince).
    """
    
    def __init__(self, tolerance: float = 1e-6, max_dt_change: float = 2.0,
                 min_dt_change: float = 0.1, safety_factor: float = 0.8):
        super().__init__("AdaptiveRK45", order=5)
        self.tolerance = tolerance
        self.max_dt_change = max_dt_change
        self.min_dt_change = min_dt_change
        self.safety_factor = safety_factor
        
        # Dormand-Prince coefficients
        self.a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
        
        # 5th order weights
        self.b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        
        # 4th order weights  
        self.b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    def integrate(self, data: FVMDataContainer2D, dt: float,
                 residual_function: Callable, **kwargs) -> Tuple[float, bool]:
        """
        Adaptive integration step with error control.
        
        Returns:
            Tuple of (suggested_next_dt, step_accepted)
        """
        
        # Store initial state
        u0 = data.state.copy()
        
        # Compute stages
        k = np.zeros((7, *data.state.shape))
        
        # k1
        k[0] = residual_function(data, **kwargs)
        
        # k2 through k7
        for stage in range(1, 7):
            data.state[:] = u0
            for j in range(stage):
                data.state[:] += dt * self.a[stage, j] * k[j]
            
            data._primitives_valid = False
            data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
            k[stage] = residual_function(data, **kwargs)
        
        # Compute 5th and 4th order solutions
        u5 = u0.copy()
        u4 = u0.copy()
        
        for j in range(7):
            u5 += dt * self.b5[j] * k[j]
            u4 += dt * self.b4[j] * k[j]
        
        # Estimate error
        error = np.max(np.abs(u5 - u4))
        
        # Check if step is acceptable
        step_accepted = error <= self.tolerance
        
        if step_accepted:
            # Accept step
            data.state[:] = u5
            data._primitives_valid = False
        else:
            # Reject step - restore original state
            data.state[:] = u0
        
        # Compute next time step
        if error > 1e-15:
            dt_scale = self.safety_factor * (self.tolerance / error)**(1.0/5.0)
        else:
            dt_scale = self.max_dt_change
        
        # Limit time step changes
        dt_scale = max(self.min_dt_change, min(self.max_dt_change, dt_scale))
        next_dt = dt * dt_scale
        
        return next_dt, step_accepted


class TimeIntegratorFactory:
    """Factory for creating time integrators"""
    
    _integrators = {
        'euler': ForwardEuler,
        'rk2': RungeKutta2,
        'rk3': RungeKutta3,
        'rk4': RungeKutta4,
        'adaptive': AdaptiveTimestepper
    }
    
    @classmethod
    def create(cls, integrator_type: str, **kwargs) -> TimeIntegrator:
        """
        Create a time integrator of specified type.
        
        Args:
            integrator_type: Type of integrator
            **kwargs: Additional parameters for integrator
            
        Returns:
            Time integrator instance
        """
        integrator_type = integrator_type.lower()
        if integrator_type not in cls._integrators:
            raise ValueError(f"Unknown integrator type: {integrator_type}")
        
        return cls._integrators[integrator_type](**kwargs)
    
    @classmethod
    def available_integrators(cls) -> list:
        """Get list of available integrator types"""
        return list(cls._integrators.keys())


class ResidualFunction:
    """
    Residual function for time integration.
    
    Computes spatial residual R(U) = -∇·F(U) + S(U)
    where F is the flux and S is the source term.
    """
    
    def __init__(self, spatial_scheme, source_function: Optional[Callable] = None):
        self.spatial_scheme = spatial_scheme
        self.source_function = source_function
    
    def __call__(self, data: FVMDataContainer2D, **kwargs) -> np.ndarray:
        """
        Compute spatial residual.
        
        Args:
            data: FVM data container
            **kwargs: Additional parameters
            
        Returns:
            Residual array with same shape as state
        """
        # Compute fluxes
        self.spatial_scheme.compute_fluxes(data, **kwargs)
        
        # Compute flux divergence
        residual = self._compute_flux_divergence(data)
        
        # Add source terms if present
        if self.source_function is not None:
            source = self.source_function(data, **kwargs)
            residual += source
        
        return residual
    
    def _compute_flux_divergence(self, data: FVMDataContainer2D) -> np.ndarray:
        """Compute flux divergence using finite volume formula"""
        residual = np.zeros_like(data.state)
        
        # X-direction flux differences
        for i in range(data.nx):
            for j in range(data.ny):
                # Flux difference: (F_{i+1/2} - F_{i-1/2}) / dx
                flux_diff_x = (data.flux_x[:, i+1, j] - data.flux_x[:, i, j]) / data.geometry.dx
                residual[:, i, j] -= flux_diff_x
        
        # Y-direction flux differences
        for i in range(data.nx):
            for j in range(data.ny):
                # Flux difference: (G_{j+1/2} - G_{j-1/2}) / dy
                flux_diff_y = (data.flux_y[:, i, j+1] - data.flux_y[:, i, j]) / data.geometry.dy
                residual[:, i, j] -= flux_diff_y
        
        return residual


class TemporalSolver:
    """
    High-level temporal solver that coordinates time integration.
    
    Manages time stepping, CFL condition, and solution advancement.
    """
    
    def __init__(self, integrator: TimeIntegrator, residual_function: ResidualFunction,
                 cfl_number: float = 0.5, adaptive_dt: bool = True):
        self.integrator = integrator
        self.residual_function = residual_function
        self.cfl_number = cfl_number
        self.adaptive_dt = adaptive_dt
        
        # Simulation state
        self.current_time = 0.0
        self.time_step = 0
        self.current_dt = 1e-6
        
    def solve_to_time(self, data: FVMDataContainer2D, final_time: float, **kwargs):
        """
        Solve from current time to final time.
        
        Args:
            data: FVM data container
            final_time: Target simulation time
            **kwargs: Additional parameters
        """
        while self.current_time < final_time:
            # Compute time step
            if self.adaptive_dt:
                dt_cfl = self.integrator.compute_cfl_timestep(data, self.cfl_number,
                                                            kwargs.get('gamma', 1.4))
                self.current_dt = min(dt_cfl, final_time - self.current_time)
            
            # Take time step
            if isinstance(self.integrator, AdaptiveTimestepper):
                # Adaptive integrator handles its own time step adjustment
                next_dt, step_accepted = self.integrator.integrate(
                    data, self.current_dt, self.residual_function, **kwargs
                )
                
                if step_accepted:
                    self.current_time += self.current_dt
                    self.time_step += 1
                
                self.current_dt = next_dt
                
            else:
                # Fixed time step integrator
                self.integrator.integrate(data, self.current_dt, self.residual_function, **kwargs)
                self.current_time += self.current_dt
                self.time_step += 1
            
            # Apply boundary conditions
            data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
    
    def solve_n_steps(self, data: FVMDataContainer2D, n_steps: int, **kwargs):
        """
        Solve for specified number of time steps.
        
        Args:
            data: FVM data container
            n_steps: Number of time steps
            **kwargs: Additional parameters
        """
        for _ in range(n_steps):
            # Compute time step
            if self.adaptive_dt:
                dt_cfl = self.integrator.compute_cfl_timestep(data, self.cfl_number,
                                                            kwargs.get('gamma', 1.4))
                self.current_dt = dt_cfl
            
            # Take time step
            self.integrator.integrate(data, self.current_dt, self.residual_function, **kwargs)
            self.current_time += self.current_dt
            self.time_step += 1
            
            # Apply boundary conditions
            data.apply_boundary_conditions(kwargs.get('boundary_type', 'periodic'))
    
    def get_status(self) -> dict:
        """Get current solver status"""
        return {
            'current_time': self.current_time,
            'time_step': self.time_step,
            'current_dt': self.current_dt,
            'integrator': self.integrator.name,
            'order': self.integrator.order
        }