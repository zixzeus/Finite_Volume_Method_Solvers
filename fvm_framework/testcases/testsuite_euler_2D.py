"""
2D Euler Equation Test Suite

This module contains standard test cases for 2D Euler equations,
adapted from classic computational fluid dynamics problems.

Based on the 1D testsuite pattern, extended to 2D scenarios.
"""

import numpy as np
from typing import Callable, Tuple

# Global constants
gamma = 1.4  # Heat capacity ratio for ideal gas

def sod_shock_tube_2d(N: int = None) -> Callable:
    """
    2D extension of Sod shock tube problem.
    
    Classic Riemann problem with shock, contact discontinuity, and rarefaction.
    Initial condition: high pressure left, low pressure right.
    
    Args:
        N: Number of grid points (if specified, returns numpy arrays)
        
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x < 0.5:  # Left state (high pressure)
            rho = 1.0
            u, v, w = 0.0, 0.0, 0.0
            p = 1.0
        else:  # Right state (low pressure)
            rho = 0.125
            u, v, w = 0.0, 0.0, 0.0
            p = 0.1
        
        # Compute total energy
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def lax_shock_tube_2d(N: int = None) -> Callable:
    """
    2D Lax shock tube problem.
    
    Another classic Riemann problem with different initial states.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x < 0.5:  # Left state
            rho = 0.445
            u, v, w = 0.698, 0.0, 0.0
            p = 3.528
        else:  # Right state
            rho = 0.5
            u, v, w = 0.0, 0.0, 0.0
            p = 0.571
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def blast_wave_2d(N: int = None) -> Callable:
    """
    2D blast wave problem.
    
    High pressure region in center, low pressure surroundings.
    Creates circular shock wave propagation.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Distance from center
        r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        
        if r < 0.1:  # Inner high pressure region
            rho = 1.0
            u, v, w = 0.0, 0.0, 0.0
            p = 10.0
        else:  # Outer low pressure region
            rho = 0.125
            u, v, w = 0.0, 0.0, 0.0
            p = 0.1
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def isentropic_vortex_2d(N: int = None) -> Callable:
    """
    2D isentropic vortex problem.
    
    Smooth solution for testing accuracy of numerical schemes.
    The vortex should propagate without distortion.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Vortex parameters
        beta = 5.0  # Vortex strength
        x_c, y_c = 0.5, 0.5  # Vortex center
        
        # Distance from vortex center
        r_sq = (x - x_c)**2 + (y - y_c)**2
        
        # Base state
        rho_inf = 1.0
        u_inf, v_inf = 1.0, 0.5
        p_inf = 1.0
        
        # Vortex perturbation
        f = beta / (2 * np.pi) * np.exp(0.5 * (1 - r_sq))
        
        # Velocity perturbation
        delta_u = -f * (y - y_c)
        delta_v = f * (x - x_c)
        
        # Temperature perturbation
        delta_T = -(gamma - 1) * f**2 / (2 * gamma)
        
        # Final state
        T = 1.0 + delta_T  # Assuming T_inf = 1
        rho = rho_inf * T**(1.0 / (gamma - 1))
        u = u_inf + delta_u
        v = v_inf + delta_v
        w = 0.0
        p = p_inf * T**(gamma / (gamma - 1))
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def riemann_2d_config1(N: int = None) -> Callable:
    """
    2D Riemann problem configuration 1.
    
    Four-quadrant initial condition creating complex wave interactions.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x > 0.5 and y > 0.5:  # Quadrant I
            rho, u, v, w, p = 0.5313, 0.0, 0.0, 0.0, 0.4
        elif x < 0.5 and y > 0.5:  # Quadrant II
            rho, u, v, w, p = 1.0, 0.7276, 0.0, 0.0, 1.0
        elif x < 0.5 and y < 0.5:  # Quadrant III
            rho, u, v, w, p = 0.8, 0.0, 0.0, 0.0, 1.0
        else:  # Quadrant IV
            rho, u, v, w, p = 1.0, 0.0, 0.7276, 0.0, 1.0
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def riemann_2d_config3(N: int = None) -> Callable:
    """
    2D Riemann problem configuration 3.
    
    Another four-quadrant problem with different wave patterns.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x > 0.5 and y > 0.5:  # Quadrant I
            rho, u, v, w, p = 1.0, -0.75, -0.5, 0.0, 1.0
        elif x < 0.5 and y > 0.5:  # Quadrant II
            rho, u, v, w, p = 2.0, -0.75, 0.5, 0.0, 1.0
        elif x < 0.5 and y < 0.5:  # Quadrant III
            rho, u, v, w, p = 1.0, 0.75, 0.5, 0.0, 1.0
        else:  # Quadrant IV
            rho, u, v, w, p = 3.0, 0.75, -0.5, 0.0, 1.0
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def kelvin_helmholtz_setup(N: int = None) -> Callable:
    """
    Kelvin-Helmholtz instability setup.
    
    Shear layer with perturbation to trigger instability.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Shear layer parameters
        y_c = 0.5  # Layer center
        delta = 0.05  # Layer thickness
        
        # Perturbation
        epsilon = 0.01
        perturbation = epsilon * np.sin(4 * np.pi * x)
        
        if abs(y - y_c) < delta:  # Shear layer
            rho = 2.0
            u = 0.5 * np.tanh((y - y_c + perturbation) / delta)
            v = perturbation * np.cos(4 * np.pi * x)
            w = 0.0
            p = 2.5
        else:  # Background
            rho = 1.0
            u = 0.5 * np.sign(y - y_c)
            v = 0.0
            w = 0.0
            p = 2.5
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def double_mach_reflection(N: int = None) -> Callable:
    """
    Double Mach reflection problem.
    
    Strong shock hitting a wedge, creating complex reflection patterns.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Shock parameters
        x_s = 1.0/6.0 + y / np.sqrt(3.0)  # Shock position
        
        if x > x_s:  # Pre-shock state
            rho = 1.4
            u = 0.0
            v = 0.0
            w = 0.0
            p = 1.0
        else:  # Post-shock state
            rho = 8.0
            u = 8.25 * np.cos(np.pi/6.0)
            v = -8.25 * np.sin(np.pi/6.0)
            w = 0.0
            p = 116.5
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

def rayleigh_taylor_instability(N: int = None) -> Callable:
    """
    Rayleigh-Taylor instability setup.
    
    Heavy fluid over light fluid in gravitational field.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Interface parameters
        y_interface = 0.5
        epsilon = 0.01  # Perturbation amplitude
        
        # Sinusoidal perturbation
        perturbation = epsilon * np.cos(2 * np.pi * x)
        interface_pos = y_interface + perturbation
        
        if y > interface_pos:  # Upper fluid (heavy)
            rho = 2.0
            u, v, w = 0.0, 0.0, 0.0
            p = 2.5 - 2.0 * y  # Hydrostatic equilibrium
        else:  # Lower fluid (light)
            rho = 1.0
            u, v, w = 0.0, 0.0, 0.0
            p = 1.0 + 1.0 * (interface_pos - y)  # Hydrostatic equilibrium
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    return initial_condition

# Test case registry
EULER_TEST_CASES = {
    'sod_shock_tube': sod_shock_tube_2d,
    'lax_shock_tube': lax_shock_tube_2d,
    'blast_wave': blast_wave_2d,
    'isentropic_vortex': isentropic_vortex_2d,
    'riemann_config1': riemann_2d_config1,
    'riemann_config3': riemann_2d_config3,
    'kelvin_helmholtz': kelvin_helmholtz_setup,
    'double_mach_reflection': double_mach_reflection,
    'rayleigh_taylor': rayleigh_taylor_instability,
}

def get_euler_test_case(name: str) -> Callable:
    """
    Get a specific Euler test case by name.
    
    Args:
        name: Test case name
        
    Returns:
        Initial condition function
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in EULER_TEST_CASES:
        available = ', '.join(EULER_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return EULER_TEST_CASES[name]()

def list_euler_test_cases() -> list:
    """List all available Euler test cases."""
    return list(EULER_TEST_CASES.keys())