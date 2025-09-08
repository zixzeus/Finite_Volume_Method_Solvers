"""
2D Burgers Equation Test Suite

This module contains standard test cases for 2D Burgers equations,
testing nonlinear wave propagation and shock formation.

Burgers equation: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
                  ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v
"""

import numpy as np
from typing import Callable, Tuple

def smooth_sine_wave_2d(N: int = None) -> Callable:
    """
    2D smooth sinusoidal initial condition.
    
    Tests accuracy of numerical scheme before shock formation.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        u = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        v = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        
        return np.array([u, v])
    
    return initial_condition

def gaussian_pulse_2d(N: int = None) -> Callable:
    """
    2D Gaussian pulse with circular symmetry.
    
    Tests radial wave propagation and steepening.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Gaussian parameters
        x_c, y_c = 0.5, 0.5  # Center
        sigma = 0.1  # Width
        amplitude = 1.0
        
        r_sq = (x - x_c)**2 + (y - y_c)**2
        u = amplitude * np.exp(-r_sq / (2 * sigma**2))
        v = 0.0  # Initially only u-component
        
        return np.array([u, v])
    
    return initial_condition

def shock_formation_2d(N: int = None) -> Callable:
    """
    2D shock formation test case.
    
    Initial smooth profile that develops into shock.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x < 0.3:
            u = 1.0
            v = 0.5
        elif x < 0.7:
            u = 0.5
            v = 0.0
        else:
            u = 0.0
            v = -0.5
        
        return np.array([u, v])
    
    return initial_condition

def rarefaction_wave_2d(N: int = None) -> Callable:
    """
    2D rarefaction wave test case.
    
    Initial condition leading to expansion wave.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x < 0.5:
            u = -0.5
            v = 0.0
        else:
            u = 0.5
            v = 0.0
        
        return np.array([u, v])
    
    return initial_condition

def vortex_burgers_2d(N: int = None) -> Callable:
    """
    2D vortical flow for Burgers equation.
    
    Tests interaction between convection and diffusion.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Vortex center
        x_c, y_c = 0.5, 0.5
        
        # Velocity components creating vortex
        u = -(y - y_c) * np.exp(-((x - x_c)**2 + (y - y_c)**2))
        v = (x - x_c) * np.exp(-((x - x_c)**2 + (y - y_c)**2))
        
        return np.array([u, v])
    
    return initial_condition

def corner_flow_2d(N: int = None) -> Callable:
    """
    2D corner flow configuration.
    
    Flow emanating from corner, testing boundary interactions.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Distance from corner
        r = np.sqrt(x**2 + y**2)
        
        if r > 1e-10:
            u = x / r * np.exp(-r)
            v = y / r * np.exp(-r)
        else:
            u = 0.0
            v = 0.0
        
        return np.array([u, v])
    
    return initial_condition

def shear_layer_2d(N: int = None) -> Callable:
    """
    2D shear layer configuration.
    
    Tests development of Kelvin-Helmholtz-like instabilities.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Shear layer parameters
        y_center = 0.5
        thickness = 0.1
        
        # Hyperbolic tangent profile
        u = np.tanh((y - y_center) / thickness)
        
        # Small perturbation
        epsilon = 0.01
        v = epsilon * np.sin(2 * np.pi * x)
        
        return np.array([u, v])
    
    return initial_condition

def converging_waves_2d(N: int = None) -> Callable:
    """
    2D converging waves test case.
    
    Multiple waves converging to create complex interactions.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Four-corner configuration
        if x < 0.5 and y < 0.5:  # Bottom-left
            u, v = 1.0, 1.0
        elif x > 0.5 and y < 0.5:  # Bottom-right
            u, v = -1.0, 1.0
        elif x < 0.5 and y > 0.5:  # Top-left
            u, v = 1.0, -1.0
        else:  # Top-right
            u, v = -1.0, -1.0
        
        return np.array([u, v])
    
    return initial_condition

def viscous_shock_2d(N: int = None) -> Callable:
    """
    2D viscous shock structure test case.
    
    Tests balance between convection and diffusion in shock.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Shock parameters
        shock_pos = 0.5
        shock_width = 0.05
        
        # Tanh shock profile
        u = 0.5 * (1.0 + np.tanh((x - shock_pos) / shock_width))
        v = 0.0
        
        return np.array([u, v])
    
    return initial_condition

def multi_shock_interaction_2d(N: int = None) -> Callable:
    """
    2D multi-shock interaction test case.
    
    Multiple shocks interacting in complex ways.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Create step functions in both directions
        if x < 0.3:
            u_base = 1.0
        elif x < 0.7:
            u_base = 0.0
        else:
            u_base = -1.0
        
        if y < 0.3:
            v_base = 1.0
        elif y < 0.7:
            v_base = 0.0
        else:
            v_base = -1.0
        
        u = u_base
        v = v_base
        
        return np.array([u, v])
    
    return initial_condition

# Test case registry
BURGERS_TEST_CASES = {
    'smooth_sine_wave': smooth_sine_wave_2d,
    'gaussian_pulse': gaussian_pulse_2d,
    'shock_formation': shock_formation_2d,
    'rarefaction_wave': rarefaction_wave_2d,
    'vortex_burgers': vortex_burgers_2d,
    'corner_flow': corner_flow_2d,
    'shear_layer': shear_layer_2d,
    'converging_waves': converging_waves_2d,
    'viscous_shock': viscous_shock_2d,
    'multi_shock_interaction': multi_shock_interaction_2d,
}

def get_burgers_test_case(name: str) -> Callable:
    """
    Get a specific Burgers test case by name.
    
    Args:
        name: Test case name
        
    Returns:
        Initial condition function
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in BURGERS_TEST_CASES:
        available = ', '.join(BURGERS_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return BURGERS_TEST_CASES[name]()

def list_burgers_test_cases() -> list:
    """List all available Burgers test cases."""
    return list(BURGERS_TEST_CASES.keys())