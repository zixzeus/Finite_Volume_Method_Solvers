"""
2D Advection Equation Test Suite

This module contains standard test cases for 2D linear advection equations,
testing pure transport without diffusion.

Advection equation: ∂u/∂t + a∂u/∂x + b∂u/∂y = 0
"""

import numpy as np
from typing import Callable

def gaussian_pulse_advection_2d() -> Callable:
    """
    2D Gaussian pulse for pure advection.
    
    The pulse should maintain its shape and translate.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Gaussian parameters
        x_c, y_c = 0.25, 0.25  # Start near corner
        sigma = 0.05
        amplitude = 1.0
        
        r_sq = (x - x_c)**2 + (y - y_c)**2
        u = amplitude * np.exp(-r_sq / (2 * sigma**2))
        
        return np.array([u])
    
    return initial_condition

def square_wave_advection_2d() -> Callable:
    """
    2D square wave for advection.
    
    Tests preservation of discontinuities and numerical diffusion.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if 0.2 <= x <= 0.4 and 0.2 <= y <= 0.4:
            u = 1.0
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

def sine_wave_advection_2d() -> Callable:
    """
    2D sine wave for advection.
    
    Smooth solution for accuracy testing.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        u = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
        
        return np.array([u])
    
    return initial_condition

def cosine_hill_2d() -> Callable:
    """
    2D cosine hill for advection.
    
    Smooth compact support function.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        x_c, y_c = 0.25, 0.5
        radius = 0.15
        
        r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
        
        if r <= radius:
            u = 0.25 * (1 + np.cos(np.pi * r / radius))
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

def slotted_cylinder_2d() -> Callable:
    """
    2D slotted cylinder for advection.
    
    Classic test case with sharp features.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        x_c, y_c = 0.5, 0.75
        radius = 0.15
        slot_width = 0.05
        
        r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
        
        # Cylinder with slot
        if r <= radius:
            if abs(x - x_c) <= slot_width and y <= y_c:
                u = 0.0  # Slot
            else:
                u = 1.0  # Cylinder
        else:
            u = 0.0  # Background
        
        return np.array([u])
    
    return initial_condition

def rotating_pulse_2d() -> Callable:
    """
    2D pulse for solid body rotation test.
    
    Tests diagonal advection and grid alignment.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        x_c, y_c = 0.5, 0.5
        radius = 0.15
        
        r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
        
        if r <= radius:
            u = 1.0 - r / radius
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

def diagonal_wave_2d() -> Callable:
    """
    2D diagonal wave for advection.
    
    Tests grid-aligned vs diagonal transport.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Diagonal wave pattern
        u = np.sin(2 * np.pi * (x + y))
        
        return np.array([u])
    
    return initial_condition

def multi_scale_waves_2d() -> Callable:
    """
    2D multi-scale wave pattern.
    
    Tests resolution of different wavelengths.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Superposition of different scales
        u = (0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) +
             0.3 * np.sin(8 * np.pi * x) * np.cos(4 * np.pi * y) +
             0.2 * np.sin(16 * np.pi * x) * np.cos(16 * np.pi * y))
        
        return np.array([u])
    
    return initial_condition

def vortex_advection_2d() -> Callable:
    """
    2D vortex in velocity field.
    
    Tests complex velocity field transport.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        x_c, y_c = 0.5, 0.5
        
        # Distance from center
        r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
        
        # Vortex profile
        if r <= 0.2:
            u = 1.0 * np.exp(-25 * r**2)
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

def step_function_2d() -> Callable:
    """
    2D step function for advection.
    
    Tests shock capturing and numerical diffusion.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x < 0.5:
            u = 1.0
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

def corner_transport_2d() -> Callable:
    """
    2D corner transport test.
    
    Scalar quantity starting in one corner.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        if x <= 0.1 and y <= 0.1:
            u = 1.0
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

def cross_pattern_2d() -> Callable:
    """
    2D cross pattern for advection.
    
    Tests transport of complex shapes.
    
    Returns:
        Function that returns initial state at any (x, y)
    """
    def initial_condition(x: float, y: float) -> np.ndarray:
        width = 0.05
        
        # Horizontal and vertical strips
        if (abs(x - 0.5) <= width and 0.3 <= y <= 0.7) or \
           (abs(y - 0.5) <= width and 0.3 <= x <= 0.7):
            u = 1.0
        else:
            u = 0.0
        
        return np.array([u])
    
    return initial_condition

# Test case registry
ADVECTION_TEST_CASES = {
    'gaussian_pulse': gaussian_pulse_advection_2d,
    'square_wave': square_wave_advection_2d,
    'sine_wave': sine_wave_advection_2d,
    'cosine_hill': cosine_hill_2d,
    'slotted_cylinder': slotted_cylinder_2d,
    'rotating_pulse': rotating_pulse_2d,
    'diagonal_wave': diagonal_wave_2d,
    'multi_scale_waves': multi_scale_waves_2d,
    'vortex_advection': vortex_advection_2d,
    'step_function': step_function_2d,
    'corner_transport': corner_transport_2d,
    'cross_pattern': cross_pattern_2d,
}

def get_advection_test_case(name: str) -> Callable:
    """
    Get a specific advection test case by name.
    
    Args:
        name: Test case name
        
    Returns:
        Initial condition function
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in ADVECTION_TEST_CASES:
        available = ', '.join(ADVECTION_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return ADVECTION_TEST_CASES[name]()

def list_advection_test_cases() -> list:
    """List all available advection test cases."""
    return list(ADVECTION_TEST_CASES.keys())