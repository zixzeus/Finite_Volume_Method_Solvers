"""
2D Burgers Equation Test Suite

This module contains standard test cases for 2D Burgers equations,
testing nonlinear wave propagation and shock formation.

Burgers equation: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
                  ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v
"""

import numpy as np

def smooth_sine_wave_2d(nx, ny):
    """
    2D smooth sinusoidal initial condition.
    
    Tests accuracy of numerical scheme before shock formation.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    v = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    return np.array([u, v])

def gaussian_vortex_2d(nx, ny):
    """
    2D Gaussian vortex for Burgers equation.
    
    Tests vortex dynamics and viscous effects.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5  # Vortex center
    sigma = 0.1
    amplitude = 1.0
    
    r_sq = (X - x_c)**2 + (Y - y_c)**2
    vortex_strength = amplitude * np.exp(-r_sq / (2 * sigma**2))
    
    # Velocity components for clockwise vortex
    u = -(Y - y_c) * vortex_strength
    v = (X - x_c) * vortex_strength
    
    return np.array([u, v])

def shock_formation_2d(nx, ny):
    """
    2D shock formation test case.
    
    Initial condition designed to form shocks quickly.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.where(X < 0.5, 1.0, -0.5)
    v = np.where(X < 0.5, 0.5, -1.0)
    
    return np.array([u, v])

def taylor_green_vortex_2d(nx, ny):
    """
    2D Taylor-Green vortex for Burgers equation.
    
    Classic vortex flow test case.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    v = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    return np.array([u, v])

def multi_vortex_2d(nx, ny):
    """
    2D multiple vortex interaction.
    
    Tests vortex merging and complex dynamics.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Multiple vortex centers
    vortices = [
        (0.3, 0.3, 1.0),   # (x_c, y_c, strength)
        (0.7, 0.3, -1.0),
        (0.5, 0.7, 1.5)
    ]
    sigma = 0.1
    
    u_total = np.zeros_like(X)
    v_total = np.zeros_like(Y)
    
    for x_c, y_c, strength in vortices:
        r_sq = (X - x_c)**2 + (Y - y_c)**2
        vortex_factor = strength * np.exp(-r_sq / (2 * sigma**2))
        
        # Add contribution from this vortex
        u_total += -(Y - y_c) * vortex_factor
        v_total += (X - x_c) * vortex_factor
    
    return np.array([u_total, v_total])

def burgers_riemann_2d(nx, ny):
    """
    2D Riemann problem for Burgers equation.
    
    Four quadrant configuration for testing discontinuous initial data.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Four quadrants with different values
    u = np.where((X <= 0.5) & (Y <= 0.5), 1.0,    # Bottom left
         np.where((X > 0.5) & (Y <= 0.5), -0.5,   # Bottom right
         np.where((X <= 0.5) & (Y > 0.5), 1.0,    # Top left
                  -0.5)))                          # Top right
    
    v = np.where((X <= 0.5) & (Y <= 0.5), 1.0,    # Bottom left
         np.where((X > 0.5) & (Y <= 0.5), 1.0,    # Bottom right
         np.where((X <= 0.5) & (Y > 0.5), -0.5,   # Top left
                  -0.5)))                          # Top right
    
    return np.array([u, v])

def viscous_shock_2d(nx, ny):
    """
    2D viscous shock layer test.
    
    Smooth transition that steepens into shock-like structure.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.tanh(10 * (X - 0.5))
    v = np.tanh(10 * (Y - 0.5))
    
    return np.array([u, v])

def diagonal_wave_2d(nx, ny):
    """
    2D diagonal wave propagation.
    
    Tests wave propagation in diagonal direction.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    wave = np.sin(2 * np.pi * (X + Y))
    u = wave
    v = wave
    
    return np.array([u, v])

# Test case registry
BURGERS_TEST_CASES = {
    'smooth_sine_wave': smooth_sine_wave_2d,
    'gaussian_vortex': gaussian_vortex_2d,
    'shock_formation': shock_formation_2d,
    'taylor_green_vortex': taylor_green_vortex_2d,
    'multi_vortex': multi_vortex_2d,
    'burgers_riemann': burgers_riemann_2d,
    'viscous_shock': viscous_shock_2d,
    'diagonal_wave': diagonal_wave_2d,
}

def get_burgers_test_case(name: str, nx: int, ny: int) -> np.ndarray:
    """
    Get a specific Burgers test case by name.
    
    Args:
        name: Test case name
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (2, nx, ny) for [u, v]
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in BURGERS_TEST_CASES:
        available = ', '.join(BURGERS_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return BURGERS_TEST_CASES[name](nx, ny)

def list_burgers_test_cases() -> list:
    """List all available Burgers test cases."""
    return list(BURGERS_TEST_CASES.keys())