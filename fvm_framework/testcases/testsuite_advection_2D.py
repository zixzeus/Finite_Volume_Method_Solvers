"""
2D Advection Equation Test Suite

This module contains standard test cases for 2D linear advection equations,
testing pure transport without diffusion.

Advection equation: ∂u/∂t + a∂u/∂x + b∂u/∂y = 0
"""

import numpy as np

def gaussian_pulse_advection_2d(nx, ny):
    """
    2D Gaussian pulse for pure advection.
    
    The pulse should maintain its shape and translate.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Gaussian parameters
    x_c, y_c = 0.5, 0.5  # Start near corner
    sigma = 0.05
    amplitude = 1.0
    
    r_sq = (X - x_c)**2 + (Y - y_c)**2
    u = amplitude * np.exp(-r_sq / (2 * sigma**2))
    
    return np.array([u])

def square_wave_advection_2d(nx, ny):
    """
    2D square wave for advection.
    
    Tests preservation of discontinuities and numerical diffusion.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.where((X >= 0.4) & (X <= 0.6) & (Y >= 0.4) & (Y <= 0.6), 1.0, 0.0)
    
    return np.array([u])

def sine_wave_advection_2d(nx, ny):
    """
    2D sine wave for advection.
    
    Smooth solution for accuracy testing.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    return np.array([u])

def cosine_hill_2d(nx, ny):
    """
    2D cosine hill for advection.
    
    Smooth compact support function.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.25, 0.5
    radius = 0.15
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    u = np.where(r <= radius, 0.25 * (1 + np.cos(np.pi * r / radius)), 0.0)
    
    return np.array([u])

def slotted_cylinder_2d(nx, ny):
    """
    2D slotted cylinder for advection.
    
    Classic test case with sharp features.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.75
    radius = 0.15
    slot_width = 0.05
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    
    # Cylinder with slot
    cylinder = r <= radius
    slot = (np.abs(X - x_c) <= slot_width) & (Y <= y_c)
    u = np.where(cylinder & ~slot, 1.0, 0.0)
    
    return np.array([u])

def rotating_pulse_2d(nx, ny):
    """
    2D pulse for solid body rotation test.
    
    Tests diagonal advection and grid alignment.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5
    radius = 0.15
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    u = np.where(r <= radius, 1.0 - r / radius, 0.0)
    
    return np.array([u])

def diagonal_wave_2d(nx, ny):
    """
    2D diagonal wave for advection.
    
    Tests grid-aligned vs diagonal transport.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.sin(2 * np.pi * (X + Y))
    
    return np.array([u])

def multi_scale_waves_2d(nx, ny):
    """
    2D multi-scale wave pattern.
    
    Tests resolution of different wavelengths.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = (0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
         0.3 * np.sin(8 * np.pi * X) * np.cos(4 * np.pi * Y) +
         0.2 * np.sin(16 * np.pi * X) * np.cos(16 * np.pi * Y))
    
    return np.array([u])

def vortex_advection_2d(nx, ny):
    """
    2D vortex in velocity field.
    
    Tests complex velocity field transport.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    
    u = np.where(r <= 0.2, 1.0 * np.exp(-25 * r**2), 0.0)
    
    return np.array([u])

def step_function_2d(nx, ny):
    """
    2D step function for advection.
    
    Tests shock capturing and numerical diffusion.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.where(X < 0.5, 1.0, 0.0)
    
    return np.array([u])

def corner_transport_2d(nx, ny):
    """
    2D corner transport test.
    
    Scalar quantity starting in one corner.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.where((X <= 0.1) & (Y <= 0.1), 1.0, 0.0)
    
    return np.array([u])

def cross_pattern_2d(nx, ny):
    """
    2D cross pattern for advection.
    
    Tests transport of complex shapes.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    width = 0.05
    
    horizontal = (np.abs(X - 0.5) <= width) & (Y >= 0.3) & (Y <= 0.7)
    vertical = (np.abs(Y - 0.5) <= width) & (X >= 0.3) & (X <= 0.7)
    u = np.where(horizontal | vertical, 1.0, 0.0)
    
    return np.array([u])

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

def get_advection_test_case(name: str, nx: int, ny: int) -> np.ndarray:
    """
    Get a specific advection test case by name.
    
    Args:
        name: Test case name
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (1, nx, ny)
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in ADVECTION_TEST_CASES:
        available = ', '.join(ADVECTION_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return ADVECTION_TEST_CASES[name](nx, ny)

def list_advection_test_cases() -> list:
    """List all available advection test cases."""
    return list(ADVECTION_TEST_CASES.keys())


if __name__ == "__main__":
    test = gaussian_pulse_advection_2d(5,5)
    print(test)