"""
2D Sound Wave Equation Test Suite

Standard test cases for 2D linear sound wave equations.

Sound wave system: ∂p/∂t + c²(∂u/∂x + ∂v/∂y) = 0
                  ∂u/∂t + ∂p/∂x = 0
                  ∂v/∂t + ∂p/∂y = 0
"""

import numpy as np

def gaussian_acoustic_pulse_2d(nx, ny):
    """
    2D Gaussian acoustic pulse.
    
    Tests symmetric wave propagation from point source.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5
    sigma = 0.1
    amplitude = 1.0
    
    r_sq = (X - x_c)**2 + (Y - y_c)**2
    p = amplitude * np.exp(-r_sq / (2 * sigma**2))
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def plane_wave_2d(nx, ny):
    """
    2D plane wave propagation.
    
    Tests directional wave propagation at angle.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    k_x, k_y = 1.0, 0.5
    amplitude = 1.0
    
    k_mag = np.sqrt(k_x**2 + k_y**2)
    phi = k_x * X + k_y * Y
    
    p = amplitude * np.cos(phi)
    u = np.where(k_mag > 1e-15, (amplitude * k_x / k_mag) * np.sin(phi), 0.0)
    v = np.where(k_mag > 1e-15, (amplitude * k_y / k_mag) * np.sin(phi), 0.0)
    
    return np.array([p, u, v])

def standing_wave_2d(nx, ny):
    """
    2D standing wave pattern.
    
    Tests modal wave patterns with pressure nodes and antinodes.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    mode_x, mode_y = 1, 1
    amplitude = 1.0
    
    p = amplitude * np.sin(mode_x * np.pi * X) * np.sin(mode_y * np.pi * Y)
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def circular_wave_2d(nx, ny):
    """
    2D circular wave expansion.
    
    Tests radial wave propagation from center.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5
    amplitude = 1.0
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    
    # Cylindrical wave with r^(-1/2) decay (approximated by exponential for numerical stability)
    p = amplitude * np.exp(-r) * np.cos(10 * np.pi * r)
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def acoustic_dipole_2d(nx, ny):
    """
    2D acoustic dipole source.
    
    Tests directional acoustic emission pattern.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5
    amplitude = 1.0
    sigma = 0.1
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    theta = np.arctan2(Y - y_c, X - x_c)
    
    # Dipole pattern: cos(theta) dependence
    p = amplitude * np.cos(theta) * np.exp(-r**2 / (2 * sigma**2))
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def acoustic_quadrupole_2d(nx, ny):
    """
    2D acoustic quadrupole source.
    
    Tests complex directional emission with four lobes.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.5, 0.5
    amplitude = 1.0
    sigma = 0.1
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    theta = np.arctan2(Y - y_c, X - x_c)
    
    # Quadrupole pattern: cos(2*theta) dependence
    p = amplitude * np.cos(2 * theta) * np.exp(-r**2 / (2 * sigma**2))
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def wave_interference_2d(nx, ny):
    """
    2D wave interference pattern.
    
    Tests superposition of multiple wave sources.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Two sources creating interference
    x1, y1 = 0.3, 0.5
    x2, y2 = 0.7, 0.5
    amplitude = 1.0
    sigma = 0.1
    
    r1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
    r2 = np.sqrt((X - x2)**2 + (Y - y2)**2)
    
    p1 = amplitude * np.exp(-r1**2 / (2 * sigma**2))
    p2 = amplitude * np.exp(-r2**2 / (2 * sigma**2))
    p = p1 + p2
    
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def acoustic_cavity_2d(nx, ny):
    """
    2D acoustic cavity mode.
    
    Tests cavity resonance with boundary conditions.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Cavity mode (m,n) = (2,1)
    m, n = 2, 1
    amplitude = 1.0
    
    p = amplitude * np.cos(m * np.pi * X) * np.cos(n * np.pi * Y)
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def pressure_pulse_2d(nx, ny):
    """
    2D localized pressure pulse.
    
    Tests sharp pressure disturbance propagation.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_c, y_c = 0.3, 0.3
    width = 0.1
    amplitude = 1.0
    
    p = np.where((np.abs(X - x_c) <= width) & (np.abs(Y - y_c) <= width), amplitude, 0.0)
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

def sound_wave_riemann_2d(nx, ny):
    """
    2D sound wave Riemann problem.
    
    Four quadrant configuration for testing wave interactions.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Four quadrants with different pressure values
    p = np.where((X <= 0.5) & (Y <= 0.5), 1.0,    # Bottom left
         np.where((X > 0.5) & (Y <= 0.5), 0.1,    # Bottom right
         np.where((X <= 0.5) & (Y > 0.5), 0.1,    # Top left
                  0.01)))                          # Top right
    
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    return np.array([p, u, v])

# Test case registry
SOUND_WAVE_TEST_CASES = {
    'gaussian_pulse': gaussian_acoustic_pulse_2d,
    'plane_wave': plane_wave_2d,
    'standing_wave': standing_wave_2d,
    'circular_wave': circular_wave_2d,
    'acoustic_dipole': acoustic_dipole_2d,
    'acoustic_quadrupole': acoustic_quadrupole_2d,
    'wave_interference': wave_interference_2d,
    'acoustic_cavity': acoustic_cavity_2d,
    'pressure_pulse': pressure_pulse_2d,
    'sound_wave_riemann': sound_wave_riemann_2d,
}

def get_sound_wave_test_case(name: str, nx: int, ny: int) -> np.ndarray:
    """
    Get a specific sound wave test case by name.
    
    Args:
        name: Test case name
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (3, nx, ny) for [p, u, v]
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in SOUND_WAVE_TEST_CASES:
        available = ', '.join(SOUND_WAVE_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return SOUND_WAVE_TEST_CASES[name](nx, ny)

def list_sound_wave_test_cases() -> list:
    """List all available sound wave test cases."""
    return list(SOUND_WAVE_TEST_CASES.keys())