"""
2D MHD Equation Test Suite

This module contains standard test cases for 2D magnetohydrodynamics equations.

Conservative variables: [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
where ρ is density, u,v,w are velocity components, E is total energy, 
and Bx,By,Bz are magnetic field components.
"""

import numpy as np

def orszag_tang_vortex_2d(nx, ny):
    """
    2D Orszag-Tang vortex problem.
    
    Classic MHD turbulence test case.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 5.0/3.0
    
    # Normalize to [0, 2π]
    x_norm = 2 * np.pi * X
    y_norm = 2 * np.pi * Y
    
    # Primitive variables
    rho = gamma**2 * np.ones_like(X)
    u = -np.sin(y_norm)
    v = np.sin(x_norm)
    w = np.zeros_like(X)
    p = gamma * np.ones_like(X)
    
    # Magnetic field
    Bx = -np.sin(y_norm) / np.sqrt(4 * np.pi)
    By = np.sin(2 * x_norm) / np.sqrt(4 * np.pi)
    Bz = np.zeros_like(X)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    
    # Total energy: E = p/(γ-1) + 0.5*ρ*(u² + v² + w²) + 0.5*(Bx² + By² + Bz²)
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def magnetic_reconnection_2d(nx, ny):
    """
    2D magnetic reconnection problem.
    
    Tests reconnection physics with X-point configuration.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 5.0/3.0
    
    # Background state
    rho = np.ones_like(X)
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    w = np.zeros_like(X)
    p = 0.5 * np.ones_like(X)
    
    # Harris current sheet magnetic field configuration
    # Bx varies with y, By = 0 initially
    Bx = np.tanh(Y / 0.1)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    
    # Add small perturbation to trigger reconnection
    psi_0 = 0.01  # Perturbation amplitude
    Bx += -psi_0 * np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    By += psi_0 * np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def mhd_rotor_2d(nx, ny):
    """
    2D MHD rotor problem.
    
    Dense rotating cylinder in magnetic field.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Rotor parameters
    x_c, y_c = 0.5, 0.5  # Center
    r0 = 0.1  # Rotor radius
    r_taper = 0.115  # Taper radius
    
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    
    # Background state
    rho_ambient = 1.0
    p_ambient = 1.0
    
    # Rotor density (10x ambient)
    rho = np.where(r <= r0, 10.0, 
           np.where(r <= r_taper, 
                   1.0 + 9.0 * (r_taper - r) / (r_taper - r0), 
                   rho_ambient))
    
    # Rotor velocity (angular velocity ω = 2)
    omega = 2.0
    u = np.where(r <= r0, -omega * (Y - y_c), 0.0)
    v = np.where(r <= r0, omega * (X - x_c), 0.0)
    w = np.zeros_like(X)
    
    # Pressure (uniform)
    p = p_ambient * np.ones_like(X)
    
    # Uniform magnetic field
    B0 = 1.0 / np.sqrt(4 * np.pi)
    Bx = B0 * np.ones_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def mhd_blast_wave_2d(nx, ny):
    """
    2D MHD blast wave problem.
    
    Spherical blast wave in magnetized medium.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    r = np.sqrt(X**2 + Y**2)
    
    # Inner high pressure, outer low pressure
    rho = np.ones_like(X)
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    w = np.zeros_like(X)
    
    # Pressure profile
    p_inner = 100.0
    p_outer = 1.0
    p = np.where(r < 0.1, p_inner, p_outer)
    
    # Uniform magnetic field
    B0 = 1.0 / np.sqrt(2.0)
    Bx = B0 * np.ones_like(X)
    By = B0 * np.ones_like(X)
    Bz = np.zeros_like(X)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def kelvin_helmholtz_mhd_2d(nx, ny):
    """
    2D MHD Kelvin-Helmholtz instability.
    
    Shear layer with magnetic field.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 5.0/3.0
    
    # Density stratification
    rho_heavy, rho_light = 2.0, 1.0
    rho = np.where(np.abs(Y) < 0.25, rho_heavy, rho_light)
    
    # Velocity shear
    u = np.where(np.abs(Y) < 0.25, -0.5, 0.5)
    w = np.zeros_like(X)
    
    # Add perturbation
    amplitude = 0.01
    v = amplitude * np.sin(4 * np.pi * X) * np.exp(-Y**2 / 0.01)
    
    # Pressure
    p = 2.5 * np.ones_like(X)
    
    # Magnetic field (longitudinal)
    Bx = 0.5 * np.ones_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def magnetic_flux_tube_2d(nx, ny):
    """
    2D magnetic flux tube problem.
    
    Tests magnetic pressure and tension forces.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 5.0/3.0
    
    r = np.sqrt(X**2 + Y**2)
    
    # Background state
    rho = np.ones_like(X)
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    w = np.zeros_like(X)
    p = np.ones_like(X)
    
    # Flux tube magnetic field
    B0 = 1.0
    flux_radius = 0.3
    
    # Magnetic field inside and outside flux tube
    Bz = np.where(r <= flux_radius, B0, 0.0)
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    
    # Add small velocity perturbation
    amplitude = 0.01
    u = amplitude * np.sin(np.pi * X) * np.sin(np.pi * Y)
    v = amplitude * np.cos(np.pi * X) * np.cos(np.pi * Y)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def current_sheet_2d(nx, ny):
    """
    2D current sheet problem.
    
    Harris current sheet equilibrium.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 5.0/3.0
    
    # Harris sheet parameters
    L = 0.5  # Sheet thickness
    B0 = 1.0
    
    # Background state
    rho = np.ones_like(X)
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    w = np.zeros_like(X)
    
    # Magnetic field (reverses across sheet)
    Bx = B0 * np.tanh(Y / L)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    
    # Pressure balance: p + B²/2 = const
    p = 0.5 + 0.5 * B0**2 * (1 - np.tanh(Y / L)**2)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

def bow_shock_2d(nx, ny):
    """
    2D bow shock problem.
    
    Supersonic flow past obstacle in magnetic field.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
    """
    x = np.linspace(0, 4, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 5.0/3.0
    
    # Upstream conditions (supersonic flow)
    rho_up = 1.0
    u_up = 2.0  # Supersonic in x-direction
    v_up = 0.0
    w_up = 0.0
    p_up = 1.0
    
    # Magnetic field
    B0 = 1.0
    Bx = B0 * np.ones_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    
    # Uniform upstream state
    rho = rho_up * np.ones_like(X)
    u = u_up * np.ones_like(X)
    v = v_up * np.ones_like(X)
    w = w_up * np.ones_like(X)
    p = p_up * np.ones_like(X)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = (p / (gamma - 1.0) + 
         0.5 * rho * (u**2 + v**2 + w**2) + 
         0.5 * (Bx**2 + By**2 + Bz**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E, Bx, By, Bz])

# Test case registry
MHD_TEST_CASES = {
    'orszag_tang_vortex': orszag_tang_vortex_2d,
    'magnetic_reconnection': magnetic_reconnection_2d,
    'mhd_rotor': mhd_rotor_2d,
    'mhd_blast_wave': mhd_blast_wave_2d,
    'kelvin_helmholtz_mhd': kelvin_helmholtz_mhd_2d,
    'magnetic_flux_tube': magnetic_flux_tube_2d,
    'current_sheet': current_sheet_2d,
    'bow_shock': bow_shock_2d,
}

def get_mhd_test_case(name: str, nx: int, ny: int) -> np.ndarray:
    """
    Get a specific MHD test case by name.
    
    Args:
        name: Test case name
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (8, nx, ny) for [ρ, ρu, ρv, ρw, E, Bx, By, Bz]
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in MHD_TEST_CASES:
        available = ', '.join(MHD_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return MHD_TEST_CASES[name](nx, ny)

def list_mhd_test_cases() -> list:
    """List all available MHD test cases."""
    return list(MHD_TEST_CASES.keys())