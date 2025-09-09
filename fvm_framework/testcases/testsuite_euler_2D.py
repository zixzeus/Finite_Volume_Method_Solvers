"""
2D Euler Equation Test Suite

This module contains standard test cases for 2D Euler equations,
testing compressible gas dynamics problems.

Conservative variables: [ρ, ρu, ρv, ρw, E]
where ρ is density, u,v,w are velocity components, and E is total energy.
"""

import numpy as np

def sod_shock_tube_2d(nx, ny):
    """
    2D Sod shock tube problem.
    
    Classic 1D Riemann problem extended to 2D.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Left and right states
    rho_left, u_left, v_left, w_left, p_left = 1.0, 0.0, 0.0, 0.0, 1.0
    rho_right, u_right, v_right, w_right, p_right = 0.125, 0.0, 0.0, 0.0, 0.1
    
    # Conservative variables
    rho = np.where(X < 0.5, rho_left, rho_right)
    rho_u = np.where(X < 0.5, rho_left * u_left, rho_right * u_right)
    rho_v = np.where(X < 0.5, rho_left * v_left, rho_right * v_right)
    rho_w = np.where(X < 0.5, rho_left * w_left, rho_right * w_right)
    
    # Total energy: E = p/(γ-1) + 0.5*ρ*(u² + v² + w²)
    E = np.where(X < 0.5, 
                 p_left / (gamma - 1.0) + 0.5 * rho_left * (u_left**2 + v_left**2 + w_left**2),
                 p_right / (gamma - 1.0) + 0.5 * rho_right * (u_right**2 + v_right**2 + w_right**2))
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def blast_wave_2d(nx, ny):
    """
    2D circular blast wave problem.
    
    High pressure region in center, low pressure outside.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    r = np.sqrt(X**2 + Y**2)
    
    # Inner high pressure, outer low pressure
    rho_inner, p_inner = 1.0, 10.0
    rho_outer, p_outer = 0.125, 0.1
    u = v = w = 0.0  # Initially at rest
    
    rho = np.where(r < 0.1, rho_inner, rho_outer)
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    
    # Total energy
    p = np.where(r < 0.1, p_inner, p_outer)
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def four_quadrant_riemann_2d(nx, ny):
    """
    2D four quadrant Riemann problem.
    
    Different states in each quadrant for testing 2D interactions.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Four quadrant states [rho, u, v, w, p]
    states = {
        'bl': [1.0, 0.0, 0.0, 0.0, 1.0],    # Bottom left
        'br': [0.5, 0.7, 0.0, 0.0, 1.0],    # Bottom right
        'tl': [0.8, 0.0, 0.7, 0.0, 1.0],    # Top left
        'tr': [0.3, 0.7, 0.7, 0.0, 1.0],    # Top right
    }
    
    # Initialize arrays
    rho = np.zeros_like(X)
    rho_u = np.zeros_like(X)
    rho_v = np.zeros_like(X)
    rho_w = np.zeros_like(X)
    E = np.zeros_like(X)
    
    # Fill quadrants
    for state_name, (rho_val, u_val, v_val, w_val, p_val) in states.items():
        if state_name == 'bl':
            mask = (X <= 0.5) & (Y <= 0.5)
        elif state_name == 'br':
            mask = (X > 0.5) & (Y <= 0.5)
        elif state_name == 'tl':
            mask = (X <= 0.5) & (Y > 0.5)
        else:  # tr
            mask = (X > 0.5) & (Y > 0.5)
        
        rho[mask] = rho_val
        rho_u[mask] = rho_val * u_val
        rho_v[mask] = rho_val * v_val
        rho_w[mask] = rho_val * w_val
        E[mask] = p_val / (gamma - 1.0) + 0.5 * rho_val * (u_val**2 + v_val**2 + w_val**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def double_mach_reflection_2d(nx, ny):
    """
    2D double Mach reflection problem.
    
    Strong shock hitting an angled wedge.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(0, 4, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Pre-shock and post-shock states
    rho_pre, u_pre, v_pre, w_pre, p_pre = 1.4, 0.0, 0.0, 0.0, 1.0
    rho_post, u_post, v_post, w_post, p_post = 8.0, 8.25, 0.0, 0.0, 116.5
    
    # Shock initially at x = 1/6, with wedge starting at (x=1/6, y=0)
    wedge_start = 1.0/6.0
    
    # Pre-shock everywhere initially
    rho = np.full_like(X, rho_pre)
    rho_u = np.full_like(X, rho_pre * u_pre)
    rho_v = np.full_like(X, rho_pre * v_pre)
    rho_w = np.full_like(X, rho_pre * w_pre)
    E = np.full_like(X, p_pre / (gamma - 1.0) + 0.5 * rho_pre * (u_pre**2 + v_pre**2 + w_pre**2))
    
    # Post-shock region (left of initial shock position)
    post_shock_mask = X < wedge_start
    rho[post_shock_mask] = rho_post
    rho_u[post_shock_mask] = rho_post * u_post
    rho_v[post_shock_mask] = rho_post * v_post
    rho_w[post_shock_mask] = rho_post * w_post
    E[post_shock_mask] = p_post / (gamma - 1.0) + 0.5 * rho_post * (u_post**2 + v_post**2 + w_post**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def kelvin_helmholtz_instability_2d(nx, ny):
    """
    2D Kelvin-Helmholtz instability.
    
    Shear layer instability test case.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Base flow with shear
    rho = np.ones_like(X)
    w = np.zeros_like(X)
    p = 2.5 * np.ones_like(X)
    
    # Velocity profile with shear layer
    u = np.where(Y < 0.5, -0.5, 0.5)
    
    # Add perturbation to trigger instability
    amplitude = 0.01
    v = amplitude * np.sin(4 * np.pi * X) * np.exp(-100 * (Y - 0.5)**2)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def rayleigh_taylor_instability_2d(nx, ny):
    """
    2D Rayleigh-Taylor instability.
    
    Gravitational instability with density stratification.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Heavy fluid on top, light fluid on bottom (unstable)
    rho_heavy, rho_light = 2.0, 1.0
    p_base = 2.5
    
    rho = np.where(Y > 1.0, rho_heavy, rho_light)
    
    # Hydrostatic pressure (simplified)
    p = p_base - 0.1 * Y  # Linear pressure gradient
    
    # Small perturbation at interface
    u = np.zeros_like(X)
    w = np.zeros_like(X)
    amplitude = 0.01
    v = amplitude * np.sin(2 * np.pi * X) * np.exp(-10 * (Y - 1.0)**2)
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def vortex_preservation_2d(nx, ny):
    """
    2D isentropic vortex preservation.
    
    Tests accuracy and vortex preservation properties.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Vortex parameters
    beta = 5.0  # Vortex strength
    r2 = X**2 + Y**2
    
    # Isentropic vortex solution
    rho_inf = 1.0
    u_inf = 1.0
    v_inf = 1.0
    p_inf = 1.0
    
    # Perturbations
    du = -(beta / (2 * np.pi)) * Y * np.exp((1 - r2) / 2)
    dv = (beta / (2 * np.pi)) * X * np.exp((1 - r2) / 2)
    dT = -(gamma - 1) * beta**2 / (8 * gamma * np.pi**2) * np.exp(1 - r2)
    
    # Flow variables
    u = u_inf + du
    v = v_inf + dv
    w = np.zeros_like(X)
    T = 1.0 + dT  # Temperature
    
    # Density and pressure from isentropic relations
    rho = rho_inf * T**(1 / (gamma - 1))
    p = p_inf * T**(gamma / (gamma - 1))
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

def shock_vortex_interaction_2d(nx, ny):
    """
    2D shock-vortex interaction.
    
    Moving shock interacting with a vortex.
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
    """
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    gamma = 1.4
    
    # Shock at x = 0.5
    shock_pos = 0.5
    
    # Pre-shock state
    rho_pre = 1.0
    u_pre = 0.0
    v_pre = 0.0
    w_pre = 0.0
    p_pre = 1.0
    
    # Post-shock state (Mach 1.3 shock)
    rho_post = 1.516
    u_post = 0.206
    v_post = 0.0
    w_post = 0.0
    p_post = 1.667
    
    # Base flow
    rho = np.where(X < shock_pos, rho_post, rho_pre)
    u = np.where(X < shock_pos, u_post, u_pre)
    v = np.where(X < shock_pos, v_post, v_pre)
    w = np.where(X < shock_pos, w_post, w_pre)
    p = np.where(X < shock_pos, p_post, p_pre)
    
    # Add vortex in pre-shock region
    vortex_center_x, vortex_center_y = 1.0, 0.5
    vortex_strength = 0.1
    vortex_radius = 0.1
    
    r_vortex = np.sqrt((X - vortex_center_x)**2 + (Y - vortex_center_y)**2)
    vortex_mask = (X > shock_pos) & (r_vortex < 0.3)
    
    # Add vortex perturbations
    theta = np.arctan2(Y - vortex_center_y, X - vortex_center_x)
    vortex_factor = vortex_strength * np.exp(-r_vortex**2 / vortex_radius**2)
    
    u[vortex_mask] += -vortex_factor[vortex_mask] * np.sin(theta[vortex_mask])
    v[vortex_mask] += vortex_factor[vortex_mask] * np.cos(theta[vortex_mask])
    
    # Conservative variables
    rho_u = rho * u
    rho_v = rho * v
    rho_w = rho * w
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    return np.array([rho, rho_u, rho_v, rho_w, E])

# Test case registry
EULER_TEST_CASES = {
    'sod_shock_tube': sod_shock_tube_2d,
    'blast_wave': blast_wave_2d,
    'four_quadrant_riemann': four_quadrant_riemann_2d,
    'double_mach_reflection': double_mach_reflection_2d,
    'kelvin_helmholtz_instability': kelvin_helmholtz_instability_2d,
    'rayleigh_taylor_instability': rayleigh_taylor_instability_2d,
    'vortex_preservation': vortex_preservation_2d,
    'shock_vortex_interaction': shock_vortex_interaction_2d,
}

def get_euler_test_case(name: str, nx: int, ny: int) -> np.ndarray:
    """
    Get a specific Euler test case by name.
    
    Args:
        name: Test case name
        nx, ny: Grid dimensions
        
    Returns:
        State array with shape (5, nx, ny) for [ρ, ρu, ρv, ρw, E]
        
    Raises:
        KeyError: If test case name not found
    """
    if name not in EULER_TEST_CASES:
        available = ', '.join(EULER_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    
    return EULER_TEST_CASES[name](nx, ny)

def list_euler_test_cases() -> list:
    """List all available Euler test cases."""
    return list(EULER_TEST_CASES.keys())