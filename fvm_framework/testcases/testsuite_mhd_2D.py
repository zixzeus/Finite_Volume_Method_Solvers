"""
2D MHD Equation Test Suite

Standard test cases for 2D magnetohydrodynamics equations.
"""

import numpy as np
from typing import Callable

gamma = 5.0/3.0  # Heat capacity ratio for MHD

def orszag_tang_vortex() -> Callable:
    """Orszag-Tang vortex - classic MHD turbulence test."""
    def initial_condition(x: float, y: float) -> np.ndarray:
        # Normalize to [0, 2π]
        x_norm = 2 * np.pi * x
        y_norm = 2 * np.pi * y
        
        rho = gamma**2
        u = -np.sin(y_norm)
        v = np.sin(x_norm)
        w = 0.0
        p = gamma
        
        B0 = 1.0 / np.sqrt(4 * np.pi)
        Bx = -B0 * np.sin(y_norm)
        By = B0 * np.sin(2 * x_norm)
        Bz = 0.0
        
        E = p/(gamma-1.0) + 0.5*rho*(u**2+v**2+w**2) + 0.5*(Bx**2+By**2+Bz**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E, Bx, By, Bz])
    
    return initial_condition

def harris_current_sheet() -> Callable:
    """Harris current sheet for magnetic reconnection."""
    def initial_condition(x: float, y: float) -> np.ndarray:
        thickness = 0.1
        B0 = 1.0
        
        Bx = B0 * np.tanh(y / thickness)
        By = 0.0
        Bz = 0.0
        
        rho = 1.0 + 0.1 / np.cosh(y / thickness)**2
        u = v = w = 0.0
        
        p = 0.5 * B0**2 - 0.5*(Bx**2+By**2+Bz**2) + 0.1
        p = max(p, 0.01)
        
        E = p/(gamma-1.0) + 0.5*rho*(u**2+v**2+w**2) + 0.5*(Bx**2+By**2+Bz**2)
        
        return np.array([rho, rho*u, rho*v, rho*w, E, Bx, By, Bz])
    
    return initial_condition

# Test case registry
MHD_TEST_CASES = {
    'orszag_tang': orszag_tang_vortex,
    'harris_sheet': harris_current_sheet,
}

def get_mhd_test_case(name: str) -> Callable:
    if name not in MHD_TEST_CASES:
        available = ', '.join(MHD_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    return MHD_TEST_CASES[name]()

def list_mhd_test_cases() -> list:
    return list(MHD_TEST_CASES.keys())