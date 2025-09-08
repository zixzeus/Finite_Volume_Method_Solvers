"""
2D Sound Wave Equation Test Suite

Standard test cases for 2D linear sound wave equations.
"""

import numpy as np
from typing import Callable

def gaussian_acoustic_pulse() -> Callable:
    """Gaussian pressure pulse."""
    def initial_condition(x: float, y: float) -> np.ndarray:
        x_c, y_c = 0.5, 0.5
        sigma = 0.1
        amplitude = 1.0
        
        r_sq = (x - x_c)**2 + (y - y_c)**2
        p = amplitude * np.exp(-r_sq / (2 * sigma**2))
        u = v = 0.0
        
        return np.array([p, u, v])
    
    return initial_condition

def plane_wave_2d() -> Callable:
    """2D plane wave."""
    def initial_condition(x: float, y: float) -> np.ndarray:
        k_x, k_y = 1.0, 0.5
        amplitude = 1.0
        
        k_mag = np.sqrt(k_x**2 + k_y**2)
        phi = k_x * x + k_y * y
        
        p = amplitude * np.cos(phi)
        u = (amplitude * k_x / k_mag) * np.sin(phi) if k_mag > 1e-15 else 0.0
        v = (amplitude * k_y / k_mag) * np.sin(phi) if k_mag > 1e-15 else 0.0
        
        return np.array([p, u, v])
    
    return initial_condition

def standing_wave_2d() -> Callable:
    """2D standing wave pattern."""
    def initial_condition(x: float, y: float) -> np.ndarray:
        mode_x, mode_y = 1, 1
        amplitude = 1.0
        
        p = amplitude * np.sin(mode_x * np.pi * x) * np.sin(mode_y * np.pi * y)
        u = v = 0.0
        
        return np.array([p, u, v])
    
    return initial_condition

# Test case registry
SOUND_WAVE_TEST_CASES = {
    'gaussian_pulse': gaussian_acoustic_pulse,
    'plane_wave': plane_wave_2d,
    'standing_wave': standing_wave_2d,
}

def get_sound_wave_test_case(name: str) -> Callable:
    if name not in SOUND_WAVE_TEST_CASES:
        available = ', '.join(SOUND_WAVE_TEST_CASES.keys())
        raise KeyError(f"Test case '{name}' not found. Available: {available}")
    return SOUND_WAVE_TEST_CASES[name]()

def list_sound_wave_test_cases() -> list:
    return list(SOUND_WAVE_TEST_CASES.keys())