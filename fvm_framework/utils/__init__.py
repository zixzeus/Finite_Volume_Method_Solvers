"""
Utility modules for FVM Framework

This module provides utility functions for profiling, visualization,
and performance analysis.
"""

from .profiling import PerformanceProfiler, benchmark_solver
from .plotting import FVMPlotter, create_physics_specific_plotter

__all__ = [
    'PerformanceProfiler',
    'benchmark_solver',
    'FVMPlotter', 
    'create_physics_specific_plotter'
]