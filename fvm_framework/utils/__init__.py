"""
Utility modules for FVM Framework

This module provides utility functions for profiling, visualization,
and performance analysis.
"""

from .profiling import PerformanceProfiler, benchmark_solver

__all__ = [
    'PerformanceProfiler',
    'benchmark_solver'
]