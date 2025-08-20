"""
Examples module for FVM Framework

This module contains standard test problems and demonstration examples
for the finite volume method framework.
"""

from .standard_problems import (
    TestProblem, SodShockTube, CircularBlastWave, KelvinHelmholtzInstability,
    RayleighTaylorInstability, DoubleMachReflection, TestSuite,
    run_validation_suite, run_quick_tests
)

__all__ = [
    'TestProblem',
    'SodShockTube',
    'CircularBlastWave', 
    'KelvinHelmholtzInstability',
    'RayleighTaylorInstability',
    'DoubleMachReflection',
    'TestSuite',
    'run_validation_suite',
    'run_quick_tests'
]