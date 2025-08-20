"""
Test suite for FVM Framework

This module contains comprehensive unit tests for the finite volume method framework,
including data containers, pipeline stages, boundary conditions, and integration tests.
"""

import unittest
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import all test modules
from .test_data_container import *
from .test_pipeline import *
from .test_boundary_conditions import *


def run_all_tests(verbosity=2):
    """
    Run all unit tests with specified verbosity.
    
    Args:
        verbosity: Test output verbosity level (0=quiet, 1=normal, 2=verbose)
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_modules = [
        'test_data_container',
        'test_pipeline', 
        'test_boundary_conditions'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(f'{__name__}.{module_name}', fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    return result


def run_performance_tests():
    """Run performance-related tests only"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add performance test cases
    from .test_data_container import TestDataContainerPerformance
    from .test_pipeline import TestPipelineIntegration
    
    suite.addTests(loader.loadTestsFromTestCase(TestDataContainerPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_basic_tests():
    """Run basic functionality tests only"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add basic test cases
    from .test_data_container import TestGridGeometry, TestFVMDataContainer2D
    from .test_pipeline import TestComputationStage, TestBoundaryStage
    from .test_boundary_conditions import TestBasicBoundaryConditions
    
    suite.addTests(loader.loadTestsFromTestCase(TestGridGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestFVMDataContainer2D))
    suite.addTests(loader.loadTestsFromTestCase(TestComputationStage))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryStage))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicBoundaryConditions))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    print("FVM Framework Test Suite")
    print("=" * 50)
    
    # Check if specific test type requested
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == 'performance':
            result = run_performance_tests()
        elif test_type == 'basic':
            result = run_basic_tests()
        else:
            result = run_all_tests()
    else:
        result = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)