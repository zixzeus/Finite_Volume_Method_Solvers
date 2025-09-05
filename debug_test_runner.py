#!/usr/bin/env python3
"""
Debug Test Runner

This script allows VS Code to debug test files with proper relative imports.
It converts file paths to module paths and runs them as modules.
"""

import sys
import os
import runpy
from pathlib import Path

def run_test_file(test_file_path):
    """Run a test file as a module to support relative imports"""
    
    # Get the absolute path
    test_file = Path(test_file_path).resolve()
    
    # Get the project root
    project_root = Path(__file__).parent.resolve()
    
    # Make sure the test file is within the project
    try:
        relative_path = test_file.relative_to(project_root)
    except ValueError:
        print(f"Error: Test file {test_file} is not within project {project_root}")
        return 1
    
    # Convert file path to module path
    # e.g., fvm_framework/tests/test_boundary_conditions.py -> fvm_framework.tests.test_boundary_conditions
    module_parts = list(relative_path.parts[:-1])  # Remove .py extension
    module_parts.append(relative_path.stem)  # Add filename without extension
    module_name = '.'.join(module_parts)
    
    print(f"Running test file as module: {module_name}")
    print(f"File path: {test_file}")
    print(f"Working directory: {project_root}")
    print("=" * 60)
    
    # Set up Python path
    sys.path.insert(0, str(project_root))
    
    # Change to project root directory
    os.chdir(project_root)
    
    try:
        # Check if this is a unittest test file
        if 'test' in module_name.lower() and module_name.endswith(('test_boundary_conditions', 'test_data_container', 'test_pipeline')):
            # Run as unittest module
            import unittest
            
            # Import the test module
            test_module = __import__(module_name, fromlist=[''])
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            return 0 if result.wasSuccessful() else 1
        else:
            # Run as regular module
            runpy.run_module(module_name, run_name='__main__')
            return 0
            
    except Exception as e:
        print(f"Error running test module {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_test_runner.py <test_file_path>")
        print("Example: python debug_test_runner.py fvm_framework/tests/test_boundary_conditions.py")
        return 1
    
    test_file_path = sys.argv[1]
    return run_test_file(test_file_path)

if __name__ == "__main__":
    sys.exit(main())