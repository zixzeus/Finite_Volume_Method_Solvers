#!/usr/bin/env python3
"""
Test runner for FVM Framework

This script provides a convenient way to run the test suite with different options.
"""

import sys
import os
import argparse
import unittest
import time

from tests import run_all_tests, run_performance_tests, run_basic_tests


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Run FVM Framework tests')
    parser.add_argument('--type', '-t', 
                       choices=['all', 'basic', 'performance'], 
                       default='all',
                       help='Type of tests to run')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q',
                       action='store_true', 
                       help='Quiet output')
    parser.add_argument('--module', '-m',
                       help='Run specific test module')
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    print("FVM Framework Test Suite")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Test type: {args.type}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        if args.module:
            # Run specific module
            loader = unittest.TestLoader()
            module_name = f'tests.{args.module}'
            module = __import__(module_name, fromlist=[''])
            suite = loader.loadTestsFromModule(module)
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
        else:
            # Run test type
            if args.type == 'basic':
                result = run_basic_tests()
            elif args.type == 'performance':
                result = run_performance_tests()
            else:
                result = run_all_tests(verbosity)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("-" * 50)
        print(f"Tests completed in {elapsed:.2f} seconds")
        
        if result.wasSuccessful():
            print("✓ All tests passed!")
            return 0
        else:
            print(f"✗ {len(result.failures)} test(s) failed")
            print(f"✗ {len(result.errors)} error(s) occurred")
            return 1
            
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())