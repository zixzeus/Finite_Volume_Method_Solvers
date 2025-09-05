#!/usr/bin/env python3
"""
Simple test script for the unified spatial discretization factory.

This script tests that all spatial discretization schemes can be created
without needing the full framework setup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fvm_framework.spatial.factory import SpatialDiscretizationFactory

def test_factory_creation():
    """Test that all spatial discretization schemes can be created"""
    print("Testing Spatial Discretization Factory...")
    print("=" * 50)
    
    # Test schemes that should work
    test_schemes = [
        ('lax_friedrichs', {}),
        ('tvdlf', {'limiter': 'minmod'}),
        ('hll', {}),
        ('hllc', {}),
        ('dg', {'polynomial_order': 1, 'riemann_solver': 'hllc'}),
        ('dg_p0', {}),
        ('dg_p1', {}),
    ]
    
    successful = 0
    total = len(test_schemes)
    
    for scheme_name, params in test_schemes:
        try:
            scheme = SpatialDiscretizationFactory.create(scheme_name, **params)
            print(f"✓ {scheme_name}: {scheme.name} (order {scheme.order})")
            successful += 1
        except Exception as e:
            print(f"✗ {scheme_name}: Failed - {e}")
    
    print(f"\nFactory test results: {successful}/{total} schemes created successfully")
    return successful == total

def test_scheme_info():
    """Test the scheme information system"""
    print("\n" + "=" * 50)
    print("Available Spatial Discretization Schemes:")
    print("=" * 50)
    
    try:
        SpatialDiscretizationFactory.print_available_schemes()
        return True
    except Exception as e:
        print(f"Error printing scheme info: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Unified Spatial Discretization Architecture")
    print("=" * 60)
    
    tests = [
        ("Factory Creation", test_factory_creation),
        ("Scheme Information", test_scheme_info),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                print(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)