#!/usr/bin/env python3
"""
Test script for the new unified spatial discretization architecture.

This script verifies that all spatial discretization methods can be created
and used correctly with the new modular design.
"""

# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fvm_framework.spatial.factory import SpatialDiscretizationFactory
from fvm_framework.examples.unified_test_runner import create_test_cases, UnifiedTestRunner

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

def test_unified_runner():
    """Test the unified test runner with new architecture"""
    print("\n" + "=" * 50)
    print("Testing Unified Test Runner...")
    print("=" * 50)
    
    try:
        # Create and run one test case to verify integration
        test_cases = create_test_cases()
        
        if test_cases:
            # Run a simple test case
            test_config = test_cases[0]  # Euler with Lax-Friedrichs
            print(f"Running test case: {test_config.name}")
            
            runner = UnifiedTestRunner(test_config)
            result = runner.run_simulation(verbose=False)
            
            if result.get('success', False):
                print(f"✓ Test case completed successfully")
                print(f"  Steps: {result['total_steps']}")
                print(f"  Time: {result['computation_time']:.3f}s")
                print(f"  Final time: {result['final_time']:.6f}")
                return True
            else:
                print(f"✗ Test case failed")
                return False
        else:
            print("✗ No test cases available")
            return False
            
    except Exception as e:
        print(f"✗ Unified runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing New Unified Spatial Discretization Architecture")
    print("=" * 60)
    
    tests = [
        ("Factory Creation", test_factory_creation),
        ("Scheme Information", test_scheme_info),
        ("Unified Test Runner", test_unified_runner),
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
        print("🎉 All tests passed! New architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    main()
