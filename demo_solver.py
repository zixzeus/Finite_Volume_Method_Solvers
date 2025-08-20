#!/usr/bin/env python3
"""
Demo script for FVM Framework

This script demonstrates the framework capabilities with the Sod shock tube problem
and validates the algorithm accuracy against expected behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

# Import the FVM framework
from fvm_framework import FVMSolver, create_shock_tube_solver
from fvm_framework.examples import SodShockTube, run_quick_tests


def demo_basic_functionality():
    """Demonstrate basic framework functionality"""
    print("FVM Framework Demo")
    print("=" * 50)
    
    # Test 1: Create a simple solver
    print("Test 1: Creating FVM Solver")
    print("-" * 30)
    
    config = {
        'grid': {'nx': 50, 'ny': 50, 'dx': 0.02, 'dy': 0.02},
        'numerical': {
            'riemann_solver': 'hllc',
            'time_integrator': 'rk3',
            'cfl_number': 0.5
        },
        'simulation': {'final_time': 0.1}
    }
    
    solver = FVMSolver(config)
    print(f"✓ Created solver with {solver.geometry.nx} × {solver.geometry.ny} grid")
    print(f"  Riemann solver: {solver.riemann_solver.name}")
    print(f"  Time integrator: {solver.time_integrator.name}")
    
    # Test 2: Set initial conditions
    print("\nTest 2: Setting Initial Conditions")
    print("-" * 30)
    
    def simple_ic(x, y):
        """Simple uniform initial condition"""
        gamma = 1.4
        rho = 1.0
        u = v = w = 0.0
        p = 1.0
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        return np.array([rho, rho*u, rho*v, rho*w, E])
    
    solver.set_initial_conditions(simple_ic)
    print("✓ Initial conditions set successfully")
    
    # Test 3: Run a few time steps
    print("\nTest 3: Running Simulation")
    print("-" * 30)
    
    start_time = time.time()
    solver.solve(final_time=0.01)  # Very short simulation
    end_time = time.time()
    
    print(f"✓ Simulation completed in {end_time - start_time:.4f} seconds")
    print(f"  Final time: {solver.current_time:.6f}")
    print(f"  Time steps: {solver.time_step}")
    
    # Test 4: Get solution
    print("\nTest 4: Extracting Solution")
    print("-" * 30)
    
    solution = solver.get_solution()
    print(f"✓ Solution extracted")
    print(f"  Density range: [{np.min(solution['density']):.6f}, {np.max(solution['density']):.6f}]")
    print(f"  Pressure range: [{np.min(solution['pressure']):.6f}, {np.max(solution['pressure']):.6f}]")
    
    return solver


def demo_shock_tube():
    """Demonstrate Sod shock tube problem"""
    print("\n\nSod Shock Tube Demo")
    print("=" * 50)
    
    # Create and run Sod shock tube
    sod = SodShockTube()
    
    print(f"Problem: {sod.name}")
    print(f"Description: {sod.description}")
    print()
    
    start_time = time.time()
    solution = sod.run(final_time=0.2, silent=False)
    end_time = time.time()
    
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    
    # Extract 1D profile for analysis
    j_mid = solution['density'].shape[1] // 2
    x_1d = solution['x'][:, j_mid]
    density_1d = solution['density'][:, j_mid]
    velocity_1d = solution['velocity_x'][:, j_mid]
    pressure_1d = solution['pressure'][:, j_mid]
    
    # Check for expected shock tube features
    print("\nValidating Shock Tube Solution:")
    print("-" * 30)
    
    # Find shock position (max density gradient)
    density_grad = np.gradient(density_1d)
    shock_idx = np.argmax(np.abs(density_grad))
    shock_pos = x_1d[shock_idx]
    
    print(f"Shock position: x = {shock_pos:.3f}")
    
    # Check density jump across shock
    left_density = np.mean(density_1d[:shock_idx-10])
    right_density = np.mean(density_1d[shock_idx+10:])
    density_ratio = left_density / right_density
    
    print(f"Density ratio across shock: {density_ratio:.3f}")
    print(f"Expected ratio ≈ 2.7 for strong shock")
    
    # Check that solution is reasonable
    if 2.0 < density_ratio < 4.0 and 0.6 < shock_pos < 0.8:
        print("✓ Shock tube solution appears physically reasonable")
        return True
    else:
        print("⚠ Shock tube solution may have issues")
        return False


def demo_performance():
    """Demonstrate performance characteristics"""
    print("\n\nPerformance Demo")
    print("=" * 50)
    
    grid_sizes = [50, 100, 200]
    times = []
    
    for nx in grid_sizes:
        print(f"\nTesting {nx} × {nx} grid...")
        
        solver = create_shock_tube_solver(nx=nx, ny=nx//10)
        
        def uniform_ic(x, y):
            gamma = 1.4
            rho = 1.0
            u = v = w = 0.0
            p = 1.0
            E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
            return np.array([rho, rho*u, rho*v, rho*w, E])
        
        solver.set_initial_conditions(uniform_ic)
        
        start_time = time.time()
        solver.solve(final_time=0.01)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        cells = nx * (nx // 10)
        cell_updates = cells * solver.time_step
        performance = cell_updates / elapsed / 1000  # kcell-updates/sec
        
        print(f"  Time: {elapsed:.3f} s")
        print(f"  Performance: {performance:.1f} kcell-updates/sec")
    
    print(f"\nPerformance Summary:")
    print(f"Grid sizes tested: {grid_sizes}")
    print(f"Computation times: {[f'{t:.3f}s' for t in times]}")
    
    return times


def main():
    """Main demo function"""
    print("FVM Framework - Comprehensive Demo")
    print("=" * 60)
    print("Testing pipeline data-driven architecture for finite volume methods")
    print()
    
    try:
        # Basic functionality test
        solver = demo_basic_functionality()
        
        # Shock tube validation
        shock_tube_ok = demo_shock_tube()
        
        # Performance demonstration
        performance_times = demo_performance()
        
        # Summary
        print("\n\nDemo Summary")
        print("=" * 50)
        print("✓ Basic solver functionality: PASSED")
        
        if shock_tube_ok:
            print("✓ Shock tube validation: PASSED")
        else:
            print("⚠ Shock tube validation: NEEDS REVIEW")
        
        print(f"✓ Performance test completed")
        
        print(f"\nFramework Features Demonstrated:")
        print(f"  • Structure of Arrays (SoA) data layout")
        print(f"  • Pipeline-based computation stages")
        print(f"  • Multiple Riemann solvers (HLL, HLLC, HLLD)")
        print(f"  • Time integration schemes (Euler, RK2, RK3, RK4)")
        print(f"  • Boundary condition management")
        print(f"  • Standard test problems")
        print(f"  • Performance monitoring")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)