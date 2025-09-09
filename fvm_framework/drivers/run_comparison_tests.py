"""
Run All Comparison Tests

This script runs all physics equation comparison tests and generates
comprehensive comparison reports and visualizations.
"""

import sys
import os
import time
import argparse
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver_advection_comparison import AdvectionComparison, ComparisonParameters as AdvectionParams
from driver_euler_comparison import EulerComparison, EulerComparisonParameters as EulerParams
from driver_burgers_comparison import BurgersComparison, BurgersComparisonParameters as BurgersParams
from driver_mhd_comparison import MHDComparison, MHDComparisonParameters as MHDParams
from driver_sound_wave_comparison import SoundWaveComparison, SoundWaveComparisonParameters as SoundParams


def create_output_directory(base_dir: str = "comparison_results") -> str:
    """Create timestamped output directory"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_advection_tests(output_dir: str, quick: bool = False) -> Dict[str, Any]:
    """Run advection equation comparison tests"""
    print("\n" + "="*60)
    print("RUNNING ADVECTION EQUATION TESTS")
    print("="*60)
    
    if quick:
        test_cases = ['gaussian_pulse', 'sine_wave']
        nx, ny = 32, 32
        final_time = 0.2
    else:
        test_cases = ['gaussian_pulse', 'square_wave', 'sine_wave', 'cosine_hill']
        nx, ny = 64, 64
        final_time = 0.3
    
    params = AdvectionParams(
        nx=nx, ny=ny,
        final_time=final_time,
        test_cases=test_cases,
        output_dir=output_dir,
        save_plots=True,
        show_plots=False
    )
    
    comparison = AdvectionComparison(params)
    start_time = time.perf_counter()
    comparison.run_all_tests()
    end_time = time.perf_counter()
    
    return {
        'name': 'Advection',
        'total_time': end_time - start_time,
        'test_cases': len(test_cases),
        'methods': len(params.spatial_methods)
    }


def run_euler_tests(output_dir: str, quick: bool = False) -> Dict[str, Any]:
    """Run Euler equation comparison tests"""
    print("\n" + "="*60)
    print("RUNNING EULER EQUATION TESTS")
    print("="*60)
    
    if quick:
        test_cases = ['sod_shock_tube']
        nx, ny = 32, 32
        final_time = 0.1
    else:
        test_cases = ['sod_shock_tube', 'explosion_2d']
        nx, ny = 64, 64
        final_time = 0.15
    
    params = EulerParams(
        nx=nx, ny=ny,
        final_time=final_time,
        test_cases=test_cases,
        output_dir=output_dir,
        save_plots=True,
        show_plots=False
    )
    
    comparison = EulerComparison(params)
    start_time = time.perf_counter()
    comparison.run_all_tests()
    end_time = time.perf_counter()
    
    return {
        'name': 'Euler',
        'total_time': end_time - start_time,
        'test_cases': len(test_cases),
        'methods': len(params.spatial_methods)
    }


def run_burgers_tests(output_dir: str, quick: bool = False) -> Dict[str, Any]:
    """Run Burgers equation comparison tests"""
    print("\n" + "="*60)
    print("RUNNING BURGERS EQUATION TESTS")
    print("="*60)
    
    if quick:
        test_cases = ['smooth_sine_wave', 'gaussian_vortex']
        nx, ny = 32, 32
        final_time = 0.2
    else:
        test_cases = ['smooth_sine_wave', 'gaussian_vortex', 'taylor_green_vortex']
        nx, ny = 64, 64
        final_time = 0.3
    
    params = BurgersParams(
        nx=nx, ny=ny,
        final_time=final_time,
        test_cases=test_cases,
        output_dir=output_dir,
        save_plots=True,
        show_plots=False
    )
    
    comparison = BurgersComparison(params)
    start_time = time.perf_counter()
    comparison.run_all_tests()
    end_time = time.perf_counter()
    
    return {
        'name': 'Burgers',
        'total_time': end_time - start_time,
        'test_cases': len(test_cases),
        'methods': len(params.spatial_methods)
    }


def run_mhd_tests(output_dir: str, quick: bool = False) -> Dict[str, Any]:
    """Run MHD equation comparison tests"""
    print("\n" + "="*60)
    print("RUNNING MHD EQUATION TESTS")
    print("="*60)
    
    if quick:
        test_cases = ['current_sheet']
        nx, ny = 32, 32
        final_time = 0.05
    else:
        test_cases = ['orszag_tang_vortex', 'current_sheet']
        nx, ny = 64, 64
        final_time = 0.08
    
    params = MHDParams(
        nx=nx, ny=ny,
        final_time=final_time,
        test_cases=test_cases,
        output_dir=output_dir,
        save_plots=True,
        show_plots=False
    )
    
    comparison = MHDComparison(params)
    start_time = time.perf_counter()
    comparison.run_all_tests()
    end_time = time.perf_counter()
    
    return {
        'name': 'MHD',
        'total_time': end_time - start_time,
        'test_cases': len(test_cases),
        'methods': len(params.spatial_methods)
    }


def run_sound_wave_tests(output_dir: str, quick: bool = False) -> Dict[str, Any]:
    """Run sound wave equation comparison tests"""
    print("\n" + "="*60)
    print("RUNNING SOUND WAVE EQUATION TESTS")
    print("="*60)
    
    if quick:
        test_cases = ['gaussian_pulse', 'plane_wave']
        nx, ny = 32, 32
        final_time = 0.2
    else:
        test_cases = ['gaussian_pulse', 'plane_wave', 'standing_wave']
        nx, ny = 64, 64
        final_time = 0.3
    
    params = SoundParams(
        nx=nx, ny=ny,
        final_time=final_time,
        test_cases=test_cases,
        output_dir=output_dir,
        save_plots=True,
        show_plots=False
    )
    
    comparison = SoundWaveComparison(params)
    start_time = time.perf_counter()
    comparison.run_all_tests()
    end_time = time.perf_counter()
    
    return {
        'name': 'Sound Wave',
        'total_time': end_time - start_time,
        'test_cases': len(test_cases),
        'methods': len(params.spatial_methods)
    }


def generate_summary_report(results: List[Dict[str, Any]], output_dir: str):
    """Generate overall summary report"""
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FVM SPATIAL DISCRETIZATION COMPARISON SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Test Configuration:\n")
        f.write("-" * 20 + "\n")
        
        total_time = sum(r['total_time'] for r in results)
        total_tests = sum(r['test_cases'] for r in results)
        
        f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
        f.write(f"Total Test Cases: {total_tests}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("Physics Equation Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(f"{result['name']:15s}: {result['total_time']:8.2f}s "
                   f"({result['test_cases']} cases, {result['methods']} methods)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Generated comparison plots and data in: " + output_dir + "\n")
        f.write("="*60 + "\n")
    
    print(f"\n📊 Summary report saved to: {summary_file}")
    
    # Also print to console
    print("\n" + "="*60)
    print("OVERALL COMPARISON SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total test cases: {total_tests}")
    print(f"Output directory: {output_dir}")
    print("\nPer-physics timing:")
    for result in results:
        print(f"  {result['name']:15s}: {result['total_time']:8.2f}s")
    print("="*60)


def main():
    """Main function to run all comparison tests"""
    parser = argparse.ArgumentParser(description='Run FVM spatial discretization comparison tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick tests with reduced grid size and fewer test cases')
    parser.add_argument('--physics', nargs='+', 
                       choices=['advection', 'euler', 'burgers', 'mhd', 'sound'], 
                       default=['advection', 'euler', 'burgers', 'mhd', 'sound'],
                       help='Physics equations to test')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Base output directory name')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    output_dir = create_output_directory(args.output_dir)
    
    print("🚀 Starting FVM Spatial Discretization Comparison Tests")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚡ Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"🔬 Physics equations: {', '.join(args.physics)}")
    
    results = []
    overall_start = time.perf_counter()
    
    # Run selected physics tests
    if 'advection' in args.physics:
        try:
            result = run_advection_tests(output_dir, args.quick)
            results.append(result)
        except Exception as e:
            print(f"❌ Advection tests failed: {e}")
    
    if 'euler' in args.physics:
        try:
            result = run_euler_tests(output_dir, args.quick)
            results.append(result)
        except Exception as e:
            print(f"❌ Euler tests failed: {e}")
    
    if 'burgers' in args.physics:
        try:
            result = run_burgers_tests(output_dir, args.quick)
            results.append(result)
        except Exception as e:
            print(f"❌ Burgers tests failed: {e}")
    
    if 'mhd' in args.physics:
        try:
            result = run_mhd_tests(output_dir, args.quick)
            results.append(result)
        except Exception as e:
            print(f"❌ MHD tests failed: {e}")
    
    if 'sound' in args.physics:
        try:
            result = run_sound_wave_tests(output_dir, args.quick)
            results.append(result)
        except Exception as e:
            print(f"❌ Sound wave tests failed: {e}")
    
    overall_end = time.perf_counter()
    
    # Generate summary report
    if results:
        generate_summary_report(results, output_dir)
        print(f"🎉 All comparison tests completed in {overall_end - overall_start:.2f} seconds!")
        print(f"📈 Generated {len(results)} physics equation comparisons")
        print(f"📊 Results saved in: {output_dir}")
    else:
        print("❌ No tests completed successfully")


if __name__ == "__main__":
    main()