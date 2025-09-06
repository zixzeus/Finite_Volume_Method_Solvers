"""
Performance Profiling Utilities

This module provides tools for profiling and benchmarking the FVM framework
to analyze performance characteristics and identify optimization opportunities.
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gc

from fvm_framework.core.solver import FVMSolver


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    wall_time: float
    cpu_time: float
    peak_memory_mb: float
    cell_updates_per_second: float
    time_steps: int
    final_time: float
    grid_size: Tuple[int, int]
    solver_config: Dict[str, Any]


class PerformanceProfiler:
    """
    Performance profiler for FVM simulations.
    
    Tracks timing, memory usage, and computational throughput.
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.current_run: Optional[Dict[str, Any]] = None
        
    def start_profiling(self, solver: FVMSolver, run_name: str = "default"):
        """Start profiling a solver run"""
        gc.collect()  # Clean up before measurement
        
        process = psutil.Process()
        
        self.current_run = {
            'name': run_name,
            'solver': solver,
            'start_wall_time': time.perf_counter(),
            'start_cpu_time': process.cpu_times().user + process.cpu_times().system,
            'start_memory_mb': process.memory_info().rss / 1024 / 1024,
            'peak_memory_mb': process.memory_info().rss / 1024 / 1024,
            'initial_time_step': solver.time_step,
            'initial_time': solver.current_time
        }
        
        print(f"Started profiling run: {run_name}")
    
    def update_memory_peak(self):
        """Update peak memory usage"""
        if self.current_run is not None:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.current_run['peak_memory_mb'] = max(
                self.current_run['peak_memory_mb'], current_memory
            )
    
    def finish_profiling(self) -> PerformanceMetrics:
        """Finish profiling and return metrics"""
        if self.current_run is None:
            raise RuntimeError("No profiling session active")
        
        process = psutil.Process()
        
        # Calculate metrics
        wall_time = time.perf_counter() - self.current_run['start_wall_time']
        current_cpu_time = process.cpu_times().user + process.cpu_times().system
        cpu_time = current_cpu_time - self.current_run['start_cpu_time']
        
        solver = self.current_run['solver']
        time_steps_taken = solver.time_step - self.current_run['initial_time_step']
        time_advanced = solver.current_time - self.current_run['initial_time']
        
        # Calculate cell updates per second
        total_cells = solver.geometry.nx * solver.geometry.ny
        total_cell_updates = total_cells * time_steps_taken
        cell_updates_per_second = total_cell_updates / wall_time if wall_time > 0 else 0
        
        # Create metrics object
        metrics = PerformanceMetrics(
            wall_time=wall_time,
            cpu_time=cpu_time,
            peak_memory_mb=self.current_run['peak_memory_mb'],
            cell_updates_per_second=cell_updates_per_second,
            time_steps=time_steps_taken,
            final_time=time_advanced,
            grid_size=(solver.geometry.nx, solver.geometry.ny),
            solver_config=solver.config.copy()
        )
        
        self.metrics.append(metrics)
        self.current_run = None
        
        print(f"Profiling completed:")
        print(f"  Wall time: {wall_time:.3f} s")
        print(f"  CPU time: {cpu_time:.3f} s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"  Performance: {cell_updates_per_second/1000:.1f} kcell-updates/s")
        
        return metrics
    
    def profile_simulation(self, solver: FVMSolver, final_time: float, 
                          run_name: str = "default") -> PerformanceMetrics:
        """Profile a complete simulation run"""
        self.start_profiling(solver, run_name)
        
        # Run with periodic memory monitoring
        original_solve = solver.solve
        
        def monitored_solve(ft):
            """Wrapper to monitor memory during solve"""
            # Override the temporal solver to monitor memory
            orig_solve_n_steps = solver.temporal_solver.solve_n_steps
            
            def monitored_solve_n_steps(*args, **kwargs):
                result = orig_solve_n_steps(*args, **kwargs)
                self.update_memory_peak()
                return result
            
            solver.temporal_solver.solve_n_steps = monitored_solve_n_steps
            
            try:
                result = original_solve(ft)
                return result
            finally:
                solver.temporal_solver.solve_n_steps = orig_solve_n_steps
        
        monitored_solve(final_time)
        
        return self.finish_profiling()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling runs"""
        if not self.metrics:
            return {"message": "No profiling data available"}
        
        # Aggregate statistics
        wall_times = [m.wall_time for m in self.metrics]
        performances = [m.cell_updates_per_second for m in self.metrics]
        memory_usage = [m.peak_memory_mb for m in self.metrics]
        
        return {
            'num_runs': len(self.metrics),
            'wall_time': {
                'mean': np.mean(wall_times),
                'std': np.std(wall_times),
                'min': np.min(wall_times),
                'max': np.max(wall_times)
            },
            'performance_kcups': {
                'mean': np.mean(performances) / 1000,
                'std': np.std(performances) / 1000,
                'min': np.min(performances) / 1000,
                'max': np.max(performances) / 1000
            },
            'memory_mb': {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage)
            }
        }
    
    def print_summary(self):
        """Print detailed summary"""
        summary = self.get_summary()
        
        if 'message' in summary:
            print(summary['message'])
            return
        
        print("Performance Profiling Summary")
        print("=" * 40)
        print(f"Number of runs: {summary['num_runs']}")
        print()
        
        print("Wall Time Statistics:")
        print(f"  Mean: {summary['wall_time']['mean']:.3f} ± {summary['wall_time']['std']:.3f} s")
        print(f"  Range: [{summary['wall_time']['min']:.3f}, {summary['wall_time']['max']:.3f}] s")
        print()
        
        print("Performance Statistics:")
        print(f"  Mean: {summary['performance_kcups']['mean']:.1f} ± {summary['performance_kcups']['std']:.1f} kcell-updates/s")
        print(f"  Range: [{summary['performance_kcups']['min']:.1f}, {summary['performance_kcups']['max']:.1f}] kcell-updates/s")
        print()
        
        print("Memory Usage Statistics:")
        print(f"  Mean: {summary['memory_mb']['mean']:.1f} ± {summary['memory_mb']['std']:.1f} MB")
        print(f"  Range: [{summary['memory_mb']['min']:.1f}, {summary['memory_mb']['max']:.1f}] MB")


def benchmark_solver(grid_sizes: List[int], 
                    riemann_solvers: List[str] = ['hll', 'hllc'],
                    time_integrators: List[str] = ['euler', 'rk3'],
                    final_time: float = 0.01) -> Dict[str, Any]:
    """
    Comprehensive benchmark of solver configurations.
    
    Args:
        grid_sizes: List of grid sizes to test (nx = ny = grid_size)
        riemann_solvers: List of Riemann solver types
        time_integrators: List of time integration schemes
        final_time: Simulation time for each benchmark
        
    Returns:
        Dictionary containing benchmark results
    """
    profiler = PerformanceProfiler()
    results = {}
    
    print("FVM Framework Comprehensive Benchmark")
    print("=" * 50)
    
    total_configs = len(grid_sizes) * len(riemann_solvers) * len(time_integrators)
    config_count = 0
    
    for grid_size in grid_sizes:
        for riemann_solver in riemann_solvers:
            for time_integrator in time_integrators:
                config_count += 1
                
                print(f"\nConfiguration {config_count}/{total_configs}:")
                print(f"  Grid: {grid_size}x{grid_size}")
                print(f"  Riemann: {riemann_solver}")
                print(f"  Time: {time_integrator}")
                
                # Create solver configuration
                config = {
                    'grid': {
                        'nx': grid_size, 'ny': grid_size,
                        'dx': 1.0/grid_size, 'dy': 1.0/grid_size
                    },
                    'numerical': {
                        'riemann_solver': riemann_solver,
                        'time_integrator': time_integrator,
                        'cfl_number': 0.5,
                        'boundary_type': 'periodic'
                    },
                    'simulation': {
                        'final_time': final_time,
                        'output_interval': final_time,
                        'monitor_interval': 1000
                    }
                }
                
                solver = FVMSolver(config)
                
                # Set simple initial conditions
                def uniform_ic(x, y):
                    gamma = 1.4
                    rho = 1.0 + 0.1 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
                    u = 0.1 * np.cos(2*np.pi*x)
                    v = 0.1 * np.cos(2*np.pi*y)
                    w = 0.0
                    p = 1.0
                    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
                    return np.array([rho, rho*u, rho*v, rho*w, E])
                
                solver.set_initial_conditions(uniform_ic)
                
                # Run benchmark
                run_name = f"{grid_size}_{riemann_solver}_{time_integrator}"
                
                try:
                    metrics = profiler.profile_simulation(solver, final_time, run_name)
                    
                    # Store results
                    key = (grid_size, riemann_solver, time_integrator)
                    results[key] = {
                        'metrics': metrics,
                        'success': True,
                        'error': None
                    }
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    results[key] = {
                        'metrics': None,
                        'success': False,
                        'error': str(e)
                    }
    
    # Generate summary report
    print(f"\n\nBenchmark Results Summary")
    print("=" * 50)
    
    successful_runs = {k: v for k, v in results.items() if v['success']}
    failed_runs = {k: v for k, v in results.items() if not v['success']}
    
    print(f"Successful runs: {len(successful_runs)}/{total_configs}")
    print(f"Failed runs: {len(failed_runs)}")
    
    if successful_runs:
        # Performance by grid size
        print(f"\nPerformance by Grid Size:")
        print("-" * 30)
        for grid_size in sorted(set(k[0] for k in successful_runs.keys())):
            grid_runs = {k: v for k, v in successful_runs.items() if k[0] == grid_size}
            performances = [v['metrics'].cell_updates_per_second/1000 for v in grid_runs.values()]
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            print(f"  {grid_size}x{grid_size}: {mean_perf:.1f} ± {std_perf:.1f} kcell-updates/s")
        
        # Performance by solver type
        print(f"\nPerformance by Riemann Solver:")
        print("-" * 30)
        for solver_type in sorted(set(k[1] for k in successful_runs.keys())):
            solver_runs = {k: v for k, v in successful_runs.items() if k[1] == solver_type}
            performances = [v['metrics'].cell_updates_per_second/1000 for v in solver_runs.values()]
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            print(f"  {solver_type.upper()}: {mean_perf:.1f} ± {std_perf:.1f} kcell-updates/s")
    
    if failed_runs:
        print(f"\nFailed Configurations:")
        print("-" * 30)
        for (grid, riemann, temporal), result in failed_runs.items():
            print(f"  {grid}x{grid}, {riemann}, {temporal}: {result['error']}")
    
    return {
        'results': results,
        'profiler': profiler,
        'summary': {
            'total_configs': total_configs,
            'successful': len(successful_runs),
            'failed': len(failed_runs)
        }
    }