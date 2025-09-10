"""
Plotting utilities for FVM Framework

Common plotting functions for visualization and comparison of simulation results.
This module provides standardized plotting interfaces used across different physics drivers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


class FVMPlotter:
    """
    Unified plotting interface for FVM simulation results
    
    Provides common plotting functionality for:
    - 2D contour plots with initial conditions
    - Cross-section comparisons 
    - Time series evolution plots
    - Multi-variable comparative plots
    """
    
    def __init__(self, params: Any):
        """
        Initialize plotter with simulation parameters
        
        Args:
            params: Parameter object containing domain_size, nx, ny, output settings
        """
        self.params = params
        self.x = np.linspace(0, self.params.domain_size, self.params.nx)
        self.y = np.linspace(0, self.params.domain_size, self.params.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
    def plot_scalar_comparison(self, 
                              test_case: str,
                              results: Dict,
                              variable_index: int = 0,
                              variable_name: str = "Variable",
                              title_suffix: str = "") -> None:
        """
        Create comparison plot for a scalar variable
        
        Args:
            test_case: Name of the test case
            results: Dictionary containing simulation results
            variable_index: Index of variable to plot from conservative state
            variable_name: Name of variable for labeling
            title_suffix: Additional text for plot title
        """
        if test_case not in results:
            print(f"No results found for test case: {test_case}")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        title = f'{title_suffix} Comparison: {test_case}' if title_suffix else f'Comparison: {test_case}'
        fig.suptitle(title, fontsize=16)
        
        # Plot initial condition
        initial_state = results[test_case]['initial_condition'][variable_index]
        
        # Initial condition contour with equal aspect ratio
        im0 = axes[0].contourf(self.X, self.Y, initial_state, levels=20, cmap='viridis')
        axes[0].set_title('Initial Condition')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_xlim(0, self.params.domain_size)
        axes[0].set_ylim(0, self.params.domain_size)
        axes[0].set_aspect('equal', adjustable='box')
        plt.colorbar(im0, ax=axes[0])
        
        # Cross-section comparison at y = domain_size/2
        y_mid_idx = self.params.ny // 2
        axes[1].plot(self.x, initial_state[:, y_mid_idx], 'k-', linewidth=2, label='Initial')
        
        # Plot all methods
        for method in self.params.spatial_methods:
            method_name = method['name']
            if method_name in results[test_case]['solutions']:
                solution = results[test_case]['solutions'][method_name]
                final_state = solution['final_solution']['conservative'][variable_index]
                
                # Cross-section
                axes[1].plot(self.x, final_state[:, y_mid_idx], 
                           color=method['color'], 
                           linestyle=method['linestyle'],
                           linewidth=1.5,
                           label=method_name)
        
        axes[1].set_title(f'Cross-section at y = {self.params.domain_size/2:.1f}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel(variable_name)
        axes[1].set_xlim(0, self.params.domain_size)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_and_show_plot(f'{test_case}_{variable_name.lower()}_comparison')

    def plot_multi_variable_comparison(self, 
                                     test_case: str,
                                     results: Dict,
                                     variables: List[Dict],
                                     title_suffix: str = "") -> None:
        """
        Create multi-panel comparison plot for multiple variables
        
        Args:
            test_case: Name of the test case
            results: Dictionary containing simulation results  
            variables: List of dicts with 'index', 'name', 'units' keys
            title_suffix: Additional text for plot title
        """
        if test_case not in results:
            print(f"No results found for test case: {test_case}")
            return
            
        n_vars = len(variables)
        fig, axes = plt.subplots(2, n_vars, figsize=(6*n_vars, 12))
        title = f'{title_suffix} Comparison: {test_case}' if title_suffix else f'Comparison: {test_case}'
        fig.suptitle(title, fontsize=16)
        
        # Ensure axes is 2D array even for single variable
        if n_vars == 1:
            axes = axes.reshape(2, 1)
            
        # Create coordinate arrays and get initial condition
        initial_condition = results[test_case]['initial_condition']
        
        for i, var_info in enumerate(variables):
            var_idx = var_info['index'] 
            var_name = var_info['name']
            var_units = var_info.get('units', '')
            
            # Initial condition contour
            initial_var = initial_condition[var_idx]
            im = axes[0, i].contourf(self.X, self.Y, initial_var, levels=20, cmap='viridis')
            axes[0, i].set_title(f'Initial {var_name}')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('y')
            axes[0, i].set_xlim(0, self.params.domain_size)
            axes[0, i].set_ylim(0, self.params.domain_size)
            axes[0, i].set_aspect('equal', adjustable='box')
            plt.colorbar(im, ax=axes[0, i])
            
            # Cross-section comparison
            y_mid_idx = self.params.ny // 2
            axes[1, i].plot(self.x, initial_var[:, y_mid_idx], 'k-', linewidth=2, label='Initial')
            
            # Plot all methods
            for method in self.params.spatial_methods:
                method_name = method['name']
                if method_name in results[test_case]['solutions']:
                    solution = results[test_case]['solutions'][method_name]
                    final_var = solution['final_solution']['conservative'][var_idx]
                    
                    axes[1, i].plot(self.x, final_var[:, y_mid_idx],
                                  color=method['color'],
                                  linestyle=method['linestyle'], 
                                  linewidth=1.5,
                                  label=method_name)
            
            ylabel = f'{var_name} [{var_units}]' if var_units else var_name
            axes[1, i].set_title(f'{var_name} at y = {self.params.domain_size/2:.1f}')
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel(ylabel)
            axes[1, i].set_xlim(0, self.params.domain_size)
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_and_show_plot(f'{test_case}_multi_variable_comparison')

    def plot_time_series(self, 
                        test_case: str,
                        method_name: str,
                        results: Dict,
                        variable_index: int = 0,
                        variable_name: str = "Variable") -> None:
        """
        Generate time series evolution plots
        
        Args:
            test_case: Name of the test case
            method_name: Name of the numerical method
            results: Dictionary containing simulation results
            variable_index: Index of variable to plot
            variable_name: Name of variable for labeling
        """
        if test_case not in results:
            print(f"No results found for test case: {test_case}")
            return
            
        if method_name not in results[test_case]['solutions']:
            print(f"No solution found for method: {method_name}")
            return
            
        solution_data = results[test_case]['solutions'][method_name]
        time_series = solution_data.get('time_series')
        
        if time_series is None:
            print(f"No time series data available for {method_name}")
            return
            
        # Check if output times are specified
        if self.params.outputtimes is None:
            print("No output times specified")
            return
            
        n_times = len(self.params.outputtimes)
        cols = min(5, n_times)  # Maximum 5 columns per row
        rows = 2 if n_times <= 5 else 3  # Add extra row for cross-sections if needed
        
        # Create figure with contour plots and cross-sections
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        fig.suptitle(f'{variable_name} Evolution: {test_case} ({method_name})', fontsize=14)
        
        # Handle single column case
        if cols == 1:
            axes = axes.reshape(-1, 1)
        elif n_times == 1:
            axes = axes.reshape(rows, 1)
            
        # Handle both time series data formats:
        # Format 1: {'times': [t1, t2, ...], 'states': [data1, data2, ...]}
        # Format 2: {t1: data1, t2: data2, ...}
        
        if 'times' in time_series and 'states' in time_series:
            # Old format: extract times and states
            times = time_series['times']
            states = time_series['states']
            time_data_pairs = list(zip(times, states))
        else:
            # New format: direct time->data mapping
            time_data_pairs = [(t, data) for t, data in time_series.items()]
            
        # Find global min/max for consistent color scaling
        all_data = []
        for t_idx, output_time in enumerate(self.params.outputtimes):
            # Find closest time in available data
            closest_data = None
            min_time_diff = float('inf')
            
            for time_val, data in time_data_pairs:
                time_diff = abs(time_val - output_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_data = data
                    
            if closest_data is not None:
                if len(closest_data.shape) > 2:  # Multi-variable data
                    var_data = closest_data[variable_index]
                else:  # Single variable data
                    var_data = closest_data
                all_data.append(var_data)
        
        if all_data:
            vmin = np.min([np.min(data) for data in all_data])
            vmax = np.max([np.max(data) for data in all_data])
        else:
            vmin, vmax = None, None
            
        # Plot time snapshots
        y_mid_idx = self.params.ny // 2
        
        for t_idx, output_time in enumerate(self.params.outputtimes[:cols]):
            col = t_idx % cols
            
            # Find closest time in available data
            closest_data = None
            closest_time = None
            min_time_diff = float('inf')
            
            for time_val, data in time_data_pairs:
                time_diff = abs(time_val - output_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_data = data
                    closest_time = time_val
                    
            if closest_data is not None:
                # Extract variable data
                if len(closest_data.shape) > 2:  # Multi-variable: shape (n_vars, nx, ny)
                    plot_data = closest_data[variable_index]
                else:  # Single variable: shape (nx, ny)
                    plot_data = closest_data
                
                # Contour plot
                im = axes[0, col].contourf(self.X, self.Y, plot_data, levels=20, 
                                         cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, col].set_title(f't = {closest_time:.3f}')
                axes[0, col].set_xlabel('x')
                axes[0, col].set_ylabel('y')
                axes[0, col].set_xlim(0, self.params.domain_size)
                axes[0, col].set_ylim(0, self.params.domain_size)
                axes[0, col].set_aspect('equal', adjustable='box')
                plt.colorbar(im, ax=axes[0, col])
                
                # Cross-section plot
                if rows >= 2:
                    axes[1, col].plot(self.x, plot_data[:, y_mid_idx], 'b-', linewidth=2)
                    axes[1, col].set_title(f'Cross-section at t = {closest_time:.3f}')
                    axes[1, col].set_xlabel('x')
                    axes[1, col].set_ylabel(variable_name)
                    axes[1, col].set_xlim(0, self.params.domain_size)
                    axes[1, col].set_ylim(vmin, vmax)
                    axes[1, col].grid(True, alpha=0.3)
            else:
                # Hide unused subplots
                axes[0, col].set_visible(False)
                if rows >= 2:
                    axes[1, col].set_visible(False)
        
        plt.tight_layout()
        self._save_and_show_plot(f'{test_case}_{method_name}_{variable_name.lower()}_time_series')

    def plot_conservation_errors(self, 
                                conservation_errors: Dict,
                                test_case: str,
                                error_types: List[str]) -> None:
        """
        Plot conservation error comparison across methods
        
        Args:
            conservation_errors: Dictionary of errors by method
            test_case: Name of test case
            error_types: List of error types to plot
        """
        if not conservation_errors:
            print("No conservation error data available")
            return
            
        methods = list(conservation_errors.keys())
        n_errors = len(error_types)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(methods))
        width = 0.8 / n_errors
        
        for i, error_type in enumerate(error_types):
            errors = [conservation_errors[method].get(error_type, 0) for method in methods]
            ax.bar(x + i * width, errors, width, label=error_type.replace('_', ' ').title())
        
        ax.set_xlabel('Numerical Method')
        ax.set_ylabel('Relative Error')
        ax.set_title(f'Conservation Errors: {test_case}')
        ax.set_xticks(x + width * (n_errors - 1) / 2)
        ax.set_xticklabels(methods, rotation=45)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_and_show_plot(f'{test_case}_conservation_errors')

    def _save_and_show_plot(self, filename_base: str) -> None:
        """
        Save and optionally show the current plot
        
        Args:
            filename_base: Base filename without extension
        """
        if self.params.save_plots:
            filename = os.path.join(self.params.output_dir, f'{filename_base}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {filename}")
        
        if self.params.show_plots:
            plt.show()
        else:
            plt.close()


def create_physics_specific_plotter(physics_type: str) -> Dict[str, Any]:
    """
    Create physics-specific plotting configurations
    
    Args:
        physics_type: Type of physics (advection, euler, burgers, etc.)
    
    Returns:
        Dictionary with variable configurations for plotting
    """
    configs = {
        'advection': {
            'primary_variable': {'index': 0, 'name': 'u', 'units': ''},
            'title_suffix': 'Advection Equation'
        },
        
        'burgers': {
            'primary_variable': {'index': 0, 'name': 'u', 'units': ''},
            'title_suffix': 'Burgers Equation'
        },
        
        'euler': {
            'variables': [
                {'index': 0, 'name': 'Density', 'units': 'kg/m³'},
                {'index': 1, 'name': 'X-Momentum', 'units': 'kg/(m²·s)'},
                {'index': 4, 'name': 'Energy', 'units': 'J/m³'}
            ],
            'title_suffix': 'Euler Equations',
            'conservation_errors': ['mass', 'momentum_x', 'momentum_y', 'energy']
        },
        
        'sound_wave': {
            'variables': [
                {'index': 0, 'name': 'Pressure', 'units': 'Pa'},
                {'index': 1, 'name': 'X-Velocity', 'units': 'm/s'},
                {'index': 2, 'name': 'Y-Velocity', 'units': 'm/s'}
            ],
            'title_suffix': 'Sound Wave Equations'
        },
        
        'mhd': {
            'variables': [
                {'index': 0, 'name': 'Density', 'units': 'kg/m³'},
                {'index': 4, 'name': 'Energy', 'units': 'J/m³'},
                {'index': 5, 'name': 'Bx', 'units': 'T'},
                {'index': 6, 'name': 'By', 'units': 'T'}
            ],
            'title_suffix': 'MHD Equations',
            'conservation_errors': ['mass', 'momentum_x', 'momentum_y', 'energy', 'magnetic']
        }
    }
    
    return configs.get(physics_type, {})