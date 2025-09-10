"""
Plotting utilities for FVM Framework

Common plotting functions for visualization and comparison of simulation results.
This module provides standardized plotting interfaces used across different physics drivers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


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

    def plot_multi_variable_time_series(self, 
                                       test_case: str,
                                       method_name: str,
                                       results: Dict,
                                       variables: List[Dict]) -> None:
        """
        Generate multi-variable time series evolution plots
        
        Args:
            test_case: Name of the test case
            method_name: Name of the numerical method
            results: Dictionary containing simulation results
            variables: List of variable configs [{'index': 0, 'name': 'u', 'units': 'm/s'}, ...]
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
            
        # Handle both time series data formats
        if 'times' in time_series and 'states' in time_series:
            times = time_series['times']
            states = time_series['states']
            time_data_pairs = list(zip(times, states))
        else:
            time_data_pairs = [(t, data) for t, data in time_series.items()]
            
        n_times = len(self.params.outputtimes)
        n_vars = len(variables)
        cols = min(5, n_times)  # Maximum 5 time columns
        rows = n_vars * 2  # Two rows per variable: contour + cross-section
        
        # Create figure with variables as rows (contour + cross-section), times as columns
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.5*rows))
        fig.suptitle(f'Multi-Variable Evolution: {test_case} ({method_name})', fontsize=14)
        
        # Ensure axes is 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        # Plot each variable across time
        for var_idx, var_info in enumerate(variables):
            var_index = var_info['index']
            var_name = var_info['name']
            var_units = var_info.get('units', '')
            
            # Find global min/max for this variable across all times
            all_var_data = []
            for t_idx, output_time in enumerate(self.params.outputtimes):
                closest_data = None
                min_time_diff = float('inf')
                
                for time_val, data in time_data_pairs:
                    time_diff = abs(time_val - output_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_data = data
                        
                if closest_data is not None:
                    if len(closest_data.shape) > 2:
                        var_data = closest_data[var_index]
                    else:
                        var_data = closest_data
                    all_var_data.append(var_data)
            
            if all_var_data:
                vmin = np.min([np.min(data) for data in all_var_data])
                vmax = np.max([np.max(data) for data in all_var_data])
            else:
                vmin, vmax = None, None
            
            # Plot time snapshots for this variable
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
                    if len(closest_data.shape) > 2:
                        plot_data = closest_data[var_index]
                    else:
                        plot_data = closest_data
                    
                    # Contour plot (first row for this variable)
                    contour_row = var_idx * 2
                    im = axes[contour_row, col].contourf(self.X, self.Y, plot_data, levels=20, 
                                                      cmap='viridis', vmin=vmin, vmax=vmax)
                    
                    # Title for contour plot
                    if var_idx == 0:
                        title = f'{var_name} Contour, t = {closest_time:.3f}'
                    else:
                        title = f't = {closest_time:.3f}'
                    axes[contour_row, col].set_title(title, fontsize=10)
                    
                    axes[contour_row, col].set_xlabel('x')
                    if col == 0:  # Only label y-axis for leftmost column
                        ylabel = f'{var_name}' if not var_units else f'{var_name} [{var_units}]'
                        axes[contour_row, col].set_ylabel(ylabel)
                    
                    axes[contour_row, col].set_xlim(0, self.params.domain_size)
                    axes[contour_row, col].set_ylim(0, self.params.domain_size)
                    axes[contour_row, col].set_aspect('equal', adjustable='box')
                    
                    # Add colorbar only for the last column of each contour row
                    if col == cols - 1:
                        plt.colorbar(im, ax=axes[contour_row, col])
                    
                    # Cross-section plot (second row for this variable)
                    cross_row = var_idx * 2 + 1
                    
                    # Extract cross-sections: middle row and middle column
                    ny, nx = plot_data.shape
                    mid_y = ny // 2
                    mid_x = nx // 2
                    
                    # Create coordinate arrays for cross-sections
                    x_coords = np.linspace(0, self.params.domain_size, nx)
                    y_coords = np.linspace(0, self.params.domain_size, ny)
                    
                    # Plot both cross-sections on the same subplot
                    axes[cross_row, col].plot(x_coords, plot_data[mid_y, :], 'b-', 
                                            label=f'y = {y_coords[mid_y]:.2f}', linewidth=1.5)
                    axes[cross_row, col].plot(y_coords, plot_data[:, mid_x], 'r--', 
                                            label=f'x = {x_coords[mid_x]:.2f}', linewidth=1.5)
                    
                    # Set consistent y-axis limits for cross-sections
                    if vmin is not None and vmax is not None and abs(vmax - vmin) > 1e-15:
                        axes[cross_row, col].set_ylim(vmin, vmax)
                    axes[cross_row, col].set_xlim(0, self.params.domain_size)
                    
                    # Title for cross-section plot
                    if var_idx == 0:
                        cross_title = f'{var_name} Cross-sections'
                    else:
                        cross_title = 'Cross-sections'
                    axes[cross_row, col].set_title(cross_title, fontsize=10)
                    
                    axes[cross_row, col].set_xlabel('Position')
                    if col == 0:  # Only label y-axis for leftmost column
                        ylabel = f'{var_name}' if not var_units else f'{var_name} [{var_units}]'
                        axes[cross_row, col].set_ylabel(ylabel)
                    
                    # Add legend only for the first column
                    if col == 0:
                        axes[cross_row, col].legend(fontsize=8)
                    
                    axes[cross_row, col].grid(True, alpha=0.3)
                    
                else:
                    # Hide unused subplots
                    contour_row = var_idx * 2
                    cross_row = var_idx * 2 + 1
                    axes[contour_row, col].set_visible(False)
                    axes[cross_row, col].set_visible(False)
        
        plt.tight_layout()
        self._save_and_show_plot(f'{test_case}_{method_name}_multi_variable_time_series')
    
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
        
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(methods))
        width = 0.8 / n_errors
        
        for i, error_type in enumerate(error_types):
            errors = [conservation_errors[method].get(error_type, 0) for method in methods]
            ax.bar(x.astype(float) + i * width, errors, width, label=error_type.replace('_', ' ').title())
        
        ax.set_xlabel('Numerical Method')
        ax.set_ylabel('Relative Error')
        ax.set_title(f'Conservation Errors: {test_case}')
        ax.set_xticks(x.astype(float) + width * (n_errors - 1) / 2)
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
            'variables': [{'index': 0, 'name': 'u', 'units': ''}],
            'title_suffix': 'Advection Equation'
        },
        
        'burgers': {
            'variables': [  # Variables for multi-variable time series
                {'index': 0, 'name': 'u', 'units': 'm/s'},
                {'index': 1, 'name': 'v', 'units': 'm/s'}
            ],
            'title_suffix': 'Burgers Equation',
            'special_metrics': ['energy_dissipation', 'enstrophy_dissipation']
        },
        
        'euler': {
            'variables': [  # All 5 Euler variables
                {'index': 0, 'name': 'ρ', 'units': 'kg/m³'},
                {'index': 1, 'name': 'ρu', 'units': 'kg/(m²·s)'},
                {'index': 2, 'name': 'ρv', 'units': 'kg/(m²·s)'},
                {'index': 3, 'name': 'ρw', 'units': 'kg/(m²·s)'},
                {'index': 4, 'name': 'E', 'units': 'J/m³'}
            ],
            'title_suffix': 'Euler Equations',
            'conservation_errors': ['mass', 'momentum_x', 'momentum_y', 'energy']
        },
        
        'sound_wave': {
            'variables': [
                {'index': 0, 'name': 'p', 'units': 'Pa'},
                {'index': 1, 'name': 'u', 'units': 'm/s'},
                {'index': 2, 'name': 'v', 'units': 'm/s'}
            ],
            'title_suffix': 'Sound Wave Equations'
        },
        
        'mhd': {
            'variables': [  # All 8 MHD variables
                {'index': 0, 'name': 'ρ', 'units': 'kg/m³'},
                {'index': 1, 'name': 'ρu', 'units': 'kg/(m²·s)'},
                {'index': 2, 'name': 'ρv', 'units': 'kg/(m²·s)'},
                {'index': 3, 'name': 'ρw', 'units': 'kg/(m²·s)'},
                {'index': 4, 'name': 'E', 'units': 'J/m³'},
                {'index': 5, 'name': 'Bx', 'units': 'T'},
                {'index': 6, 'name': 'By', 'units': 'T'},
                {'index': 7, 'name': 'Bz', 'units': 'T'}
            ],
            'title_suffix': 'MHD Equations',
            'conservation_errors': ['mass', 'momentum_x', 'momentum_y', 'energy', 'magnetic']
        }
    }
    
    return configs.get(physics_type, {})