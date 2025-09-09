# FVM Spatial Discretization Comparison Drivers

This directory contains comprehensive comparison driver files for different spatial discretization methods across various physics equations implemented in the FVM framework.

## Driver Structure

The comparison drivers follow a unified pattern inspired by the reference implementation, allowing systematic comparison of different numerical methods on the same physics problems.

### Available Driver Files

| Driver File | Physics Equation | Variables | Description |
|-----------|-----------------|-----------|-------------|
| `driver_advection_comparison.py` | Linear Advection | `[u]` | Pure transport without diffusion |
| `driver_euler_comparison.py` | Compressible Euler | `[ρ, ρu, ρv, ρw, E]` | Compressible gas dynamics |
| `driver_burgers_comparison.py` | Burgers Equation | `[u, v]` | Nonlinear wave propagation |
| `driver_mhd_comparison.py` | Magnetohydrodynamics | `[ρ, ρu, ρv, ρw, E, Bx, By, Bz]` | Plasma physics with magnetic fields |
| `driver_sound_wave_comparison.py` | Linear Sound Wave | `[p, u, v]` | Acoustic wave propagation |

### Spatial Methods Compared

Each test compares multiple spatial discretization methods:

1. **First Order Methods**
   - Constant reconstruction + Lax-Friedrichs flux
   - Low numerical diffusion but first-order accurate

2. **Second Order Methods** 
   - Slope limiter reconstruction + HLL/HLLC flux
   - Balance between accuracy and stability

3. **High Order Methods**
   - WENO5 reconstruction + HLLC/HLLD flux
   - High accuracy with shock capturing

4. **Specialized Flux Methods**
   - HLL, HLLC, HLLD Riemann solvers
   - Physics-specific flux calculations

## Quick Start

### Run Individual Physics Drivers

```bash
# Run advection equation comparison
python driver_advection_comparison.py

# Run Euler equation comparison  
python driver_euler_comparison.py

# Run Burgers equation comparison
python driver_burgers_comparison.py

# Run MHD equation comparison
python driver_mhd_comparison.py

# Run sound wave equation comparison
python driver_sound_wave_comparison.py
```

### Run All Drivers with Master Script

```bash
# Run all physics equations with full test suite
python run_comparison_tests.py

# Quick test mode (smaller grids, fewer test cases)
python run_comparison_tests.py --quick

# Run specific physics equations only
python run_comparison_tests.py --physics advection euler

# Custom output directory
python run_comparison_tests.py --output-dir my_results
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--quick` | Fast testing with reduced resolution | `--quick` |
| `--physics` | Select specific physics equations | `--physics advection euler` |
| `--output-dir` | Custom output directory base name | `--output-dir results` |

## Test Cases by Physics

### Advection Equation Tests
- **gaussian_pulse**: Gaussian blob transport
- **square_wave**: Discontinuous transport
- **sine_wave**: Smooth periodic wave
- **cosine_hill**: Compact support function
- **slotted_cylinder**: Sharp feature preservation

### Euler Equation Tests
- **sod_shock_tube**: Classic 1D shock tube
- **explosion_2d**: Circular blast wave
- **double_mach_reflection**: Complex shock interaction
- **riemann_2d**: 2D Riemann problems

### Burgers Equation Tests
- **smooth_sine_wave**: Pre-shock smooth evolution
- **gaussian_vortex**: Vortical structures
- **shock_formation**: Nonlinear steepening
- **taylor_green_vortex**: Classical vortex flow

### MHD Tests
- **orszag_tang_vortex**: MHD turbulence benchmark
- **brio_wu_shock**: MHD shock tube
- **current_sheet**: Magnetic reconnection
- **magnetic_reconnection**: Plasma instabilities

### Sound Wave Tests
- **gaussian_pulse**: Acoustic pulse propagation
- **plane_wave**: Directional wave propagation
- **standing_wave**: Resonant modes
- **circular_wave**: Radial wave expansion
- **acoustic_dipole**: Directional emission

## Output and Visualization

### Generated Outputs

Each test produces:

1. **Comparison Plots**: PNG files showing side-by-side method comparisons
2. **Performance Data**: Timing and accuracy metrics
3. **Error Analysis**: Conservation errors and numerical dissipation
4. **Summary Reports**: Text summaries of all results

### Plot Types Generated

- **Initial Conditions**: Reference starting states
- **Cross-sections**: 1D cuts through 2D solutions  
- **Error Comparisons**: L1, L2, L∞ norms
- **Conservation Plots**: Mass, momentum, energy preservation
- **Performance Charts**: Computation time and step counts
- **Physics-specific Metrics**:
  - Vorticity and enstrophy (Burgers)
  - ∇·B constraint (MHD) 
  - Wave amplitude preservation (Sound waves)

### Directory Structure

```
comparison_results_YYYYMMDD_HHMMSS/
├── advection_comparison_gaussian_pulse.png
├── advection_comparison_square_wave.png
├── euler_comparison_sod_shock_tube.png
├── burgers_comparison_gaussian_vortex.png
├── mhd_comparison_orszag_tang_vortex.png
├── sound_wave_comparison_plane_wave.png
└── comparison_summary.txt
```

## Customization

### Adding New Test Cases

1. Add test case to appropriate testsuite file in `/testcases/`
2. Update `test_cases` list in comparison parameter class
3. Run comparison to automatically include new case

### Adding New Spatial Methods

Update the `spatial_methods` list in parameter classes:

```python
{
    'name': 'My Method',
    'reconstruction_type': 'my_reconstruction',
    'flux_type': 'my_flux', 
    'flux_params': {'param': 'value'},
    'color': 'orange',
    'linestyle': '--'
}
```

### Modifying Grid Resolution

Adjust `nx` and `ny` parameters in test parameter classes or use command line arguments.

### Physics Parameters

Each physics equation has customizable parameters:
- **Advection**: `advection_x`, `advection_y`
- **Euler**: `gamma` 
- **Burgers**: `viscosity`
- **MHD**: `gamma`
- **Sound**: `sound_speed`, `density`

## Integration with Framework

The comparison tests integrate seamlessly with the FVM framework:

- Uses the same `FVMSolver` interface as production code
- Leverages the modular pipeline architecture
- Automatically adapts to new spatial discretization methods
- Provides performance profiling for optimization

## Best Practices

1. **Start with Quick Mode**: Use `--quick` for initial testing
2. **Focus on Relevant Physics**: Select specific equations with `--physics`
3. **Check Conservation**: Monitor conservation error plots
4. **Performance Analysis**: Use timing data to optimize methods
5. **Visual Validation**: Always check the generated plots
6. **Systematic Testing**: Run full test suite before releases

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure parent directory is in Python path
2. **Memory Issues**: Reduce grid size for large test suites
3. **Plot Display**: Set `show_plots=False` for headless systems
4. **Failed Tests**: Check individual physics equation imports

### Performance Tips

- Use `--quick` mode for development testing
- Run tests on compute nodes for large grids
- Monitor memory usage for high-resolution tests
- Use profiling data to identify bottlenecks

This comprehensive test suite enables systematic validation and comparison of spatial discretization methods across the full range of physics equations supported by the FVM framework.