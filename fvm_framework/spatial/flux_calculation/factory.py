"""
Flux Calculator Factory

This module provides a factory for creating various numerical flux calculators.
"""

from typing import Dict, Any
from .base_flux import FluxCalculator
from .lax_friedrichs_flux import LaxFriedrichsFlux
from .riemann_flux import RiemannFlux, HLLFlux, HLLCFlux, HLLDFlux, ExactRiemannFlux


class FluxCalculatorFactory:
    """Factory for creating flux calculators"""
    
    @staticmethod
    def create(flux_type: str, **kwargs) -> FluxCalculator:
        """
        Create a flux calculator.
        
        Args:
            flux_type: Type of flux calculator ('lax_friedrichs', 'riemann', 'hll', 'hllc', etc.)
            **kwargs: Additional parameters for the flux calculator
            
        Returns:
            FluxCalculator instance
        """
        flux_type = flux_type.lower().replace('-', '_').replace(' ', '_')
        
        # Lax-Friedrichs type flux calculators
        if flux_type in ['lax_friedrichs', 'lf', 'lax_f']:
            return LaxFriedrichsFlux()
        
        # Generic Riemann solver flux calculator
        elif flux_type in ['riemann', 'riemann_solver']:
            solver_type = kwargs.get('riemann_solver', kwargs.get('solver_type', 'hllc'))
            return RiemannFlux(solver_type)
        
        # Specific Riemann solvers
        elif flux_type in ['hll', 'hll_riemann']:
            return HLLFlux()
            
        elif flux_type in ['hllc', 'hllc_riemann']:
            return HLLCFlux()
            
        elif flux_type in ['hlld', 'hlld_riemann']:
            return HLLDFlux()
            
        elif flux_type in ['exact', 'exact_riemann']:
            return ExactRiemannFlux()
        
        # Future extensions:
        # elif flux_type in ['central', 'central_difference']:
        #     return CentralFlux()
        #     
        # elif flux_type in ['upwind', 'first_order_upwind']:
        #     return UpwindFlux()
        #     
        # elif flux_type in ['godunov', 'godunov_flux']:
        #     return GodunovFlux()
        
        else:
            raise ValueError(f"Unknown flux calculator type: {flux_type}")
    
    @staticmethod
    def get_available_flux_calculators() -> Dict[str, Dict[str, Any]]:
        """Get dictionary of available flux calculators with descriptions"""
        return {
            # Dissipative methods
            'lax_friedrichs': {
                'description': 'Lax-Friedrichs flux (stable, dissipative, first-order)',
                'type': 'dissipative',
                'needs_wave_speed': True,
                'parameters': {},
                'aliases': ['lf', 'lax_f']
            },
            
            # Riemann solver-based methods
            'riemann': {
                'description': 'Generic Riemann solver flux calculator',
                'type': 'riemann_based',
                'needs_wave_speed': False,
                'parameters': {
                    'riemann_solver': 'Riemann solver type (hll, hllc, hlld, exact)'
                },
                'aliases': ['riemann_solver']
            },
            'hll': {
                'description': 'HLL Riemann solver flux (robust, moderately accurate)',
                'type': 'riemann_based',
                'needs_wave_speed': False,
                'parameters': {},
                'aliases': ['hll_riemann']
            },
            'hllc': {
                'description': 'HLLC Riemann solver flux (good for Euler equations)',
                'type': 'riemann_based',
                'needs_wave_speed': False,
                'parameters': {},
                'aliases': ['hllc_riemann']
            },
            'hlld': {
                'description': 'HLLD Riemann solver flux (specialized for MHD)',
                'type': 'riemann_based',
                'needs_wave_speed': False,
                'parameters': {},
                'physics_support': ['mhd', 'magnetohydrodynamics'],
                'aliases': ['hlld_riemann']
            },
            'exact': {
                'description': 'Exact Riemann solver flux (accurate but expensive)',
                'type': 'riemann_based',
                'needs_wave_speed': False,
                'parameters': {},
                'aliases': ['exact_riemann']
            }
            
            # Future flux calculators:
            # 'central': {
            #     'description': 'Central difference flux (non-dissipative)',
            #     'type': 'central',
            #     'needs_wave_speed': False,
            #     'parameters': {},
            #     'aliases': ['central_difference']
            # },
            # 'upwind': {
            #     'description': 'First-order upwind flux',
            #     'type': 'upwind',
            #     'needs_wave_speed': True,
            #     'parameters': {},
            #     'aliases': ['first_order_upwind']
            # }
        }
    
    @staticmethod
    def get_flux_calculators_by_type(calc_type: str) -> Dict[str, Dict[str, Any]]:
        """Get flux calculators filtered by type"""
        all_calculators = FluxCalculatorFactory.get_available_flux_calculators()
        return {name: info for name, info in all_calculators.items() 
                if info['type'] == calc_type}
    
    @staticmethod
    def get_flux_calculators_for_physics(physics_type: str) -> Dict[str, Dict[str, Any]]:
        """Get flux calculators that support specific physics type"""
        all_calculators = FluxCalculatorFactory.get_available_flux_calculators()
        compatible = {}
        
        for name, info in all_calculators.items():
            # If physics_support is specified, check compatibility
            if 'physics_support' in info:
                if physics_type.lower() in info['physics_support']:
                    compatible[name] = info
            else:
                # If no specific physics support listed, assume general compatibility
                compatible[name] = info
        
        return compatible
    
    @staticmethod
    def print_available_flux_calculators():
        """Print a formatted list of available flux calculators"""
        calculators = FluxCalculatorFactory.get_available_flux_calculators()
        
        print("Available Flux Calculators:")
        print("=" * 50)
        
        # Group by type
        types = {
            'dissipative': 'Dissipative Methods',
            'riemann_based': 'Riemann Solver-based Methods',
            'central': 'Central Methods',
            'upwind': 'Upwind Methods'
        }
        
        for calc_type, type_name in types.items():
            type_calculators = FluxCalculatorFactory.get_flux_calculators_by_type(calc_type)
            if type_calculators:
                print(f"\n{type_name}:")
                print("-" * len(type_name))
                
                for name, info in type_calculators.items():
                    wave_speed_str = "(needs wave speed)" if info['needs_wave_speed'] else ""
                    print(f"  {name:<15} - {info['description']} {wave_speed_str}")
                    
                    if info['aliases']:
                        aliases_str = ', '.join(info['aliases'])
                        print(f"  {'':<15}   Aliases: {aliases_str}")
                    
                    if info['parameters']:
                        print(f"  {'':<15}   Parameters:")
                        for param, desc in info['parameters'].items():
                            print(f"  {'':<15}     {param}: {desc}")
                    
                    if 'physics_support' in info:
                        physics_str = ', '.join(info['physics_support'])
                        print(f"  {'':<15}   Physics: {physics_str}")
        
        print("\nExample usage:")
        print("  factory.create('lax_friedrichs')")
        print("  factory.create('hllc')")
        print("  factory.create('riemann', riemann_solver='hll')")


# Convenience function
def create_flux_calculator(flux_type: str, **kwargs) -> FluxCalculator:
    """Convenience function to create flux calculators"""
    return FluxCalculatorFactory.create(flux_type, **kwargs)