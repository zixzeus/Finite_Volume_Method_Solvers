"""
Spatial Discretization Factory

This module provides a unified factory for creating all types of spatial 
discretization schemes, including finite volume, Riemann-based, and 
discontinuous Galerkin methods.
"""

from typing import Dict, Any
from .base import SpatialDiscretization


class SpatialDiscretizationFactory:
    """Factory for creating spatial discretization schemes"""
    
    @staticmethod
    def create(scheme_type: str, **kwargs) -> SpatialDiscretization:
        """
        Create a spatial discretization scheme.
        
        Args:
            scheme_type: Type of scheme ('lax_friedrichs', 'tvdlf', 'hll', 'dg', etc.)
            **kwargs: Additional parameters for the scheme
            
        Returns:
            SpatialDiscretization instance
        """
        scheme_type = scheme_type.lower().replace('-', '_').replace(' ', '_')
        
        # Finite Volume Methods
        if scheme_type in ['lax_friedrichs', 'lf', 'lax_f']:
            from .lax_friedrichs import LaxFriedrichsScheme
            return LaxFriedrichsScheme()
            
        elif scheme_type in ['tvdlf', 'tvd_lf', 'tvd_lax_friedrichs']:
            from .tvd_lax_friedrichs import TVDLFScheme
            limiter = kwargs.get('limiter', 'minmod')
            return TVDLFScheme(limiter)
        
        # Riemann Solver-based Methods
        elif scheme_type in ['hll', 'hll_riemann']:
            from .riemann_schemes import HLLRiemannScheme
            return HLLRiemannScheme()
            
        elif scheme_type in ['hllc', 'hllc_riemann']:
            from .riemann_schemes import HLLCRiemannScheme
            return HLLCRiemannScheme()
            
        elif scheme_type in ['hlld', 'hlld_riemann']:
            from .riemann_schemes import HLLDRiemannScheme
            return HLLDRiemannScheme()
        
        # Discontinuous Galerkin Methods
        elif scheme_type in ['dg', 'discontinuous_galerkin']:
            from .dg_scheme import DGScheme
            polynomial_order = kwargs.get('polynomial_order', kwargs.get('order', 1))
            riemann_solver = kwargs.get('riemann_solver', 'hllc')
            return DGScheme(polynomial_order, riemann_solver)
            
        elif scheme_type in ['dg_p0', 'dg0']:
            from .dg_scheme import DGScheme
            riemann_solver = kwargs.get('riemann_solver', 'hllc')
            return DGScheme(0, riemann_solver)
            
        elif scheme_type in ['dg_p1', 'dg1']:
            from .dg_scheme import DGScheme
            riemann_solver = kwargs.get('riemann_solver', 'hllc')
            return DGScheme(1, riemann_solver)
            
        elif scheme_type in ['dg_p2', 'dg2']:
            from .dg_scheme import DGScheme
            riemann_solver = kwargs.get('riemann_solver', 'hllc')
            return DGScheme(2, riemann_solver)
        
        else:
            raise ValueError(f"Unknown spatial discretization scheme: {scheme_type}")
    
    @staticmethod
    def get_available_schemes() -> Dict[str, Dict[str, Any]]:
        """Get dictionary of available schemes with descriptions and parameters"""
        return {
            # Finite Volume Methods
            'lax_friedrichs': {
                'description': 'First-order Lax-Friedrichs scheme (stable, dissipative)',
                'order': 1,
                'type': 'finite_volume',
                'parameters': {},
                'aliases': ['lf', 'lax_f']
            },
            'tvdlf': {
                'description': 'Second-order TVD Lax-Friedrichs scheme with flux limiters',
                'order': 2,
                'type': 'finite_volume',
                'parameters': {
                    'limiter': 'Flux limiter type (minmod, superbee, van_leer, mc)'
                },
                'aliases': ['tvd_lf', 'tvd_lax_friedrichs']
            },
            
            # Riemann Solver-based Methods
            'hll': {
                'description': 'HLL Riemann solver (robust, moderately accurate)',
                'order': 1,
                'type': 'riemann_based',
                'parameters': {},
                'aliases': ['hll_riemann']
            },
            'hllc': {
                'description': 'HLLC Riemann solver (good for Euler equations)',
                'order': 1,
                'type': 'riemann_based',
                'parameters': {},
                'aliases': ['hllc_riemann']
            },
            'hlld': {
                'description': 'HLLD Riemann solver (specialized for MHD)',
                'order': 1,
                'type': 'riemann_based',
                'parameters': {},
                'aliases': ['hlld_riemann']
            },
            
            # Discontinuous Galerkin Methods
            'dg': {
                'description': 'Discontinuous Galerkin method (high-order)',
                'order': 'variable',
                'type': 'discontinuous_galerkin',
                'parameters': {
                    'polynomial_order': 'Polynomial order (0, 1, 2)',
                    'riemann_solver': 'Riemann solver for numerical fluxes'
                },
                'aliases': ['discontinuous_galerkin']
            },
            'dg_p0': {
                'description': 'DG with P0 polynomials (equivalent to FV)',
                'order': 1,
                'type': 'discontinuous_galerkin',
                'parameters': {
                    'riemann_solver': 'Riemann solver for numerical fluxes'
                },
                'aliases': ['dg0']
            },
            'dg_p1': {
                'description': 'DG with P1 polynomials (piecewise linear)',
                'order': 2,
                'type': 'discontinuous_galerkin',
                'parameters': {
                    'riemann_solver': 'Riemann solver for numerical fluxes'
                },
                'aliases': ['dg1']
            },
            'dg_p2': {
                'description': 'DG with P2 polynomials (piecewise quadratic)',
                'order': 3,
                'type': 'discontinuous_galerkin',
                'parameters': {
                    'riemann_solver': 'Riemann solver for numerical fluxes'
                },
                'aliases': ['dg2']
            }
        }
    
    @staticmethod
    def get_schemes_by_type(scheme_type: str) -> Dict[str, Dict[str, Any]]:
        """Get schemes filtered by type"""
        all_schemes = SpatialDiscretizationFactory.get_available_schemes()
        return {name: info for name, info in all_schemes.items() 
                if info['type'] == scheme_type}
    
    @staticmethod
    def get_schemes_by_order(order: int) -> Dict[str, Dict[str, Any]]:
        """Get schemes filtered by order"""
        all_schemes = SpatialDiscretizationFactory.get_available_schemes()
        return {name: info for name, info in all_schemes.items() 
                if info['order'] == order or info['order'] == 'variable'}
    
    @staticmethod
    def print_available_schemes():
        """Print a formatted list of available schemes"""
        schemes = SpatialDiscretizationFactory.get_available_schemes()
        
        print("Available Spatial Discretization Schemes:")
        print("=" * 50)
        
        # Group by type
        types = {'finite_volume': 'Finite Volume Methods',
                'riemann_based': 'Riemann Solver-based Methods', 
                'discontinuous_galerkin': 'Discontinuous Galerkin Methods'}
        
        for scheme_type, type_name in types.items():
            type_schemes = SpatialDiscretizationFactory.get_schemes_by_type(scheme_type)
            if type_schemes:
                print(f"\n{type_name}:")
                print("-" * len(type_name))
                
                for name, info in type_schemes.items():
                    order_str = f"Order {info['order']}" if info['order'] != 'variable' else 'Variable order'
                    print(f"  {name:<15} - {info['description']} ({order_str})")
                    
                    if info['aliases']:
                        aliases_str = ', '.join(info['aliases'])
                        print(f"  {'':<15}   Aliases: {aliases_str}")
                    
                    if info['parameters']:
                        print(f"  {'':<15}   Parameters:")
                        for param, desc in info['parameters'].items():
                            print(f"  {'':<15}     {param}: {desc}")
        
        print("\nExample usage:")
        print("  factory.create('lax_friedrichs')")
        print("  factory.create('tvdlf', limiter='minmod')")
        print("  factory.create('dg', polynomial_order=2, riemann_solver='hllc')")


# Convenience function
def create_spatial_scheme(scheme_type: str, **kwargs) -> SpatialDiscretization:
    """Convenience function to create spatial schemes"""
    return SpatialDiscretizationFactory.create(scheme_type, **kwargs)