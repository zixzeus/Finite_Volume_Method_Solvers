"""
Reconstruction Factory

This module provides a factory for creating various spatial reconstruction schemes.
"""

from typing import Dict, Any
from .base_reconstruction import ReconstructionScheme
from .constant_reconstruction import ConstantReconstruction
from .slope_limiter_reconstruction import SlopeLimiterReconstruction


class ReconstructionFactory:
    """Factory for creating reconstruction schemes"""
    
    @staticmethod
    def create(reconstruction_type: str, **kwargs) -> ReconstructionScheme:
        """
        Create a reconstruction scheme.
        
        Args:
            reconstruction_type: Type of reconstruction ('constant', 'slope_limiter', 'muscl', 'weno', etc.)
            **kwargs: Additional parameters for the reconstruction scheme
            
        Returns:
            ReconstructionScheme instance
        """
        reconstruction_type = reconstruction_type.lower().replace('-', '_').replace(' ', '_')
        
        # First-order reconstruction methods
        if reconstruction_type in ['constant', 'first_order', 'zero_order']:
            return ConstantReconstruction()
            
        # Second-order reconstruction methods
        elif reconstruction_type in ['slope_limiter', 'tvd', 'muscl_basic', 'limited']:
            limiter = kwargs.get('limiter', kwargs.get('limiter_type', 'minmod'))
            return SlopeLimiterReconstruction(limiter)
            
        # Specific limiter-based reconstructions
        elif reconstruction_type in ['minmod', 'superbee', 'van_leer', 'mc']:
            return SlopeLimiterReconstruction(reconstruction_type)
        
        # High-order reconstruction methods
        elif reconstruction_type in ['weno', 'weno3', 'weno5']:
            from .weno_reconstruction import WENOReconstruction
            if 'weno3' in reconstruction_type:
                order = 3
            elif 'weno5' in reconstruction_type:
                order = 5
            else:
                order = kwargs.get('order', 3)
            return WENOReconstruction(order)
            
        elif reconstruction_type in ['muscl', 'muscl_hancock']:
            from .muscl_reconstruction import MUSCLReconstruction
            limiter = kwargs.get('limiter', 'van_leer')
            kappa = kwargs.get('kappa', 1.0/3.0)
            return MUSCLReconstruction(limiter, kappa)
        
        else:
            raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")
    
    @staticmethod
    def get_available_reconstructions() -> Dict[str, Dict[str, Any]]:
        """Get dictionary of available reconstruction schemes with descriptions"""
        return {
            # First-order methods
            'constant': {
                'description': 'First-order constant reconstruction (piecewise constant)',
                'order': 1,
                'parameters': {},
                'aliases': ['first_order', 'zero_order']
            },
            
            # Second-order methods
            'slope_limiter': {
                'description': 'Second-order TVD reconstruction with slope limiters',
                'order': 2,
                'parameters': {
                    'limiter': 'Limiter type (minmod, superbee, van_leer, mc)'
                },
                'aliases': ['tvd', 'muscl_basic', 'limited']
            },
            'minmod': {
                'description': 'TVD reconstruction with MinMod limiter (most dissipative)',
                'order': 2,
                'parameters': {},
                'aliases': []
            },
            'superbee': {
                'description': 'TVD reconstruction with Superbee limiter (least dissipative)',
                'order': 2,
                'parameters': {},
                'aliases': []
            },
            'van_leer': {
                'description': 'TVD reconstruction with Van Leer limiter (smooth)',
                'order': 2,
                'parameters': {},
                'aliases': []
            },
            'mc': {
                'description': 'TVD reconstruction with Monotonized Central limiter',
                'order': 2,
                'parameters': {},
                'aliases': []
            },
            
            # High-order methods
            'weno': {
                'description': 'WENO reconstruction (high-order, non-oscillatory)',
                'order': 'variable',
                'parameters': {
                    'order': 'WENO order (3 or 5)',
                    'epsilon': 'Small parameter to avoid division by zero'
                },
                'aliases': ['weno3', 'weno5']
            },
            'weno3': {
                'description': 'WENO3 reconstruction (3rd order)',
                'order': 3,
                'parameters': {
                    'epsilon': 'Small parameter to avoid division by zero'
                },
                'aliases': []
            },
            'weno5': {
                'description': 'WENO5 reconstruction (5th order)',
                'order': 5,
                'parameters': {
                    'epsilon': 'Small parameter to avoid division by zero'
                },
                'aliases': []
            },
            'muscl': {
                'description': 'MUSCL reconstruction (second-order with parameter)',
                'order': 2,
                'parameters': {
                    'limiter': 'Limiter type for MUSCL scheme',
                    'kappa': 'MUSCL parameter (-1 ≤ κ ≤ 1)'
                },
                'aliases': ['muscl_hancock']
            }
        }
    
    @staticmethod
    def get_reconstructions_by_order(order: int) -> Dict[str, Dict[str, Any]]:
        """Get reconstruction schemes filtered by order"""
        all_reconstructions = ReconstructionFactory.get_available_reconstructions()
        return {name: info for name, info in all_reconstructions.items() 
                if info['order'] == order or info['order'] == 'variable'}
    
    @staticmethod
    def print_available_reconstructions():
        """Print a formatted list of available reconstruction schemes"""
        reconstructions = ReconstructionFactory.get_available_reconstructions()
        
        print("Available Spatial Reconstruction Schemes:")
        print("=" * 50)
        
        # Group by order
        orders = [1, 2]  # Will add 'variable' when high-order methods are implemented
        
        for order in orders:
            order_reconstructions = ReconstructionFactory.get_reconstructions_by_order(order)
            if order_reconstructions:
                print(f"\nOrder {order} Methods:")
                print("-" * 20)
                
                for name, info in order_reconstructions.items():
                    print(f"  {name:<15} - {info['description']}")
                    
                    if info['aliases']:
                        aliases_str = ', '.join(info['aliases'])
                        print(f"  {'':<15}   Aliases: {aliases_str}")
                    
                    if info['parameters']:
                        print(f"  {'':<15}   Parameters:")
                        for param, desc in info['parameters'].items():
                            print(f"  {'':<15}     {param}: {desc}")
        
        print("\nExample usage:")
        print("  factory.create('constant')")
        print("  factory.create('slope_limiter', limiter='minmod')")
        print("  factory.create('superbee')")
        print("  factory.create('muscl', limiter='van_leer', kappa=0.33)")
        print("  factory.create('weno5', epsilon=1e-6)")


# Convenience function
def create_reconstruction(reconstruction_type: str, **kwargs) -> ReconstructionScheme:
    """Convenience function to create reconstruction schemes"""
    return ReconstructionFactory.create(reconstruction_type, **kwargs)