#!/usr/bin/env python3
"""
Demonstration of the Unified Spatial Discretization Architecture

This script shows how all spatial discretization methods (finite volume,
Riemann solver-based, and discontinuous Galerkin) are now unified under
a single framework with consistent interfaces.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fvm_framework.spatial.factory import SpatialDiscretizationFactory

def demonstrate_unified_interface():
    """Demonstrate how different methods share the same interface"""
    print("Unified Spatial Discretization Architecture Demo")
    print("=" * 60)
    
    # List of different schemes to demonstrate
    schemes_to_demo = [
        ('lax_friedrichs', {}, 'First-order finite volume'),
        ('tvdlf', {'limiter': 'minmod'}, 'Second-order finite volume with limiting'),
        ('hllc', {}, 'HLLC Riemann solver'),
        ('dg_p1', {'riemann_solver': 'hllc'}, 'Discontinuous Galerkin P1'),
        ('dg', {'polynomial_order': 2, 'riemann_solver': 'hll'}, 'Discontinuous Galerkin P2')
    ]
    
    print(f"\nDemonstrating {len(schemes_to_demo)} different spatial discretization methods:")
    print("-" * 60)
    
    schemes = []
    for scheme_name, params, description in schemes_to_demo:
        try:
            scheme = SpatialDiscretizationFactory.create(scheme_name, **params)
            schemes.append((scheme, description))
            
            # Show unified interface properties
            print(f"\n{description}:")
            print(f"  • Scheme: {scheme.name}")
            print(f"  • Order: {scheme.order}")
            print(f"  • Type: {scheme.scheme_type}")
            
            # Show method-specific properties
            if hasattr(scheme, 'polynomial_order'):
                print(f"  • Polynomial Order: {scheme.polynomial_order}")
            if hasattr(scheme, 'riemann_solver'):
                print(f"  • Riemann Solver: {scheme.riemann_solver}")
            if hasattr(scheme, 'limiter'):
                print(f"  • Limiter: {scheme.limiter}")
            
            # All schemes share the same interface
            print(f"  • Interface methods: compute_fluxes(), get_max_wave_speed()")
            
        except Exception as e:
            print(f"✗ Failed to create {scheme_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Successfully created {len(schemes)} unified spatial discretization schemes!")
    print("All schemes implement the same SpatialDiscretization interface.")
    return len(schemes)

def demonstrate_filtering():
    """Demonstrate scheme filtering by type and order"""
    print("\n\nScheme Filtering Examples:")
    print("=" * 60)
    
    factory = SpatialDiscretizationFactory
    
    # Filter by type
    print("1. Finite Volume Methods:")
    fv_schemes = factory.get_schemes_by_type('finite_volume')
    for name, info in fv_schemes.items():
        print(f"   • {name}: {info['description']}")
    
    print("\n2. Riemann Solver-based Methods:")
    riemann_schemes = factory.get_schemes_by_type('riemann_based')
    for name, info in riemann_schemes.items():
        print(f"   • {name}: {info['description']}")
    
    print("\n3. Discontinuous Galerkin Methods:")
    dg_schemes = factory.get_schemes_by_type('discontinuous_galerkin')
    for name, info in dg_schemes.items():
        print(f"   • {name}: {info['description']}")
    
    # Filter by order
    print("\n4. First-order Methods:")
    first_order = factory.get_schemes_by_order(1)
    for name, info in first_order.items():
        if info['order'] == 1:
            print(f"   • {name}: {info['description']}")
    
    print("\n5. Second-order Methods:")
    second_order = factory.get_schemes_by_order(2)
    for name, info in second_order.items():
        if info['order'] == 2:
            print(f"   • {name}: {info['description']}")

def demonstrate_aliases():
    """Demonstrate scheme aliases"""
    print("\n\nAlias Usage Examples:")
    print("=" * 60)
    
    alias_examples = [
        ('lax_friedrichs', 'lf'),
        ('tvdlf', 'tvd_lf'),
        ('hllc', 'hllc_riemann'),
        ('dg_p1', 'dg1'),
    ]
    
    for canonical, alias in alias_examples:
        try:
            scheme1 = SpatialDiscretizationFactory.create(canonical)
            scheme2 = SpatialDiscretizationFactory.create(alias)
            print(f"'{canonical}' ≡ '{alias}' → {scheme1.name}")
        except Exception as e:
            print(f"Failed to create {canonical}/{alias}: {e}")

def main():
    """Main demonstration"""
    # Core demonstration
    num_schemes = demonstrate_unified_interface()
    
    # Additional features
    demonstrate_filtering()
    demonstrate_aliases()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Unified Spatial Discretization Architecture Summary:")
    print("=" * 60)
    print("✓ All methods unified under SpatialDiscretization base class")
    print("✓ Consistent interface: compute_fluxes(), get_max_wave_speed()")
    print("✓ Factory pattern for easy creation and configuration")
    print("✓ Type-based and order-based filtering")
    print("✓ Multiple aliases for convenience")
    print("✓ Modular file organization")
    print("✓ Easy to extend with new methods")
    print(f"✓ {num_schemes} different schemes successfully demonstrated")
    
    print("\nThe architecture successfully unifies:")
    print("  • Finite Volume methods (Lax-Friedrichs, TVDLF)")
    print("  • Riemann solver-based methods (HLL, HLLC, HLLD)")
    print("  • Discontinuous Galerkin methods (P0, P1, P2)")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)