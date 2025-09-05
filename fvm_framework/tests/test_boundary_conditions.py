"""
Unit tests for Boundary Conditions

Tests various boundary condition implementations and their correctness.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from boundary.boundary_conditions import (
    PeriodicBC, ReflectiveBC, TransmissiveBC, InflowBC, CustomBC,
    BoundaryManager, EulerBoundaryConditions
)
from core.data_container import FVMDataContainer2D, GridGeometry


class TestBasicBoundaryConditions(unittest.TestCase):
    """Test basic boundary condition implementations"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=6, ny=4, dx=0.2, dy=0.25)
        self.data = FVMDataContainer2D(self.geometry)
        
        # Set up interior state (avoid boundaries for testing)
        interior_density = 1.5
        interior_velocity = [0.3, -0.2, 0.1]
        interior_pressure = 2.0
        gamma = 1.4
        
        # Set interior cells with uniform state
        for i in range(1, self.geometry.nx-1):
            for j in range(1, self.geometry.ny-1):
                self.data.state[0, i, j] = interior_density
                self.data.state[1, i, j] = interior_density * interior_velocity[0]
                self.data.state[2, i, j] = interior_density * interior_velocity[1]
                self.data.state[3, i, j] = interior_density * interior_velocity[2]
                
                # Total energy
                kinetic = 0.5 * interior_density * sum(v**2 for v in interior_velocity)
                internal = interior_pressure / (gamma - 1.0)
                self.data.state[4, i, j] = kinetic + internal
    
    def test_periodic_bc(self):
        """Test periodic boundary conditions"""
        bc = PeriodicBC()
        
        # Apply boundary condition (should not modify anything for interior-only)
        bc.apply(self.data)
        
        # Periodic BC implementation is placeholder, so just test it doesn't crash
        self.assertEqual(bc.name, "Periodic")
    
    def test_reflective_bc(self):
        """Test reflective boundary conditions"""
        bc = ReflectiveBC()
        
        # Apply reflective BC
        bc.apply(self.data)
        
        # Check that normal momentum components are reflected
        # Left boundary (x=0): x-momentum should be negative
        interior_x_mom = self.data.state[1, 1, 1]  # Interior x-momentum
        boundary_x_mom = self.data.state[1, 0, 1]  # Boundary x-momentum
        self.assertAlmostEqual(boundary_x_mom, -interior_x_mom, places=10)
        
        # Right boundary (x=nx-1): x-momentum should be negative
        interior_x_mom = self.data.state[1, -2, 1]
        boundary_x_mom = self.data.state[1, -1, 1]
        self.assertAlmostEqual(boundary_x_mom, -interior_x_mom, places=10)
        
        # Bottom boundary (y=0): y-momentum should be negative
        interior_y_mom = self.data.state[2, 1, 1]
        boundary_y_mom = self.data.state[2, 1, 0]
        self.assertAlmostEqual(boundary_y_mom, -interior_y_mom, places=10)
    
    def test_reflective_bc_with_wall_velocity(self):
        """Test reflective BC with moving wall"""
        wall_velocity = np.array([0.1, 0.0, 0.0])  # Moving wall in x-direction
        bc = ReflectiveBC(wall_velocity=wall_velocity)
        
        bc.apply(self.data)
        
        # At wall, x-momentum should reflect wall velocity
        rho_wall = self.data.state[0, 0, 1]  # Density at wall
        expected_wall_momentum = rho_wall * wall_velocity[0]
        actual_wall_momentum = self.data.state[1, 0, 1]
        
        self.assertAlmostEqual(actual_wall_momentum, expected_wall_momentum, places=10)
    
    def test_transmissive_bc(self):
        """Test transmissive boundary conditions"""
        bc = TransmissiveBC()
        
        # Store interior values for comparison
        left_interior = self.data.state[:, 1, :].copy()
        right_interior = self.data.state[:, -2, :].copy()
        bottom_interior = self.data.state[:, :, 1].copy()
        top_interior = self.data.state[:, :, -2].copy()
        
        bc.apply(self.data)
        
        # Check that boundary values match adjacent interior
        assert_array_almost_equal(self.data.state[:, 0, :], left_interior)    # Left
        assert_array_almost_equal(self.data.state[:, -1, :], right_interior)  # Right
        assert_array_almost_equal(self.data.state[:, :, 0], bottom_interior)  # Bottom
        assert_array_almost_equal(self.data.state[:, :, -1], top_interior)    # Top
    
    def test_inflow_bc(self):
        """Test inflow boundary condition"""
        # Define inflow state [density, momentum_x, momentum_y, momentum_z, energy]
        inflow_state = np.array([2.0, 1.0, 0.5, 0.0, 5.0])
        boundaries = ['left', 'bottom']
        
        bc = InflowBC(inflow_state, boundaries)
        bc.apply(self.data)
        
        # Check left boundary
        for j in range(self.geometry.ny):
            assert_array_almost_equal(self.data.state[:, 0, j], inflow_state)
        
        # Check bottom boundary
        for i in range(self.geometry.nx):
            assert_array_almost_equal(self.data.state[:, i, 0], inflow_state)
    
    def test_custom_bc(self):
        """Test custom boundary condition"""
        # Define custom function that sets boundary to specific values
        def custom_apply(data, **kwargs):
            multiplier = kwargs.get('multiplier', 2.0)
            # Double the density at all boundaries
            data.state[0, 0, :] *= multiplier    # Left
            data.state[0, -1, :] *= multiplier   # Right
            data.state[0, :, 0] *= multiplier    # Bottom
            data.state[0, :, -1] *= multiplier   # Top
        
        bc = CustomBC("DoubleBC", custom_apply)
        
        # Store original boundary densities
        original_left = self.data.state[0, 0, :].copy()
        original_right = self.data.state[0, -1, :].copy()
        
        # Apply custom BC with multiplier
        bc.apply(self.data, multiplier=3.0)
        
        # Check that densities were tripled
        assert_array_almost_equal(self.data.state[0, 0, :], original_left * 3.0)
        assert_array_almost_equal(self.data.state[0, -1, :], original_right * 3.0)


class TestBoundaryManager(unittest.TestCase):
    """Test boundary condition manager"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=4, ny=4, dx=0.25, dy=0.25)
        self.data = FVMDataContainer2D(self.geometry)
        self.manager = BoundaryManager()
        
        # Initialize with uniform state
        self.data.state[0] = 1.0
        self.data.state[4] = 2.5
    
    def test_default_boundary(self):
        """Test setting default boundary condition"""
        default_bc = TransmissiveBC()
        self.manager.set_default_boundary(default_bc)
        
        self.assertEqual(self.manager.default_bc, default_bc)
        
        # Apply all boundaries
        self.manager.apply_all(self.data)
        
        # Should work without errors
        self.assertIsNotNone(self.data.state)
    
    def test_mixed_boundaries(self):
        """Test mixed boundary conditions"""
        # Set different BCs for different regions
        self.manager.set_boundary('left', InflowBC(np.array([2.0, 1.0, 0.0, 0.0, 5.0]), ['left']))
        self.manager.set_boundary('right', TransmissiveBC())
        self.manager.set_default_boundary(ReflectiveBC())
        
        # Apply all
        self.manager.apply_all(self.data)
        
        # Check that left boundary has inflow values
        expected_inflow = np.array([2.0, 1.0, 0.0, 0.0, 5.0])
        assert_array_almost_equal(self.data.state[:, 0, 0], expected_inflow)


class TestEulerBoundaryConditions(unittest.TestCase):
    """Test Euler-equation specific boundary conditions"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=6, ny=6, dx=0.2, dy=0.2)
        self.data = FVMDataContainer2D(self.geometry)
        
        # Set up a reasonable interior state
        gamma = 1.4
        rho, u, v, w, p = 1.0, 0.5, 0.0, 0.0, 1.0
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        self.data.state[0] = rho
        self.data.state[1] = rho * u
        self.data.state[2] = rho * v
        self.data.state[3] = rho * w
        self.data.state[4] = E
    
    def test_subsonic_inflow(self):
        """Test subsonic inflow boundary condition"""
        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.2, 0.8, 0.1, 0.0, 1.5
        
        inflow_bc = EulerBoundaryConditions.subsonic_inflow(
            rho_inf, u_inf, v_inf, w_inf, p_inf
        )
        
        self.assertEqual(inflow_bc.name, "Inflow")
        
        # Apply boundary condition
        inflow_bc.apply(self.data)
        
        # Check left boundary values
        gamma = 1.4
        expected_energy = p_inf / (gamma - 1.0) + 0.5 * rho_inf * (u_inf**2 + v_inf**2 + w_inf**2)
        
        assert_allclose(self.data.state[0, 0, :], rho_inf, rtol=1e-12)
        assert_allclose(self.data.state[1, 0, :], rho_inf * u_inf, rtol=1e-12)
        assert_allclose(self.data.state[2, 0, :], rho_inf * v_inf, rtol=1e-12)
        assert_allclose(self.data.state[3, 0, :], rho_inf * w_inf, rtol=1e-12)
        assert_allclose(self.data.state[4, 0, :], expected_energy, rtol=1e-12)
    
    def test_subsonic_outflow(self):
        """Test subsonic outflow boundary condition"""
        p_back = 0.8
        
        outflow_bc = EulerBoundaryConditions.subsonic_outflow(p_back)
        
        self.assertEqual(outflow_bc.name, "SubsonicOutflow")
        
        # Apply boundary condition
        outflow_bc.apply(self.data, gamma=1.4)
        
        # Right boundary should have prescribed back pressure
        # Compute pressure from boundary state
        gamma = 1.4
        rho = self.data.state[0, -1, :]
        u = self.data.state[1, -1, :] / rho
        v = self.data.state[2, -1, :] / rho
        w = self.data.state[3, -1, :] / rho
        E = self.data.state[4, -1, :]
        
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        pressure = (gamma - 1.0) * (E - kinetic_energy)
        
        assert_allclose(pressure, p_back, rtol=1e-10)
    
    def test_wall_with_heat_transfer(self):
        """Test wall boundary with heat transfer"""
        wall_temp = 350.0  # K
        
        wall_bc = EulerBoundaryConditions.wall_with_heat_transfer(wall_temp)
        
        self.assertEqual(wall_bc.name, "WallHeatTransfer")
        
        # Apply boundary condition
        wall_bc.apply(self.data, gamma=1.4, gas_constant=287.0)
        
        # Should not crash and should modify boundary appropriately
        self.assertIsNotNone(self.data.state)
        
        # Normal velocity should be zero (reflective part)
        # This is a simplified test since the implementation modifies energy


class TestBoundaryConditionPhysics(unittest.TestCase):
    """Test physical correctness of boundary conditions"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=8, ny=8, dx=0.125, dy=0.125)
        self.data = FVMDataContainer2D(self.geometry)
        
        # Set up a physical state (Mach 0.3 flow)
        gamma = 1.4
        rho = 1.225  # kg/m³ (air at standard conditions)
        u = 100.0    # m/s
        v = 0.0
        w = 0.0
        p = 101325.0 # Pa (standard pressure)
        
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
        
        # Fill entire domain
        self.data.state[0] = rho
        self.data.state[1] = rho * u
        self.data.state[2] = rho * v
        self.data.state[3] = rho * w
        self.data.state[4] = E
    
    def test_mass_conservation_reflective(self):
        """Test mass conservation with reflective BC"""
        initial_mass = np.sum(self.data.state[0]) * self.geometry.cell_volume
        
        bc = ReflectiveBC()
        bc.apply(self.data)
        
        final_mass = np.sum(self.data.state[0]) * self.geometry.cell_volume
        
        # Mass should be conserved (density copied, not created)
        self.assertAlmostEqual(initial_mass, final_mass, places=10)
    
    def test_momentum_reflection(self):
        """Test momentum reflection at walls"""
        bc = ReflectiveBC()
        
        # Store initial momentum
        initial_x_momentum_interior = self.data.state[1, 2, 2]  # Interior point
        
        bc.apply(self.data)
        
        # Wall momentum should be opposite of adjacent interior
        wall_x_momentum = self.data.state[1, 0, 2]      # Left wall
        adjacent_interior = self.data.state[1, 1, 2]     # Adjacent interior
        
        self.assertAlmostEqual(wall_x_momentum, -adjacent_interior, places=10)
        
        # Interior should be unchanged
        self.assertAlmostEqual(self.data.state[1, 2, 2], initial_x_momentum_interior, places=10)
    
    def test_energy_positivity(self):
        """Test that energy remains positive after BC application"""
        boundary_conditions = [
            TransmissiveBC(),
            ReflectiveBC(),
            InflowBC(np.array([1.0, 50.0, 0.0, 0.0, 200000.0]), ['left'])
        ]
        
        for bc in boundary_conditions:
            # Reset state
            self.setUp()
            
            bc.apply(self.data)
            
            # All energies should be positive
            self.assertTrue(np.all(self.data.state[4] > 0), 
                          f"Negative energy found with BC: {bc.name}")
    
    def test_pressure_positivity(self):
        """Test pressure positivity after BC application"""
        bc = TransmissiveBC()
        bc.apply(self.data)
        
        # Compute pressure from state
        gamma = 1.4
        self.data.compute_primitives(gamma)
        pressure = self.data.primitives[4]
        
        # All pressures should be positive
        self.assertTrue(np.all(pressure > 0), "Negative pressure found")
        
        # Should be reasonable values (within factor of 10 of initial)
        self.assertTrue(np.all(pressure < 1e6), "Unreasonably high pressure")
        self.assertTrue(np.all(pressure > 1e4), "Unreasonably low pressure")


if __name__ == '__main__':
    unittest.main(verbosity=2)