"""
Unit tests for FVMDataContainer2D

Tests the core data container functionality including memory layout,
state management, and primitive variable computations.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose


from core.data_container import FVMDataContainer2D, GridGeometry


class TestGridGeometry(unittest.TestCase):
    """Test cases for GridGeometry class"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=10, ny=8, dx=0.1, dy=0.125, x_min=-0.5, y_min=-0.5)
    
    def test_initialization(self):
        """Test basic geometry initialization"""
        self.assertEqual(self.geometry.nx, 10)
        self.assertEqual(self.geometry.ny, 8)
        self.assertEqual(self.geometry.dx, 0.1)
        self.assertEqual(self.geometry.dy, 0.125)
        
    def test_domain_bounds(self):
        """Test domain boundary calculations"""
        self.assertEqual(self.geometry.x_max, 0.5)   # -0.5 + 10*0.1
        self.assertEqual(self.geometry.y_max, 0.5)   # -0.5 + 8*0.125
        
    def test_cell_volume(self):
        """Test cell volume calculation"""
        expected_volume = 0.1 * 0.125
        self.assertEqual(self.geometry.cell_volume, expected_volume)


class TestFVMDataContainer2D(unittest.TestCase):
    """Test cases for FVMDataContainer2D class"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=4, ny=3, dx=0.25, dy=0.5)
        self.data = FVMDataContainer2D(self.geometry, num_vars=5)
        
    def test_initialization(self):
        """Test proper initialization of data container"""
        self.assertEqual(self.data.nx, 4)
        self.assertEqual(self.data.ny, 3)
        self.assertEqual(self.data.num_vars, 5)
        self.assertEqual(self.data.total_cells, 12)
        
        # Check array shapes
        self.assertEqual(self.data.state.shape, (5, 4, 3))
        self.assertEqual(self.data.state_new.shape, (5, 4, 3))
        self.assertEqual(self.data.flux_x.shape, (5, 5, 3))  # nx+1
        self.assertEqual(self.data.flux_y.shape, (5, 4, 4))  # ny+1
        
    def test_memory_layout(self):
        """Test Structure of Arrays memory layout"""
        # Arrays should be C-contiguous for vectorization
        self.assertTrue(self.data.state.flags['C_CONTIGUOUS'])
        self.assertTrue(self.data.flux_x.flags['C_CONTIGUOUS'])
        self.assertTrue(self.data.flux_y.flags['C_CONTIGUOUS'])
        
    def test_state_management(self):
        """Test state array management operations"""
        # Initialize with test data
        self.data.state[0] = 1.0  # density
        self.data.state[1] = 2.0  # momentum_x
        
        # Test copy state
        self.data.copy_state()
        assert_array_almost_equal(self.data.state_new, self.data.state)
        
        # Test swap states
        old_state = self.data.state.copy()
        old_state_new = self.data.state_new.copy()
        self.data.swap_states()
        assert_array_almost_equal(self.data.state, old_state_new)
        assert_array_almost_equal(self.data.state_new, old_state)
        
        # Test reset
        self.data.reset_state()
        assert_array_almost_equal(self.data.state, np.zeros_like(self.data.state))
    
    def test_block_access(self):
        """Test block-wise data access"""
        # Set up test data
        test_data = np.random.random((5, 4, 3))
        self.data.state[:] = test_data
        
        # Get a block
        block = self.data.get_block(1, 3, 0, 2)
        expected = test_data[:, 1:3, 0:2]
        assert_array_almost_equal(block, expected)
        
        # Set a block
        new_block = np.ones((5, 2, 2)) * 5.0
        self.data.set_block(1, 3, 0, 2, new_block)
        assert_array_almost_equal(self.data.state[:, 1:3, 0:2], new_block)
    
    def test_primitive_variables(self):
        """Test primitive variable computation"""
        gamma = 1.4
        
        # Set up a uniform state (density=1, velocity=(1,0.5,0), pressure=1)
        rho = 1.0
        u, v, w = 1.0, 0.5, 0.0
        p = 1.0
        E = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v + w*w)
        
        self.data.state[0] = rho
        self.data.state[1] = rho * u
        self.data.state[2] = rho * v
        self.data.state[3] = rho * w
        self.data.state[4] = E
        
        # Compute primitives
        primitives = self.data.get_primitives(gamma)
        
        # Check results
        assert_allclose(primitives[0], rho, rtol=1e-12)      # density
        assert_allclose(primitives[1], u, rtol=1e-12)        # velocity_x
        assert_allclose(primitives[2], v, rtol=1e-12)        # velocity_y
        assert_allclose(primitives[3], w, rtol=1e-12)        # velocity_z
        assert_allclose(primitives[4], p, rtol=1e-12)        # pressure
    
    def test_sound_speed(self):
        """Test sound speed calculation"""
        gamma = 1.4
        
        # Set uniform state: rho=1, p=1
        self.data.state[0] = 1.0
        self.data.state[1] = 0.0
        self.data.state[2] = 0.0
        self.data.state[3] = 0.0
        self.data.state[4] = 1.0 / (gamma - 1.0)  # E = p/(γ-1) for zero velocity
        
        sound_speed = self.data.get_sound_speed(gamma)
        expected = np.sqrt(gamma * 1.0 / 1.0)  # sqrt(gamma * p / rho)
        
        assert_allclose(sound_speed, expected, rtol=1e-12)
    
    def test_max_wave_speed(self):
        """Test maximum wave speed calculation"""
        gamma = 1.4
        
        # Create non-uniform state with different velocities
        self.data.state[0] = 1.0  # density
        self.data.state[1, 0, 0] = 2.0  # high x-velocity at (0,0)
        self.data.state[2, 1, 1] = 1.5  # high y-velocity at (1,1)
        self.data.state[3] = 0.0  # zero z-velocity
        self.data.state[4] = 1.0 / (gamma - 1.0) + 0.5 * (
            self.data.state[1]**2 + self.data.state[2]**2 + self.data.state[3]**2
        )
        
        max_speed = self.data.get_max_wave_speed(gamma)
        
        # Should be positive and reasonable
        self.assertGreater(max_speed, 0.0)
        self.assertLess(max_speed, 10.0)  # Sanity check
    
    def test_conservation_error(self):
        """Test conservation error calculation"""
        # Set uniform state
        self.data.state[0] = 2.0  # density
        self.data.state[1] = 1.0  # momentum_x
        self.data.state[2] = 0.5  # momentum_y
        self.data.state[3] = 0.0  # momentum_z
        self.data.state[4] = 3.0  # energy
        
        conservation = self.data.get_conservation_error()
        
        # Expected total for each variable
        cell_volume = self.geometry.cell_volume
        total_cells = self.data.total_cells
        expected = np.array([2.0, 1.0, 0.5, 0.0, 3.0]) * cell_volume * total_cells
        
        assert_allclose(conservation, expected, rtol=1e-12)
    
    def test_boundary_conditions(self):
        """Test boundary condition application"""
        # Set interior state
        self.data.state[0, 1:-1, 1:-1] = 1.0  # Interior density
        self.data.state[1, 1:-1, 1:-1] = 0.5  # Interior x-momentum
        
        # Test transmissive BC
        self.data.apply_boundary_conditions('transmissive')
        
        # Boundary should match adjacent interior
        assert_array_almost_equal(self.data.state[:, 0, :], self.data.state[:, 1, :])  # Left
        assert_array_almost_equal(self.data.state[:, -1, :], self.data.state[:, -2, :])  # Right
        
        # Test reflective BC
        self.data.state[1, 1:-1, 1:-1] = 1.0  # Reset x-momentum
        self.data.apply_boundary_conditions('reflective')
        
        # Normal momentum should be reflected (negative)
        self.assertLess(self.data.state[1, 0, 1], 0)  # Left boundary x-momentum
        self.assertLess(self.data.state[1, -1, 1], 0)  # Right boundary x-momentum


class TestDataContainerPerformance(unittest.TestCase):
    """Performance-related tests for data container"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=10, ny=10, dx=0.1, dy=0.1)
        self.data = FVMDataContainer2D(self.geometry)
    
    def test_large_array_creation(self):
        """Test creation of larger arrays for performance"""
        large_geometry = GridGeometry(nx=100, ny=100, dx=0.01, dy=0.01)
        large_data = FVMDataContainer2D(large_geometry)
        
        self.assertEqual(large_data.total_cells, 10000)
        self.assertEqual(large_data.state.shape, (5, 100, 100))
        
        # Test that arrays are properly initialized
        self.assertTrue(np.all(large_data.state == 0))
    
    def test_memory_alignment(self):
        """Test memory alignment for vectorization"""
        # Check if arrays are properly aligned (implementation dependent)
        alignment = self.data.state.__array_interface__['data'][0]
        # Should be aligned to at least 8 bytes (double precision)
        self.assertEqual(alignment % 8, 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)