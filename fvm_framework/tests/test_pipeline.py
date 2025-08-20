"""
Unit tests for FVM Pipeline Framework

Tests the pipeline orchestration, stage execution, and performance monitoring.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import time

from ..core.pipeline import (
    ComputationStage, BoundaryStage, ReconstructionStage, 
    FluxStage, SourceStage, TemporalStage, FVMPipeline, PipelineMonitor
)
from ..core.data_container import FVMDataContainer2D, GridGeometry


class MockStage(ComputationStage):
    """Mock computation stage for testing"""
    
    def __init__(self, name: str, execution_time: float = 0.001):
        super().__init__(name)
        self.execution_time = execution_time
        self.process_calls = 0
        
    def process(self, data: FVMDataContainer2D, **kwargs) -> None:
        def _mock_process():
            time.sleep(self.execution_time)  # Simulate computation
            self.process_calls += 1
            
        self._time_execution(_mock_process)


class TestComputationStage(unittest.TestCase):
    """Test base computation stage functionality"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=4, ny=4, dx=0.25, dy=0.25)
        self.data = FVMDataContainer2D(self.geometry)
        self.stage = MockStage("TestStage", execution_time=0.002)
    
    def test_stage_creation(self):
        """Test stage creation and initialization"""
        self.assertEqual(self.stage.name, "TestStage")
        self.assertEqual(self.stage._call_count, 0)
        self.assertEqual(self.stage._execution_time, 0.0)
    
    def test_stage_execution(self):
        """Test stage execution and timing"""
        # Execute stage
        self.stage.process(self.data)
        
        # Check that process was called
        self.assertEqual(self.stage.process_calls, 1)
        self.assertEqual(self.stage._call_count, 1)
        self.assertGreater(self.stage._execution_time, 0.001)  # Should be at least 2ms
    
    def test_performance_stats(self):
        """Test performance statistics collection"""
        # Execute multiple times
        for _ in range(3):
            self.stage.process(self.data)
        
        stats = self.stage.get_performance_stats()
        
        self.assertEqual(stats['call_count'], 3)
        self.assertGreater(stats['total_time'], 0.005)  # At least 6ms total
        self.assertGreater(stats['avg_time'], 0.001)    # At least 2ms average


class TestBoundaryStage(unittest.TestCase):
    """Test boundary condition stage"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=4, ny=4, dx=0.25, dy=0.25)
        self.data = FVMDataContainer2D(self.geometry)
        self.data.state[0] = 1.0  # Set uniform density
    
    def test_periodic_boundary(self):
        """Test periodic boundary condition application"""
        stage = BoundaryStage('periodic')
        stage.process(self.data)
        
        # For periodic BC, should not change interior values significantly
        self.assertTrue(np.allclose(self.data.state[0, 1:-1, 1:-1], 1.0))
    
    def test_reflective_boundary(self):
        """Test reflective boundary condition"""
        # Set up momentum
        self.data.state[1, :, :] = 0.5  # x-momentum
        
        stage = BoundaryStage('reflective')
        stage.process(self.data)
        
        # Performance measurement should work
        stats = stage.get_performance_stats()
        self.assertEqual(stats['call_count'], 1)
    
    def test_transmissive_boundary(self):
        """Test transmissive boundary condition"""
        stage = BoundaryStage('transmissive')
        stage.process(self.data)
        
        stats = stage.get_performance_stats()
        self.assertEqual(stats['call_count'], 1)


class TestFVMPipeline(unittest.TestCase):
    """Test complete FVM pipeline"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=8, ny=8, dx=0.125, dy=0.125)
        self.data = FVMDataContainer2D(self.geometry)
        
        # Initialize with simple uniform state
        self.data.state[0] = 1.0   # density
        self.data.state[1] = 0.1   # momentum_x
        self.data.state[2] = 0.05  # momentum_y
        self.data.state[3] = 0.0   # momentum_z
        self.data.state[4] = 2.5   # energy
    
    def test_pipeline_creation(self):
        """Test pipeline creation with default stages"""
        pipeline = FVMPipeline()
        
        self.assertEqual(len(pipeline.stages), 5)
        self.assertEqual(pipeline.stages[0].name, "BoundaryConditions")
        self.assertEqual(pipeline.stages[1].name, "SpatialReconstruction")
        self.assertEqual(pipeline.stages[2].name, "FluxComputation")
        self.assertEqual(pipeline.stages[3].name, "SourceTerms")
        self.assertEqual(pipeline.stages[4].name, "TemporalIntegration")
    
    def test_pipeline_custom_configuration(self):
        """Test pipeline with custom configuration"""
        pipeline = FVMPipeline(
            boundary_type='reflective',
            reconstruction_type='weno',
            riemann_solver='hll',
            time_scheme='rk4'
        )
        
        self.assertEqual(len(pipeline.stages), 5)
        # Check that configuration was applied (stages should exist)
        self.assertIsNotNone(pipeline.stages[0])  # BoundaryStage
        self.assertIsNotNone(pipeline.stages[2])  # FluxStage
    
    def test_single_time_step(self):
        """Test execution of single time step"""
        pipeline = FVMPipeline()
        initial_state = self.data.state.copy()
        
        # Execute time step
        dt = 0.001
        pipeline.execute_time_step(self.data, dt, gamma=1.4)
        
        # Pipeline should have executed
        self.assertEqual(pipeline.total_steps, 1)
        self.assertGreater(pipeline.total_time, 0.0)
    
    def test_multiple_time_steps(self):
        """Test multiple time step execution"""
        pipeline = FVMPipeline()
        
        # Execute multiple steps
        dt = 0.001
        num_steps = 5
        
        for _ in range(num_steps):
            pipeline.execute_time_step(self.data, dt)
        
        self.assertEqual(pipeline.total_steps, num_steps)
        self.assertGreater(pipeline.total_time, 0.0)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        pipeline = FVMPipeline()
        
        # Execute some steps
        for _ in range(3):
            pipeline.execute_time_step(self.data, 0.001)
        
        summary = pipeline.get_performance_summary()
        
        self.assertEqual(summary['total_steps'], 3)
        self.assertGreater(summary['total_time'], 0.0)
        self.assertGreater(summary['avg_step_time'], 0.0)
        self.assertIn('stage_breakdown', summary)
        self.assertEqual(len(summary['stage_breakdown']), 5)
    
    def test_performance_reset(self):
        """Test performance counter reset"""
        pipeline = FVMPipeline()
        
        # Execute steps
        pipeline.execute_time_step(self.data, 0.001)
        self.assertGreater(pipeline.total_steps, 0)
        
        # Reset and check
        pipeline.reset_performance_counters()
        self.assertEqual(pipeline.total_steps, 0)
        self.assertEqual(pipeline.total_time, 0.0)


class TestPipelineMonitor(unittest.TestCase):
    """Test pipeline monitoring functionality"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=6, ny=6, dx=1.0/6, dy=1.0/6)
        self.data = FVMDataContainer2D(self.geometry)
        self.monitor = PipelineMonitor(monitor_interval=2)
        
        # Set up a conservation-preserving state
        self.data.state[0] = 1.0   # density
        self.data.state[1] = 0.0   # zero momentum
        self.data.state[2] = 0.0
        self.data.state[3] = 0.0
        self.data.state[4] = 1.0   # energy
    
    def test_monitor_creation(self):
        """Test monitor creation"""
        self.assertEqual(self.monitor.monitor_interval, 2)
        self.assertEqual(self.monitor.step_count, 0)
        self.assertEqual(len(self.monitor.conservation_history), 0)
    
    def test_monitor_update(self):
        """Test monitor update functionality"""
        dt = 0.01
        
        # Update multiple times
        for i in range(5):
            self.monitor.update(self.data, dt)
        
        self.assertEqual(self.monitor.step_count, 5)
        # Should have 2 conservation measurements (at steps 2 and 4)
        self.assertEqual(len(self.monitor.conservation_history), 2)
        self.assertEqual(len(self.monitor.max_wave_speed_history), 2)
    
    def test_conservation_drift(self):
        """Test conservation drift calculation"""
        dt = 0.01
        
        # Update to get initial measurement
        for _ in range(2):
            self.monitor.update(self.data, dt)
        
        # Slightly modify state (simulate numerical drift)
        self.data.state[0] *= 1.0001
        
        # Update again to get second measurement
        for _ in range(2):
            self.monitor.update(self.data, dt)
        
        drift = self.monitor.get_conservation_drift()
        self.assertEqual(len(drift), 5)  # 5 conservation variables
        self.assertGreater(drift[0], 0.0)  # Should detect drift in density


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests combining pipeline and monitor"""
    
    def setUp(self):
        self.geometry = GridGeometry(nx=10, ny=10, dx=0.1, dy=0.1)
        self.data = FVMDataContainer2D(self.geometry)
        
        # Set up blast wave initial condition
        center_x, center_y = self.geometry.nx // 2, self.geometry.ny // 2
        for i in range(self.geometry.nx):
            for j in range(self.geometry.ny):
                r = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if r < 3:
                    # High pressure region
                    self.data.state[0, i, j] = 1.0      # density
                    self.data.state[4, i, j] = 10.0     # high energy
                else:
                    # Low pressure region
                    self.data.state[0, i, j] = 0.125    # density
                    self.data.state[4, i, j] = 1.0      # low energy
    
    def test_pipeline_with_monitor(self):
        """Test pipeline execution with monitoring"""
        pipeline = FVMPipeline(boundary_type='transmissive')
        monitor = PipelineMonitor(monitor_interval=5)
        
        # Simulate time evolution
        dt = 0.0001
        num_steps = 20
        
        for step in range(num_steps):
            pipeline.execute_time_step(self.data, dt, gamma=1.4)
            monitor.update(self.data, dt, gamma=1.4)
        
        # Check that both pipeline and monitor ran
        self.assertEqual(pipeline.total_steps, num_steps)
        self.assertEqual(monitor.step_count, num_steps)
        
        # Should have some conservation measurements
        expected_measurements = num_steps // monitor.monitor_interval
        self.assertEqual(len(monitor.conservation_history), expected_measurements)
        
        # Get final summary
        summary = pipeline.get_performance_summary()
        self.assertIn('total_steps', summary)
        self.assertIn('stage_breakdown', summary)


if __name__ == '__main__':
    # Run with high verbosity to see detailed test results
    unittest.main(verbosity=2)