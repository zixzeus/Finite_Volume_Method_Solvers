"""
Core module for FVM Framework

This module contains the fundamental data structures and pipeline framework
for high-performance finite volume method computations.
"""

from .data_container import FVMDataContainer2D, GridGeometry
from .pipeline import (
    FVMPipeline, ComputationStage, BoundaryStage, ReconstructionStage,
    FluxStage, SourceStage, TemporalStage, PipelineMonitor
)
# Solver is imported separately to avoid circular imports
# from .solver import FVMSolver, create_blast_wave_solver, create_shock_tube_solver

__all__ = [
    'FVMDataContainer2D', 
    'GridGeometry',
    'FVMPipeline',
    'ComputationStage',
    'BoundaryStage', 
    'ReconstructionStage',
    'FluxStage',
    'SourceStage', 
    'TemporalStage',
    'PipelineMonitor'
    # 'FVMSolver',
    # 'create_blast_wave_solver', 
    # 'create_shock_tube_solver'
]