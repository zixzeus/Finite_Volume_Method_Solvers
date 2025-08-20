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
]