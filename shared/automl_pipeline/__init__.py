"""
AutoML Pipeline Generator for Q2 Platform

This module provides cutting-edge automated machine learning capabilities:
- Automated pipeline generation and optimization
- Neural architecture search (NAS)
- Hyperparameter optimization
- Feature engineering automation
- Model selection and ensemble generation
- Performance monitoring and auto-retraining
"""

from .pipeline_generator import *

__all__ = [
    # Pipeline Generator
    "AutoMLPipelineGenerator",
    "PipelineConfiguration",
    "MLPipeline",
    "OptimizationObjective",
    "TaskType",
    "DatasetProfile",
    "OptimizationResult",
]