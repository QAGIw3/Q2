"""
Edge AI Deployment Manager for Q2 Platform

This module provides cutting-edge edge AI capabilities:
- Model deployment and optimization for edge devices
- Real-time inference on resource-constrained hardware
- Model compression and quantization
- Edge device fleet management
- Over-the-air model updates
- Performance monitoring and optimization
"""

from .deployment_manager import *

__all__ = [
    # Deployment Manager
    "EdgeDeploymentManager",
    "DeploymentStrategy",
    "EdgeDevice",
    "DeploymentTarget",
    "ModelDeployment",
    "DeviceCapabilities",
    "ResourceConstraints",
    "InferenceRequest",
    "InferenceResult",
]