"""
Federated Learning Framework for Q2 Platform

This module provides cutting-edge federated learning capabilities:
- Distributed model training across edge devices
- Privacy-preserving machine learning
- Secure aggregation protocols
- Edge device management and orchestration
- Differential privacy mechanisms
"""

from .federation_manager import *

__all__ = [
    # Federation Manager
    "FederatedLearningManager",
    "FederationConfig",
    "TrainingRound",
    "ParticipantNode",
    "FLModelType",
    "AggregationStrategy",
    "ParticipantStatus",
    "PrivacyLevel",
    "ModelUpdate",
]