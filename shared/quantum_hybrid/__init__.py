"""
Quantum-Classical Hybrid Computing Module for Q2 Platform v2.0.0

Advanced quantum computing capabilities integrated with classical systems:
- Quantum Machine Learning Pipeline (QVNNs, QRL, QGANs)
- Quantum-Classical Hybrid Optimization Algorithms
- Variational quantum eigensolvers (VQE) and QAOA
- Quantum advantage detection and benchmarking
- Classical-quantum resource management
"""

from .hybrid_optimizer import *
from .quantum_ml_pipeline import (
    quantum_ml_pipeline, 
    QuantumMLAlgorithm, 
    QuantumMLPipeline,
    QuantumVariationalNeuralNetwork,
    QuantumReinforcementLearning,
    QuantumGenerativeAdversarialNetwork,
    QuantumFeatureMap
)

__version__ = "2.0.0"
__author__ = "Q2 Platform Team - 254STUDIOZ"

__all__ = [
    # Hybrid Optimizer
    "QuantumClassicalOptimizer",
    "HybridAlgorithm",
    "OptimizationProblem",
    "HybridResult",
    "QuantumBackend",
    "ClassicalBackend",
    "QuantumResource",
    "ClassicalResource",
    
    # Quantum ML Pipeline
    "quantum_ml_pipeline",
    "QuantumMLAlgorithm", 
    "QuantumMLPipeline",
    "QuantumVariationalNeuralNetwork",
    "QuantumReinforcementLearning", 
    "QuantumGenerativeAdversarialNetwork",
    "QuantumFeatureMap"
]