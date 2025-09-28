"""
Quantum-Classical Hybrid Optimizer for Q2 Platform

This module provides cutting-edge quantum-classical hybrid computing capabilities:
- Quantum-classical hybrid optimization algorithms
- Variational quantum eigensolvers (VQE) and algorithms  
- Quantum approximate optimization algorithm (QAOA)
- Hybrid neural-quantum networks
- Quantum advantage detection and routing
- Classical-quantum resource management
"""

from .hybrid_optimizer import *

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
]