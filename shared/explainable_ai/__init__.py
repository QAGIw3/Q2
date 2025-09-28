"""
Explainable AI (XAI) Engine for Q2 Platform

This module provides cutting-edge explainability capabilities:
- Model interpretability and decision explanations
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature attribution and importance analysis
- Visual explanation generation
- Counterfactual explanations
- Fairness and bias detection
"""

from .explanation_engine import *

__all__ = [
    # Explanation Engine
    "ExplanationEngine",
    "ExplanationType",
    "ExplanationMethod",
    "ModelExplanation",
    "FeatureImportance",
    "SHAPExplainer",
    "LIMEExplainer",
    "PermutationExplainer",
    "CounterfactualExample",
    "BiasReport",
]