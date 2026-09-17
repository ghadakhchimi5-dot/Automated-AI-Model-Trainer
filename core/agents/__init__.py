"""Agentic reasoning layer for GHD Auto Trainer.

Each agent wraps a deterministic pipeline step and adds:
- Structured reasoning about its decision
- Warnings and recommendations surfaced to the user
- Adaptive adjustments based on data characteristics
"""

from .data_agent import DataAgent, DataAnalysis
from .model_agent import ModelAgent, ModelDecision
from .optimization_agent import OptimizationAgent, OptimizationPlan
from .training_agent import TrainingAgent, TrainingOutcome, TrainingHealth
from .evaluation_agent import EvaluationAgent, EvaluationVerdict

__all__ = [
    "DataAgent",
    "DataAnalysis",
    "ModelAgent",
    "ModelDecision",
    "OptimizationAgent",
    "OptimizationPlan",
    "TrainingAgent",
    "TrainingOutcome",
    "TrainingHealth",
    "EvaluationAgent",
    "EvaluationVerdict",
]
