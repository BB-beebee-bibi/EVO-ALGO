"""
Evaluation module for the TRISOLARIS framework.

This module provides components for assessing solution fitness and enforcing ethical boundaries.
"""

from trisolaris.evaluation.fitness import FitnessEvaluator
from trisolaris.evaluation.ethical_filter import EthicalBoundaryEnforcer

__all__ = [
    'FitnessEvaluator',
    'EthicalBoundaryEnforcer'
] 