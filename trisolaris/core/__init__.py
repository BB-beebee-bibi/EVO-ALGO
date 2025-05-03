"""
Core module for the TRISOLARIS framework.

This module provides the fundamental components for evolutionary computation,
including the evolution engine, code genome representation, and adaptive landscape.
"""

from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import CodeGenome
from trisolaris.core.landscape import AdaptiveLandscape

__all__ = [
    'EvolutionEngine',
    'CodeGenome',
    'AdaptiveLandscape'
] 