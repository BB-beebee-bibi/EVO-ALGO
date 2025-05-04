"""
Managers package for the TRISOLARIS framework.

This package provides components for managing various aspects of the evolutionary process,
including resources, diversity, and island ecosystems.
"""

from trisolaris.managers.resource import ResourceSteward
from trisolaris.managers.diversity import DiversityGuardian
from trisolaris.managers.island import IslandEcosystemManager
from trisolaris.managers.repository import GenomeRepository

__all__ = [
    'ResourceSteward',
    'DiversityGuardian',
    'IslandEcosystemManager',
    'GenomeRepository'
]
