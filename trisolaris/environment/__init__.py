"""
Environment module for the TRISOLARIS framework.

This module provides sandboxed execution environments and resource simulation
for safe evolution of code.
"""

from trisolaris.environment.sandbox import SandboxedEnvironment, SandboxViolationError, ResourceLimitExceededError, ExecutionTimeoutError
from trisolaris.environment.simulator import ResourceSimulator, SimulatedFileSystem, SimulatedNetwork, SimulatedResources, SimulatedIO

__all__ = [
    'SandboxedEnvironment',
    'SandboxViolationError',
    'ResourceLimitExceededError',
    'ExecutionTimeoutError',
    'ResourceSimulator',
    'SimulatedFileSystem',
    'SimulatedNetwork',
    'SimulatedResources',
    'SimulatedIO'
]