"""
Task interfaces for the TRISOLARIS framework.

This module defines interfaces for evolving different types of tasks
using the TRISOLARIS evolutionary framework.
"""

from trisolaris.tasks.base import TaskInterface, TrisolarisBoundary
from trisolaris.tasks.drive_scanner import DriveScannerTask

__all__ = ['TaskInterface', 'TrisolarisBoundary', 'DriveScannerTask']
