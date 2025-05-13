"""
Progrémon - A CLI-based UX component for Trisolaris evolutionary algorithm framework.
Provides an intuitive interface for users to guide the evolutionary algorithm process
through natural language feedback and simple commands.
"""

from .cli import ProgrémonCLI
from .config import ProgrémonConfig
from .core import ProgrémonCore

__version__ = "0.1.0"
__all__ = ["ProgrémonCLI", "ProgrémonConfig", "ProgrémonCore"] 