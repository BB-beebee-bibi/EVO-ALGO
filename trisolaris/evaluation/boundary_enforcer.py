"""
Ethical Boundary Enforcer for the TRISOLARIS framework.

This module provides the EthicalBoundaryEnforcer class that enforces various ethical boundaries
on evolved code to prevent harmful or unethical behavior.
"""

from typing import Dict, Any, List, Optional
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EthicalBoundaryEnforcer:
    """
    Enforces ethical boundaries on evolved code.
    
    This class provides methods to check if code meets various ethical requirements
    and constraints.
    """
    
    def __init__(self):
        """Initialize the boundary enforcer."""
        self.boundaries = {}
        
    def add_boundary(self, name: str, **params):
        """
        Add a new ethical boundary.
        
        Args:
            name: Name of the boundary
            **params: Additional parameters for the boundary
        """
        self.boundaries[name] = params
    
    def check(self, genome) -> bool:
        """
        Check if a genome passes all ethical boundaries.
        
        Args:
            genome: The genome to check
            
        Returns:
            True if all boundaries pass, False otherwise
        """
        try:
            # Parse the code
            tree = ast.parse(genome.code)
            
            # Check each boundary
            for name, params in self.boundaries.items():
                if name == "no_eval_exec":
                    if not self._check_no_eval_exec(tree):
                        return False
                elif name == "max_execution_time":
                    if not self._check_execution_time(genome, params.get("max_execution_time", 1.0)):
                        return False
                elif name == "max_memory_usage":
                    if not self._check_memory_usage(genome, params.get("max_memory_usage", 100)):
                        return False
                elif name == "allow_external_libraries":
                    if not self._check_external_libraries(tree, params.get("allowed_libraries", [])):
                        return False
                else:
                    logger.warning(f"Unknown boundary: {name}")
                    continue
            
            return True
        except Exception as e:
            logger.error(f"Error checking boundaries: {e}")
            return False
    
    def _check_no_eval_exec(self, tree: ast.AST) -> bool:
        """
        Check if the code contains eval() or exec() calls.
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Expr)):
                if isinstance(node, ast.Call):
                    func = node.func
                else:
                    func = node.value
                    
                if isinstance(func, ast.Name) and func.id in ['eval', 'exec']:
                    return False
        return True
    
    def _check_execution_time(self, genome, max_time: float) -> bool:
        """
        Check if the code executes within the maximum allowed time.
        """
        try:
            import time
            start_time = time.time()
            exec(genome.code)
            execution_time = time.time() - start_time
            return execution_time <= max_time
        except Exception:
            return False
    
    def _check_memory_usage(self, genome, max_memory: float) -> bool:
        """
        Check if the code uses more than the maximum allowed memory.
        """
        try:
            import tracemalloc
            tracemalloc.start()
            exec(genome.code)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak / (1024 * 1024) <= max_memory  # Convert bytes to MB
        except Exception:
            return False
    
    def _check_external_libraries(self, tree: ast.AST, allowed_libraries: List[str]) -> bool:
        """
        Check if the code uses external libraries.
        """
        # Get all import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        
        # Check if any imports are not allowed
        for imp in imports:
            if imp not in allowed_libraries:
                return False
        
        return True
