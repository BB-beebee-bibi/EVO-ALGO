"""
Code validation and repair for TRISOLARIS.

This module provides utilities to validate and repair Python code during the
evolutionary process, ensuring that genetic operations produce valid code.
"""

import ast
import astor
import logging
from typing import Tuple, Optional, Dict, Any, List
from ..utils.ast_utils import (
    get_function_nodes, get_block_nodes, get_statement_nodes,
    are_nodes_compatible, get_node_dependencies, get_node_definitions
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeValidator:
    """Validates and repairs Python code."""
    
    def __init__(self, repair_strategies: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable repair strategy weights.
        
        Args:
            repair_strategies: Dictionary mapping repair strategies to probabilities.
                Default: {'syntax': 0.4, 'semantic': 0.3, 'structural': 0.3}
        """
        self.repair_strategies = repair_strategies or {
            'syntax': 0.4,    # Syntax-level repairs
            'semantic': 0.3,  # Semantic-level repairs
            'structural': 0.3 # Structural-level repairs
        }
    
    def is_valid_syntax(self, code_ast: ast.AST) -> bool:
        """
        Check if the AST represents valid Python syntax.
        
        Args:
            code_ast: The AST to validate
            
        Returns:
            True if the syntax is valid, False otherwise
        """
        try:
            # Try to convert back to source code
            astor.to_source(code_ast)
            return True
        except:
            return False
    
    def is_valid_semantic(self, code_ast: ast.AST) -> bool:
        """
        Check if the AST represents semantically valid code.
        
        Args:
            code_ast: The AST to validate
            
        Returns:
            True if the semantics are valid, False otherwise
        """
        try:
            # Check for undefined variables
            undefined = self._find_undefined_variables(code_ast)
            if undefined:
                return False
            
            # Check for invalid operations
            invalid_ops = self._find_invalid_operations(code_ast)
            if invalid_ops:
                return False
            
            return True
        except:
            return False
    
    def is_valid_structure(self, code_ast: ast.AST) -> bool:
        """
        Check if the AST has a valid structure.
        
        Args:
            code_ast: The AST to validate
            
        Returns:
            True if the structure is valid, False otherwise
        """
        try:
            # Check module structure
            if not isinstance(code_ast, ast.Module):
                return False
            
            # Check function structure
            for node in ast.walk(code_ast):
                if isinstance(node, ast.FunctionDef):
                    if not node.body:
                        return False
                    if not isinstance(node.body[0], ast.Expr):
                        return False
            
            return True
        except:
            return False
    
    def repair(self, code_ast: ast.AST) -> ast.AST:
        """
        Attempt to repair an invalid AST.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            A repaired AST if possible, otherwise the original AST
        """
        # Try each repair strategy in order
        for strategy in self.repair_strategies.values():
            try:
                repaired_ast = strategy(code_ast)
                if self.is_valid_syntax(repaired_ast):
                    return repaired_ast
            except Exception as e:
                logger.error(f"Repair strategy failed: {e}")
                continue
        return code_ast
    
    def _create_default_ast(self) -> ast.AST:
        """Create a default valid AST."""
        return ast.Module(
            body=[
                ast.FunctionDef(
                    name='default_function',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]
                    ),
                    body=[ast.Return(value=ast.Constant(value=None))],
                    decorator_list=[],
                    returns=None
                )
            ],
            type_ignores=[]
        )
    
    def _repair_syntax(self, code_ast: ast.AST) -> ast.AST:
        """Repair syntax-level issues."""
        # Fix missing locations
        ast.fix_missing_locations(code_ast)
        
        # Ensure all nodes have proper parent references
        for node in ast.walk(code_ast):
            if not hasattr(node, 'parent'):
                node.parent = None
        
        return code_ast
    
    def _repair_semantic(self, code_ast: ast.AST) -> ast.AST:
        """Repair semantic-level issues."""
        # Fix undefined variables
        undefined = self._find_undefined_variables(code_ast)
        for var in undefined:
            self._add_variable_definition(code_ast, var)
        
        # Fix invalid operations
        invalid_ops = self._find_invalid_operations(code_ast)
        for op in invalid_ops:
            self._fix_invalid_operation(code_ast, op)
        
        return code_ast
    
    def _repair_structural(self, code_ast: ast.AST) -> ast.AST:
        """Repair structural issues."""
        # Ensure module has proper structure
        if not isinstance(code_ast, ast.Module):
            code_ast = ast.Module(body=[code_ast], type_ignores=[])
        
        # Ensure all blocks have proper indentation
        for node in ast.walk(code_ast):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef)):
                if not node.body:
                    node.body = [ast.Pass()]
                if hasattr(node, 'orelse') and not node.orelse:
                    node.orelse = []
        
        return code_ast
    
    def _find_undefined_variables(self, code_ast: ast.AST) -> List[str]:
        """Find undefined variables in the AST."""
        undefined = set()
        defined = set()
        
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in defined:
                        undefined.add(node.id)
        
        return list(undefined)
    
    def _find_invalid_operations(self, code_ast: ast.AST) -> List[ast.AST]:
        """Find invalid operations in the AST."""
        invalid = []
        
        for node in ast.walk(code_ast):
            if isinstance(node, ast.BinOp):
                # Check for invalid arithmetic operations
                if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                    if isinstance(node.right, ast.Constant) and node.right.value == 0:
                        invalid.append(node)
            elif isinstance(node, ast.Compare):
                # Check for invalid comparisons
                if len(node.comparators) == 0:
                    invalid.append(node)
        
        return invalid
    
    def _add_variable_definition(self, code_ast: ast.AST, var_name: str) -> None:
        """Add a variable definition to the AST."""
        # Find the first function definition
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef):
                # Add variable definition at the start of the function
                node.body.insert(0, ast.Assign(
                    targets=[ast.Name(id=var_name, ctx=ast.Store())],
                    value=ast.Constant(value=None)
                ))
                break
    
    def _fix_invalid_operation(self, code_ast: ast.AST, op_node: ast.AST) -> None:
        """Fix an invalid operation in the AST."""
        if isinstance(op_node, ast.BinOp):
            # Replace division by zero with a safe value
            if isinstance(op_node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                op_node.right = ast.Constant(value=1)
        elif isinstance(op_node, ast.Compare):
            # Add a default comparator
            if len(op_node.comparators) == 0:
                op_node.comparators = [ast.Constant(value=0)] 