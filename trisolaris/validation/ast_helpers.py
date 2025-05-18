"""
Validation utilities for ASTs.
"""
import ast
from typing import Tuple, Optional

def validate_ast(tree: ast.AST) -> Tuple[bool, Optional[str]]:
    """
    Validate an AST for syntax and basic semantics.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check for basic syntax by compiling
        compile(tree, filename="<ast>", mode="exec")
        return True, None
    except (SyntaxError, TypeError) as e:
        return False, f"Compilation error: {str(e)}"

def fix_missing_locations(tree: ast.AST) -> ast.AST:
    """Add line numbers and column offsets to AST nodes."""
    ast.fix_missing_locations(tree)
    return tree 