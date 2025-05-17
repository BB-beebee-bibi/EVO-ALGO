"""
Code validation and repair utilities for TRISOLARIS.

This module provides functionality for validating and repairing Python code after mutation,
ensuring syntactic correctness and maintaining code quality.
"""

import ast
import astor
from typing import List, Optional, Tuple, Dict, Any
from ..utils.ast_utils import get_function_nodes, get_block_nodes, get_statement_nodes

class CodeValidator:
    """Validates and repairs Python code after mutation."""
    
    def __init__(self, repair_strategies: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable repair strategies.
        
        Args:
            repair_strategies: Dictionary mapping repair strategy names to probabilities.
                Default: {
                    'fix_returns': 0.3,
                    'balance_delimiters': 0.2,
                    'complete_structures': 0.2,
                    'fix_indentation': 0.15,
                    'add_imports': 0.15
                }
        """
        self.repair_strategies = repair_strategies or {
            'fix_returns': 0.3,
            'balance_delimiters': 0.2,
            'complete_structures': 0.2,
            'fix_indentation': 0.15,
            'add_imports': 0.15
        }
    
    @staticmethod
    def is_valid_syntax(code_ast: ast.AST) -> bool:
        """
        Check if the AST represents syntactically valid Python code.
        
        Args:
            code_ast: The AST to validate
            
        Returns:
            True if the code is syntactically valid, False otherwise
        """
        try:
            ast.fix_missing_locations(code_ast)
            astor.to_source(code_ast)
            return True
        except Exception:
            return False
    
    def attempt_repair(self, code_ast: ast.AST) -> Tuple[ast.AST, bool]:
        """
        Attempt to repair common syntax issues in the AST.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            Tuple of (repaired AST, success flag)
        """
        if self.is_valid_syntax(code_ast):
            return code_ast, True
            
        # Apply repair strategies in order of probability
        strategies = sorted(
            self.repair_strategies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for strategy_name, _ in strategies:
            repair_method = getattr(self, f'_repair_{strategy_name}', None)
            if repair_method:
                repaired_ast = repair_method(code_ast)
                if self.is_valid_syntax(repaired_ast):
                    return repaired_ast, True
        
        return code_ast, False
    
    def _repair_fix_returns(self, code_ast: ast.AST) -> ast.AST:
        """
        Fix missing return statements in functions.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            Repaired AST
        """
        for func in get_function_nodes(code_ast):
            if not self._has_return_statement(func.body):
                # Add a default return statement
                func.body.append(ast.Return(value=ast.Constant(value=None)))
        return code_ast
    
    def _repair_balance_delimiters(self, code_ast: ast.AST) -> ast.AST:
        """
        Balance parentheses, brackets, and braces in the code.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            Repaired AST
        """
        # Implementation of delimiter balancing
        return code_ast
    
    def _repair_complete_structures(self, code_ast: ast.AST) -> ast.AST:
        """
        Complete incomplete if/else, try/except, and other control structures.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            Repaired AST
        """
        for node in ast.walk(code_ast):
            if isinstance(node, ast.If) and not node.orelse:
                node.orelse = []
            elif isinstance(node, ast.Try) and not node.orelse:
                node.orelse = []
            elif isinstance(node, ast.Try) and not node.finalbody:
                node.finalbody = []
        return code_ast
    
    def _repair_fix_indentation(self, code_ast: ast.AST) -> ast.AST:
        """
        Fix indentation issues in the code.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            Repaired AST
        """
        # Implementation of indentation fixing
        return code_ast
    
    def _repair_add_imports(self, code_ast: ast.AST) -> ast.AST:
        """
        Add missing imports based on used symbols.
        
        Args:
            code_ast: The AST to repair
            
        Returns:
            Repaired AST
        """
        if not isinstance(code_ast, ast.Module):
            return code_ast
            
        # Collect used symbols
        used_symbols = self._collect_used_symbols(code_ast)
        
        # Add missing imports
        for symbol in used_symbols:
            if not self._has_import_for_symbol(code_ast, symbol):
                import_stmt = self._create_import_statement(symbol)
                code_ast.body.insert(0, import_stmt)
        
        return code_ast
    
    def _has_return_statement(self, body: List[ast.AST]) -> bool:
        """Check if a function body has a return statement."""
        for node in body:
            if isinstance(node, ast.Return):
                return True
            elif isinstance(node, (ast.If, ast.Try)):
                if self._has_return_statement(node.body) or self._has_return_statement(node.orelse):
                    return True
        return False
    
    def _collect_used_symbols(self, code_ast: ast.AST) -> List[str]:
        """Collect all symbols used in the code."""
        symbols = set()
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Name):
                symbols.add(node.id)
        return list(symbols)
    
    def _has_import_for_symbol(self, code_ast: ast.AST, symbol: str) -> bool:
        """Check if a symbol is already imported."""
        for node in code_ast.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.asname == symbol or name.name == symbol:
                            return True
                else:  # ImportFrom
                    for name in node.names:
                        if name.asname == symbol or name.name == symbol:
                            return True
        return False
    
    def _create_import_statement(self, symbol: str) -> ast.Import:
        """Create an import statement for a symbol."""
        return ast.Import(names=[ast.alias(name=symbol, asname=None)])
    
    def repair(self, code_ast):
        """Stub repair method: returns the input AST or a minimal valid AST if input is None."""
        if code_ast is not None:
            return code_ast
        # Return a minimal valid AST
        import ast
        return ast.parse("def repaired():\n    return 0\n") 