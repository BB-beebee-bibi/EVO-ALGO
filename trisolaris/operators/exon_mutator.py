"""
Exon-like mutation operators for TRISOLARIS.

This module implements biologically-inspired mutation operators that operate at multiple
granularity levels, from function-level to fine-grained AST node mutations.
"""

import ast
import random
import copy
from typing import List, Dict, Any, Optional, Tuple
import astor
from ..utils.ast_utils import (
    get_function_nodes, get_block_nodes, get_statement_nodes,
    get_node_parent_map, get_node_children_map, are_nodes_compatible,
    get_node_dependencies, get_node_definitions
)

class ExonMutator:
    """Implements exon-like mutation operations on Python ASTs."""
    
    def __init__(self, granularity_weights: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable weights for different mutation granularities.
        
        Args:
            granularity_weights: Dictionary mapping granularity levels to probabilities.
                Default: {'function': 0.5, 'block': 0.3, 'statement': 0.15, 'node': 0.05}
        """
        self.granularity_weights = granularity_weights or {
            'function': 0.5,  # Function-level mutations
            'block': 0.3,     # Block-level mutations (if/else, loops)
            'statement': 0.15, # Statement-level mutations
            'node': 0.05      # Fine-grained AST node mutations
        }
        
        # Initialize mutation operators
        self.mutation_operators = {
            'function': [
                self.replace_function,
                self.reverse_function,
                self.chain_functions
            ],
            'block': [
                self.transpose_block,
                self.duplicate_block
            ],
            'statement': [
                self.replace_statement,
                self.swap_statements
            ],
            'node': [
                self.mutate_node
            ]
        }
    
    def select_granularity(self) -> str:
        """Select mutation granularity based on configured weights."""
        return random.choices(
            list(self.granularity_weights.keys()),
            weights=list(self.granularity_weights.values()),
            k=1
        )[0]
    
    def mutate(self, code_ast: ast.AST) -> ast.AST:
        """
        Apply a randomly selected mutation at an appropriate granularity.
        
        Args:
            code_ast: The AST to mutate
            
        Returns:
            A new AST with the mutation applied
        """
        granularity = self.select_granularity()
        mutation_op = random.choice(self.mutation_operators[granularity])
        return mutation_op(code_ast)
    
    def replace_function(self, code_ast: ast.AST) -> ast.AST:
        """
        Replace a function body with another compatible function body.
        
        Args:
            code_ast: The AST containing functions to mutate
            
        Returns:
            A new AST with a function body replaced
        """
        functions = get_function_nodes(code_ast)
        if not functions:
            return code_ast
            
        # Select a random function to replace
        target_func = random.choice(functions)
        
        # Create a new function body with compatible signature
        new_body = self._generate_compatible_function_body(target_func)
        
        # Replace the function body
        target_func.body = new_body
        return code_ast
    
    def reverse_function(self, code_ast: ast.AST) -> ast.AST:
        """
        Create an inverse operation of a selected function.
        
        Args:
            code_ast: The AST containing functions to mutate
            
        Returns:
            A new AST with a function's operations reversed
        """
        functions = get_function_nodes(code_ast)
        if not functions:
            return code_ast
            
        target_func = random.choice(functions)
        
        # Reverse the operations in the function body
        reversed_body = self._reverse_function_operations(target_func.body)
        target_func.body = reversed_body
        
        return code_ast
    
    def chain_functions(self, code_ast: ast.AST) -> ast.AST:
        """
        Chain two compatible functions in sequence.
        
        Args:
            code_ast: The AST containing functions to mutate
            
        Returns:
            A new AST with functions chained together
        """
        functions = get_function_nodes(code_ast)
        if len(functions) < 2:
            return code_ast
            
        # Select two random functions to chain
        func1, func2 = random.sample(functions, 2)
        
        # Create a new function that chains the two functions
        chained_func = self._create_chained_function(func1, func2)
        
        # Add the new function to the AST
        if isinstance(code_ast, ast.Module):
            code_ast.body.append(chained_func)
        
        return code_ast
    
    def transpose_block(self, code_ast: ast.AST) -> ast.AST:
        """
        Move a logical block to another compatible location.
        
        Args:
            code_ast: The AST containing blocks to mutate
            
        Returns:
            A new AST with a block transposed
        """
        blocks = get_block_nodes(code_ast)
        if not blocks:
            return code_ast
            
        # Select a random block to transpose
        target_block = random.choice(blocks)
        
        # Find a compatible location to move the block
        new_location = self._find_compatible_location(code_ast, target_block)
        if new_location:
            self._move_block(target_block, new_location)
        
        return code_ast
    
    def duplicate_block(self, code_ast: ast.AST) -> ast.AST:
        """
        Duplicate a logical block with potential modifications.
        
        Args:
            code_ast: The AST containing blocks to mutate
            
        Returns:
            A new AST with a block duplicated
        """
        blocks = get_block_nodes(code_ast)
        if not blocks:
            return code_ast
            
        # Select a random block to duplicate
        target_block = random.choice(blocks)
        
        # Create a modified copy of the block
        duplicate = self._create_modified_block_copy(target_block)
        
        # Insert the duplicate at a compatible location
        new_location = self._find_compatible_location(code_ast, duplicate)
        if new_location:
            self._insert_block(duplicate, new_location)
        
        return code_ast
    
    def replace_statement(self, code_ast: ast.AST) -> ast.AST:
        """
        Replace a statement with a compatible alternative.
        
        Args:
            code_ast: The AST containing statements to mutate
            
        Returns:
            A new AST with a statement replaced
        """
        statements = get_statement_nodes(code_ast)
        if not statements:
            return code_ast
            
        # Select a random statement to replace
        target_stmt = random.choice(statements)
        
        # Generate a compatible replacement statement
        new_stmt = self._generate_compatible_statement(target_stmt)
        
        # Replace the statement
        self._replace_node(target_stmt, new_stmt)
        
        return code_ast
    
    def swap_statements(self, code_ast: ast.AST) -> ast.AST:
        """
        Swap two compatible statements.
        
        Args:
            code_ast: The AST containing statements to mutate
            
        Returns:
            A new AST with two statements swapped
        """
        statements = get_statement_nodes(code_ast)
        if len(statements) < 2:
            return code_ast
            
        # Select two random statements to swap
        stmt1, stmt2 = random.sample(statements, 2)
        
        # Check if the statements are compatible for swapping
        if self._are_statements_compatible(stmt1, stmt2):
            self._swap_nodes(stmt1, stmt2)
        
        return code_ast
    
    def mutate_node(self, code_ast: ast.AST) -> ast.AST:
        """
        Apply fine-grained mutation to an AST node.
        
        Args:
            code_ast: The AST containing nodes to mutate
            
        Returns:
            A new AST with a node mutated
        """
        # Get all nodes in the AST
        nodes = list(ast.walk(code_ast))
        if not nodes:
            return code_ast
            
        # Select a random node to mutate
        target_node = random.choice(nodes)
        
        # Apply mutation based on node type
        if isinstance(target_node, ast.Num):
            target_node.n += random.randint(-10, 10)
        elif isinstance(target_node, ast.Str):
            target_node.s = target_node.s[::-1]  # Reverse string
        elif isinstance(target_node, ast.Name):
            target_node.id = f"var_{random.randint(0, 1000)}"
        
        return code_ast
    
    # Helper methods for mutation operations
    def _generate_compatible_function_body(self, target_func: ast.FunctionDef) -> List[ast.AST]:
        """Generate a new function body compatible with the target function's signature."""
        # Create a simple function body that returns a default value
        return [
            ast.Return(value=ast.Constant(value=None))
        ]
    
    def _reverse_function_operations(self, body: List[ast.AST]) -> List[ast.AST]:
        """Reverse the operations in a function body."""
        reversed_body = []
        for node in reversed(body):
            if isinstance(node, ast.If):
                # Reverse the condition
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                # Swap if and else bodies
                node.body, node.orelse = node.orelse, node.body
            reversed_body.append(node)
        return reversed_body
    
    def _create_chained_function(self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> ast.FunctionDef:
        """Create a new function that chains two functions together."""
        # Create a new function with the same signature as func1
        chained_func = ast.FunctionDef(
            name=f"chained_{func1.name}_{func2.name}",
            args=func1.args,
            body=[],
            decorator_list=[],
            returns=None
        )
        
        # Add function calls to the body
        chained_func.body.extend([
            ast.Assign(
                targets=[ast.Name(id='result', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=func1.name, ctx=ast.Load()),
                    args=[ast.Name(id='lst', ctx=ast.Load())],
                    keywords=[]
                )
            ),
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id=func2.name, ctx=ast.Load()),
                    args=[ast.Name(id='result', ctx=ast.Load())],
                    keywords=[]
                )
            )
        ])
        
        return chained_func
    
    def _find_compatible_location(self, code_ast: ast.AST, block: ast.AST) -> Optional[ast.AST]:
        """Find a compatible location to move or insert a block."""
        # Get all potential locations (other blocks)
        potential_locations = get_block_nodes(code_ast)
        if not potential_locations:
            return None
            
        # Filter for compatible locations
        compatible_locations = [
            loc for loc in potential_locations
            if are_nodes_compatible(block, loc)
        ]
        
        return random.choice(compatible_locations) if compatible_locations else None
    
    def _move_block(self, block: ast.AST, new_location: ast.AST) -> None:
        """Move a block to a new location."""
        # Get parent maps
        parent_map = get_node_parent_map(block)
        new_parent = parent_map.get(new_location)
        
        if new_parent:
            # Remove block from old location
            if block in new_parent.body:
                new_parent.body.remove(block)
            
            # Add block to new location
            new_location.body.append(block)
    
    def _create_modified_block_copy(self, block: ast.AST) -> ast.AST:
        """Create a modified copy of a block."""
        # Create a deep copy of the block
        duplicate = copy.deepcopy(block)
        
        # Modify the duplicate
        if isinstance(duplicate, ast.If):
            # Reverse the condition
            duplicate.test = ast.UnaryOp(op=ast.Not(), operand=duplicate.test)
        elif isinstance(duplicate, (ast.For, ast.While)):
            # Modify the loop variable
            if isinstance(duplicate.target, ast.Name):
                duplicate.target.id = f"i_{random.randint(0, 1000)}"
        
        return duplicate
    
    def _insert_block(self, block: ast.AST, location: ast.AST) -> None:
        """Insert a block at a specific location."""
        if hasattr(location, 'body'):
            location.body.append(block)
    
    def _generate_compatible_statement(self, target_stmt: ast.AST) -> ast.AST:
        """Generate a compatible replacement statement."""
        if isinstance(target_stmt, ast.Return):
            return ast.Return(value=ast.Constant(value=None))
        elif isinstance(target_stmt, ast.Assign):
            return ast.Assign(
                targets=[ast.Name(id=f"var_{random.randint(0, 1000)}", ctx=ast.Store())],
                value=ast.Constant(value=None)
            )
        elif isinstance(target_stmt, ast.Expr):
            return ast.Expr(value=ast.Constant(value=None))
        return target_stmt
    
    def _replace_node(self, old_node: ast.AST, new_node: ast.AST) -> None:
        """Replace a node in the AST."""
        parent_map = get_node_parent_map(old_node)
        parent = parent_map.get(old_node)
        
        if parent and hasattr(parent, 'body'):
            idx = parent.body.index(old_node)
            parent.body[idx] = new_node
    
    def _are_statements_compatible(self, stmt1: ast.AST, stmt2: ast.AST) -> bool:
        """Check if two statements are compatible for swapping."""
        return are_nodes_compatible(stmt1, stmt2)
    
    def _swap_nodes(self, node1: ast.AST, node2: ast.AST) -> None:
        """Swap two nodes in the AST."""
        parent_map = get_node_parent_map(node1)
        parent1 = parent_map.get(node1)
        parent2 = parent_map.get(node2)
        
        if parent1 and parent2 and hasattr(parent1, 'body') and hasattr(parent2, 'body'):
            idx1 = parent1.body.index(node1)
            idx2 = parent2.body.index(node2)
            
            parent1.body[idx1], parent2.body[idx2] = parent2.body[idx2], parent1.body[idx1] 