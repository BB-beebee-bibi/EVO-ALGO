"""
Exon-like mutation operators for TRISOLARIS.

This module implements biologically-inspired mutation operators that operate at multiple
granularity levels, from function-level to fine-grained AST node mutations.
"""

import ast
import random
import copy
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import astor
from ..utils.ast_utils import (
    get_function_nodes, get_block_nodes, get_statement_nodes,
    get_node_parent_map, get_node_children_map, are_nodes_compatible,
    get_node_dependencies, get_node_definitions
)
from .code_validator import CodeValidator

# Configure logging
logger = logging.getLogger(__name__)

class ExonMutator:
    """Implements exon-like mutation operations on Python ASTs."""
    
    def __init__(self, granularity_weights: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable mutation granularity weights.
        
        Args:
            granularity_weights: Dictionary mapping granularity levels to probabilities.
                Default: {'function': 0.4, 'block': 0.3, 'statement': 0.2, 'node': 0.1}
        """
        self.granularity_weights = granularity_weights or {
            'function': 0.4,  # Function-level mutations
            'block': 0.3,     # Block-level mutations
            'statement': 0.2, # Statement-level mutations
            'node': 0.1       # Node-level mutations
        }
        
        # Initialize mutation operators
        self.mutation_operators = {
            'function': [
                self.merge_functions,
                self.split_function,
                self.reverse_function_operations
            ],
            'block': [
                self.merge_blocks,
                self.split_block,
                self.insert_statement,
                self.remove_statement
            ],
            'statement': [
                self.swap_statements,
                self.duplicate_statement,
                self.remove_statement
            ],
            'node': [
                self.mutate_constant,
                self.swap_nodes,
                self.insert_node
            ]
        }
        
        # Initialize repair strategies
        self.repair_strategies = {
            'syntax': self._repair_syntax,
            'semantic': self._repair_semantic,
            'structural': self._repair_structure
        }
    
    def mutate(self, code_ast: ast.AST) -> ast.AST:
        """
        Apply a random mutation to the AST.
        
        Args:
            code_ast: The AST to mutate
            
        Returns:
            A new AST with the mutation applied
        """
        # Select mutation granularity
        granularity = self._select_granularity()
        
        # Select mutation operator
        operator = random.choice(self.mutation_operators[granularity])
        
        # Apply mutation
        try:
            mutated_ast = operator(code_ast)
            
            # Validate and repair if necessary
            validator = CodeValidator()
            if not validator.is_valid_syntax(mutated_ast):
                mutated_ast = self._repair_ast(mutated_ast)
            
            return mutated_ast
        except Exception as e:
            logger.error(f"Mutation failed: {str(e)}")
            return code_ast
    
    def _select_granularity(self) -> str:
        """Select mutation granularity based on weights."""
        granularities = list(self.granularity_weights.keys())
        weights = list(self.granularity_weights.values())
        return random.choices(granularities, weights=weights)[0]
    
    def mutate_constant(self, code_ast: ast.AST) -> ast.AST:
        """Mutate a constant value in the AST."""
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    # Mutate numeric constant
                    node.value = node.value + random.choice([-1, 1])
                elif isinstance(node.value, str):
                    # Mutate string constant
                    if len(node.value) > 0:
                        if random.random() < 0.5:
                            # Add character
                            node.value += random.choice('abcdefghijklmnopqrstuvwxyz')
                        else:
                            # Remove character
                            node.value = node.value[:-1]
                elif isinstance(node.value, bool):
                    # Toggle boolean
                    node.value = not node.value
                break
        return code_ast
    
    def _repair_ast(self, code_ast: ast.AST) -> ast.AST:
        """Attempt to repair an invalid AST."""
        for strategy in self.repair_strategies.values():
            try:
                repaired_ast = strategy(code_ast)
                if CodeValidator().is_valid_syntax(repaired_ast):
                    return repaired_ast
            except:
                continue
        return code_ast
    
    def _repair_syntax(self, code_ast: ast.AST) -> ast.AST:
        """Repair syntax-level issues."""
        # Create a default valid AST if the current one is invalid
        if not isinstance(code_ast, ast.Module):
            return ast.Module(body=[], type_ignores=[])
        
        # Ensure all nodes have proper parent references
        for node in ast.walk(code_ast):
            for child in ast.iter_child_nodes(node):
                if not hasattr(child, 'parent'):
                    child.parent = node
        
        return code_ast
    
    def _repair_semantic(self, code_ast: ast.AST) -> ast.AST:
        """Repair semantic-level issues."""
        # Find undefined variables
        undefined_vars = self._find_undefined_variables(code_ast)
        
        # Add variable definitions
        for var in undefined_vars:
            # Create assignment node
            assign = ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Constant(value=None)
            )
            # Add to module body
            if isinstance(code_ast, ast.Module):
                code_ast.body.insert(0, assign)
        
        return code_ast
    
    def _repair_structure(self, code_ast: ast.AST) -> ast.AST:
        """Repair structural-level issues."""
        # Ensure function definitions have proper structure
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef):
                if not node.body:
                    node.body = [ast.Pass()]
                if not node.args.args:
                    node.args.args = [ast.arg(arg='self', annotation=None)]
        
        return code_ast
    
    def _find_undefined_variables(self, code_ast: ast.AST) -> Set[str]:
        """Find undefined variables in the AST."""
        defined_vars = set()
        used_vars = set()
        
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
        
        return used_vars - defined_vars
    
    def merge_functions(self, code_ast: ast.AST) -> ast.AST:
        """Merge two compatible functions into one."""
        functions = get_function_nodes(code_ast)
        if len(functions) < 2:
            return code_ast
            
        # Select two random functions
        func1, func2 = random.sample(functions, 2)
        
        # Check if functions are compatible
        if not are_nodes_compatible(func1, func2):
            return code_ast
            
        # Merge function bodies
        merged_body = func1.body + func2.body
        
        # Create new function
        new_func = ast.FunctionDef(
            name=f"merged_{func1.name}_{func2.name}",
            args=func1.args,
            body=merged_body,
            decorator_list=[],
            returns=None
        )
        
        # Replace one function with the merged one
        func1.body = [new_func]
        
        return code_ast
    
    def split_function(self, code_ast: ast.AST) -> ast.AST:
        """
        Split a function into two parts.
        
        Args:
            code_ast: The AST containing functions to mutate
            
        Returns:
            A new AST with a function split into two parts
        """
        functions = get_function_nodes(code_ast)
        if not functions:
            return code_ast
            
        # Select a random function
        target_func = random.choice(functions)
        
        # Only split if function has enough statements
        if len(target_func.body) < 2:
            return code_ast
            
        # Split point
        split_idx = random.randint(1, len(target_func.body) - 1)
        
        # Create new function for second part
        new_func = ast.FunctionDef(
            name=f"{target_func.name}_part2",
            args=target_func.args,
            body=target_func.body[split_idx:],
            decorator_list=[],
            returns=None
        )
        
        # Update original function
        target_func.body = target_func.body[:split_idx]
        
        # Add new function to module
        if isinstance(code_ast, ast.Module):
            code_ast.body.append(new_func)
        
        return code_ast
    
    def reverse_function_operations(self, code_ast: ast.AST) -> ast.AST:
        """
        Reverse the operations in a function.
        
        Args:
            code_ast: The AST containing functions to mutate
            
        Returns:
            A new AST with a function's operations reversed
        """
        functions = get_function_nodes(code_ast)
        if not functions:
            return code_ast
            
        # Select a random function
        target_func = random.choice(functions)
        
        # Reverse the operations in the function body
        reversed_body = []
        for node in reversed(target_func.body):
            if isinstance(node, ast.If):
                # Reverse the condition
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                # Swap if and else bodies
                node.body, node.orelse = node.orelse, node.body
            reversed_body.append(node)
        
        target_func.body = reversed_body
        return code_ast
    
    def split_block(self, code_ast: ast.AST) -> ast.AST:
        """Split a block into two parts."""
        blocks = get_block_nodes(code_ast)
        if not blocks:
            return code_ast
            
        # Select a random block
        block = random.choice(blocks)
        
        # Only split if block has enough statements
        if len(block.body) < 2:
            return code_ast
            
        # Split point
        split_idx = random.randint(1, len(block.body) - 1)
        
        # Create new block
        new_block = ast.If(
            test=ast.Constant(value=True),
            body=block.body[split_idx:],
            orelse=[]
        )
        
        # Update original block
        block.body = block.body[:split_idx] + [new_block]
        
        return code_ast
    
    def merge_blocks(self, code_ast: ast.AST) -> ast.AST:
        """Merge two adjacent blocks."""
        blocks = get_block_nodes(code_ast)
        if len(blocks) < 2:
            return code_ast
            
        # Find adjacent blocks
        for i in range(len(blocks) - 1):
            block1, block2 = blocks[i], blocks[i + 1]
            
            # Check if blocks are compatible
            if are_nodes_compatible(block1, block2):
                # Merge block bodies
                block1.body.extend(block2.body)
                
                # Remove second block
                if isinstance(block1.parent, ast.Module):
                    block1.parent.body.remove(block2)
                elif isinstance(block1.parent, ast.FunctionDef):
                    block1.parent.body.remove(block2)
                
                return code_ast
        
        return code_ast
    
    def insert_statement(self, code_ast: ast.AST) -> ast.AST:
        """Insert a new statement into a block."""
        blocks = get_block_nodes(code_ast)
        if not blocks:
            return code_ast
            
        # Select a random block
        block = random.choice(blocks)
        
        # Create a simple statement
        new_stmt = ast.Assign(
            targets=[ast.Name(id='temp', ctx=ast.Store())],
            value=ast.Constant(value=0)
        )
        
        # Insert at random position
        insert_idx = random.randint(0, len(block.body))
        block.body.insert(insert_idx, new_stmt)
        
        return code_ast
    
    def remove_statement(self, code_ast: ast.AST) -> ast.AST:
        """Remove a statement from a block."""
        blocks = get_block_nodes(code_ast)
        if not blocks:
            return code_ast
            
        # Select a random block
        block = random.choice(blocks)
        
        # Only remove if block has statements
        if not block.body:
            return code_ast
            
        # Remove random statement
        remove_idx = random.randint(0, len(block.body) - 1)
        block.body.pop(remove_idx)
        
        return code_ast
    
    def swap_nodes(self, code_ast: ast.AST) -> ast.AST:
        """Swap two compatible nodes."""
        nodes = list(ast.walk(code_ast))
        if len(nodes) < 2:
            return code_ast
            
        # Select two random nodes
        node1, node2 = random.sample(nodes, 2)
        
        # Check if nodes are compatible
        if not are_nodes_compatible(node1, node2):
            return code_ast
            
        # Swap node values
        if isinstance(node1, ast.Num) and isinstance(node2, ast.Num):
            node1.n, node2.n = node2.n, node1.n
        elif isinstance(node1, ast.Str) and isinstance(node2, ast.Str):
            node1.s, node2.s = node2.s, node1.s
        elif isinstance(node1, ast.Name) and isinstance(node2, ast.Name):
            node1.id, node2.id = node2.id, node1.id
        
        return code_ast
    
    def insert_node(self, code_ast: ast.AST) -> ast.AST:
        """Insert a new node into the AST."""
        nodes = list(ast.walk(code_ast))
        if not nodes:
            return code_ast
            
        # Select a random node
        target = random.choice(nodes)
        
        # Create a simple node
        new_node = ast.Constant(value=0)
        
        # Insert as child of target
        if isinstance(target, ast.BinOp):
            target.right = new_node
        elif isinstance(target, ast.Compare):
            target.comparators.append(new_node)
        
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
    
    def duplicate_statement(self, code_ast: ast.AST) -> ast.AST:
        """
        Duplicate a random statement in a block.
        
        Args:
            code_ast: The AST containing statements to mutate
            
        Returns:
            A new AST with a statement duplicated
        """
        blocks = get_block_nodes(code_ast)
        if not blocks:
            return code_ast
        
        # Select a random block
        block = random.choice(blocks)
        if not block.body:
            return code_ast
        
        # Select a random statement to duplicate
        stmt = random.choice(block.body)
        # Insert a copy of the statement at a random position
        insert_idx = random.randint(0, len(block.body))
        block.body.insert(insert_idx, copy.deepcopy(stmt))
        return code_ast 