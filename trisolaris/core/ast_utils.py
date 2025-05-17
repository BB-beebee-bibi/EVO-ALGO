"""
AST utility classes and functions for the TRISOLARIS framework.

This module provides AST-related functionality with modern Python 3.8+ compatibility,
ensuring future-proofing for Python 3.14+ where deprecated AST node types
(ast.Num, ast.Str, ast.NameConstant) will be removed.
"""

import ast
import random
import copy
from typing import List, Tuple, Union, Optional

class ModernMutationTransformer(ast.NodeTransformer):
    """
    AST transformer that applies random mutations using modern ast.Constant
    for all literal values (Python 3.8+ compatible).
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the mutation transformer.
        
        Args:
            mutation_rate: Probability of mutation for each node
        """
        self.mutation_rate = mutation_rate
        
        # Define possible literal mutations
        self.num_mutators = [
            lambda n: n + 1,
            lambda n: n - 1,
            lambda n: n * 2,
            lambda n: n // 2 if n != 0 else 1
        ]
        
        self.str_mutators = [
            lambda s: s + "X",
            lambda s: s[:-1] if len(s) > 1 else s,
            lambda s: "X" + s,
            lambda s: s.replace("a", "A").replace("e", "E")
        ]
        
        self.bool_mutators = [
            lambda b: not b
        ]
    
    def generic_visit(self, node):
        """Visit all nodes and possibly apply mutations."""
        # Call the parent method to continue traversal
        node = super().generic_visit(node)
        
        # Randomly decide whether to mutate this node
        if random.random() < self.mutation_rate:
            node = self.mutate_node(node)
        
        return node
    
    def mutate_node(self, node):
        """Apply appropriate mutation based on node type."""
        # Handle different node types with modern Python 3.8+ approach
        if isinstance(node, ast.Constant):
            # Python 3.8+ uses ast.Constant for all literals
            if isinstance(node.value, (int, float)):
                # Number mutation
                mutator = random.choice(self.num_mutators)
                try:
                    new_value = mutator(node.value)
                    return ast.Constant(value=new_value)
                except:
                    return node
            elif isinstance(node.value, str):
                # String mutation - preserve newlines and quotes
                if '\n' in node.value:
                    # For multiline strings, only mutate content between newlines
                    lines = node.value.split('\n')
                    mutated_lines = []
                    for line in lines:
                        if random.random() < self.mutation_rate:
                            mutator = random.choice(self.str_mutators)
                            try:
                                mutated_lines.append(mutator(line))
                            except:
                                mutated_lines.append(line)
                        else:
                            mutated_lines.append(line)
                    return ast.Constant(value='\n'.join(mutated_lines))
                else:
                    # For single-line strings, apply normal mutation
                    mutator = random.choice(self.str_mutators)
                    try:
                        new_value = mutator(node.value)
                        return ast.Constant(value=new_value)
                    except:
                        return node
            elif isinstance(node.value, bool):
                # Boolean mutation
                return ast.Constant(value=not node.value)
        elif isinstance(node, ast.BinOp):
            return self.mutate_binop(node)
        elif isinstance(node, ast.Compare):
            return self.mutate_compare(node)
        elif isinstance(node, ast.Name):
            return self.mutate_name(node)
        
        # Return unchanged for other node types
        return node
    
    def mutate_binop(self, node):
        """Mutate a binary operation by changing the operator."""
        op_map = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
            ast.Div: ast.Mult,
            ast.FloorDiv: ast.Div,
            ast.Mod: ast.FloorDiv
        }
        
        if type(node.op) in op_map:
            node.op = op_map[type(node.op)]()
        
        return node
    
    def mutate_compare(self, node):
        """Mutate a comparison by changing the operator."""
        if not node.ops:
            return node
            
        op_map = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.Gt,
            ast.Gt: ast.Lt,
            ast.LtE: ast.GtE,
            ast.GtE: ast.LtE,
            ast.Is: ast.IsNot,
            ast.IsNot: ast.Is
        }
        
        if type(node.ops[0]) in op_map:
            node.ops[0] = op_map[type(node.ops[0])]()
        
        return node
    
    def mutate_name(self, node):
        """Potentially add a prefix or suffix to a variable name."""
        # Be careful about changing names - only do it for certain contexts
        if random.random() < 0.1:  # Very low probability
            # Only modify names that appear to be variables, not built-ins
            if not node.id.startswith('__') and node.id not in dir(__builtins__):
                node.id = f"var_{node.id}"
        
        return node


class ModernAstCrossover:
    """
    Implements crossover operations for AST trees using modern AST nodes.
    This class is compatible with Python 3.8+ and future versions.
    """
    
    def crossover(self, tree1: ast.AST, tree2: ast.AST) -> Tuple[ast.AST, ast.AST]:
        """
        Perform crossover between two AST trees.
        
        Args:
            tree1: First parent AST
            tree2: Second parent AST
            
        Returns:
            Tuple of two child ASTs
        """
        # Create deep copies to avoid modifying the originals
        child1 = copy.deepcopy(tree1)
        child2 = copy.deepcopy(tree2)
        
        # Get all suitable subtrees for crossover
        subtrees1 = self._collect_subtrees(child1)
        subtrees2 = self._collect_subtrees(child2)
        
        if not subtrees1 or not subtrees2:
            # Not enough subtrees for crossover
            return child1, child2
        
        # Select random subtrees
        parent1_node, parent1_field_or_list, parent1_index = random.choice(subtrees1)
        parent2_node, parent2_field_or_list, parent2_index = random.choice(subtrees2)
        
        # Extract the subtrees to swap
        # If parent1_index is not None, it's a list element swap
        if parent1_index is not None:
            subtree1 = parent1_field_or_list[parent1_index]
        else: # It's an attribute swap
            subtree1 = getattr(parent1_node, parent1_field_or_list)
            
        if parent2_index is not None:
            subtree2 = parent2_field_or_list[parent2_index]
        else: # It's an attribute swap
            subtree2 = getattr(parent2_node, parent2_field_or_list)
        
        # Ensure the types are compatible for swap if possible (simple check)
        if type(subtree1) != type(subtree2):
             # If types don't match, might be risky to swap; return clones
             # A more sophisticated check could allow swapping compatible types
             return child1, child2

        # Perform the swap
        if parent1_index is not None:
            parent1_field_or_list[parent1_index] = subtree2
        else:
            setattr(parent1_node, parent1_field_or_list, subtree2)
            
        if parent2_index is not None:
            parent2_field_or_list[parent2_index] = subtree1
        else:
            setattr(parent2_node, parent2_field_or_list, subtree1)
        
        # Fix the AST structure
        ast.fix_missing_locations(child1)
        ast.fix_missing_locations(child2)
        
        return child1, child2
    
    def _collect_subtrees(self, tree: ast.AST) -> List[Tuple[ast.AST, Union[str, list], Optional[int]]]:
        """
        Collect all valid subtrees for crossover.
        Subtrees can be direct attributes or elements within list attributes.
        
        Args:
            tree: The AST to analyze
            
        Returns:
            List of tuples (parent_node, field_name_or_list, index_or_None)
        """
        collector = ModernSubtreeCollector()
        collector.visit(tree)
        return collector.subtrees


class ModernSubtreeCollector(ast.NodeVisitor):
    """
    Collects subtrees from an AST that are suitable for crossover.
    Compatible with Python 3.8+ AST nodes.
    """
    
    def __init__(self):
        # Stores tuples: (parent_node, field_name_or_list, index_or_None)
        self.subtrees: List[Tuple[ast.AST, Union[str, list], Optional[int]]] = []

    def visit(self, node):
        """Override visit to process attributes and list elements."""
        # First, visit children to collect deeper subtrees
        super().visit(node)

        # Then, process the current node's direct attributes and list elements
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                # Add direct AST attributes as potential swap points
                self.subtrees.append((node, field, None)) 
            elif isinstance(value, list) and value and all(isinstance(x, ast.AST) for x in value):
                # Add individual elements of AST lists as potential swap points
                for i, item in enumerate(value):
                     # Ensure the item is actually an AST node before adding
                    if isinstance(item, ast.AST):
                         self.subtrees.append((node, value, i)) # Store the list itself and the index 