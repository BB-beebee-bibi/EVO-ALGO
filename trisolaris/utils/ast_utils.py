"""
AST utility functions for TRISOLARIS.

This module provides utility functions for working with Python ASTs,
including functions to extract and manipulate different types of nodes.
"""

import ast
from typing import List, Optional, Set, Dict, Any, Tuple
from collections import defaultdict

def get_function_nodes(code_ast: ast.AST) -> List[ast.FunctionDef]:
    """
    Extract all function definition nodes from an AST.
    
    Args:
        code_ast: The AST to analyze
        
    Returns:
        List of function definition nodes
    """
    return [node for node in ast.walk(code_ast) if isinstance(node, ast.FunctionDef)]

def get_block_nodes(code_ast: ast.AST) -> List[ast.AST]:
    """
    Extract all block-level nodes from an AST.
    
    Args:
        code_ast: The AST to analyze
        
    Returns:
        List of block-level nodes (if/else, loops, try/except, etc.)
    """
    block_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)
    return [node for node in ast.walk(code_ast) if isinstance(node, block_types)]

def get_statement_nodes(code_ast: ast.AST) -> List[ast.AST]:
    """
    Extract all statement-level nodes from an AST.
    
    Args:
        code_ast: The AST to analyze
        
    Returns:
        List of statement nodes
    """
    stmt_types = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return, ast.Delete,
                 ast.Pass, ast.Break, ast.Continue, ast.Raise, ast.Assert, ast.Expr)
    return [node for node in ast.walk(code_ast) if isinstance(node, stmt_types)]

def get_node_parent_map(code_ast: ast.AST) -> Dict[ast.AST, ast.AST]:
    """
    Create a mapping of nodes to their parent nodes.
    
    Args:
        code_ast: The AST to analyze
        
    Returns:
        Dictionary mapping nodes to their parent nodes
    """
    parent_map = {}
    
    for node in ast.walk(code_ast):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node
    
    return parent_map

def get_node_children_map(code_ast: ast.AST) -> Dict[ast.AST, List[ast.AST]]:
    """
    Create a mapping of nodes to their child nodes.
    
    Args:
        code_ast: The AST to analyze
        
    Returns:
        Dictionary mapping nodes to their child nodes
    """
    children_map = defaultdict(list)
    
    for node in ast.walk(code_ast):
        for child in ast.iter_child_nodes(node):
            children_map[node].append(child)
    
    return dict(children_map)

def get_node_context(code_ast: ast.AST, target_node: ast.AST) -> Dict[str, Any]:
    """
    Get the context information for a node.
    
    Args:
        code_ast: The AST containing the node
        target_node: The node to analyze
        
    Returns:
        Dictionary containing context information:
        - parent: Parent node
        - siblings: Sibling nodes
        - depth: Depth in the AST
        - path: Path from root to node
    """
    parent_map = get_node_parent_map(code_ast)
    children_map = get_node_children_map(code_ast)
    
    context = {
        'parent': parent_map.get(target_node),
        'siblings': [],
        'depth': 0,
        'path': []
    }
    
    # Get siblings
    if context['parent'] in children_map:
        context['siblings'] = [
            child for child in children_map[context['parent']]
            if child is not target_node
        ]
    
    # Calculate depth and path
    current = target_node
    while current in parent_map:
        context['path'].append(current)
        current = parent_map[current]
        context['depth'] += 1
    
    context['path'].reverse()
    
    return context

def are_nodes_compatible(node1: ast.AST, node2: ast.AST) -> bool:
    """
    Check if two nodes are compatible for mutation operations.
    
    Two nodes are compatible if:
    1. They are of the same type
    2. They have compatible dependencies
    3. They don't create circular dependencies
    """
    if type(node1) != type(node2):
        return False
        
    # For function definitions, check argument compatibility
    if isinstance(node1, ast.FunctionDef):
        # Functions are compatible if they have the same number of arguments
        return len(node1.args.args) == len(node2.args.args)
    
    # For if statements, they are always compatible
    if isinstance(node1, ast.If):
        return True
    
    # For loops, check target compatibility
    if isinstance(node1, (ast.For, ast.While)):
        if isinstance(node1.target, ast.Name) and isinstance(node2.target, ast.Name):
            return True
        if isinstance(node1.target, ast.Tuple) and isinstance(node2.target, ast.Tuple):
            return len(node1.target.elts) == len(node2.target.elts)
        return False
    
    # For try blocks, check handler compatibility
    if isinstance(node1, ast.Try):
        return len(node1.handlers) == len(node2.handlers)
    
    # For basic statements, they are compatible if they have the same structure
    if isinstance(node1, (ast.Expr, ast.Assign, ast.AugAssign, ast.Return)):
        return True
    
    # For other nodes, check dependencies
    deps1 = get_node_dependencies(node1)
    deps2 = get_node_dependencies(node2)
    
    # Get definitions for both nodes
    defs1 = get_node_definitions(node1)
    defs2 = get_node_definitions(node2)
    
    # Check for circular dependencies
    if deps1 & defs2 or deps2 & defs1:
        return False
        
    # Check if dependencies are satisfied
    if not (deps1.issubset(defs2) and deps2.issubset(defs1)):
        return False
        
    return True

def get_node_dependencies(node: ast.AST) -> Set[str]:
    """
    Get the set of names that a node depends on.
    
    Args:
        node: The node to analyze
        
    Returns:
        Set of names that the node depends on
    """
    deps = set()
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            deps.add(child.id)
    
    return deps

def get_node_definitions(node: ast.AST) -> Set[str]:
    """
    Get the set of names that a node defines.
    
    Args:
        node: The node to analyze
        
    Returns:
        Set of names that the node defines
    """
    defs = set()
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            defs.add(child.id)
        elif isinstance(child, ast.FunctionDef):
            defs.add(child.name)
        elif isinstance(child, ast.ClassDef):
            defs.add(child.name)
    
    return defs

def get_node_scope(node: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> Optional[ast.AST]:
    """Get the scope (function or class) that contains a node."""
    current = node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, (ast.FunctionDef, ast.ClassDef)):
            return current
    return None

def get_scope_variables(scope: ast.AST) -> Set[str]:
    """Get all variable names defined in a scope."""
    return get_node_definitions(scope)

def get_scope_dependencies(scope: ast.AST) -> Set[str]:
    """Get all variable names that a scope depends on."""
    return get_node_dependencies(scope)

def is_valid_ast(node: ast.AST) -> bool:
    """Check if an AST node is valid Python syntax."""
    try:
        ast.fix_missing_locations(node)
        return True
    except:
        return False

def get_node_type(node: ast.AST) -> str:
    """Get the type name of an AST node."""
    return node.__class__.__name__

def get_node_attributes(node: ast.AST) -> Dict[str, Any]:
    """Get all attributes of an AST node."""
    return {attr: getattr(node, attr) for attr in dir(node) 
            if not attr.startswith('_') and not callable(getattr(node, attr))}

def get_node_location(node: ast.AST) -> Tuple[int, int]:
    """Get the line number and column offset of a node."""
    return (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))

def get_node_end_location(node: ast.AST) -> Tuple[int, int]:
    """Get the end line number and column offset of a node."""
    return (getattr(node, 'end_lineno', 0), getattr(node, 'end_col_offset', 0))

def get_node_source(node: ast.AST, source_code: str) -> str:
    """Get the source code corresponding to a node."""
    start_line, start_col = get_node_location(node)
    end_line, end_col = get_node_end_location(node)
    
    lines = source_code.splitlines()
    if start_line == end_line:
        return lines[start_line - 1][start_col:end_col]
    else:
        result = [lines[start_line - 1][start_col:]]
        result.extend(lines[start_line:end_line - 1])
        result.append(lines[end_line - 1][:end_col])
        return '\n'.join(result)

def get_node_docstring(node: ast.AST) -> Optional[str]:
    """Get the docstring of a node if it exists."""
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
        return None
        
    if not node.body:
        return None
        
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Str):
        return first.value.s
        
    return None

def get_node_decorators(node: ast.AST) -> List[ast.AST]:
    """Get the decorators of a node if it has any."""
    if not hasattr(node, 'decorator_list'):
        return []
    return node.decorator_list

def get_node_returns(node: ast.AST) -> Optional[ast.AST]:
    """Get the return type annotation of a node if it has one."""
    if not hasattr(node, 'returns'):
        return None
    return node.returns

def get_node_args(node: ast.AST) -> List[ast.arg]:
    """Get the arguments of a node if it has any."""
    if not hasattr(node, 'args'):
        return []
    return node.args.args

def get_node_keywords(node: ast.AST) -> List[ast.keyword]:
    """Get the keyword arguments of a node if it has any."""
    if not hasattr(node, 'keywords'):
        return []
    return node.keywords

def get_node_defaults(node: ast.AST) -> List[ast.AST]:
    """Get the default values of a node's arguments if it has any."""
    if not hasattr(node, 'defaults'):
        return []
    return node.defaults

def get_node_kw_defaults(node: ast.AST) -> List[Optional[ast.AST]]:
    """Get the default values of a node's keyword arguments if it has any."""
    if not hasattr(node, 'kw_defaults'):
        return []
    return node.kw_defaults

def get_node_vararg(node: ast.AST) -> Optional[ast.arg]:
    """Get the variable argument of a node if it has one."""
    if not hasattr(node, 'vararg'):
        return None
    return node.vararg

def get_node_kwarg(node: ast.AST) -> Optional[ast.arg]:
    """Get the keyword argument of a node if it has one."""
    if not hasattr(node, 'kwarg'):
        return None
    return node.kwarg

def get_node_posonlyargs(node: ast.AST) -> List[ast.arg]:
    """Get the positional-only arguments of a node if it has any."""
    if not hasattr(node, 'posonlyargs'):
        return []
    return node.posonlyargs

def get_node_kwonlyargs(node: ast.AST) -> List[ast.arg]:
    """Get the keyword-only arguments of a node if it has any."""
    if not hasattr(node, 'kwonlyargs'):
        return []
    return node.kwonlyargs 