import ast
from typing import List, Optional, Tuple
import astor
import copy

def get_all_nodes(node: ast.AST) -> List[ast.AST]:
    """Get all nodes in the AST."""
    nodes = [node]
    for child in ast.iter_child_nodes(node):
        nodes.extend(get_all_nodes(child))
    return nodes

def get_leaf_nodes(node: ast.AST) -> List[ast.AST]:
    """Get all leaf nodes (nodes with no children) in the AST."""
    nodes = []
    for child in ast.iter_child_nodes(node):
        if len(list(ast.iter_child_nodes(child))) == 0:
            nodes.append(child)
        else:
            nodes.extend(get_leaf_nodes(child))
    return nodes

def get_subtrees(node: ast.AST, max_depth: int = 3) -> List[ast.AST]:
    """Get all subtrees up to a certain depth."""
    def get_subtrees_recursive(node: ast.AST, current_depth: int) -> List[ast.AST]:
        if current_depth >= max_depth:
            return []
        subtrees = [node]
        for child in ast.iter_child_nodes(node):
            subtrees.extend(get_subtrees_recursive(child, current_depth + 1))
        return subtrees
    return get_subtrees_recursive(node, 0)

def validate_ast(ast_tree: ast.AST) -> Tuple[bool, Optional[str]]:
    """
    Validate that an AST is valid and can be compiled to Python code.
    
    Args:
        ast_tree: The AST to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # First check if the AST can be compiled
        ast.fix_missing_locations(ast_tree)
        compile(astor.to_source(ast_tree), '<string>', 'exec')
        
        # Additional validation checks
        if not isinstance(ast_tree, ast.Module):
            return False, "Root node must be a Module"
            
        # Check for required function definition
        has_sort_files = False
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'sort_files':
                has_sort_files = True
                # Validate function signature
                if not node.args.args or node.args.args[0].arg != 'file_list':
                    return False, "sort_files function must take file_list as first argument"
                break
                
        if not has_sort_files:
            return False, "Program must contain a sort_files function"
            
        return True, None
        
    except Exception as e:
        return False, str(e)

def clone_ast(ast_node: ast.AST) -> ast.AST:
    """Clone an AST node using source round-trip to avoid recursion issues."""
    return ast.parse(astor.to_source(ast_node))

def replace_subtree(ast_root: ast.AST, target_node: ast.AST, replacement_node: ast.AST) -> ast.AST:
    """
    Replace a specific subtree in the AST with a new subtree.
    
    Args:
        ast_root: The root of the AST to modify
        target_node: The node to replace
        replacement_node: The node to insert
        
    Returns:
        Modified AST with the replacement applied
    """
    new_ast = ast.copy_location(
        ast.fix_missing_locations(clone_ast(ast_root)),
        ast_root
    )
    
    for parent in ast.walk(new_ast):
        for field, old_value in ast.iter_fields(parent):
            if isinstance(old_value, list):
                for i, item in enumerate(old_value):
                    if item is target_node:
                        old_value[i] = replacement_node
            elif old_value is target_node:
                setattr(parent, field, replacement_node)
                
    return new_ast 