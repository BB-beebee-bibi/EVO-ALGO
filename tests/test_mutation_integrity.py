import pytest
import ast
from trisolaris.core.program_representation import ProgramAST
from trisolaris.core.ast_helpers import validate_ast

def test_program_ast_initialization():
    """Test that ProgramAST initializes with valid AST."""
    program = ProgramAST()
    is_valid, error = validate_ast(program.ast_tree)
    assert is_valid, f"Invalid AST: {error}"

def test_point_mutation():
    """Test that point mutations produce valid programs."""
    program = ProgramAST()
    original_source = program.to_source()
    
    # Test multiple mutations
    for _ in range(10):
        mutated = program.mutate(mutation_rate=1.0)  # Force mutation
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after point mutation: {error}"
        assert mutated.to_source() != original_source, "Mutation did not change the program"

def test_subtree_mutation():
    """Test that subtree mutations produce valid programs."""
    program = ProgramAST()
    original_source = program.to_source()
    
    # Test multiple mutations
    for _ in range(10):
        mutated = program.mutate(mutation_rate=1.0)  # Force mutation
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after subtree mutation: {error}"
        assert mutated.to_source() != original_source, "Mutation did not change the program"

def test_functional_mutation():
    """Test that functional mutations produce valid programs."""
    program = ProgramAST()
    original_source = program.to_source()
    
    # Test multiple mutations
    for _ in range(10):
        mutated = program.mutate(mutation_rate=1.0)  # Force mutation
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after functional mutation: {error}"
        assert mutated.to_source() != original_source, "Mutation did not change the program"

def test_mutation_preserves_structure():
    """Test that mutations preserve essential program structure."""
    program = ProgramAST()
    
    # Test multiple mutations
    for _ in range(10):
        mutated = program.mutate(mutation_rate=1.0)
        
        # Check that sort_files function still exists
        has_sort_files = False
        for node in ast.walk(mutated.ast_tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'sort_files':
                has_sort_files = True
                # Check function signature
                assert node.args.args[0].arg == 'file_list', "sort_files signature changed"
                break
        assert has_sort_files, "sort_files function was lost during mutation"

def test_mutation_rate():
    """Test that mutation rate controls mutation frequency."""
    program = ProgramAST()
    original_source = program.to_source()
    
    # Test with zero mutation rate
    mutated = program.mutate(mutation_rate=0.0)
    assert mutated.to_source() == original_source, "Program changed with zero mutation rate"
    
    # Test with high mutation rate
    mutated = program.mutate(mutation_rate=1.0)
    assert mutated.to_source() != original_source, "Program did not change with high mutation rate"

def test_invalid_mutation_recovery():
    """Test that invalid mutations are caught and recovered from."""
    program = ProgramAST()
    
    # Force an invalid mutation by modifying the AST directly
    for node in ast.walk(program.ast_tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'sort_files':
            node.name = 'invalid_name'  # This should make the AST invalid
            break
    
    # Attempt to mutate the invalid program
    with pytest.raises(ValueError):
        program.mutate(mutation_rate=1.0) 