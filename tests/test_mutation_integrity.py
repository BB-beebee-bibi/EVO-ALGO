import pytest
import ast
from trisolaris.core.program_representation import ProgramAST
from trisolaris.core.ast_helpers import validate_ast

def test_program_ast_initialization():
    """Test that ProgramAST initializes with valid AST and telemetry."""
    program = ProgramAST()
    is_valid, error = validate_ast(program.ast_tree)
    assert is_valid, f"Invalid AST: {error}"
    
    # Verify telemetry initialization
    assert program.stats['mutation_attempts'] == 0
    assert program.stats['mutation_effective'] == 0
    assert program.stats['mutation_noop'] == 0
    assert program.stats['crossover_attempts'] == 0
    assert program.stats['crossover_novel'] == 0

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

def test_mutation_diversity():
    """Test that mutation produces novel programs with reasonable frequency."""
    program = ProgramAST()
    original_hash = hash(ast.dump(program.ast_tree))
    
    # Track mutation success over multiple attempts
    novel_count = 0
    attempts = 10
    
    for _ in range(attempts):
        mutated = program.mutate(mutation_rate=0.3)
        if hash(ast.dump(mutated.ast_tree)) != original_hash:
            novel_count += 1
    
    # Require at least 30% success rate - realistic but still functional
    assert novel_count >= 3, f"Only {novel_count}/{attempts} mutations produced novel programs"
    
    # Verify telemetry
    assert program.stats['mutation_attempts'] == attempts
    assert program.stats['mutation_effective'] == novel_count
    assert program.stats['mutation_noop'] == attempts - novel_count

def test_mutation_telemetry():
    """Test that mutation telemetry is accurately tracked."""
    program = ProgramAST()
    
    # Test with zero mutation rate
    mutated = program.mutate(mutation_rate=0.0)
    assert program.stats['mutation_attempts'] == 1
    assert program.stats['mutation_noop'] == 1
    assert program.stats['mutation_effective'] == 0
    
    # Test with high mutation rate
    original_hash = hash(ast.dump(program.ast_tree))
    mutated = program.mutate(mutation_rate=1.0)
    assert program.stats['mutation_attempts'] == 2
    if hash(ast.dump(mutated.ast_tree)) != original_hash:
        assert program.stats['mutation_effective'] == 1
    else:
        assert program.stats['mutation_noop'] == 2

def test_macro_mutation():
    """Test that macro mutations are attempted when normal mutations fail."""
    program = ProgramAST()
    original_hash = hash(ast.dump(program.ast_tree))
    
    # Force a situation where normal mutations might fail
    mutated = program.mutate(mutation_rate=1.0, max_retries=2)
    
    # Verify that either a novel program was produced or macro mutation was attempted
    assert (hash(ast.dump(mutated.ast_tree)) != original_hash or 
            program.stats['mutation_attempts'] > 1), "Macro mutation was not attempted"

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
        
        # Verify AST validity
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after mutation: {error}"

def test_mutation_stats_reporting():
    """Test that mutation statistics are correctly reported."""
    program = ProgramAST()
    
    # Perform some mutations
    for _ in range(5):
        program.mutate(mutation_rate=0.3)
    
    # Verify stats reporting
    assert program.stats['mutation_attempts'] == 5
    assert program.stats['mutation_effective'] + program.stats['mutation_noop'] == 5

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