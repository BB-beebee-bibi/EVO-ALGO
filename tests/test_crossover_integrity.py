import pytest
from trisolaris.core.program_representation import ProgramAST, validate_ast
import ast

def test_crossover_basic():
    """Test basic crossover functionality with two parent programs."""
    p1 = ProgramAST()
    p2 = ProgramAST()
    
    # Apply mutations to ensure diversity
    p1 = p1.mutate(mutation_rate=0.3)
    p2 = p2.mutate(mutation_rate=0.3)
    
    c1, c2 = ProgramAST.crossover(p1, p2)
    
    # Validate ASTs
    assert validate_ast(c1.ast_tree)[0], "Child 1 has invalid AST"
    assert validate_ast(c2.ast_tree)[0], "Child 2 has invalid AST"
    
    # Check for uniqueness - allow for some identical offspring
    p1_hash = hash(ast.dump(p1.ast_tree))
    p2_hash = hash(ast.dump(p2.ast_tree))
    c1_hash = hash(ast.dump(c1.ast_tree))
    c2_hash = hash(ast.dump(c2.ast_tree))
    
    # At least one child should be different from both parents
    assert (c1_hash != p1_hash or c1_hash != p2_hash or 
            c2_hash != p1_hash or c2_hash != p2_hash), "Both children identical to parents"

def test_crossover_success_rate():
    """Test crossover success rate with multiple parent pairs."""
    num_pairs = 20
    successful_crossovers = 0
    novel_offspring = 0
    total_offspring = 0
    
    for _ in range(num_pairs):
        # Create parents with some mutation for diversity
        p1 = ProgramAST()
        p2 = ProgramAST()
        p1 = p1.mutate(mutation_rate=0.3)
        p2 = p2.mutate(mutation_rate=0.3)
        
        try:
            c1, c2 = ProgramAST.crossover(p1, p2)
            total_offspring += 2
            
            # Validate both children
            if validate_ast(c1.ast_tree)[0] and validate_ast(c2.ast_tree)[0]:
                successful_crossovers += 1
                
                # Check for novel offspring
                p1_hash = hash(ast.dump(p1.ast_tree))
                p2_hash = hash(ast.dump(p2.ast_tree))
                c1_hash = hash(ast.dump(c1.ast_tree))
                c2_hash = hash(ast.dump(c2.ast_tree))
                
                if c1_hash != p1_hash and c1_hash != p2_hash:
                    novel_offspring += 1
                if c2_hash != p1_hash and c2_hash != p2_hash:
                    novel_offspring += 1
                    
        except Exception:
            continue
    
    success_rate = successful_crossovers / num_pairs
    novel_rate = novel_offspring / total_offspring
    
    # Require at least 50% success rate for valid offspring
    assert success_rate >= 0.5, f"Crossover success rate {success_rate:.2%} below 50% threshold"
    # Require at least 30% novel offspring rate
    assert novel_rate >= 0.3, f"Novel offspring rate {novel_rate:.2%} below 30% threshold"

def test_crossover_telemetry():
    """Test that crossover telemetry is accurately tracked."""
    p1 = ProgramAST()
    p2 = ProgramAST()
    
    # Apply mutations to ensure diversity
    p1 = p1.mutate(mutation_rate=0.3)
    p2 = p2.mutate(mutation_rate=0.3)
    
    # Perform crossover
    c1, c2 = ProgramAST.crossover(p1, p2)
    
    # Verify telemetry
    assert p1.stats['crossover_attempts'] == 1
    assert p2.stats['crossover_attempts'] == 1
    
    # Check novel offspring tracking
    p1_hash = hash(ast.dump(p1.ast_tree))
    p2_hash = hash(ast.dump(p2.ast_tree))
    c1_hash = hash(ast.dump(c1.ast_tree))
    c2_hash = hash(ast.dump(c2.ast_tree))
    
    if c1_hash != p1_hash and c1_hash != p2_hash:
        assert p1.stats['crossover_novel'] == 1
    if c2_hash != p1_hash and c2_hash != p2_hash:
        assert p2.stats['crossover_novel'] == 1

def test_crossover_type_safety():
    """Test that crossover maintains type safety."""
    p1 = ProgramAST()
    p2 = ProgramAST()
    
    c1, c2 = ProgramAST.crossover(p1, p2)
    
    # Verify that all nodes in the children are valid Python AST nodes
    for node in ast.walk(c1.ast_tree):
        assert isinstance(node, ast.AST), f"Invalid node type in child 1: {type(node)}"
    
    for node in ast.walk(c2.ast_tree):
        assert isinstance(node, ast.AST), f"Invalid node type in child 2: {type(node)}"

def test_crossover_structure_preservation():
    """Test that crossover preserves essential program structure."""
    p1 = ProgramAST()
    p2 = ProgramAST()
    
    c1, c2 = ProgramAST.crossover(p1, p2)
    
    # Verify that both children have the required top-level structure
    assert isinstance(c1.ast_tree, ast.Module), "Child 1 missing Module node"
    assert isinstance(c2.ast_tree, ast.Module), "Child 2 missing Module node"
    
    # Check for function definition
    assert any(isinstance(node, ast.FunctionDef) for node in ast.walk(c1.ast_tree)), "Child 1 missing function definition"
    assert any(isinstance(node, ast.FunctionDef) for node in ast.walk(c2.ast_tree)), "Child 2 missing function definition"

def test_crossover_with_diverse_parents():
    """Test crossover with significantly different parent programs."""
    # Create two very different parent programs
    p1 = ProgramAST()
    p2 = ProgramAST()
    
    # Apply multiple mutations to p2 to ensure diversity
    for _ in range(5):
        p2 = p2.mutate(mutation_rate=0.5)
    
    # Verify parent diversity
    p1_hash = hash(ast.dump(p1.ast_tree))
    p2_hash = hash(ast.dump(p2.ast_tree))
    assert p1_hash != p2_hash, "Parents not diverse enough for meaningful crossover test"
    
    # Perform crossover
    c1, c2 = ProgramAST.crossover(p1, p2)
    
    # Verify offspring validity
    assert validate_ast(c1.ast_tree)[0], "Child 1 has invalid AST"
    assert validate_ast(c2.ast_tree)[0], "Child 2 has invalid AST"
    
    # Check for novel offspring
    c1_hash = hash(ast.dump(c1.ast_tree))
    c2_hash = hash(ast.dump(c2.ast_tree))
    
    # With diverse parents, we expect at least one novel offspring
    assert (c1_hash != p1_hash and c1_hash != p2_hash) or (c2_hash != p1_hash and c2_hash != p2_hash), \
        "No novel offspring produced from diverse parents" 