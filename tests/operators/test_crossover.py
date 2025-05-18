import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ast
import logging
import pytest
from trisolaris.core.program_representation import ProgramAST
from trisolaris.core.ast_helpers import validate_ast
from hypothesis import given, settings
import hypothesis.strategies as st

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_crossover")

# Helper: Generate random valid ProgramASTs
def random_program():
    p = ProgramAST()
    for _ in range(3):
        p = p.mutate(mutation_rate=0.5)
    return p

def test_structural_equality_replacement():
    """Test that crossover uses structural equality for subtree replacement."""
    p1 = ProgramAST()
    p2 = ProgramAST()
    p1 = p1.mutate(mutation_rate=0.5)
    p2 = p2.mutate(mutation_rate=0.5)
    c1, c2 = ProgramAST.crossover(p1, p2)
    # At least one child should differ structurally from both parents
    assert ast.dump(c1.ast_tree) != ast.dump(p1.ast_tree) or ast.dump(c2.ast_tree) != ast.dump(p2.ast_tree)


def test_location_tracking_swaps_correct_subtrees():
    """Test that crossover swaps subtrees at the correct location."""
    p1 = ProgramAST()
    p2 = ProgramAST()
    p1 = p1.mutate(mutation_rate=0.5)
    p2 = p2.mutate(mutation_rate=0.5)
    # Get compatible subtrees
    subtrees1 = ProgramAST.get_compatible_subtrees(p1.ast_tree)
    subtrees2 = ProgramAST.get_compatible_subtrees(p2.ast_tree)
    assert subtrees1 and subtrees2
    # Pick a specific subtree and swap
    s1, parent1, field1, idx1 = subtrees1[0]
    s2, parent2, field2, idx2 = subtrees2[0]
    child = ProgramAST.replace_subtree_with_location(p1.ast_tree, s1, s2, parent1, field1, idx1)
    # The swapped location should now match s2
    found = False
    for node in ast.walk(child):
        for f, v in ast.iter_fields(node):
            if f == field1:
                if isinstance(v, list) and idx1 is not None and len(v) > idx1:
                    if ast.dump(v[idx1]) == ast.dump(s2):
                        found = True
                elif isinstance(v, ast.AST):
                    if ast.dump(v) == ast.dump(s2):
                        found = True
    assert found, "Subtree was not swapped at the correct location"

def test_invalid_node_types_are_never_swapped():
    """Test that invalid node types (ctx, operator tokens) are never selected for swapping."""
    p = ProgramAST()
    subtrees = ProgramAST.get_compatible_subtrees(p.ast_tree)
    for node, parent, field, idx in subtrees:
        # ctx and operator tokens should not appear
        assert not isinstance(node, (ast.Load, ast.Store, ast.Del, ast.operator, ast.cmpop, ast.boolop, ast.unaryop)), f"Invalid node type {type(node)} selected"

def test_debug_crossover_logs(tmp_path, caplog):
    """Test that DEBUG_CROSSOVER emits before/after AST dumps and code snippets."""
    os.environ['DEBUG_CROSSOVER'] = 'true'
    caplog.set_level(logging.INFO)
    p1 = random_program()
    p2 = random_program()
    c1, c2 = ProgramAST.crossover(p1, p2)
    # Check logs for AST dumps and unparse
    logs = caplog.text
    assert "Crossover debug mode enabled" in logs
    assert "Parent 1 AST:" in logs
    assert "Parent 2 AST:" in logs
    assert "Successfully produced novel offspring" in logs or "Failed to produce novel offspring" in logs
    # Clean up
    del os.environ['DEBUG_CROSSOVER']

@given(st.integers())
def test_property_based_crossover_diversity(_):
    """Property-based: Crossover should produce novel offspring at least 30% of the time."""
    p1 = random_program()
    p2 = random_program()
    c1, c2 = ProgramAST.crossover(p1, p2)
    # At least one child should be novel
    assert ast.dump(c1.ast_tree) != ast.dump(p1.ast_tree) or ast.dump(c2.ast_tree) != ast.dump(p2.ast_tree) 