"""
Tests for the exon-like mutation system.

This module contains tests for the ExonMutator, CodeValidator, and ExonEvolutionEngine
classes, verifying their functionality and integration.
"""

import ast
import pytest
from trisolaris.operators import ExonMutator, CodeValidator, ExonEvolutionEngine
from trisolaris.utils.ast_utils import (
    get_function_nodes, get_block_nodes, get_statement_nodes,
    are_nodes_compatible, get_node_dependencies, get_node_definitions
)

# Sample code for testing
SAMPLE_CODE = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

def filter_data(data):
    return [x for x in data if x > 0]

def combine_results(a, b):
    return a + b
"""

def test_exon_mutator_initialization():
    """Test ExonMutator initialization with default and custom weights."""
    # Test with default weights
    mutator = ExonMutator()
    assert mutator.granularity_weights == {
        'function': 0.5,
        'block': 0.3,
        'statement': 0.15,
        'node': 0.05
    }
    
    # Test with custom weights
    custom_weights = {
        'function': 0.7,
        'block': 0.2,
        'statement': 0.08,
        'node': 0.02
    }
    mutator = ExonMutator(granularity_weights=custom_weights)
    assert mutator.granularity_weights == custom_weights

def test_code_validator_initialization():
    """Test CodeValidator initialization with default and custom strategies."""
    # Test with default strategies
    validator = CodeValidator()
    assert validator.repair_strategies['fix_returns'] == 0.3
    assert validator.repair_strategies['balance_delimiters'] == 0.2
    assert validator.repair_strategies['complete_structures'] == 0.2
    assert validator.repair_strategies['fix_indentation'] == 0.15
    assert validator.repair_strategies['add_imports'] == 0.15
    
    # Test with custom strategies
    custom_strategies = {
        'fix_returns': 0.4,
        'balance_delimiters': 0.3,
        'complete_structures': 0.2,
        'fix_indentation': 0.05,
        'add_imports': 0.05
    }
    validator = CodeValidator(custom_strategies)
    assert validator.repair_strategies == custom_strategies

def test_evolution_engine_initialization():
    """Test ExonEvolutionEngine initialization with default and custom config."""
    # Test with default config
    engine = ExonEvolutionEngine()
    assert engine.population_size == 100
    assert engine.invalid_retention_rate == 0.1
    
    # Test with custom config
    custom_config = {
        'population_size': 50,
        'invalid_retention_rate': 0.2,
        'granularity_weights': {
            'function': 0.6,
            'block': 0.2,
            'statement': 0.15,
            'node': 0.05
        }
    }
    engine = ExonEvolutionEngine(custom_config)
    assert engine.population_size == 50
    assert engine.invalid_retention_rate == 0.2
    assert engine.mutator.granularity_weights == custom_config['granularity_weights']

def test_function_mutation():
    """Test function-level mutations."""
    # Create a simple function AST
    source = """
def add(a, b):
    return a + b
"""
    code_ast = ast.parse(source)
    mutator = ExonMutator()
    
    # Test function replacement
    mutated_ast = mutator.replace_function(code_ast)
    functions = get_function_nodes(mutated_ast)
    assert len(functions) == 1
    assert isinstance(functions[0].body[0], ast.Return)
    
    # Test function reversal
    mutated_ast = mutator.reverse_function(code_ast)
    functions = get_function_nodes(mutated_ast)
    assert len(functions) == 1
    assert isinstance(functions[0].body[0], ast.Return)

def test_block_mutation():
    """Test block-level mutations."""
    # Create a function with blocks
    source = """
def process_list(lst):
    result = []
    for item in lst:
        if item > 0:
            result.append(item)
    return result
"""
    code_ast = ast.parse(source)
    mutator = ExonMutator()
    
    # Test block transposition
    mutated_ast = mutator.transpose_block(code_ast)
    blocks = get_block_nodes(mutated_ast)
    assert len(blocks) > 0
    
    # Test block duplication
    mutated_ast = mutator.duplicate_block(code_ast)
    blocks = get_block_nodes(mutated_ast)
    assert len(blocks) > 0

def test_statement_mutation():
    """Test statement-level mutations."""
    # Create a function with multiple statements
    source = """
def calculate(x, y):
    a = x + y
    b = x - y
    return a * b
"""
    code_ast = ast.parse(source)
    mutator = ExonMutator()
    
    # Test statement replacement
    mutated_ast = mutator.replace_statement(code_ast)
    statements = get_statement_nodes(mutated_ast)
    assert len(statements) > 0
    
    # Test statement swapping
    mutated_ast = mutator.swap_statements(code_ast)
    statements = get_statement_nodes(mutated_ast)
    assert len(statements) > 0

def test_node_mutation():
    """Test fine-grained node mutations."""
    # Create a simple expression
    source = """
def test():
    x = 42
    y = "hello"
    z = x + len(y)
"""
    code_ast = ast.parse(source)
    mutator = ExonMutator()
    
    # Test node mutation
    mutated_ast = mutator.mutate_node(code_ast)
    assert isinstance(mutated_ast, ast.Module)

def test_node_compatibility():
    """Test node compatibility checking."""
    # Create two compatible functions
    source1 = """
def func1(x):
    return x + 1
"""
    source2 = """
def func2(y):
    return y - 1
"""
    ast1 = ast.parse(source1)
    ast2 = ast.parse(source2)
    
    func1 = get_function_nodes(ast1)[0]
    func2 = get_function_nodes(ast2)[0]
    
    assert are_nodes_compatible(func1, func2)
    
    # Create incompatible functions
    source3 = """
def func3(x, y):
    return x + y
"""
    ast3 = ast.parse(source3)
    func3 = get_function_nodes(ast3)[0]
    
    assert not are_nodes_compatible(func1, func3)

def test_dependency_analysis():
    """Test dependency and definition analysis."""
    source = """
def process(x, y):
    z = x + y
    result = z * 2
    return result
"""
    code_ast = ast.parse(source)
    func = get_function_nodes(code_ast)[0]
    
    # Test dependencies
    deps = get_node_dependencies(func)
    assert 'x' in deps
    assert 'y' in deps
    assert 'z' in deps
    assert 'result' in deps
    
    # Test definitions
    defs = get_node_definitions(func)
    assert 'z' in defs
    assert 'result' in defs
    assert 'process' in defs

def test_mutation_operators():
    """Test that all mutation operators are properly initialized."""
    mutator = ExonMutator()
    
    assert 'function' in mutator.mutation_operators
    assert 'block' in mutator.mutation_operators
    assert 'statement' in mutator.mutation_operators
    assert 'node' in mutator.mutation_operators
    
    assert len(mutator.mutation_operators['function']) > 0
    assert len(mutator.mutation_operators['block']) > 0
    assert len(mutator.mutation_operators['statement']) > 0
    assert len(mutator.mutation_operators['node']) > 0

def test_granularity_selection():
    """Test that granularity selection follows the configured weights."""
    mutator = ExonMutator()
    granularities = [mutator.select_granularity() for _ in range(1000)]
    
    # Count occurrences
    counts = {}
    for g in granularities:
        counts[g] = counts.get(g, 0) + 1
    
    # Check that the relative frequencies are roughly correct
    total = sum(counts.values())
    for g, count in counts.items():
        expected = mutator.granularity_weights[g] * total
        assert abs(count - expected) < 0.1 * total  # Allow 10% deviation

def test_mutation_preserves_syntax():
    """Test that mutations preserve Python syntax."""
    source = """
def complex_function(x, y):
    if x > 0:
        result = x + y
    else:
        result = x - y
    return result
"""
    code_ast = ast.parse(source)
    mutator = ExonMutator()
    
    # Test each mutation type
    for _ in range(10):
        mutated_ast = mutator.mutate(code_ast)
        # If this doesn't raise an exception, the syntax is valid
        ast.fix_missing_locations(mutated_ast)

def test_code_validation():
    """Test code validation and repair."""
    # Test valid code
    source = """
def valid_function(x, y):
    if x > y:
        return x
    else:
        return y
"""
    code_ast = ast.parse(source)
    validator = CodeValidator()
    assert validator.is_valid_syntax(code_ast)
    
    # Test invalid code
    invalid_code = """
def broken_function(x):
    if x > 0
        return x
    return 0
"""
    try:
        invalid_ast = ast.parse(invalid_code)
        assert False, "Should have raised a SyntaxError"
    except SyntaxError:
        # Test repair with None
        repaired_ast = validator.repair(None)
        assert validator.is_valid_syntax(repaired_ast)

def test_evolution_process():
    """Test the complete evolution process."""
    # Create initial code
    initial_code = """
def foo(x):
    return x + 1
"""
    # Define a simple fitness function
    def fitness_function(code):
        return code.count('return')
    # Initialize and run evolution
    engine = ExonEvolutionEngine({
        'population_size': 10,
        'invalid_retention_rate': 0.1
    })
    engine.initialize_population(initial_code)
    best_solution = engine.evolve(fitness_function, generations=5)
    assert best_solution is not None

if __name__ == '__main__':
    pytest.main([__file__]) 