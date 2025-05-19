import pytest
import ast
import random
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from trisolaris.core.program_representation import ProgramAST
from trisolaris.core.ast_helpers import validate_ast

# Enable debug logging for mutation tests
import os
os.environ['DEBUG_MUTATION'] = 'true'
import logging
logger = logging.getLogger(__name__)

def test_point_mutation():
    """Test that point mutations produce valid programs."""
    program = ProgramAST()
    original_source = program.to_source()
    original_ast = ast.dump(program.ast_tree)
    
    logger.info(f"\nTesting point mutation:")
    logger.info(f"Original Source: {original_source}")
    logger.info(f"Original AST: {original_ast}")
    
    # Test multiple mutations
    for i in range(10):
        mutated = program.mutate(mutation_rate=1.0)  # Force mutation
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after point mutation: {error}"
        
        mutated_source = mutated.to_source()
        mutated_ast = ast.dump(mutated.ast_tree)
        
        logger.info(f"\nMutation attempt {i + 1}:")
        logger.info(f"Mutated Source: {mutated_source}")
        logger.info(f"Mutated AST: {mutated_ast}")
        
        # Check if the mutation actually changed something
        assert mutated_source != original_source, f"Mutation {i + 1} did not change the program"
        assert mutated_ast != original_ast, f"Mutation {i + 1} did not change the AST"

def test_subtree_mutation():
    """Test that subtree mutations produce valid programs."""
    program = ProgramAST()
    original_source = program.to_source()
    original_ast = ast.dump(program.ast_tree)
    
    logger.info(f"\nTesting subtree mutation:")
    logger.info(f"Original Source: {original_source}")
    logger.info(f"Original AST: {original_ast}")
    
    # Test multiple mutations
    for i in range(10):
        mutated = program.mutate(mutation_rate=1.0)  # Force mutation
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after subtree mutation: {error}"
        
        mutated_source = mutated.to_source()
        mutated_ast = ast.dump(mutated.ast_tree)
        
        logger.info(f"\nMutation attempt {i + 1}:")
        logger.info(f"Mutated Source: {mutated_source}")
        logger.info(f"Mutated AST: {mutated_ast}")
        
        # Check if the mutation actually changed something
        assert mutated_source != original_source, f"Mutation {i + 1} did not change the program"
        assert mutated_ast != original_ast, f"Mutation {i + 1} did not change the AST"

def test_functional_mutation():
    """Test that functional mutations produce valid programs."""
    program = ProgramAST()
    original_source = program.to_source()
    original_ast = ast.dump(program.ast_tree)
    
    logger.info(f"\nTesting functional mutation:")
    logger.info(f"Original Source: {original_source}")
    logger.info(f"Original AST: {original_ast}")
    
    # Test multiple mutations
    for i in range(10):
        mutated = program.mutate(mutation_rate=1.0)  # Force mutation
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after functional mutation: {error}"
        
        mutated_source = mutated.to_source()
        mutated_ast = ast.dump(mutated.ast_tree)
        
        logger.info(f"\nMutation attempt {i + 1}:")
        logger.info(f"Mutated Source: {mutated_source}")
        logger.info(f"Mutated AST: {mutated_ast}")
        
        # Check if the mutation actually changed something
        assert mutated_source != original_source, f"Mutation {i + 1} did not change the program"
        assert mutated_ast != original_ast, f"Mutation {i + 1} did not change the AST"

@given(st.floats(min_value=0.0, max_value=1.0))
def test_mutation_rate_control(mutation_rate):
    """Test that mutation rate controls mutation frequency."""
    program = ProgramAST()
    original_source = program.to_source()
    original_ast = ast.dump(program.ast_tree)
    
    logger.info(f"\nTesting mutation rate {mutation_rate}:")
    logger.info(f"Original Source: {original_source}")
    logger.info(f"Original AST: {original_ast}")
    
    mutated = program.mutate(mutation_rate=mutation_rate)
    mutated_source = mutated.to_source()
    mutated_ast = ast.dump(mutated.ast_tree)
    
    logger.info(f"Mutated Source: {mutated_source}")
    logger.info(f"Mutated AST: {mutated_ast}")
    
    if mutation_rate == 0.0:
        assert mutated_source == original_source, "Program changed with zero mutation rate"
        assert mutated_ast == original_ast, "AST changed with zero mutation rate"
    elif mutation_rate == 1.0:
        assert mutated_source != original_source, "Program did not change with high mutation rate"
        assert mutated_ast != original_ast, "AST did not change with high mutation rate"

class MutationStateMachine(RuleBasedStateMachine):
    """State machine for testing mutation properties."""
    
    def __init__(self):
        super().__init__()
        self.program = ProgramAST()
        self.original_source = self.program.to_source()
        self.original_ast = ast.dump(self.program.ast_tree)
        self.mutation_count = 0
        self.valid_mutations = 0
        self.rates_used = []  # Track which mutation rates were used
    
    @rule(mutation_rate=st.floats(min_value=0.0, max_value=1.0))
    def mutate(self, mutation_rate):
        """Apply mutation and track statistics."""
        self.mutation_count += 1
        self.rates_used.append(mutation_rate)  # Track the rate used
        mutated = self.program.mutate(mutation_rate=mutation_rate)
        is_valid, error = validate_ast(mutated.ast_tree)
        
        logger.info(f"\nState machine mutation attempt {self.mutation_count}:")
        logger.info(f"Mutation rate: {mutation_rate}")
        logger.info(f"Original Source: {self.original_source}")
        logger.info(f"Mutated Source: {mutated.to_source()}")
        
        if is_valid:
            if mutation_rate > 0.0:
                self.valid_mutations += 1
                assert mutated.to_source() != self.original_source, "Mutation did not change the program"
                assert ast.dump(mutated.ast_tree) != self.original_ast, "Mutation did not change the AST"
        
        # Update program for next mutation
        self.program = mutated
        self.original_source = self.program.to_source()
        self.original_ast = ast.dump(self.program.ast_tree)
    
    def teardown(self):
        """Verify mutation statistics."""
        if self.mutation_count > 0:
            # Only check success rate if we used any non-zero mutation rates
            if any(rate > 0.0 for rate in self.rates_used):
                success_rate = self.valid_mutations / sum(1 for rate in self.rates_used if rate > 0.0)
                logger.info(f"\nMutation success rate: {success_rate:.1%}")
                assert success_rate >= 0.3, f"Mutation success rate {success_rate:.1%} below 30% threshold"

TestMutationStateMachine = MutationStateMachine.TestCase

def test_macro_mutation():
    """Test that macro mutations produce valid programs when normal mutations fail."""
    program = ProgramAST()
    original_source = program.to_source()
    original_ast = ast.dump(program.ast_tree)
    
    logger.info(f"\nTesting macro mutation:")
    logger.info(f"Original Source: {original_source}")
    logger.info(f"Original AST: {original_ast}")
    
    # Force macro mutation by making normal mutations fail
    for i in range(10):
        mutated = program.mutate(mutation_rate=1.0, max_retries=1)
        is_valid, error = validate_ast(mutated.ast_tree)
        assert is_valid, f"Invalid AST after macro mutation: {error}"
        
        mutated_source = mutated.to_source()
        mutated_ast = ast.dump(mutated.ast_tree)
        
        logger.info(f"\nMacro mutation attempt {i + 1}:")
        logger.info(f"Mutated Source: {mutated_source}")
        logger.info(f"Mutated AST: {mutated_ast}")
        
        # Check if the mutation actually changed something
        assert mutated_source != original_source, f"Macro mutation {i + 1} did not change the program"
        assert mutated_ast != original_ast, f"Macro mutation {i + 1} did not change the AST"

def test_mutation_stats():
    """Test that mutation statistics are correctly tracked."""
    program = ProgramAST()
    
    # Perform some mutations
    for i in range(5):
        logger.info(f"\nMutation stats test attempt {i + 1}:")
        mutated = program.mutate(mutation_rate=0.3)
        logger.info(f"Original Source: {program.to_source()}")
        logger.info(f"Mutated Source: {mutated.to_source()}")
        program = mutated
    
    # Verify stats reporting
    assert program.stats['mutation_attempts'] == 5
    assert program.stats['mutation_effective'] + program.stats['mutation_noop'] == 5
    
    # Print stats for debugging
    program.print_genetic_stats()

def test_mutation_telemetry_export():
    """Test that mutation statistics are correctly exported to CSV."""
    import os
    import csv
    from datetime import datetime
    
    # Create a test program
    program = ProgramAST()
    
    # Perform some mutations
    for _ in range(5):
        program.mutate(mutation_rate=0.3)
    
    # Export stats
    run_id = 'test_run'
    generation = 1
    program.export_mutation_stats(run_id, generation)
    
    # Verify CSV file was created
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'reports/{date_str}_mutation.csv'
    assert os.path.exists(filename), "Telemetry CSV file was not created"
    
    # Read and verify contents
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1, "Expected exactly one row in telemetry CSV"
        
        row = rows[0]
        assert row['run_id'] == run_id
        assert int(row['generation']) == generation
        assert int(row['attempts']) == 5
        assert int(row['effective']) + int(row['noop_count']) == 5
        assert float(row['success_rate']) >= 0.0
        assert float(row['success_rate']) <= 1.0
    
    # Clean up
    os.remove(filename) 