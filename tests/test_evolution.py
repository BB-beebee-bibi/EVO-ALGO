"""
Tests for the evolutionary algorithm.
"""
import pytest
from typing import List, Dict, Any
from trisolaris.core.evolution import EvolutionaryAlgorithm
from trisolaris.core.program_representation import ProgramAST
from trisolaris.core.fitness import FitnessEvaluator

def create_simple_test_cases() -> List[Dict[str, Any]]:
    """Create simple test cases for testing."""
    return [
        {
            'input': [1, 2, 3],
            'expected': [1, 2, 3]
        },
        {
            'input': [3, 2, 1],
            'expected': [1, 2, 3]
        }
    ]

def test_evolutionary_algorithm_initialization():
    """Test initialization of the evolutionary algorithm."""
    ea = EvolutionaryAlgorithm(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism=0.1
    )
    
    # Test initialization
    test_cases = create_simple_test_cases()
    ea.initialize(test_cases)
    
    # Check that population and evaluator are created
    assert ea.population is not None
    assert ea.evaluator is not None
    assert ea.population.size == 10
    
    # Check that best program is initialized
    assert ea.best_program is not None
    assert isinstance(ea.best_program, ProgramAST)

def test_evolutionary_algorithm_evolution():
    """Test evolution process."""
    ea = EvolutionaryAlgorithm(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism=0.1
    )
    
    # Initialize
    test_cases = create_simple_test_cases()
    ea.initialize(test_cases)
    
    # Get initial best fitness
    initial_best, initial_fitness = ea.get_best_program()
    
    # Run evolution
    ea.evolve()
    
    # Get final best fitness
    final_best, final_fitness = ea.get_best_program()
    
    # Check that evolution occurred
    assert final_fitness >= initial_fitness
    
    # Check statistics
    stats = ea.get_statistics()
    assert stats['generations'] <= 5
    assert stats['population_size'] == 10
    assert stats['best_fitness'] >= 0.0
    assert stats['elapsed_time'] >= 0.0

def test_fitness_evaluator():
    """Test fitness evaluation."""
    test_cases = create_simple_test_cases()
    evaluator = FitnessEvaluator(test_cases)
    
    # Create a simple program
    program = ProgramAST(source="""
def main(lst):
    return sorted(lst)
""")
    
    # Evaluate fitness
    fitness = evaluator.evaluate(program)
    
    # Check fitness score
    assert 0.0 <= fitness <= 1.0
    
    # Test with incorrect program
    incorrect_program = ProgramAST(source="""
def main(lst):
    return lst
""")
    
    incorrect_fitness = evaluator.evaluate(incorrect_program)
    assert incorrect_fitness < fitness

def test_population_evolution():
    """Test population evolution."""
    from trisolaris.core.population import Population
    
    # Create population
    population = Population(
        size=10,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism=0.1
    )
    
    # Create evaluator
    test_cases = create_simple_test_cases()
    evaluator = FitnessEvaluator(test_cases)
    
    # Initialize population
    population.initialize(evaluator)
    
    # Get initial statistics
    initial_stats = population.get_statistics()
    
    # Evolve population
    population.evolve(evaluator)
    
    # Get final statistics
    final_stats = population.get_statistics()
    
    # Check that evolution occurred
    assert final_stats['generation'] > initial_stats['generation']
    assert final_stats['best_fitness'] >= initial_stats['best_fitness'] 