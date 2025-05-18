#!/usr/bin/env python3
"""
Example script demonstrating the evolutionary algorithm by evolving a sorting function.
"""
import sys
import logging
from typing import List, Dict, Any
from trisolaris.core.evolution import EvolutionaryAlgorithm
from trisolaris.core.program_representation import ProgramAST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for sorting."""
    return [
        {
            'input': [3, 1, 4, 1, 5, 9, 2, 6],
            'expected': [1, 1, 2, 3, 4, 5, 6, 9]
        },
        {
            'input': [1],
            'expected': [1]
        },
        {
            'input': [],
            'expected': []
        },
        {
            'input': [1, 2, 3, 4, 5],
            'expected': [1, 2, 3, 4, 5]
        },
        {
            'input': [5, 4, 3, 2, 1],
            'expected': [1, 2, 3, 4, 5]
        }
    ]

def main():
    """Run the evolutionary algorithm to evolve a sorting function."""
    # Create test cases
    test_cases = create_test_cases()
    
    # Create evolutionary algorithm
    ea = EvolutionaryAlgorithm(
        population_size=50,
        generations=100,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elitism=0.1,
        fitness_weights={
            'correctness': 0.7,
            'performance': 0.2,
            'complexity': 0.1
        }
    )
    
    # Initialize algorithm
    ea.initialize(test_cases)
    
    # Run evolution
    ea.evolve()
    
    # Get best program
    best_program, fitness = ea.get_best_program()
    
    # Print results
    logger.info(f"\nBest program found (fitness: {fitness:.4f}):")
    logger.info("\nSource code:")
    print(best_program.to_source())
    
    # Print statistics
    stats = ea.get_statistics()
    logger.info("\nStatistics:")
    logger.info(f"Generations: {stats['generations']}")
    logger.info(f"Population size: {stats['population_size']}")
    logger.info(f"Best fitness: {stats['best_fitness']:.4f}")
    logger.info(f"Elapsed time: {stats['elapsed_time']:.1f}s")

if __name__ == '__main__':
    main() 