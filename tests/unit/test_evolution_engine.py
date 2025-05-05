#!/usr/bin/env python3
"""
Unit tests for the TRISOLARIS Evolution Engine
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import trisolaris modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator


class MockEvaluator:
    """Mock fitness evaluator for testing."""
    
    def __init__(self, fitness_values=None):
        """Initialize with predefined fitness values for testing."""
        self.fitness_values = fitness_values or {}
        self.call_count = 0
        
    def evaluate(self, genome):
        """Return predefined fitness or incremental fitness based on call count."""
        self.call_count += 1
        if genome in self.fitness_values:
            return self.fitness_values[genome]
        # Return increasing fitness for each call to simulate improvement
        return 0.1 * self.call_count


class TestEvolutionEngine(unittest.TestCase):
    """Test cases for the Evolution Engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_evaluator = MockEvaluator()
        self.engine = EvolutionEngine(
            population_size=10,
            evaluator=self.mock_evaluator,
            mutation_rate=0.1,
            crossover_rate=0.7,
            genome_class=CodeGenome
        )
    
    def test_initialization(self):
        """Test that the engine initializes correctly."""
        self.assertEqual(len(self.engine.population), 10)
        self.assertEqual(self.engine.mutation_rate, 0.1)
        self.assertEqual(self.engine.crossover_rate, 0.7)
        
    def test_evaluation(self):
        """Test population evaluation."""
        # Populate with simple genomes
        self.engine.population = [CodeGenome() for _ in range(10)]
        
        # Evaluate population
        self.engine.evaluate_population()
        
        # Check all genomes were evaluated
        self.assertEqual(self.mock_evaluator.call_count, 10)
        
        # Check fitness values were assigned
        for genome in self.engine.population:
            self.assertTrue(hasattr(genome, 'fitness'))
            self.assertIsNotNone(genome.fitness)
    
    def test_selection(self):
        """Test parent selection."""
        # Create population with known fitness values
        self.engine.population = [CodeGenome() for _ in range(10)]
        for i, genome in enumerate(self.engine.population):
            genome.fitness = i / 10.0  # Fitness from 0.0 to 0.9
        
        # Select parents
        parents = self.engine.select_parents()
        
        # Should have selected parents based on fitness
        self.assertGreaterEqual(len(parents), 1)
        
        # Higher fitness genomes should be more likely to be selected
        avg_fitness = sum(genome.fitness for genome in parents) / len(parents)
        self.assertGreater(avg_fitness, 0.3)  # Average fitness should be higher than random
    
    def test_crossover(self):
        """Test crossover operation."""
        # Create parent genomes
        parent1 = CodeGenome.from_source("def func1(): return 1")
        parent2 = CodeGenome.from_source("def func2(): return 2")
        
        # Force crossover
        with patch('random.random', return_value=0.1):  # Below crossover rate
            offspring = self.engine._crossover(parent1, parent2)
        
        # Offspring should be different from both parents
        self.assertNotEqual(offspring.to_source(), parent1.to_source())
        self.assertNotEqual(offspring.to_source(), parent2.to_source())
    
    def test_mutation(self):
        """Test mutation operation."""
        # Create genome
        genome = CodeGenome.from_source("def func(): return 1")
        original_source = genome.to_source()
        
        # Force mutation
        with patch('random.random', return_value=0.05):  # Below mutation rate
            mutated = self.engine._mutate(genome)
        
        # Mutated genome should be different
        self.assertNotEqual(mutated.to_source(), original_source)
    
    def test_evolution_improvement(self):
        """Test that evolution improves fitness over generations."""
        # Create initial population
        self.engine.population = [CodeGenome() for _ in range(10)]
        
        # Mock evaluator to return increasing fitness for newer generations
        generation_fitnesses = []
        
        # Run evolution for a few generations
        for gen in range(3):
            self.engine.evaluate_population()
            best_fitness = max(genome.fitness for genome in self.engine.population)
            generation_fitnesses.append(best_fitness)
            
            # Create next generation
            parents = self.engine.select_parents()
            offspring = self.engine.create_offspring(parents)
            self.engine.population = self.engine.select_survivors(offspring)
        
        # Verify fitness increased
        self.assertGreater(generation_fitnesses[-1], generation_fitnesses[0])


if __name__ == "__main__":
    unittest.main() 