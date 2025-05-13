"""
Unit tests for the evolutionary_math module.

This module tests the mathematical functions that form the theoretical foundation
of the evolutionary algorithms in the TRISOLARIS framework.
"""

import unittest
import numpy as np
from typing import List, Dict, Any

from trisolaris.core.evolutionary_math import (
    calculate_price_equation,
    calculate_fisher_theorem,
    calculate_selection_gradient,
    calculate_fitness_landscape
)
from trisolaris.core.genome import CodeGenome

class MockGenome:
    """Mock genome class for testing."""
    
    def __init__(self, source_code: str = None, fitness: float = 0.5):
        self._source_code = source_code or "def test(): return 0"
        self._fitness = fitness
    
    def to_source(self) -> str:
        """Return the source code."""
        return self._source_code
    
    def clone(self):
        """Create a copy of this genome."""
        return MockGenome(self._source_code, self._fitness)
    
    def mutate(self, rate: float = 0.1):
        """Apply a mock mutation."""
        if rate > 0:
            # Simple mock mutation: add a comment
            self._source_code += f"\n# Mutated with rate {rate}"


class TestPriceEquation(unittest.TestCase):
    """Tests for the Price equation calculation."""
    
    def test_basic_calculation(self):
        """Test basic Price equation calculation with simple values."""
        # Create a simple population
        population = [MockGenome(f"def test{i}(): return {i}") for i in range(5)]
        fitness_values = [0.5, 0.7, 0.6, 0.8, 0.9]
        trait_values = [10, 12, 11, 15, 18]
        
        # Calculate Price equation
        total_change, selection, transmission = calculate_price_equation(
            population, fitness_values, trait_values
        )
        
        # Verify results are of the correct type
        self.assertIsInstance(total_change, float)
        self.assertIsInstance(selection, float)
        self.assertIsInstance(transmission, float)
        
        # Verify the total change equals the sum of components
        self.assertAlmostEqual(total_change, selection + transmission)
    
    def test_empty_population(self):
        """Test Price equation with empty population."""
        total_change, selection, transmission = calculate_price_equation([], [], [])
        
        # All components should be zero for empty population
        self.assertEqual(total_change, 0.0)
        self.assertEqual(selection, 0.0)
        self.assertEqual(transmission, 0.0)
    
    def test_input_validation(self):
        """Test input validation for Price equation."""
        population = [MockGenome() for _ in range(3)]
        fitness_values = [0.5, 0.7]  # One less than population
        trait_values = [10, 12, 11]
        
        # Should raise ValueError due to mismatched lengths
        with self.assertRaises(ValueError):
            calculate_price_equation(population, fitness_values, trait_values)
    
    def test_zero_mean_fitness(self):
        """Test Price equation with zero mean fitness."""
        population = [MockGenome() for _ in range(3)]
        fitness_values = [0.0, 0.0, 0.0]  # Zero mean fitness
        trait_values = [10, 12, 11]
        
        # Should handle zero mean fitness gracefully
        total_change, selection, transmission = calculate_price_equation(
            population, fitness_values, trait_values
        )
        
        # All components should be zero when mean fitness is zero
        self.assertEqual(total_change, 0.0)
        self.assertEqual(selection, 0.0)
        self.assertEqual(transmission, 0.0)


class TestFisherTheorem(unittest.TestCase):
    """Tests for Fisher's Fundamental Theorem calculation."""
    
    def test_basic_calculation(self):
        """Test basic Fisher theorem calculation with simple values."""
        # Create a simple population
        population = [MockGenome() for _ in range(5)]
        fitness_values = [0.5, 0.7, 0.6, 0.8, 0.9]
        additive_genetic_variance = 0.05
        
        # Calculate Fisher theorem
        rate_of_increase = calculate_fisher_theorem(
            population, fitness_values, additive_genetic_variance
        )
        
        # Verify result is of the correct type
        self.assertIsInstance(rate_of_increase, float)
        
        # Verify the result is positive (for positive variance)
        self.assertGreater(rate_of_increase, 0.0)
        
        # Verify the calculation is correct
        expected_rate = additive_genetic_variance / np.mean(fitness_values)
        self.assertAlmostEqual(rate_of_increase, expected_rate)
    
    def test_empty_population(self):
        """Test Fisher theorem with empty population."""
        rate_of_increase = calculate_fisher_theorem([], [], 0.05)
        
        # Rate should be zero for empty population
        self.assertEqual(rate_of_increase, 0.0)
    
    def test_input_validation(self):
        """Test input validation for Fisher theorem."""
        population = [MockGenome() for _ in range(3)]
        fitness_values = [0.5, 0.7]  # One less than population
        additive_genetic_variance = 0.05
        
        # Should raise ValueError due to mismatched lengths
        with self.assertRaises(ValueError):
            calculate_fisher_theorem(population, fitness_values, additive_genetic_variance)
    
    def test_zero_mean_fitness(self):
        """Test Fisher theorem with zero mean fitness."""
        population = [MockGenome() for _ in range(3)]
        fitness_values = [0.0, 0.0, 0.0]  # Zero mean fitness
        additive_genetic_variance = 0.05
        
        # Should handle zero mean fitness gracefully
        rate_of_increase = calculate_fisher_theorem(
            population, fitness_values, additive_genetic_variance
        )
        
        # Rate should be zero when mean fitness is zero
        self.assertEqual(rate_of_increase, 0.0)


class TestSelectionGradient(unittest.TestCase):
    """Tests for selection gradient calculation."""
    
    def test_basic_calculation(self):
        """Test basic selection gradient calculation."""
        # Create a simple population
        population = [
            MockGenome("def test(): return 0", 0.5),
            MockGenome("def test(): return 1", 0.7),
            MockGenome("def test(): return 2", 0.9)
        ]
        
        # Define a simple fitness landscape function
        def fitness_landscape(genome):
            return 0.5 + 0.1 * len(genome.to_source())
        
        # Calculate selection gradient
        gradients = calculate_selection_gradient(population, fitness_landscape)
        
        # Verify results
        self.assertEqual(len(gradients), len(population))
        for gradient in gradients:
            self.assertIsInstance(gradient, float)
    
    def test_empty_population(self):
        """Test selection gradient with empty population."""
        def fitness_landscape(genome):
            return 0.5
        
        gradients = calculate_selection_gradient([], fitness_landscape)
        
        # Should return empty list for empty population
        self.assertEqual(gradients, [])
    
    def test_with_real_genome(self):
        """Test selection gradient with actual CodeGenome instances."""
        # Create real CodeGenome instances
        population = [
            CodeGenome(source_code="def func1(): return 1"),
            CodeGenome(source_code="def func2(): return 2"),
            CodeGenome(source_code="def func3(): return 3")
        ]
        
        # Define a simple fitness landscape function
        def fitness_landscape(genome):
            code = genome.to_source()
            return 0.5 + 0.01 * code.count('return')
        
        # Calculate selection gradient
        gradients = calculate_selection_gradient(population, fitness_landscape)
        
        # Verify results
        self.assertEqual(len(gradients), len(population))
        for gradient in gradients:
            self.assertIsInstance(gradient, float)


class TestFitnessLandscape(unittest.TestCase):
    """Tests for fitness landscape calculation."""
    
    def test_basic_calculation(self):
        """Test basic fitness landscape calculation."""
        # Create a simple population
        population = [
            MockGenome("def test(): return 0", 0.5),
            MockGenome("def test(): if True: return 1", 0.7),
            MockGenome("def test(): for i in range(3): return 2", 0.9)
        ]
        
        # Define environment parameters
        environment = {
            'mutation_rate': 0.1,
            'selection_pressure': 0.7
        }
        
        # Calculate fitness landscape
        landscape_info = calculate_fitness_landscape(population, environment)
        
        # Verify results
        self.assertIsInstance(landscape_info, dict)
        expected_keys = [
            'ruggedness', 'num_peaks', 'mean_gradient', 
            'max_fitness', 'min_fitness', 'fitness_range'
        ]
        for key in expected_keys:
            self.assertIn(key, landscape_info)
            self.assertIsInstance(landscape_info[key], (int, float))
    
    def test_empty_population(self):
        """Test fitness landscape with empty population."""
        environment = {'mutation_rate': 0.1}
        
        landscape_info = calculate_fitness_landscape([], environment)
        
        # Should return default values for empty population
        self.assertEqual(landscape_info['ruggedness'], 0.0)
        self.assertEqual(landscape_info['num_peaks'], 0)
        self.assertEqual(landscape_info['mean_gradient'], 0.0)
        self.assertEqual(landscape_info['max_fitness'], 0.0)
        self.assertEqual(landscape_info['min_fitness'], 0.0)
        self.assertEqual(landscape_info['fitness_range'], 0.0)
    
    def test_with_real_genome(self):
        """Test fitness landscape with actual CodeGenome instances."""
        # Create real CodeGenome instances
        population = [
            CodeGenome(source_code="def func1(): return 1"),
            CodeGenome(source_code="def func2(): if True: return 2"),
            CodeGenome(source_code="def func3(): for i in range(3): return i")
        ]
        
        # Define environment parameters
        environment = {
            'mutation_rate': 0.1,
            'selection_pressure': 0.5
        }
        
        # Calculate fitness landscape
        landscape_info = calculate_fitness_landscape(population, environment)
        
        # Verify results
        self.assertIsInstance(landscape_info, dict)
        self.assertGreaterEqual(landscape_info['max_fitness'], landscape_info['min_fitness'])
        self.assertEqual(
            landscape_info['fitness_range'], 
            landscape_info['max_fitness'] - landscape_info['min_fitness']
        )


if __name__ == '__main__':
    unittest.main()