"""
Main evolutionary algorithm implementation.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from .program_representation import ProgramAST
from .fitness import FitnessEvaluator
from .population import Population

logger = logging.getLogger(__name__)

class EvolutionaryAlgorithm:
    """Main evolutionary algorithm implementation."""
    
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism: float = 0.1,
                 fitness_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the evolutionary algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Fraction of best individuals to preserve
            fitness_weights: Optional weights for fitness components
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.fitness_weights = fitness_weights
        
        self.population = None
        self.evaluator = None
        self.best_program = None
        self.best_fitness = 0.0
        self.generation = 0
        self.start_time = 0
        
    def initialize(self, test_cases: List[Dict[str, Any]]) -> None:
        """Initialize the algorithm with test cases."""
        # Create fitness evaluator
        self.evaluator = FitnessEvaluator(test_cases, self.fitness_weights)
        
        # Create population
        self.population = Population(
            size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism
        )
        
        # Initialize population
        self.population.initialize(self.evaluator)
        
        # Get initial best program
        self.best_program, self.best_fitness = self.population.get_best_program()
        
        # Start timing
        self.start_time = time.time()
        
    def evolve(self) -> None:
        """Run the evolutionary algorithm."""
        if not self.population or not self.evaluator:
            raise ValueError("Algorithm not initialized")
            
        logger.info("Starting evolution...")
        
        for generation in range(self.generations):
            # Evolve population
            self.population.evolve(self.evaluator)
            
            # Update best program
            current_best, current_fitness = self.population.get_best_program()
            if current_fitness > self.best_fitness:
                self.best_program = current_best
                self.best_fitness = current_fitness
                
            # Log progress
            stats = self.population.get_statistics()
            elapsed_time = time.time() - self.start_time
            
            logger.info(
                f"Generation {generation + 1}/{self.generations} - "
                f"Best: {self.best_fitness:.4f} - "
                f"Avg: {stats['avg_fitness']:.4f} - "
                f"Time: {elapsed_time:.1f}s"
            )
            
            # Check for convergence
            if self._check_convergence():
                logger.info("Population converged, stopping evolution")
                break
                
        logger.info("Evolution completed")
        
    def _check_convergence(self) -> bool:
        """Check if the population has converged."""
        if not self.population:
            return False
            
        stats = self.population.get_statistics()
        
        # Check if best fitness is close to 1.0
        if stats['best_fitness'] > 0.95:
            return True
            
        # Check if average fitness is close to best fitness
        if stats['best_fitness'] - stats['avg_fitness'] < 0.01:
            return True
            
        return False
        
    def get_best_program(self) -> Tuple[ProgramAST, float]:
        """Get the best program and its fitness score."""
        if not self.best_program:
            raise ValueError("No best program found")
        return self.best_program, self.best_fitness
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        if not self.population:
            return {
                'generations': self.generations,
                'population_size': self.population_size,
                'best_fitness': 0.0,
                'elapsed_time': 0.0
            }
            
        stats = self.population.get_statistics()
        stats['elapsed_time'] = time.time() - self.start_time
        
        return stats 