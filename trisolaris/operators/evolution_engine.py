"""
Evolution engine for TRISOLARIS implementing exon-like mutations.

This module provides the main evolution engine that integrates mutation operators
with code validation and repair mechanisms.
"""

import ast
import random
from typing import List, Dict, Any, Optional, Tuple
from .exon_mutator import ExonMutator
from .code_validator import CodeValidator

class ExonEvolutionEngine:
    """Manages the evolutionary process using exon-like mutations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters:
                - population_size: Size of the population (default: 100)
                - invalid_retention_rate: Rate at which invalid individuals are retained (default: 0.1)
                - granularity_weights: Weights for mutation granularity levels
                - repair_strategies: Weights for repair strategies
        """
        self.config = config or {}
        self.population_size = self.config.get('population_size', 100)
        self.invalid_retention_rate = self.config.get('invalid_retention_rate', 0.1)
        
        # Initialize components
        self.mutator = ExonMutator(
            granularity_weights=self.config.get('granularity_weights')
        )
        self.validator = CodeValidator(
            repair_strategies=self.config.get('repair_strategies')
        )
        
        # Initialize state
        self.population: List[ast.AST] = []
        self.generation = 0
        self.best_solution: Optional[ast.AST] = None
        self.best_fitness: float = float('-inf')
    
    def initialize_population(self, initial_code: str) -> None:
        """
        Initialize the population with variations of the initial code.
        
        Args:
            initial_code: The initial code to use as a base
        """
        base_ast = ast.parse(initial_code)
        self.population = [base_ast]
        
        # Generate variations
        while len(self.population) < self.population_size:
            variant = self.mutator.mutate(base_ast)
            if self.validator.is_valid_syntax(variant):
                self.population.append(variant)
    
    def evaluate_fitness(self, code_ast: ast.AST, fitness_func: callable) -> float:
        """
        Evaluate the fitness of a code AST.
        
        Args:
            code_ast: The AST to evaluate
            fitness_func: Function that takes code as string and returns fitness
            
        Returns:
            The fitness score
        """
        try:
            code = ast.unparse(code_ast)
            return fitness_func(code)
        except:
            return float('-inf')
    
    def select_parents(self, fitness_scores: List[float]) -> Tuple[ast.AST, ast.AST]:
        """
        Select two parents using tournament selection.
        
        Args:
            fitness_scores: List of fitness scores for the population
            
        Returns:
            Tuple of two selected parent ASTs
        """
        def tournament_select():
            candidates = random.sample(range(len(self.population)), 3)
            winner = max(candidates, key=lambda i: fitness_scores[i])
            return self.population[winner]
        
        return tournament_select(), tournament_select()
    
    def crossover(self, parent1: ast.AST, parent2: ast.AST) -> ast.AST:
        """
        Perform crossover between two parent ASTs.
        
        Args:
            parent1: First parent AST
            parent2: Second parent AST
            
        Returns:
            A new child AST
        """
        # For now, just return a mutated copy of parent1
        # TODO: Implement proper crossover
        return self.mutator.mutate(parent1)
    
    def evolve(self, fitness_func: callable, generations: int = 100) -> ast.AST:
        """
        Run the evolutionary process.
        
        Args:
            fitness_func: Function that takes code as string and returns fitness
            generations: Number of generations to evolve
            
        Returns:
            The best solution found
        """
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            fitness_scores = [
                self.evaluate_fitness(ind, fitness_func)
                for ind in self.population
            ]
            
            # Update best solution
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_solution = self.population[best_idx]
            
            # Create next generation
            new_population = []
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(fitness_scores)
                
                # Create child
                child = self.crossover(parent1, parent2)
                
                # Mutate child
                child = self.mutator.mutate(child)
                
                # Validate and repair if necessary
                if not self.validator.is_valid_syntax(child):
                    if random.random() < self.invalid_retention_rate:
                        child = self.validator.repair(child)
                    else:
                        continue
                
                new_population.append(child)
            
            self.population = new_population
        
        return self.best_solution 