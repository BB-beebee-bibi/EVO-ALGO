"""
Population management for evolutionary algorithms.
"""
import random
import logging
from typing import List, Tuple, Optional, Dict, Any
from .program_representation import ProgramAST
from .fitness import FitnessEvaluator

logger = logging.getLogger(__name__)

class Population:
    """Manages a population of programs for evolution."""
    
    def __init__(self,
                 size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism: float = 0.1):
        """
        Initialize the population.
        
        Args:
            size: Population size
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Fraction of best individuals to preserve
        """
        self.size = size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        
        self.programs: List[ProgramAST] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        
    def initialize(self, evaluator: FitnessEvaluator) -> None:
        """Initialize the population with random programs."""
        self.programs = []
        self.fitness_scores = []
        
        # Create initial population
        for _ in range(self.size):
            program = ProgramAST()  # Creates minimal valid program
            self.programs.append(program)
            
        # Evaluate initial fitness
        self._evaluate_fitness(evaluator)
        
    def evolve(self, evaluator: FitnessEvaluator) -> None:
        """Evolve the population for one generation."""
        # Select parents
        parents = self._select_parents()
        
        # Create new population
        new_population = []
        
        # Elitism: Keep best individuals
        elite_count = int(self.size * self.elitism)
        elite_indices = sorted(range(len(self.fitness_scores)), 
                             key=lambda i: self.fitness_scores[i],
                             reverse=True)[:elite_count]
        
        for idx in elite_indices:
            new_population.append(self.programs[idx])
            
        # Create offspring
        while len(new_population) < self.size:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = parent1.crossover(parent2)
                
                # Mutate children
                if random.random() < self.mutation_rate:
                    child1.mutate()
                if random.random() < self.mutation_rate:
                    child2.mutate()
                    
                new_population.extend([child1, child2])
            else:
                # Mutation only
                parent = random.choice(parents)
                child = ProgramAST(tree=parent.tree)
                
                if random.random() < self.mutation_rate:
                    child.mutate()
                    
                new_population.append(child)
                
        # Update population
        self.programs = new_population[:self.size]
        self._evaluate_fitness(evaluator)
        self.generation += 1
        
    def _evaluate_fitness(self, evaluator: FitnessEvaluator) -> None:
        """Evaluate fitness of all programs."""
        self.fitness_scores = []
        for program in self.programs:
            score = evaluator.evaluate(program)
            self.fitness_scores.append(score)
            
    def _select_parents(self) -> List[ProgramAST]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = 3
        
        while len(parents) < self.size:
            # Select tournament participants
            tournament = random.sample(list(zip(self.programs, self.fitness_scores)), 
                                    tournament_size)
            
            # Select winner
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
            
        return parents
        
    def get_best_program(self) -> Tuple[ProgramAST, float]:
        """Get the best program and its fitness score."""
        if not self.programs:
            raise ValueError("Population is empty")
            
        best_idx = max(range(len(self.fitness_scores)), 
                      key=lambda i: self.fitness_scores[i])
        return self.programs[best_idx], self.fitness_scores[best_idx]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self.fitness_scores:
            return {
                'generation': self.generation,
                'size': self.size,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'worst_fitness': 0.0
            }
            
        return {
            'generation': self.generation,
            'size': self.size,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': sum(self.fitness_scores) / len(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores)
        } 