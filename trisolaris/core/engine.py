"""
Main Evolutionary Engine for the TRISOLARIS framework.

This module implements the core evolutionary loop that drives the optimization process.
"""

import random
import time
import logging
from typing import List, Callable, Optional, Any, Dict, Tuple

from trisolaris.core.genome import CodeGenome

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionEngine:
    """
    The main evolutionary engine that drives the TRISOLARIS optimization process.
    
    This class manages the overall evolutionary process, including population management,
    selection, variation, and tracking progress over generations.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        evaluator: Any = None,
        genome_class: type = CodeGenome,
        selection_pressure: float = 0.7,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_ratio: float = 0.1,
        resource_monitor = None,
        diversity_guardian = None,
        repository = None,
    ):
        """
        Initialize the Evolution Engine with specified parameters.
        
        Args:
            population_size: Size of the population to evolve
            evaluator: FitnessEvaluator object to assess solutions
            genome_class: Class to use for representing individual solutions
            selection_pressure: How strongly selection favors better solutions (0-1)
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover between parents
            elitism_ratio: Proportion of best solutions carried to next generation
            resource_monitor: Optional resource monitoring component
            diversity_guardian: Optional diversity maintenance component
            repository: Optional repository for storing solutions
        """
        self.population_size = population_size
        self.evaluator = evaluator
        self.genome_class = genome_class
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        # Optional components
        self.resource_monitor = resource_monitor
        self.diversity_guardian = diversity_guardian
        self.repository = repository
        
        # Runtime variables
        self.population = []
        self.fitness_scores = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.generation = 0
        self.start_time = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
    
    def initialize_population(self):
        """Initialize a random population of solutions."""
        self.population = [self.genome_class() for _ in range(self.population_size)]
        logger.info(f"Initialized population with {self.population_size} individuals")
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals in the population."""
        if self.evaluator is None:
            raise ValueError("Evaluator not set. Cannot evaluate population.")
        
        # Check if resource monitor allows evaluation
        if self.resource_monitor and not self.resource_monitor.can_proceed():
            logger.warning("Resource constraints exceeded. Throttling evaluation.")
            self.fitness_scores = [s if i < len(self.fitness_scores) else 0 
                                  for i, s in enumerate(self.population)]
            return
        
        # Evaluate each individual
        self.fitness_scores = []
        for genome in self.population:
            # Apply ethical filter if available
            if hasattr(self.evaluator, 'check_ethical_boundaries'):
                if not self.evaluator.check_ethical_boundaries(genome):
                    self.fitness_scores.append(float('-inf'))
                    continue
            
            # Compute fitness
            fitness = self.evaluator.evaluate(genome)
            self.fitness_scores.append(fitness)
            
            # Update best solution if needed
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = genome.clone()
                if self.repository:
                    self.repository.store_solution(genome, fitness, self.generation)
        
        # Update history
        avg_fitness = sum(f for f in self.fitness_scores if f != float('-inf')) / len(self.fitness_scores)
        self.history['best_fitness'].append(self.best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        
        logger.info(f"Generation {self.generation}: Best fitness = {self.best_fitness}, Avg fitness = {avg_fitness}")
    
    def select_parents(self) -> List[CodeGenome]:
        """Select parents for reproduction using tournament selection."""
        tournament_size = max(2, int(self.population_size * self.selection_pressure * 0.1))
        parents = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitnesses = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def create_offspring(self, parents: List[CodeGenome]) -> List[CodeGenome]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        # Ensure even number of parents for crossover
        if len(parents) % 2 == 1:
            parents.append(parents[0])
        
        # Apply crossover
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                # Safety check for index out of bounds
                break
                
            parent1, parent2 = parents[i], parents[i+1]
            
            # Validate parents are proper CodeGenome instances
            if not isinstance(parent1, CodeGenome) or not isinstance(parent2, CodeGenome):
                # Skip invalid pairs and log error
                logging.error(f"Invalid parent types: {type(parent1)}, {type(parent2)}")
                # Create random genomes instead
                child1, child2 = CodeGenome(), CodeGenome()
            else:
                try:
                    # Apply crossover if probability check passes
                    if random.random() < self.crossover_rate:
                        child1, child2 = parent1.crossover(parent2)
                    else:
                        child1, child2 = parent1.clone(), parent2.clone()
                    
                    # Apply mutation
                    child1.mutate(self.mutation_rate)
                    child2.mutate(self.mutation_rate)
                except Exception as e:
                    # Log error and fallback to clones or random genomes
                    logging.error(f"Error during crossover or mutation: {str(e)}")
                    try:
                        child1, child2 = parent1.clone(), parent2.clone()
                    except:
                        child1, child2 = CodeGenome(), CodeGenome()
            
            offspring.append(child1)
            offspring.append(child2)
        
        return offspring[:self.population_size]
    
    def select_survivors(self, offspring: List[CodeGenome]) -> List[CodeGenome]:
        """Select survivors for the next generation using elitism."""
        # Calculate how many elites to keep
        num_elites = max(1, int(self.population_size * self.elitism_ratio))
        
        # Get indices of the best individuals
        elite_indices = sorted(range(len(self.fitness_scores)), 
                              key=lambda i: self.fitness_scores[i], 
                              reverse=True)[:num_elites]
        
        # Create new population with elites
        new_population = [self.population[i].clone() for i in elite_indices]
        
        # Fill the rest with offspring
        new_population.extend(offspring[:self.population_size - num_elites])
        
        return new_population
    
    def maintain_diversity(self):
        """Apply diversity maintenance if diversity guardian is available."""
        if not self.diversity_guardian:
            return
        
        diversity = self.diversity_guardian.measure_diversity(self.population)
        self.history['diversity'].append(diversity)
        
        if diversity < self.diversity_guardian.min_diversity:
            logger.info(f"Diversity too low ({diversity}). Applying diversity maintenance.")
            self.population = self.diversity_guardian.inject_diversity(self.population)
    
    def evolve(self, generations: int = 100, max_time: int = None, target_fitness: float = None) -> CodeGenome:
        """
        Run the evolutionary process for a specified number of generations.
        
        Args:
            generations: Maximum number of generations to evolve
            max_time: Maximum time in seconds for the evolutionary process
            target_fitness: Target fitness to achieve (stops when reached)
            
        Returns:
            The best solution found
        """
        self.start_time = time.time()
        
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        # Main evolutionary loop
        for self.generation in range(generations):
            # Check termination conditions
            elapsed_time = time.time() - self.start_time
            if max_time and elapsed_time > max_time:
                logger.info(f"Time limit of {max_time}s reached after {self.generation} generations")
                break
                
            if target_fitness and self.best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached in generation {self.generation}")
                break
            
            # Evaluate current population
            self.evaluate_population()
            
            # Create next generation
            parents = self.select_parents()
            offspring = self.create_offspring(parents)
            self.population = self.select_survivors(offspring)
            
            # Maintain diversity if needed
            self.maintain_diversity()
        
        return self.get_best_solution()
    
    def get_best_solution(self) -> CodeGenome:
        """Return the best solution found so far."""
        return self.best_solution
    
    def set_selection_pressure(self, pressure: float):
        """Set the selection pressure parameter."""
        if 0 <= pressure <= 1:
            self.selection_pressure = pressure
        else:
            raise ValueError("Selection pressure must be between 0 and 1")
    
    def set_mutation_rate(self, rate: float):
        """Set the mutation rate parameter."""
        if 0 <= rate <= 1:
            self.mutation_rate = rate
        else:
            raise ValueError("Mutation rate must be between 0 and 1")
    
    def set_crossover_rate(self, rate: float):
        """Set the crossover rate parameter."""
        if 0 <= rate <= 1:
            self.crossover_rate = rate
        else:
            raise ValueError("Crossover rate must be between 0 and 1") 