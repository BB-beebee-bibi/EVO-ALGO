"""
Island Ecosystem Manager for the TRISOLARIS framework.

This module implements the island model for evolutionary computation, maintaining 
multiple subpopulations with different selection pressures and enabling migration between them.
"""

import random
import copy
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IslandEcosystemManager:
    """
    Manages multiple subpopulations with different selection pressures and facilitates migration.
    
    This class implements the island model of evolutionary computation, maintaining separate
    subpopulations (islands) that evolve in parallel with occasional migration between them.
    """
    
    def __init__(
        self,
        num_islands: int = 3,
        migration_interval: int = 5,
        migration_rate: float = 0.1,
        island_sizes: List[int] = None,
        selection_pressures: List[float] = None,
        mutation_rates: List[float] = None,
        crossover_rates: List[float] = None,
    ):
        """
        Initialize the Island Ecosystem Manager.
        
        Args:
            num_islands: Number of islands (subpopulations)
            migration_interval: How often migration occurs (in generations)
            migration_rate: Fraction of individuals that migrate between islands
            island_sizes: List of population sizes for each island
            selection_pressures: List of selection pressures for each island
            mutation_rates: List of mutation rates for each island
            crossover_rates: List of crossover rates for each island
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        
        # Set up island configurations
        self._configure_islands(
            island_sizes=island_sizes,
            selection_pressures=selection_pressures,
            mutation_rates=mutation_rates,
            crossover_rates=crossover_rates
        )
        
        # Storage for populations and fitness scores
        self.islands = [[] for _ in range(self.num_islands)]
        self.fitness_scores = [[] for _ in range(self.num_islands)]
        
        # Track generation and migration history
        self.generation = 0
        self.migration_history = []
        self.island_statistics = {i: {'avg_fitness': [], 'best_fitness': []} for i in range(self.num_islands)}
    
    def _configure_islands(
        self,
        island_sizes: List[int] = None,
        selection_pressures: List[float] = None,
        mutation_rates: List[float] = None,
        crossover_rates: List[float] = None
    ):
        """Configure parameters for each island."""
        # Island sizes - default is equal sizes
        if island_sizes is None:
            # Default to equal sizes
            self.island_sizes = [20] * self.num_islands
        else:
            if len(island_sizes) != self.num_islands:
                raise ValueError(f"Expected {self.num_islands} island sizes, got {len(island_sizes)}")
            self.island_sizes = island_sizes
        
        # Selection pressures - vary from low to high across islands
        if selection_pressures is None:
            # Default to uniform spread from 0.3 (low pressure) to 0.9 (high pressure)
            low, high = 0.3, 0.9
            step = (high - low) / max(1, self.num_islands - 1)
            self.selection_pressures = [low + i * step for i in range(self.num_islands)]
        else:
            if len(selection_pressures) != self.num_islands:
                raise ValueError(f"Expected {self.num_islands} selection pressures, got {len(selection_pressures)}")
            self.selection_pressures = selection_pressures
        
        # Mutation rates - vary from high to low across islands
        if mutation_rates is None:
            # Default to uniform spread from 0.2 (high mutation) to 0.02 (low mutation)
            high, low = 0.2, 0.02
            step = (high - low) / max(1, self.num_islands - 1)
            self.mutation_rates = [high - i * step for i in range(self.num_islands)]
        else:
            if len(mutation_rates) != self.num_islands:
                raise ValueError(f"Expected {self.num_islands} mutation rates, got {len(mutation_rates)}")
            self.mutation_rates = mutation_rates
        
        # Crossover rates - default is uniform
        if crossover_rates is None:
            # Default to uniform 0.7 crossover rate
            self.crossover_rates = [0.7] * self.num_islands
        else:
            if len(crossover_rates) != self.num_islands:
                raise ValueError(f"Expected {self.num_islands} crossover rates, got {len(crossover_rates)}")
            self.crossover_rates = crossover_rates
        
        # Log island configurations
        for i in range(self.num_islands):
            logger.info(
                f"Island {i}: size={self.island_sizes[i]}, "
                f"selection={self.selection_pressures[i]:.2f}, "
                f"mutation={self.mutation_rates[i]:.3f}, "
                f"crossover={self.crossover_rates[i]:.2f}"
            )
    
    def initialize_islands(self, population, genome_class):
        """
        Initialize all islands with copies of the initial population.
        
        Args:
            population: Initial population of genomes
            genome_class: Class to use for creating genomes
        """
        if not population:
            # Create a random population
            for i in range(self.num_islands):
                self.islands[i] = [genome_class() for _ in range(self.island_sizes[i])]
        else:
            # Distribute the provided population across islands
            for i in range(self.num_islands):
                # Take individuals from the provided population if available
                if len(population) >= self.island_sizes[i]:
                    # Take a sample from the population
                    indices = random.sample(range(len(population)), self.island_sizes[i])
                    self.islands[i] = [copy.deepcopy(population[j]) for j in indices]
                else:
                    # Take all available and supplement with random individuals
                    existing = copy.deepcopy(population)
                    additional = [genome_class() for _ in range(self.island_sizes[i] - len(existing))]
                    self.islands[i] = existing + additional
        
        logger.info(f"Initialized {self.num_islands} islands with populations of sizes: {[len(island) for island in self.islands]}")
    
    def evaluate_islands(self, evaluator):
        """
        Evaluate all individuals on all islands.
        
        Args:
            evaluator: FitnessEvaluator object to assess solutions
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        for island_idx in range(self.num_islands):
            island = self.islands[island_idx]
            fitness_scores = []
            
            for genome in island:
                try:
                    # Apply ethical filter if available
                    if hasattr(evaluator, 'check_ethical_boundaries'):
                        if not evaluator.check_ethical_boundaries(genome):
                            fitness_scores.append(float('-inf'))
                            continue
                    
                    # Compute fitness
                    fitness = evaluator.evaluate(genome)
                    fitness_scores.append(fitness)
                    
                    # Store fitness on the genome for later use
                    if hasattr(genome, 'set_fitness'):
                        genome.set_fitness(fitness)
                    else:
                        genome.fitness = fitness
                        
                except Exception as e:
                    logger.error(f"Error evaluating genome: {str(e)}")
                    fitness_scores.append(float('-inf'))
            
            self.fitness_scores[island_idx] = fitness_scores
            
            # Calculate statistics
            valid_scores = [s for s in fitness_scores if s != float('-inf')]
            if valid_scores:
                avg_fitness = sum(valid_scores) / len(valid_scores)
                best_fitness = max(valid_scores)
                
                self.island_statistics[island_idx]['avg_fitness'].append(avg_fitness)
                self.island_statistics[island_idx]['best_fitness'].append(best_fitness)
                
                results[f'island_{island_idx}'] = {
                    'avg_fitness': avg_fitness,
                    'best_fitness': best_fitness,
                    'valid_solutions': len(valid_scores),
                    'total_solutions': len(fitness_scores)
                }
            else:
                self.island_statistics[island_idx]['avg_fitness'].append(0.0)
                self.island_statistics[island_idx]['best_fitness'].append(float('-inf'))
                
                results[f'island_{island_idx}'] = {
                    'avg_fitness': 0.0,
                    'best_fitness': float('-inf'),
                    'valid_solutions': 0,
                    'total_solutions': len(fitness_scores)
                }
        
        # Find best overall
        best_island_idx = -1
        best_solution_idx = -1
        best_fitness = float('-inf')
        
        for i, fitness_list in enumerate(self.fitness_scores):
            if not fitness_list:
                continue
                
            island_best_idx = fitness_list.index(max(fitness_list)) if fitness_list else -1
            if island_best_idx >= 0 and fitness_list[island_best_idx] > best_fitness:
                best_island_idx = i
                best_solution_idx = island_best_idx
                best_fitness = fitness_list[island_best_idx]
        
        results['best_solution'] = {
            'island': best_island_idx,
            'index': best_solution_idx,
            'fitness': best_fitness
        }
        
        return results
    
    def evolve_islands(self, evaluator, evolution_engine):
        """
        Evolve all islands for one generation.
        
        Args:
            evaluator: FitnessEvaluator object
            evolution_engine: EvolutionEngine to use for evolution
        """
        self.generation += 1
        
        # Evaluate current populations
        evaluation_results = self.evaluate_islands(evaluator)
        
        # Perform migration if it's time
        if self.generation % self.migration_interval == 0:
            self.perform_migration()
        
        # Evolve each island with its own parameters
        for island_idx in range(self.num_islands):
            # Set island-specific parameters
            evolution_engine.set_selection_pressure(self.selection_pressures[island_idx])
            evolution_engine.set_mutation_rate(self.mutation_rates[island_idx])
            evolution_engine.set_crossover_rate(self.crossover_rates[island_idx])
            
            # Get the current population and fitness scores for this island
            island = self.islands[island_idx]
            fitness_scores = self.fitness_scores[island_idx]
            
            # Perform selection, crossover, mutation within this island
            try:
                # Use the evolution engine to evolve this island
                evolution_engine.population = island
                evolution_engine.fitness_scores = fitness_scores
                
                # Select parents
                parents = evolution_engine.select_parents()
                
                # Create offspring
                offspring = evolution_engine.create_offspring(parents)
                
                # Select survivors
                new_population = evolution_engine.select_survivors(offspring)
                
                # Update island population
                self.islands[island_idx] = new_population
                
                logger.debug(f"Evolved island {island_idx} with {len(new_population)} individuals")
            except Exception as e:
                logger.error(f"Error evolving island {island_idx}: {str(e)}")
        
        return evaluation_results
    
    def perform_migration(self):
        """
        Perform migration between islands according to migration topology and rate.
        """
        logger.info(f"Performing migration at generation {self.generation}")
        
        # Record migration statistics
        migration_stats = {
            'generation': self.generation,
            'migrations': []
        }
        
        # For each island
        for from_idx in range(self.num_islands):
            # Get source population and fitness scores
            source_pop = self.islands[from_idx]
            source_fitness = self.fitness_scores[from_idx]
            
            if not source_pop or not source_fitness:
                continue
            
            # Calculate how many individuals to migrate from this island
            num_migrants = max(1, int(len(source_pop) * self.migration_rate))
            
            # Sort by fitness (descending)
            sorted_indices = sorted(
                range(len(source_fitness)), 
                key=lambda i: source_fitness[i],
                reverse=True
            )
            
            # Select migrants (best individuals)
            migrants = [copy.deepcopy(source_pop[i]) for i in sorted_indices[:num_migrants]]
            
            # Determine target island (ring topology)
            to_idx = (from_idx + 1) % self.num_islands
            
            # Get target population and fitness
            target_pop = self.islands[to_idx]
            target_fitness = self.fitness_scores[to_idx]
            
            if not target_pop or not target_fitness:
                continue
            
            # Replace worst individuals in target island
            if target_fitness:
                # Sort target by fitness (ascending, to get worst first)
                target_sorted = sorted(
                    range(len(target_fitness)), 
                    key=lambda i: target_fitness[i]
                )
                
                # Replace worst individuals with migrants
                for i, migrant_idx in enumerate(target_sorted[:num_migrants]):
                    if i < len(migrants):
                        target_pop[migrant_idx] = migrants[i]
            
            # Record migration for statistics
            migration_stats['migrations'].append({
                'from': from_idx,
                'to': to_idx,
                'count': min(num_migrants, len(migrants)),
                'source_size': len(source_pop),
                'target_size': len(target_pop)
            })
        
        # Add migration record to history
        self.migration_history.append(migration_stats)
    
    def get_best_solution(self):
        """
        Get the best solution across all islands.
        
        Returns:
            Tuple of (best_genome, best_fitness)
        """
        best_fitness = float('-inf')
        best_genome = None
        best_island = -1
        
        for island_idx, (island, fitness_list) in enumerate(zip(self.islands, self.fitness_scores)):
            if not island or not fitness_list:
                continue
                
            max_fitness = max(fitness_list)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = fitness_list.index(max_fitness)
                best_genome = island[best_idx]
                best_island = island_idx
        
        logger.info(f"Best solution found on island {best_island} with fitness {best_fitness}")
        return best_genome, best_fitness
    
    def get_island_best_solutions(self):
        """
        Get the best solution from each island.
        
        Returns:
            List of (island_idx, genome, fitness) tuples
        """
        best_solutions = []
        
        for island_idx, (island, fitness_list) in enumerate(zip(self.islands, self.fitness_scores)):
            if not island or not fitness_list:
                continue
                
            max_fitness = max(fitness_list)
            best_idx = fitness_list.index(max_fitness)
            best_genome = island[best_idx]
            
            best_solutions.append((island_idx, best_genome, max_fitness))
        
        return best_solutions
    
    def merge_islands(self):
        """
        Merge all islands into a single population.
        
        Returns:
            Combined list of genomes from all islands
        """
        combined = []
        
        for island_idx, island in enumerate(self.islands):
            if not island:
                continue
                
            combined.extend(copy.deepcopy(island))
        
        return combined
    
    def get_population_size(self):
        """
        Get the total population size across all islands.
        
        Returns:
            Total number of genomes across all islands
        """
        return sum(len(island) for island in self.islands)
    
    def get_statistics(self):
        """
        Get statistics about the island ecosystem.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'generation': self.generation,
            'num_islands': self.num_islands,
            'total_population': self.get_population_size(),
            'island_sizes': [len(island) for island in self.islands],
            'migration_count': len(self.migration_history),
            'island_stats': {},
            'migration_events': len(self.migration_history),
        }
        
        # Add per-island statistics
        for i in range(self.num_islands):
            island_stats = self.island_statistics[i]
            if island_stats['best_fitness']:
                best_fitness = max(f for f in island_stats['best_fitness'] if f != float('-inf'))
                stats['island_stats'][i] = {
                    'current_size': len(self.islands[i]),
                    'best_fitness': best_fitness,
                    'current_fitness': self.island_statistics[i]['best_fitness'][-1] if self.island_statistics[i]['best_fitness'] else float('-inf'),
                    'parameters': {
                        'selection_pressure': self.selection_pressures[i],
                        'mutation_rate': self.mutation_rates[i],
                        'crossover_rate': self.crossover_rates[i]
                    }
                }
        
        return stats
    
    def get_diversity_between_islands(self):
        """
        Calculate diversity between islands using representative samples.
        
        Returns:
            Float representing inter-island diversity (0.0 to 1.0)
        """
        if self.num_islands < 2:
            return 0.0
        
        # Calculate island fitness centroid (normalized)
        centroids = []
        for i, fitness_list in enumerate(self.fitness_scores):
            valid_fitness = [f for f in fitness_list if f != float('-inf')]
            if not valid_fitness:
                centroids.append(0.0)
                continue
            
            # Normalize to 0-1 range
            min_fit = min(valid_fitness)
            range_fit = max(valid_fitness) - min_fit
            if range_fit > 0:
                normalized = [(f - min_fit) / range_fit for f in valid_fitness]
            else:
                normalized = [0.5] * len(valid_fitness)
            
            # Calculate centroid (average)
            centroids.append(sum(normalized) / len(normalized))
        
        # Calculate diversity as coefficient of variation of centroids
        if not centroids:
            return 0.0
            
        mean = sum(centroids) / len(centroids)
        if mean == 0:
            return 0.0
            
        variance = sum((c - mean) ** 2 for c in centroids) / len(centroids)
        std_dev = variance ** 0.5
        cv = std_dev / mean if mean > 0 else 0
        
        # Map to 0-1 range
        diversity = min(1.0, cv / 0.5)
        return diversity
    
    def report(self):
        """
        Generate a text report of island ecosystem status.
        
        Returns:
            Report text
        """
        stats = self.get_statistics()
        diversity = self.get_diversity_between_islands()
        
        lines = []
        lines.append("=== ISLAND ECOSYSTEM REPORT ===")
        lines.append(f"Generation: {stats['generation']}")
        lines.append(f"Islands: {stats['num_islands']}")
        lines.append(f"Total population: {stats['total_population']}")
        lines.append(f"Inter-island diversity: {diversity:.4f}")
        lines.append(f"Migration events: {stats['migration_events']}")
        lines.append("")
        lines.append("Island statistics:")
        
        for i, island_stats in stats['island_stats'].items():
            lines.append(f"  Island {i}:")
            lines.append(f"    Size: {island_stats['current_size']}")
            lines.append(f"    Best fitness: {island_stats['best_fitness']:.4f}")
            lines.append(f"    Current fitness: {island_stats['current_fitness']:.4f}")
            lines.append(f"    Selection pressure: {island_stats['parameters']['selection_pressure']:.2f}")
            lines.append(f"    Mutation rate: {island_stats['parameters']['mutation_rate']:.3f}")
            lines.append(f"    Crossover rate: {island_stats['parameters']['crossover_rate']:.2f}")
        
        lines.append("=============================")
        return "\n".join(lines)
    
    def __str__(self):
        """String representation of the island ecosystem."""
        return (
            f"IslandEcosystemManager(islands={self.num_islands}, "
            f"generation={self.generation}, "
            f"population={self.get_population_size()}, "
            f"migrations={len(self.migration_history)})"
        )
