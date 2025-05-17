"""
Main Evolutionary Engine for the TRISOLARIS framework.

This module implements the core evolutionary loop that drives the optimization process
within a sandboxed environment for safe and controlled evolution.
"""

import os
import random
import time
import logging
import multiprocessing
import functools
from typing import List, Callable, Optional, Any, Dict, Tuple, Set
from pathlib import Path
import hashlib

from trisolaris.core.genome import CodeGenome
from trisolaris.environment.sandbox import SandboxedEnvironment, ResourceLimitExceededError, ExecutionTimeoutError
from trisolaris.environment.simulator import ResourceSimulator
from trisolaris.managers.resource_scheduler import ResourceScheduler, BatchProcessor
from trisolaris.config import get_config, BaseConfig

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
        evaluator: Any = None,
        genome_class: type = CodeGenome,
        resource_monitor = None,
        diversity_guardian = None,
        repository = None,
        config: Optional[BaseConfig] = None,
        component_name: str = "evolution_engine",
        run_id: Optional[str] = None
    ):
        """
        Initialize the Evolution Engine with specified parameters.
        
        Args:
            evaluator: FitnessEvaluator object to assess solutions
            genome_class: Class to use for representing individual solutions
            resource_monitor: Optional resource monitoring component
            diversity_guardian: Optional diversity maintenance component
            repository: Optional repository for storing solutions
            config: Configuration object (if None, will be loaded from global config)
            component_name: Name of this component for configuration lookup
            run_id: Optional run ID for configuration lookup
        """
        # Load configuration
        self.config = config or get_config(component_name, run_id)
        self.component_name = component_name
        self.run_id = run_id
        
        # Core parameters from configuration
        self.population_size = self.config.evolution.population_size
        self.selection_pressure = self.config.evolution.selection_pressure
        self.mutation_rate = self.config.evolution.mutation_rate
        self.crossover_rate = self.config.evolution.crossover_rate
        self.elitism_ratio = self.config.evolution.elitism_ratio
        
        # Provided parameters
        self.evaluator = evaluator
        self.genome_class = genome_class
        
        # Optional components
        self.resource_monitor = resource_monitor
        self.diversity_guardian = diversity_guardian
        self.repository = repository
        
        # Sandbox configuration
        self.sandbox_dir = self.config.sandbox.base_dir
        self.max_cpu_percent = self.config.sandbox.resource_limits.max_cpu_percent
        self.max_memory_percent = self.config.sandbox.resource_limits.max_memory_percent
        self.max_execution_time = self.config.sandbox.resource_limits.max_execution_time
        self.use_sandbox = True  # Always use sandbox for safety
        self.sandbox = None
        self.simulator = None
        
        # Performance optimization parameters
        self.parallel_evaluation = self.config.evolution.parallel_evaluation
        self.max_workers = self.config.evolution.max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.use_caching = self.config.evolution.use_caching
        self.early_stopping = self.config.evolution.early_stopping
        self.early_stopping_generations = self.config.evolution.early_stopping_generations
        self.early_stopping_threshold = self.config.evolution.early_stopping_threshold
        self.resource_aware = self.config.evolution.resource_aware
        
        # Initialize resource scheduler if resource-aware mode is enabled
        self.resource_scheduler = None
        if self.resource_aware:
            self.resource_scheduler = ResourceScheduler(
                target_cpu_usage=self.config.resource_scheduler.target_cpu_usage,
                target_memory_usage=self.config.resource_scheduler.target_memory_usage,
                min_cpu_available=self.config.resource_scheduler.min_cpu_available,
                min_memory_available=self.config.resource_scheduler.min_memory_available,
                adaptive_batch_size=self.config.resource_scheduler.adaptive_batch_size,
                initial_batch_size=min(self.config.resource_scheduler.initial_batch_size,
                                      self.population_size // 4)
            )
        
        # Initialize sandbox if enabled
        if self.use_sandbox:
            self._initialize_sandbox()
        
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
            'diversity': [],
            'execution_time': [],
            'resource_usage': []
        }
        
        # Fitness caching
        self.fitness_cache = {}
        self.generations_without_improvement = 0
        self.previous_best_fitness = float('-inf')
    
    def initialize_population(self):
        """Initialize a random population of solutions."""
        self.population = [self.genome_class() for _ in range(self.population_size)]
        logger.info(f"Initialized population with {self.population_size} individuals")
    
    def _initialize_sandbox(self):
        """Initialize the sandbox environment."""
        try:
            # The SandboxedEnvironment constructor already pulls resource limits
            # from the provided `BaseConfig`; passing individual limit parameters
            # would raise a `TypeError`. Therefore, we only forward the `base_dir`
            # (allowing the sandbox path to be overridden) together with the active
            # configuration object.
            self.sandbox = SandboxedEnvironment(
                base_dir=self.sandbox_dir,
                config=self.config,
                component_name="sandbox",
                run_id=self.run_id
            )
            
            # Initialize resource simulator
            self.simulator = ResourceSimulator(self.sandbox.base_dir)
            
            logger.info(f"Initialized sandbox environment at {self.sandbox.base_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {str(e)}")
            self.use_sandbox = False
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals in the population."""
        if self.evaluator is None:
            raise ValueError("Evaluator not set. Cannot evaluate population.")
        
        # Check if resource monitor allows evaluation
        if self.resource_monitor and not self.resource_monitor.can_proceed():
            logger.warning("Resource constraints exceeded. Throttling evaluation.")
            self.fitness_scores = [float('-inf')] * len(self.population)
            return
        
        # Start timing the evaluation
        eval_start_time = time.time()
        
        # Start resource monitoring if resource-aware mode is enabled
        if self.resource_aware and self.resource_scheduler:
            self.resource_scheduler.start_monitoring()
        
        try:
            # Evaluate each individual (either in parallel or sequentially)
            if self.parallel_evaluation and len(self.population) > 1:
                self.fitness_scores = self._parallel_evaluate_population()
            else:
                self.fitness_scores = self._sequential_evaluate_population()
        finally:
            # Stop resource monitoring if it was started
            if self.resource_aware and self.resource_scheduler:
                self.resource_scheduler.stop_monitoring()
        
        # Update best solution
        self._update_best_solution()
        
        # Update history
        valid_scores = [f for f in self.fitness_scores if f != float('-inf')]
        avg_fitness = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        self.history['best_fitness'].append(self.best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        
        # Check for early stopping
        if self.early_stopping:
            self._check_early_stopping()
        
        # Log evaluation time
        eval_time = time.time() - eval_start_time
        logger.info(f"Generation {self.generation}: Best fitness = {self.best_fitness}, "
                   f"Avg fitness = {avg_fitness}, Evaluation time = {eval_time:.2f}s")
    
    def _sequential_evaluate_population(self) -> List[float]:
        """Evaluate the population sequentially."""
        fitness_scores = []
        for genome in self.population:
            fitness = self._evaluate_genome(genome)
            genome.set_fitness(fitness)
            fitness_scores.append(fitness)
        return fitness_scores
    
    def _parallel_evaluate_population(self) -> List[float]:
        """Evaluate the population in parallel using multiprocessing."""
        try:
            # Determine optimal worker count based on resource availability
            worker_count = self.max_workers
            if self.resource_aware and self.resource_scheduler:
                worker_count = self.resource_scheduler.get_optimal_worker_count()
                logger.info(f"Using {worker_count} workers based on resource availability")
            
            # If resource-aware scheduling is enabled, use batch processing
            if self.resource_aware and self.resource_scheduler:
                return self._resource_aware_parallel_evaluate()
            
            # Otherwise, use standard multiprocessing
            with multiprocessing.Pool(processes=worker_count) as pool:
                # Map the evaluation function to each genome
                fitness_scores = pool.map(self._evaluate_genome_wrapper, self.population)
                # Set fitness values for each genome
                for genome, fitness in zip(self.population, fitness_scores):
                    genome.set_fitness(fitness)
            return fitness_scores
        except Exception as e:
            logger.error(f"Error in parallel evaluation: {str(e)}")
            # Fall back to sequential evaluation
            return self._sequential_evaluate_population()
    
    def _resource_aware_parallel_evaluate(self) -> List[float]:
        """
        Evaluate the population using resource-aware parallel processing.
        
        This method uses the ResourceScheduler to optimize resource usage
        during parallel evaluation.
        
        Returns:
            List of fitness scores
        """
        # Create a batch processor with the resource scheduler
        batch_processor = BatchProcessor(
            scheduler=self.resource_scheduler,
            items=self.population,
            process_func=self._evaluate_genome_wrapper
        )
        
        # Process all genomes in resource-aware batches
        return batch_processor.process_all()
    
    def _evaluate_genome_wrapper(self, genome):
        """Wrapper for parallel evaluation to handle exceptions."""
        try:
            return self._evaluate_genome(genome)
        except Exception as e:
            logger.error(f"Error evaluating genome in parallel: {str(e)}")
            return float('-inf')
    
    def _evaluate_genome(self, genome) -> float:
        """
        Evaluate a single genome, with caching if enabled.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Fitness score
        """
        # Check cache first if enabled
        if self.use_caching:
            cache_key = self._get_genome_hash(genome)
            if cache_key in self.fitness_cache:
                return self.fitness_cache[cache_key]
        
        # Compute fitness (with or without sandbox)
        if self.use_sandbox and self.sandbox:
            fitness = self._evaluate_in_sandbox(genome)
        else:
            # Apply ethical filter if available (only when not using sandbox)
            if hasattr(self.evaluator, 'check_ethical_boundaries'):
                if not self.evaluator.check_ethical_boundaries(genome):
                    return float('-inf')
            
            # Compute fitness directly
            fitness = self.evaluator.evaluate(genome)
        
        # Cache the result if caching is enabled
        if self.use_caching:
            self.fitness_cache[cache_key] = fitness
        
        return fitness
    
    def _get_genome_hash(self, genome) -> str:
        """
        Generate a hash for a genome to use as a cache key.
        
        Args:
            genome: The genome to hash
            
        Returns:
            Hash string
        """
        try:
            source_code = genome.to_source()
            return hashlib.md5(source_code.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating genome hash: {str(e)}")
            # Fallback to object ID if hashing fails
            return str(id(genome))
    
    def _update_best_solution(self):
        """Update the best solution based on current fitness scores."""
        if not self.fitness_scores:
            return
        
        current_best_idx = max(range(len(self.fitness_scores)),
                              key=lambda i: self.fitness_scores[i])
        current_best_fitness = self.fitness_scores[current_best_idx]
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = self.population[current_best_idx].clone()
            if self.repository:
                self.repository.store_solution(
                    self.population[current_best_idx],
                    current_best_fitness,
                    self.generation
                )
    
    def _check_early_stopping(self):
        """Check if early stopping criteria are met."""
        if self.best_fitness > self.previous_best_fitness + self.early_stopping_threshold:
            # Reset counter if there's significant improvement
            self.generations_without_improvement = 0
        else:
            # Increment counter if no significant improvement
            self.generations_without_improvement += 1
        
        # Update previous best fitness
        self.previous_best_fitness = self.best_fitness
        
        # Log early stopping status
        if self.generations_without_improvement > 0:
            logger.info(f"Generations without significant improvement: {self.generations_without_improvement}/{self.early_stopping_generations}")
    
    def _evaluate_in_sandbox(self, genome) -> float:
        """
        Evaluate a genome within the sandbox environment.
        
        Args:
            genome: The genome to evaluate
            
        Returns:
            Fitness score
        """
        try:
            # Save genome to sandbox
            source_code = genome.to_source()
            code_path = self.sandbox.save_file(source_code, 'code', f'genome_{id(genome)}.py')
            
            # Define evaluation function to run in sandbox
            def sandboxed_evaluation():
                # Track execution time
                start_time = time.time()
                
                try:
                    # Evaluate the genome
                    fitness = self.evaluator.evaluate(genome)
                    
                    # Track resource usage
                    execution_time = time.time() - start_time
                    
                    return {
                        'fitness': fitness,
                        'execution_time': execution_time,
                        'success': True
                    }
                except Exception as e:
                    logger.warning(f"Error in sandboxed evaluation: {str(e)}")
                    return {
                        'fitness': float('-inf'),
                        'execution_time': time.time() - start_time,
                        'success': False,
                        'error': str(e)
                    }
            
            # Execute evaluation in sandbox
            try:
                result = self.sandbox.execute_in_sandbox(sandboxed_evaluation)
                
                # Log execution metrics
                self.sandbox.log_execution(
                    "evaluate_genome",
                    {
                        'genome_id': id(genome),
                        'execution_time': result.get('execution_time', 0),
                        'success': result.get('success', False)
                    }
                )
                
                # Update resource usage history
                resource_usage = self.sandbox.get_resource_usage_report()
                self.history['execution_time'].append(result.get('execution_time', 0))
                self.history['resource_usage'].append(resource_usage)
                
                return result.get('fitness', float('-inf'))
                
            except ResourceLimitExceededError as e:
                logger.warning(f"Resource limit exceeded during evaluation: {str(e)}")
                return float('-inf')
                
            except ExecutionTimeoutError as e:
                logger.warning(f"Execution timeout during evaluation: {str(e)}")
                return float('-inf')
                
            except Exception as e:
                logger.error(f"Error in sandbox execution: {str(e)}")
                return float('-inf')
                
        except Exception as e:
            logger.error(f"Error preparing sandboxed evaluation: {str(e)}")
            return float('-inf')
    
    def select_parents(self) -> List[CodeGenome]:
        """
        Select parents for reproduction using optimized tournament selection.
        
        This implementation uses a more efficient tournament selection algorithm
        that avoids redundant fitness lookups and uses a faster selection method.
        """
        tournament_size = max(2, int(self.population_size * self.selection_pressure * 0.1))
        parents = []
        
        # Create fitness lookup dictionary for faster access
        fitness_lookup = {i: self.fitness_scores[i] for i in range(len(self.fitness_scores))}
        
        # Pre-compute valid indices (exclude individuals with -inf fitness)
        valid_indices = [i for i, f in enumerate(self.fitness_scores) if f != float('-inf')]
        if not valid_indices:
            # If no valid individuals, return random selection
            return random.choices(self.population, k=self.population_size)
        
        for _ in range(self.population_size):
            # Tournament selection with sampling from valid indices
            if len(valid_indices) <= tournament_size:
                # If we have fewer valid individuals than tournament size, use all valid ones
                tournament_indices = valid_indices
            else:
                # Otherwise, sample from valid indices
                tournament_indices = random.sample(valid_indices, tournament_size)
            
            # Find the winner directly without creating intermediate lists
            winner_idx = max(tournament_indices, key=lambda i: fitness_lookup[i])
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
    
    def evolve(self, generations: Optional[int] = None, max_time: Optional[int] = None,
              target_fitness: Optional[float] = None) -> CodeGenome:
        """
        Run the evolutionary process for a specified number of generations.
        
        Args:
            generations: Maximum number of generations to evolve (defaults to 100)
            max_time: Maximum time in seconds for the evolutionary process
            target_fitness: Target fitness to achieve (stops when reached)
            
        Returns:
            The best solution found
        """
        # Use default generations if not specified
        if generations is None:
            generations = 100
        self.start_time = time.time()
        
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        try:
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
                
                # Check early stopping condition
                if (self.early_stopping and
                    self.generations_without_improvement >= self.early_stopping_generations):
                    logger.info(f"Early stopping triggered after {self.generations_without_improvement} "
                               f"generations without improvement")
                    break
                
                # Evaluate current population
                self.evaluate_population()
                
                # Create next generation
                parents = self.select_parents()
                offspring = self.create_offspring(parents)
                self.population = self.select_survivors(offspring)
                
                # Apply resource-aware throttling if needed
                if self.resource_aware and self.resource_scheduler and self.resource_scheduler.should_throttle():
                    logger.info("Resource constraints detected. Applying resource-aware throttling.")
                    params = self.resource_scheduler.get_throttle_parameters()
                    
                    # Adjust population size based on resource availability
                    if params['population_scale_factor'] < 0.9:  # Only adjust if significant reduction is needed
                        effective_pop_size = max(10, int(self.population_size * params['population_scale_factor']))
                        logger.info(f"Reducing effective population size to {effective_pop_size} due to resource constraints")
                        
                        # Trim population to effective size
                        if len(self.population) > effective_pop_size:
                            # Keep the best individuals
                            indices = sorted(range(len(self.fitness_scores)),
                                           key=lambda i: self.fitness_scores[i],
                                           reverse=True)[:effective_pop_size]
                            self.population = [self.population[i] for i in indices]
                            self.fitness_scores = [self.fitness_scores[i] for i in indices]
                    
                    # Wait for resources if needed
                    if params['population_scale_factor'] < 0.5:  # Severe resource constraints
                        logger.info("Severe resource constraints. Waiting for resources to become available.")
                        wait_timeout = self.config.resource_scheduler.check_interval * 5
                        self.resource_scheduler.wait_for_resources(timeout=wait_timeout)
                
                # Maintain diversity if needed
                self.maintain_diversity()
                
                # Periodically clean the cache to prevent memory bloat
                if self.use_caching and self.generation % 10 == 0 and self.generation > 0:
                    self._prune_fitness_cache()
        
        finally:
            # Clean up sandbox resources if used
            if self.use_sandbox and self.sandbox:
                # Don't actually clean up the sandbox directory to allow for inspection
                # but do clean up any open resources
                if self.simulator:
                    self.simulator.cleanup()
        
        return self.get_best_solution()
    
    def _prune_fitness_cache(self):
        """Prune the fitness cache to prevent memory bloat."""
        if len(self.fitness_cache) > self.population_size * 10:
            logger.info(f"Pruning fitness cache from {len(self.fitness_cache)} entries")
            
            # Get current genome hashes
            current_hashes = set()
            for genome in self.population:
                try:
                    current_hashes.add(self._get_genome_hash(genome))
                except Exception:
                    pass
            
            # Keep only entries for current population plus some margin
            new_cache = {}
            for genome_hash, fitness in self.fitness_cache.items():
                if genome_hash in current_hashes or len(new_cache) < self.population_size * 2:
                    new_cache[genome_hash] = fitness
            
            self.fitness_cache = new_cache
            logger.info(f"Fitness cache pruned to {len(self.fitness_cache)} entries")
    
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
            
    def update_config(self, config: BaseConfig) -> None:
        """
        Update the engine's configuration.
        
        Args:
            config: New configuration object
        """
        self.config = config
        
        # Update core parameters
        self.population_size = self.config.evolution.population_size
        self.selection_pressure = self.config.evolution.selection_pressure
        self.mutation_rate = self.config.evolution.mutation_rate
        self.crossover_rate = self.config.evolution.crossover_rate
        self.elitism_ratio = self.config.evolution.elitism_ratio
        
        # Update sandbox configuration
        self.sandbox_dir = self.config.sandbox.base_dir
        self.max_cpu_percent = self.config.sandbox.resource_limits.max_cpu_percent
        self.max_memory_percent = self.config.sandbox.resource_limits.max_memory_percent
        self.max_execution_time = self.config.sandbox.resource_limits.max_execution_time
        
        # Update performance optimization parameters
        self.parallel_evaluation = self.config.evolution.parallel_evaluation
        self.max_workers = self.config.evolution.max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.use_caching = self.config.evolution.use_caching
        self.early_stopping = self.config.evolution.early_stopping
        self.early_stopping_generations = self.config.evolution.early_stopping_generations
        self.early_stopping_threshold = self.config.evolution.early_stopping_threshold
        self.resource_aware = self.config.evolution.resource_aware
        
        # Update resource scheduler if it exists
        if self.resource_scheduler:
            self.resource_scheduler = ResourceScheduler(
                target_cpu_usage=self.config.resource_scheduler.target_cpu_usage,
                target_memory_usage=self.config.resource_scheduler.target_memory_usage,
                min_cpu_available=self.config.resource_scheduler.min_cpu_available,
                min_memory_available=self.config.resource_scheduler.min_memory_available,
                adaptive_batch_size=self.config.resource_scheduler.adaptive_batch_size,
                initial_batch_size=min(self.config.resource_scheduler.initial_batch_size,
                                      self.population_size // 4)
            )

    def get_sandbox_path(self) -> Optional[Path]:
        """
        Get the path to the sandbox directory.
        
        Returns:
            Path to the sandbox directory or None if sandbox is not used
        """
        if self.use_sandbox and self.sandbox:
            return self.sandbox.base_dir
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about execution time and resource usage.
        
        Returns:
            Dictionary with execution statistics
        """
        stats = {
            'generations': self.generation,
            'total_time': time.time() - self.start_time if self.start_time else 0,
        }
        
        if self.use_sandbox and self.sandbox:
            # Add sandbox-specific stats
            stats.update({
                'sandbox_path': str(self.sandbox.base_dir),
                'max_cpu_percent': self.max_cpu_percent,
                'max_memory_percent': self.max_memory_percent,
                'max_execution_time': self.max_execution_time,
            })
            
            # Add resource usage history if available
            if 'execution_time' in self.history and self.history['execution_time']:
                stats.update({
                    'avg_execution_time': sum(self.history['execution_time']) / len(self.history['execution_time']),
                    'max_execution_time_observed': max(self.history['execution_time']),
                })
        
        # Add optimization-related stats
        stats.update({
            'parallel_evaluation': self.parallel_evaluation,
            'max_workers': self.max_workers,
            'use_caching': self.use_caching,
            'cache_size': len(self.fitness_cache) if self.use_caching else 0,
            'early_stopping': self.early_stopping,
            'generations_without_improvement': self.generations_without_improvement,
            'resource_aware': self.resource_aware,
        })
        
        # Add resource scheduler stats if available
        if self.resource_aware and self.resource_scheduler:
            stats.update({
                'resource_summary': self.resource_scheduler.get_resource_summary()
            })
        
        return stats