#!/usr/bin/env python3
"""
TRISOLARIS Task Runner Class

A wrapper class that uses the TRISOLARIS framework to evolve
solutions for any task that implements the TaskInterface.
"""

import os
import sys
import json
import datetime
import logging
import importlib
import shutil
from typing import Dict, Any, Tuple, List, Optional

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
from trisolaris.managers.resource import ResourceSteward
from trisolaris.managers.repository import GenomeRepository
from trisolaris.managers.diversity import DiversityGuardian
from trisolaris.managers.island import IslandEcosystemManager
from trisolaris.tasks import TaskInterface
from trisolaris.tasks.network_scanner import NetworkScannerTask
from trisolaris.utils.paths import create_timestamped_output_dir, create_generation_dir
import trisolaris_task_runner as task_runner_impl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trisolaris_evolution.log")
    ]
)
logger = logging.getLogger(__name__)

class TaskRunner:
    """
    A class that manages the evolution process for a task.
    
    This wraps the functionality from trisolaris_task_runner.py to provide
    a clean interface for scripts like evolve_network_scanner.py.
    """
    
    def __init__(
        self,
        task: TaskInterface,
        output_dir: str = "outputs",
        num_generations: int = 10,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        ethics_level: str = "basic",
        resource_monitoring: bool = False,
        use_git: bool = False,
        use_islands: bool = False,
        islands: int = 3,
        migration_interval: int = 3,
        diversity_threshold: float = 0.3,
        show_resource_report: bool = False
    ):
        """
        Initialize the task runner.
        
        Args:
            task: The task to evolve a solution for
            output_dir: Directory to save outputs
            num_generations: Number of generations to run
            population_size: Population size for each generation
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
            ethics_level: Ethical filter level (none, basic, full)
            resource_monitoring: Whether to enable resource monitoring
            use_git: Whether to use Git for version control
            use_islands: Whether to use island model for evolution
            islands: Number of islands when using island model
            migration_interval: Number of generations between migrations
            diversity_threshold: Diversity threshold for injection
            show_resource_report: Whether to show resource usage report
        """
        self.task = task
        self.output_dir = output_dir
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.ethics_level = ethics_level
        self.resource_monitoring = resource_monitoring
        self.use_git = use_git
        self.use_islands = use_islands
        self.islands = islands
        self.migration_interval = migration_interval
        self.diversity_threshold = diversity_threshold
        self.show_resource_report = show_resource_report
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized TaskRunner for {task.get_name()}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Generations: {num_generations}, Population: {population_size}")
    
    def run(self) -> Tuple[CodeGenome, Dict[str, Any]]:
        """
        Run the evolution process.
        
        Returns:
            A tuple containing (best_genome, statistics)
        """
        logger.info(f"Starting evolution for {self.task.get_name()}")
        
        # Create timestamped output directory
        task_output_dir = os.path.join(self.output_dir, f"{self.task.get_name()}_evolution")
        run_dir = create_timestamped_output_dir(task_output_dir)
        logger.info(f"Created output directory: {run_dir}")
        
        # Initialize repository for genome storage
        repository = GenomeRepository(
            base_dir=task_output_dir,
            run_id=os.path.basename(run_dir),
            use_git=self.use_git
        )
        
        # Initialize resource monitoring if enabled
        resource_monitor = None
        if self.resource_monitoring:
            resource_monitor = ResourceSteward(
                min_available_memory=0.25,
                min_available_cpu=0.25
            )
            logger.info("Resource monitoring enabled")
            logger.info(resource_monitor.get_system_info())
        
        # Setup ethical filter based on chosen level
        ethical_filter = task_runner_impl.setup_ethical_filter(
            self.task, self.ethics_level, run_dir
        )
        if ethical_filter:
            logger.info(f"Ethical filter enabled at level: {self.ethics_level}")
            for boundary in ethical_filter.get_active_boundaries():
                logger.info(f"  - Boundary: {boundary}")
        
        # Set up the fitness evaluator
        evaluator = task_runner_impl.setup_fitness_evaluator(self.task, ethical_filter)
        
        # Initialize diversity guardian
        diversity_guardian = DiversityGuardian()
        
        # Create the initial population
        population = []
        
        try:
            # Get the template code from the task
            template_code = self.task.get_template()
            
            # Create the first genome from the template
            base_genome = CodeGenome.from_source(template_code)
            population.append(base_genome)
            
            # Create variants with small mutations to ensure diversity
            for i in range(1, self.population_size):
                # Clone the base genome
                genome = base_genome.clone()
                
                # Apply some mutations manually
                if hasattr(genome, 'mutate'):
                    try:
                        genome.mutate()  # No parameters, use default mutation rate
                    except Exception as e:
                        logger.warning(f"Error during mutation: {e}")
                
                population.append(genome)
                
        except Exception as e:
            logger.error(f"Error creating initial population: {e}")
            
            # If we can't create a population from the template, create empty genomes
            for i in range(self.population_size):
                population.append(CodeGenome())
        
        # Set up either island model or regular evolution engine
        if self.use_islands:
            logger.info(f"Using island model with {self.islands} islands")
            
            # Create island model manager
            # Calculate island sizes - distribute population evenly
            island_size = self.population_size // self.islands
            island_sizes = [island_size] * self.islands
            
            # Create the island manager with appropriate parameters
            island_manager = IslandEcosystemManager(
                num_islands=self.islands,
                island_sizes=island_sizes,
                migration_interval=self.migration_interval
            )
            
            # Initialize the islands with the initial population
            island_manager.initialize_islands(population, CodeGenome)
            
            # Use the island manager as our evolution engine
            engine = island_manager
        else:
            # Set up the regular evolution engine
            engine = EvolutionEngine(
                population_size=self.population_size,
                evaluator=evaluator,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                genome_class=CodeGenome,
                resource_monitor=resource_monitor,
                repository=repository,
                diversity_guardian=diversity_guardian
            )
            
            # Set the initial population
            engine.population = population
        
        
        # Run the evolution
        logger.info(f"Starting evolution with population size {self.population_size} for {self.num_generations} generations")
        logger.info(f"Ethics level: {self.ethics_level}")
        
        start_time = datetime.datetime.now()
        
        # Track generations
        best_fitness = 0.0
        best_genome = None
        
        metadata = {
            "task": self.task.get_name(),
            "description": self.task.get_description(),
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "ethics_level": self.ethics_level,
            "use_islands": self.use_islands,
            "start_time": start_time.isoformat(),
            "end_time": None,
            "best_fitness": None,
            "generation_metrics": []
        }
        
        for gen in range(self.num_generations):
            gen_start_time = datetime.datetime.now()
            
            # Check if resource monitor allows us to proceed
            if resource_monitor and not resource_monitor.can_proceed():
                logger.warning("Resource constraints exceeded. Waiting for resources...")
                resource_monitor.wait_for_resources(timeout=60)  # Wait up to 60 seconds
            
            # Create generation directory
            gen_dir = create_generation_dir(run_dir, gen)
            
            # Evaluate the current generation
            if self.use_islands:
                # For island model
                evaluation_results = engine.evaluate_islands(evaluator)
                best_of_gen, gen_fitness = engine.get_best_solution()
            else:
                # For regular evolution engine
                engine.evaluate_population()
                
                # Check diversity and inject if needed
                if hasattr(engine, 'check_diversity'):
                    engine.check_diversity()
                
                # Save the best solution of this generation
                best_of_gen = engine.get_best_solution()
                gen_fitness = engine.best_fitness
            
            if best_of_gen:
                # Update best overall solution
                if gen_fitness > best_fitness:
                    best_fitness = gen_fitness
                    best_genome = best_of_gen
                
                # Save to repository
                repository.store_solution(
                    genome=best_of_gen,
                    fitness=gen_fitness,
                    generation=gen,
                    metadata={"generation_time": (datetime.datetime.now() - gen_start_time).total_seconds()}
                )
                
                # Save to generation directory
                best_gen_path = os.path.join(gen_dir, "best.py")
                task_runner_impl.post_process_solution(
                    self.task, best_of_gen.to_source(), best_gen_path
                )
            
            # Log progress
            gen_time = (datetime.datetime.now() - gen_start_time).total_seconds()
            logger.info(f"Generation {gen} complete in {gen_time:.2f}s. Best fitness: {gen_fitness:.4f}")
            
            # Store metrics for this generation
            metadata["generation_metrics"].append({
                "generation": gen,
                "best_fitness": gen_fitness,
                "time_seconds": gen_time,
                "diversity": None  # We'll implement diversity metrics later
            })
            
            # Create the next generation (unless it's the last generation)
            if gen < self.num_generations - 1:
                # Apply throttling if resource monitor recommends it
                if resource_monitor and resource_monitor.get_throttle_level() > 0:
                    params = resource_monitor.get_throttling_parameters()
                    effective_pop_size = max(5, int(self.population_size * params['population_scale_factor']))
                    logger.info(f"Throttling: Reducing population size to {effective_pop_size} for next generation")
                    
                    # Delegate to engine, which will handle differently for island vs regular
                    if hasattr(engine, 'handle_throttling'):
                        engine.handle_throttling(params)
                    else:
                        # Standard throttling for regular engine
                        parents = engine.select_parents()[:effective_pop_size]
                        offspring = engine.create_offspring(parents)
                        engine.population = engine.select_survivors(offspring)[:effective_pop_size]
                else:
                    # Normal operation
                    if self.use_islands:
                        # Island model evolution
                        engine.evolve_islands(evaluator, EvolutionEngine(
                            population_size=10,  # This is per island
                            evaluator=evaluator,
                            mutation_rate=self.mutation_rate,
                            crossover_rate=self.crossover_rate,
                            genome_class=CodeGenome
                        ))
                    else:
                        # Standard evolution for regular engine
                        parents = engine.select_parents()
                        offspring = engine.create_offspring(parents)
                        engine.population = engine.select_survivors(offspring)
        
        # Save the final best solution
        if best_genome:
            # Update metadata
            end_time = datetime.datetime.now()
            metadata["end_time"] = end_time.isoformat()
            metadata["best_fitness"] = best_fitness
            metadata["duration_seconds"] = (end_time - start_time).total_seconds()
            
            # Save to repository
            repository.store_best_solution(
                genome=best_genome,
                fitness=best_fitness,
                generation=self.num_generations - 1
            )
            
            # Save to run directory
            best_path = os.path.join(run_dir, f"best_{self.task.get_name()}.py")
            task_runner_impl.post_process_solution(self.task, best_genome.to_source(), best_path)
            
            logger.info(f"Best solution saved to {best_path}")
            logger.info(f"Best fitness: {best_fitness:.4f}")
            
            # Copy to a standard location for easy access
            final_path = os.path.join(self.output_dir, f"evolved_{self.task.get_name()}.py")
            shutil.copy2(best_path, final_path)
            os.chmod(final_path, 0o755)
            logger.info(f"Best solution also copied to {final_path}")
        else:
            logger.warning("No valid solution found")
            metadata["end_time"] = datetime.datetime.now().isoformat()
            metadata["best_fitness"] = 0.0
        
        # Save metadata
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Generate a report
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Evolution complete in {total_time:.2f}s")
        
        # Show resource report if requested
        if self.show_resource_report and resource_monitor:
            try:
                resource_history = resource_monitor.get_resource_history()
                logger.info("Resource Usage Report:")
                
                if resource_history and isinstance(resource_history, list) and len(resource_history) > 0:
                    # Check if entries are dictionaries with expected keys
                    if all(isinstance(entry, dict) for entry in resource_history):
                        avg_cpu = sum(entry.get('cpu_percent', 0) for entry in resource_history) / len(resource_history)
                        avg_memory = sum(entry.get('memory_percent', 0) for entry in resource_history) / len(resource_history)
                        max_cpu = max(entry.get('cpu_percent', 0) for entry in resource_history)
                        max_memory = max(entry.get('memory_percent', 0) for entry in resource_history)
                        
                        logger.info(f"  Average CPU usage: {avg_cpu:.2f}%")
                        logger.info(f"  Maximum CPU usage: {max_cpu:.2f}%")
                        logger.info(f"  Average memory usage: {avg_memory:.2f}%")
                        logger.info(f"  Maximum memory usage: {max_memory:.2f}%")
                    else:
                        logger.info(f"  Resource data available but in unexpected format")
                else:
                    logger.info(f"  No resource monitoring data available")
            except Exception as e:
                logger.error(f"Error generating resource report: {str(e)}")
        
        # Save the repository summary
        repository_summary = repository.get_summary()
        logger.info(f"Repository summary: {repository_summary}")
        
        # Return best genome and statistics
        stats = {
            "best_fitness": best_fitness,
            "generations": self.num_generations,
            "duration_seconds": (end_time - start_time).total_seconds() if 'end_time' in locals() else 0,
            "output_path": best_path if 'best_path' in locals() else None,
            "metrics": metadata["generation_metrics"] if "generation_metrics" in metadata else []
        }
        
        return best_genome, stats
