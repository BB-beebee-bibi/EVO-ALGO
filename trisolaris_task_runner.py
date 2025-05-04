#!/usr/bin/env python3
"""
TRISOLARIS Task Runner

A generic task runner that uses the TRISOLARIS framework to evolve
solutions for any task that implements the TaskInterface.
"""

import os
import sys
import json
import datetime
import argparse
import importlib
import shutil
import logging
from typing import Dict, Any, List, Optional, Type

from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
from trisolaris.managers.resource import ResourceSteward
from trisolaris.managers.repository import GenomeRepository
from trisolaris.managers.diversity import DiversityGuardian
from trisolaris.managers.island import IslandEcosystemManager
from trisolaris.utils.paths import create_timestamped_output_dir, create_generation_dir
from trisolaris.tasks import TaskInterface, DriveScannerTask

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

# Dictionary mapping task names to their implementing classes
TASK_REGISTRY = {
    "drive_scanner": DriveScannerTask
}

def setup_ethical_filter(task: TaskInterface, level: str, output_dir: str) -> Optional[EthicalBoundaryEnforcer]:
    """
    Create an ethical filter based on the task's requirements and the specified level.
    
    Args:
        task: The task being evolved
        level: Ethical filter level (none, basic, full)
        output_dir: Output directory for results
        
    Returns:
        Configured EthicalBoundaryEnforcer or None if level is "none"
    """
    if level == "none":
        return None
    
    enforcer = EthicalBoundaryEnforcer()
    
    # Add boundaries based on task requirements
    if level == "basic":
        # Add required boundaries from task
        required_boundaries = task.get_required_boundaries()
        for boundary_name, params in required_boundaries.items():
            enforcer.add_boundary(boundary_name, **params)
    
    elif level == "full":
        # Full ethics includes all task requirements plus additional Gurbani-inspired boundaries
        required_boundaries = task.get_required_boundaries()
        for boundary_name, params in required_boundaries.items():
            enforcer.add_boundary(boundary_name, **params)
        
        # Add additional Gurbani-inspired boundaries
        enforcer.add_boundary("universal_equity")
        enforcer.add_boundary("truthful_communication")
        enforcer.add_boundary("humble_code")
        enforcer.add_boundary("service_oriented")
        enforcer.add_boundary("harmony_with_environment")
    
    # Add import restrictions based on task's allowed imports
    allowed_imports = task.get_allowed_imports()
    enforcer.add_boundary("no_imports", allowed_imports=set(allowed_imports))
    
    return enforcer

def setup_fitness_evaluator(task: TaskInterface, ethical_filter: Optional[EthicalBoundaryEnforcer]) -> FitnessEvaluator:
    """
    Set up the fitness evaluator with task-specific settings.
    
    Args:
        task: The task being evolved
        ethical_filter: Optional ethical filter
        
    Returns:
        Configured FitnessEvaluator
    """
    # Create a fitness evaluator with the ethical filter
    evaluator = FitnessEvaluator(ethical_filter=ethical_filter)
    
    # Configure the weights for objectives from task-specific settings
    fitness_weights = task.get_fitness_weights()
    evaluator.set_weights(**fitness_weights)
    
    # Add a custom alignment measure using the task's fitness evaluator
    def evaluate_task_fitness(genome):
        if hasattr(genome, 'to_source'):
            code = genome.to_source()
        else:
            code = str(genome)
        
        fitness, _ = task.evaluate_fitness(code)
        return fitness
    
    evaluator.add_alignment_measure(
        evaluate_task_fitness,
        weight=1.0,
        name=f"{task.get_name()}_fitness"
    )
    
    return evaluator

def create_initial_population(task: TaskInterface, population_size: int) -> list:
    """
    Create an initial population based on the task's template.
    
    Args:
        task: The task being evolved
        population_size: Size of the population to create
        
    Returns:
        List of CodeGenome instances
    """
    population = []
    
    try:
        # Get the template code from the task
        template_code = task.get_template()
        
        # Create the first genome from the template
        base_genome = CodeGenome.from_source(template_code)
        population.append(base_genome)
        
        # Create variants with different mutation rates to ensure diversity
        for i in range(1, population_size):
            # Clone the base genome
            genome = base_genome.clone()
            
            # Apply mutations with increasing intensity
            mutation_intensity = 0.05 * (i / (population_size - 1) * 5)
            genome.mutate(mutation_rate=mutation_intensity)
            
            population.append(genome)
            
    except Exception as e:
        logger.error(f"Error creating initial population: {e}")
        
        # If we can't create a population from the template, create random genomes
        for i in range(population_size):
            population.append(CodeGenome())
    
    return population

def post_process_solution(task: TaskInterface, solution: str, output_path: str) -> None:
    """
    Apply post-processing to the evolved solution and save it.
    
    Args:
        task: The task that was evolved
        solution: Source code of the evolved solution
        output_path: Path to save the post-processed solution
    """
    # Apply task-specific post-processing
    processed_solution = task.post_process(solution)
    
    # Save the post-processed solution
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(processed_solution)
    
    # Make the solution executable
    os.chmod(output_path, 0o755)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evolve code solutions for specific tasks using the TRISOLARIS framework"
    )
    
    # Task specification
    task_group = parser.add_argument_group("Task")
    task_group.add_argument(
        "task",
        choices=list(TASK_REGISTRY.keys()),
        help="The task to evolve a solution for"
    )
    task_group.add_argument(
        "--template",
        help="Path to a custom template file to start evolution from"
    )
    
    # Evolution parameters
    evo_group = parser.add_argument_group("Evolution Parameters")
    evo_group.add_argument(
        "--output-dir",
        "-o",
        default="outputs",
        help="Directory to save evolved code (default: outputs)"
    )
    evo_group.add_argument(
        "--pop-size",
        "-p",
        type=int,
        default=None,
        help="Population size (default: task-specific recommendation)"
    )
    evo_group.add_argument(
        "--gens",
        "-g",
        type=int,
        default=None,
        help="Number of generations (default: task-specific recommendation)"
    )
    evo_group.add_argument(
        "--mutation-rate",
        "-m",
        type=float,
        default=None,
        help="Mutation rate (default: task-specific recommendation)"
    )
    evo_group.add_argument(
        "--crossover-rate",
        "-c",
        type=float,
        default=None,
        help="Crossover rate (default: task-specific recommendation)"
    )
    
    # Ethical parameters
    ethics_group = parser.add_argument_group("Ethical Parameters")
    ethics_group.add_argument(
        "--ethics-level",
        "-e",
        choices=["none", "basic", "full"],
        default="basic",
        help="Ethical filter level (none, basic, full) (default: basic)"
    )
    
    # Advanced parameters
    advanced_group = parser.add_argument_group("Advanced Parameters")
    advanced_group.add_argument(
        "--resource-monitoring",
        action="store_true",
        help="Enable resource monitoring and throttling"
    )
    advanced_group.add_argument(
        "--use-git",
        action="store_true",
        help="Use Git for version control of solution history"
    )
    advanced_group.add_argument(
        "--use-islands",
        action="store_true",
        help="Use island model for evolution"
    )
    advanced_group.add_argument(
        "--islands",
        type=int,
        default=3,
        help="Number of islands when using island model (default: 3)"
    )
    advanced_group.add_argument(
        "--migration-interval",
        type=int,
        default=3,
        help="Number of generations between migrations (default: 3)"
    )
    advanced_group.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.3,
        help="Diversity threshold for triggering diversity injection (default: 0.3)"
    )
    
    args = parser.parse_args()

    # Initialize the task
    try:
        task_class = TASK_REGISTRY[args.task]
        task = task_class(template_path=args.template)
        logger.info(f"Initialized task: {task.get_name()} - {task.get_description()}")
    except KeyError:
        logger.error(f"Unknown task: {args.task}")
        return

    # Get task-specific parameters and override with command line arguments if provided
    task_params = task.get_evolution_params()
    
    population_size = args.pop_size if args.pop_size is not None else task_params.get("population_size", 20)
    num_generations = args.gens if args.gens is not None else task_params.get("num_generations", 10)
    mutation_rate = args.mutation_rate if args.mutation_rate is not None else task_params.get("mutation_rate", 0.1)
    crossover_rate = args.crossover_rate if args.crossover_rate is not None else task_params.get("crossover_rate", 0.7)
    
    # Create timestamped output directory
    task_output_dir = os.path.join(args.output_dir, f"{task.get_name()}_evolution")
    run_dir = create_timestamped_output_dir(task_output_dir)
    logger.info(f"Created output directory: {run_dir}")
    
    # Initialize repository for genome storage
    repository = GenomeRepository(
        base_dir=task_output_dir,
        run_id=os.path.basename(run_dir),
        use_git=args.use_git
    )
    
    # Initialize resource monitoring if enabled
    resource_monitor = None
    if args.resource_monitoring:
        resource_monitor = ResourceSteward(
            min_available_memory=0.25,
            min_available_cpu=0.25
        )
        logger.info("Resource monitoring enabled")
        logger.info(resource_monitor.get_system_info())
    
    # Setup ethical filter based on chosen level
    ethical_filter = setup_ethical_filter(task, args.ethics_level, run_dir)
    if ethical_filter:
        logger.info(f"Ethical filter enabled at level: {args.ethics_level}")
        for boundary in ethical_filter.get_active_boundaries():
            logger.info(f"  - Boundary: {boundary}")
    
    # Set up the fitness evaluator
    evaluator = setup_fitness_evaluator(task, ethical_filter)
    
    # Initialize diversity guardian
    diversity_guardian = DiversityGuardian(
        threshold=args.diversity_threshold
    )
    
    # Set up either island model or regular evolution engine
    if args.use_islands:
        logger.info(f"Using island model with {args.islands} islands")
        
        # Create island model manager
        engine = IslandEcosystemManager(
            num_islands=args.islands,
            island_population_size=population_size // args.islands,
            evaluator=evaluator,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            genome_class=CodeGenome,
            migration_interval=args.migration_interval,
            resource_monitor=resource_monitor,
            repository=repository,
            diversity_guardian=diversity_guardian
        )
    else:
        # Set up the regular evolution engine
        engine = EvolutionEngine(
            population_size=population_size,
            evaluator=evaluator,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            genome_class=CodeGenome,
            resource_monitor=resource_monitor,
            repository=repository,
            diversity_guardian=diversity_guardian
        )
    
    # Create the initial population
    engine.population = create_initial_population(task, population_size)
    
    # Run the evolution
    logger.info(f"Starting evolution with population size {population_size} for {num_generations} generations")
    logger.info(f"Ethics level: {args.ethics_level}")
    
    start_time = datetime.datetime.now()
    
    # Track generations
    best_fitness = 0.0
    best_genome = None
    
    metadata = {
        "task": task.get_name(),
        "description": task.get_description(),
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "ethics_level": args.ethics_level,
        "use_islands": args.use_islands,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "best_fitness": None,
        "generation_metrics": []
    }
    
    for gen in range(num_generations):
        gen_start_time = datetime.datetime.now()
        
        # Check if resource monitor allows us to proceed
        if resource_monitor and not resource_monitor.can_proceed():
            logger.warning("Resource constraints exceeded. Waiting for resources...")
            resource_monitor.wait_for_resources(timeout=60)  # Wait up to 60 seconds
        
        # Create generation directory
        gen_dir = create_generation_dir(run_dir, gen)
        
        # Evaluate the current generation
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
            post_process_solution(task, best_of_gen.to_source(), best_gen_path)
        
        # Log progress
        gen_time = (datetime.datetime.now() - gen_start_time).total_seconds()
        logger.info(f"Generation {gen} complete in {gen_time:.2f}s. Best fitness: {gen_fitness:.4f}")
        
        # Store metrics for this generation
        metadata["generation_metrics"].append({
            "generation": gen,
            "best_fitness": gen_fitness,
            "time_seconds": gen_time,
            "diversity": diversity_guardian.get_diversity() if diversity_guardian else None
        })
        
        # Create the next generation (unless it's the last generation)
        if gen < num_generations - 1:
            # Apply throttling if resource monitor recommends it
            if resource_monitor and resource_monitor.get_throttle_level() > 0:
                params = resource_monitor.get_throttling_parameters()
                effective_pop_size = max(5, int(population_size * params['population_scale_factor']))
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
                if hasattr(engine, 'evolve_generation'):
                    # Islands handle this differently
                    engine.evolve_generation()
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
            generation=num_generations - 1
        )
        
        # Save to run directory
        best_path = os.path.join(run_dir, f"best_{task.get_name()}.py")
        post_process_solution(task, best_genome.to_source(), best_path)
        
        logger.info(f"Best solution saved to {best_path}")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        
        # Copy to a standard location for easy access
        final_path = os.path.join(args.output_dir, f"evolved_{task.get_name()}.py")
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
    
    # Save the repository summary
    repository_summary = repository.get_summary()
    logger.info(f"Repository summary: {repository_summary}")

if __name__ == "__main__":
    main()
