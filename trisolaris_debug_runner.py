#!/usr/bin/env python3
"""
TRISOLARIS Debug Task Runner

An enhanced version of the task runner that includes comprehensive debugging
and performance monitoring capabilities.
"""

import os
import sys
import json
import datetime
import argparse
import importlib
import shutil
import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Type

# Ensure the trisolaris package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
from trisolaris.managers.resource import ResourceSteward
from trisolaris.managers.repository import GenomeRepository
from trisolaris.managers.diversity import DiversityGuardian
from trisolaris.managers.island import IslandEcosystemManager
from trisolaris.utils.paths import create_timestamped_output_dir, create_generation_dir
from trisolaris.tasks import TaskInterface, DriveScannerTask, NetworkScannerTask, BluetoothScannerTask
from trisolaris.utils.debug import (
    initialize_debug, debug_decorator, debug_log, debug_exception,
    log_genome_details, log_fitness_evaluation, log_ethical_check,
    log_resource_usage, log_evolution_progress, generate_performance_report,
    save_performance_report, thread_safe_logger
)

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
    "drive_scanner": DriveScannerTask,
    "network_scanner": NetworkScannerTask,
    "bluetooth_scanner": BluetoothScannerTask
}

@debug_decorator
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
    debug_log(f"Setting up ethical filter with level: {level}", level=logging.INFO)
    
    if level == "none":
        debug_log("Ethical filter disabled", level=logging.INFO)
        return None
    
    enforcer = EthicalBoundaryEnforcer()
    
    # Add boundaries based on task requirements
    if level == "basic":
        # Add required boundaries from task
        required_boundaries = task.get_required_boundaries()
        for boundary_name, params in required_boundaries.items():
            debug_log(f"Adding required boundary: {boundary_name}", level=logging.DEBUG, params=params)
            enforcer.add_boundary(boundary_name, **params)
    
    elif level == "full":
        # Full ethics includes all task requirements plus additional Gurbani-inspired boundaries
        required_boundaries = task.get_required_boundaries()
        for boundary_name, params in required_boundaries.items():
            debug_log(f"Adding required boundary: {boundary_name}", level=logging.DEBUG, params=params)
            enforcer.add_boundary(boundary_name, **params)
        
        # Add additional Gurbani-inspired boundaries
        gurbani_boundaries = [
            "universal_equity",
            "truthful_communication",
            "humble_code",
            "service_oriented",
            "harmony_with_environment"
        ]
        for boundary in gurbani_boundaries:
            debug_log(f"Adding Gurbani-inspired boundary: {boundary}", level=logging.DEBUG)
            enforcer.add_boundary(boundary)
    
    # Add import restrictions based on task's allowed imports
    allowed_imports = task.get_allowed_imports()
    debug_log(f"Setting allowed imports", level=logging.DEBUG, imports=allowed_imports)
    enforcer.add_boundary("no_imports", allowed_imports=set(allowed_imports))
    
    # Log all active boundaries
    active_boundaries = enforcer.get_active_boundaries()
    debug_log(f"Ethical filter configured with {len(active_boundaries)} boundaries", 
             level=logging.INFO, boundaries=active_boundaries)
    
    return enforcer

@debug_decorator
def setup_fitness_evaluator(task: TaskInterface, ethical_filter: Optional[EthicalBoundaryEnforcer]) -> FitnessEvaluator:
    """
    Set up the fitness evaluator with task-specific settings.
    
    Args:
        task: The task being evolved
        ethical_filter: Optional ethical filter
        
    Returns:
        Configured FitnessEvaluator
    """
    debug_log(f"Setting up fitness evaluator for task: {task.get_name()}", level=logging.INFO)
    
    # Create a fitness evaluator with the ethical filter
    evaluator = FitnessEvaluator(ethical_filter=ethical_filter)
    
    # Configure the weights for objectives from task-specific settings
    fitness_weights = task.get_fitness_weights()
    debug_log(f"Setting fitness weights", level=logging.DEBUG, weights=fitness_weights)
    evaluator.set_weights(**fitness_weights)
    
    # Add a custom alignment measure using the task's fitness evaluator
    def evaluate_task_fitness(genome):
        if hasattr(genome, 'to_source'):
            code = genome.to_source()
        else:
            code = str(genome)
        
        start_time = time.time()
        fitness, details = task.evaluate_fitness(code)
        elapsed = time.time() - start_time
        
        # Log the fitness evaluation
        log_fitness_evaluation(genome, fitness, details)
        debug_log(f"Task fitness evaluation", level=logging.DEBUG, 
                 fitness=fitness, time=elapsed)
        
        return fitness
    
    evaluator.add_alignment_measure(
        evaluate_task_fitness,
        weight=1.0,
        name=f"{task.get_name()}_fitness"
    )
    
    debug_log(f"Fitness evaluator configured", level=logging.INFO)
    return evaluator

@debug_decorator
def create_initial_population(task: TaskInterface, population_size: int) -> list:
    """
    Create an initial population based on the task's template.
    
    Args:
        task: The task being evolved
        population_size: Size of the population to create
        
    Returns:
        List of CodeGenome instances
    """
    debug_log(f"Creating initial population of size {population_size}", level=logging.INFO)
    population = []
    
    try:
        # Get the template code from the task
        template_code = task.get_template()
        debug_log(f"Got template code of length {len(template_code)}", level=logging.DEBUG)
        
        # Create the first genome from the template
        base_genome = CodeGenome.from_source(template_code)
        population.append(base_genome)
        log_genome_details(base_genome, generation=0, fitness=None)
        
        # Create variants with different mutation rates to ensure diversity
        for i in range(1, population_size):
            # Clone the base genome
            genome = base_genome.clone()
            
            # Apply mutations with increasing intensity
            mutation_intensity = 0.05 * (i / (population_size - 1) * 5)
            debug_log(f"Creating variant {i} with mutation intensity {mutation_intensity:.4f}", 
                     level=logging.DEBUG)
            
            start_time = time.time()
            genome.mutate(mutation_rate=mutation_intensity)
            elapsed = time.time() - start_time
            
            debug_log(f"Mutation completed", level=logging.DEBUG, time=elapsed)
            population.append(genome)
            
        debug_log(f"Created {len(population)} initial genomes", level=logging.INFO)
            
    except Exception as e:
        debug_exception(e, context="create_initial_population")
        logger.error(f"Error creating initial population: {e}")
        
        # If we can't create a population from the template, create random genomes
        debug_log(f"Falling back to random genome creation", level=logging.WARNING)
        for i in range(population_size):
            population.append(CodeGenome())
    
    return population

@debug_decorator
def post_process_solution(task: TaskInterface, solution: str, output_path: str) -> None:
    """
    Apply post-processing to the evolved solution and save it.
    
    Args:
        task: The task that was evolved
        solution: Source code of the evolved solution
        output_path: Path to save the post-processed solution
    """
    debug_log(f"Post-processing solution", level=logging.INFO, output_path=output_path)
    
    # Apply task-specific post-processing
    processed_solution = task.post_process(solution)
    
    # Save the post-processed solution
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(processed_solution)
    
    # Make the solution executable
    os.chmod(output_path, 0o755)
    debug_log(f"Solution saved and made executable", level=logging.INFO)

@debug_decorator
def run_evolution(
    task: TaskInterface,
    output_dir: str,
    num_generations: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    ethics_level: str,
    resource_monitoring: bool,
    use_git: bool,
    use_islands: bool,
    islands: int,
    migration_interval: int,
    diversity_threshold: float,
    debug_level: str
) -> tuple:
    """
    Run the evolutionary process with comprehensive debugging.
    
    Args:
        task: The task to evolve
        output_dir: Directory to save outputs
        num_generations: Number of generations to run
        population_size: Population size
        mutation_rate: Mutation rate
        crossover_rate: Crossover rate
        ethics_level: Ethical filter level
        resource_monitoring: Whether to enable resource monitoring
        use_git: Whether to use Git for version control
        use_islands: Whether to use island model
        islands: Number of islands
        migration_interval: Migration interval
        diversity_threshold: Diversity threshold
        debug_level: Debug logging level
        
    Returns:
        Tuple of (best_genome, statistics)
    """
    # Set up debug level
    debug_log_level = {
        "minimal": logging.WARNING,
        "normal": logging.INFO,
        "verbose": logging.DEBUG,
        "trace": logging.DEBUG  # With additional function tracing
    }.get(debug_level, logging.INFO)
    
    # Initialize debug system
    initialize_debug(
        enabled=True,
        log_level=debug_log_level,
        log_file=os.path.join(output_dir, "trisolaris_debug.log"),
        log_to_console=True,
        log_to_file=True,
        log_evolution_details=True,
        log_genome_details=(debug_level in ["verbose", "trace"]),
        log_fitness_details=(debug_level in ["verbose", "trace"]),
        log_resource_usage=resource_monitoring,
        log_ethical_checks=(ethics_level != "none"),
        log_performance_metrics=True,
        capture_exceptions=True,
        trace_function_calls=(debug_level == "trace")
    )
    
    debug_log(f"Starting evolution for {task.get_name()}", level=logging.INFO)
    debug_log(f"Evolution parameters", level=logging.INFO,
             generations=num_generations,
             population=population_size,
             mutation_rate=mutation_rate,
             crossover_rate=crossover_rate,
             ethics=ethics_level,
             islands=use_islands)
    
    # Create timestamped output directory
    task_output_dir = os.path.join(output_dir, f"{task.get_name()}_evolution")
    run_dir = create_timestamped_output_dir(task_output_dir)
    debug_log(f"Created output directory: {run_dir}", level=logging.INFO)
    
    # Initialize repository for genome storage
    repository = GenomeRepository(
        base_dir=task_output_dir,
        run_id=os.path.basename(run_dir),
        use_git=use_git
    )
    
    # Initialize resource monitoring if enabled
    resource_monitor = None
    if resource_monitoring:
        resource_monitor = ResourceSteward(
            min_available_memory=0.25,
            min_available_cpu=0.25
        )
        debug_log("Resource monitoring enabled", level=logging.INFO)
        system_info = resource_monitor.get_system_info()
        debug_log("System information", level=logging.INFO, **system_info)
    
    # Setup ethical filter based on chosen level
    ethical_filter = setup_ethical_filter(task, ethics_level, run_dir)
    if ethical_filter:
        debug_log(f"Ethical filter enabled at level: {ethics_level}", level=logging.INFO)
        for boundary in ethical_filter.get_active_boundaries():
            debug_log(f"Active boundary: {boundary}", level=logging.INFO)
    
    # Set up the fitness evaluator
    evaluator = setup_fitness_evaluator(task, ethical_filter)
    
    # Initialize diversity guardian
    diversity_guardian = DiversityGuardian(
        threshold=diversity_threshold
    )
    debug_log(f"Diversity guardian initialized with threshold {diversity_threshold}", level=logging.INFO)
    
    # Set up either island model or regular evolution engine
    if use_islands:
        debug_log(f"Using island model with {islands} islands", level=logging.INFO)
        
        # Create island model manager
        engine = IslandEcosystemManager(
            num_islands=islands,
            island_population_size=population_size // islands,
            evaluator=evaluator,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            genome_class=CodeGenome,
            migration_interval=migration_interval,
            resource_monitor=resource_monitor,
            repository=repository,
            diversity_guardian=diversity_guardian
        )
    else:
        # Set up the regular evolution engine
        debug_log(f"Using standard evolution engine", level=logging.INFO)
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
    debug_log(f"Starting evolution process", level=logging.INFO)
    
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
        "ethics_level": ethics_level,
        "use_islands": use_islands,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "best_fitness": None,
        "generation_metrics": []
    }
    
    for gen in range(num_generations):
        gen_start_time = datetime.datetime.now()
        
        debug_log(f"Starting generation {gen}", level=logging.INFO)
        
        # Check if resource monitor allows us to proceed
        if resource_monitor:
            log_resource_usage(resource_monitor)
            if not resource_monitor.can_proceed():
                debug_log("Resource constraints exceeded. Waiting for resources...", level=logging.WARNING)
                resource_monitor.wait_for_resources(timeout=60)  # Wait up to 60 seconds
        
        # Create generation directory
        gen_dir = create_generation_dir(run_dir, gen)
        
        # Evaluate the current generation
        debug_log(f"Evaluating population for generation {gen}", level=logging.INFO)
        
        evaluation_start = time.time()
        engine.evaluate_population()
        evaluation_time = time.time() - evaluation_start
        
        debug_log(f"Population evaluation completed", level=logging.INFO, 
                 time=evaluation_time)
        
        # Check diversity and inject if needed
        if hasattr(engine, 'check_diversity'):
            diversity = engine.check_diversity()
            debug_log(f"Population diversity", level=logging.INFO, diversity=diversity)
        
        # Save the best solution of this generation
        best_of_gen = engine.get_best_solution()
        gen_fitness = engine.best_fitness
        
        # Calculate average fitness
        avg_fitness = 0.0
        if hasattr(engine, 'fitness_scores') and engine.fitness_scores:
            valid_scores = [f for f in engine.fitness_scores if f != float('-inf')]
            if valid_scores:
                avg_fitness = sum(valid_scores) / len(valid_scores)
        
        # Log progress
        log_evolution_progress(gen, gen_fitness, avg_fitness, 
                              (datetime.datetime.now() - gen_start_time).total_seconds(), 
                              population_size)
        
        if best_of_gen:
            # Log the best genome
            log_genome_details(best_of_gen, generation=gen, fitness=gen_fitness)
            
            # Update best overall solution
            if gen_fitness > best_fitness:
                debug_log(f"New best solution found", level=logging.INFO, 
                         fitness=gen_fitness, generation=gen)
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
        
        # Store metrics for this generation
        gen_time = (datetime.datetime.now() - gen_start_time).total_seconds()
        
        # Get diversity if available
        diversity_value = None
        if diversity_guardian:
            diversity_value = diversity_guardian.get_diversity()
        
        metadata["generation_metrics"].append({
            "generation": gen,
            "best_fitness": gen_fitness,
            "avg_fitness": avg_fitness,
            "time_seconds": gen_time,
            "diversity": diversity_value
        })
        
        # Create the next generation (unless it's the last generation)
        if gen < num_generations - 1:
            debug_log(f"Creating next generation", level=logging.INFO)
            
            # Apply throttling if resource monitor recommends it
            if resource_monitor and resource_monitor.get_throttle_level() > 0:
                params = resource_monitor.get_throttling_parameters()
                effective_pop_size = max(5, int(population_size * params['population_scale_factor']))
                debug_log(f"Throttling evolution", level=logging.WARNING, 
                         throttle_level=resource_monitor.get_throttle_level(),
                         reduced_population=effective_pop_size)
                
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
        
        debug_log(f"Evolution completed successfully", level=logging.INFO,
                 generations=num_generations,
                 best_fitness=best_fitness,
                 duration=(end_time - start_time).total_seconds())
        
        # Save to repository
        repository.store_best_solution(
            genome=best_genome,
            fitness=best_fitness,
            generation=num_generations - 1
        )
        
        # Save to run directory
        best_path = os.path.join(run_dir, f"best_{task.get_name()}.py")
        post_process_solution(task, best_genome.to_source(), best_path)
        
        debug_log(f"Best solution saved", level=logging.INFO, path=best_path)
        
        # Copy to a standard location for easy access
        final_path = os.path.join(output_dir, f"evolved_{task.get_name()}.py")
        shutil.copy2(best_path, final_path)
        os.chmod(final_path, 0o755)
        debug_log(f"Best solution copied to standard location", level=logging.INFO, path=final_path)
    else:
        debug_log(f"No valid solution found", level=logging.WARNING)
        metadata["end_time"] = datetime.datetime.now().isoformat()
        metadata["best_fitness"] = 0.0
    
    # Save metadata
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Generate and save performance report
    performance_report_path = os.path.join(run_dir, "performance_report.txt")
    save_performance_report(performance_report_path)
    
    # Save the repository summary
    repository_summary = repository.get_summary()
    debug_log(f"Repository summary", level=logging.INFO, **repository_summary)
    
    # Return best genome and statistics
    stats = {
        "best_fitness": best_fitness,
        "generations": num_generations,
        "duration_seconds": (end_time - start_time).total_seconds() if 'end_time' in locals() else 0,
        "output_path": best_path if 'best_path' in locals() else None,
        "metrics": metadata["generation_metrics"] if "generation_metrics" in metadata else [],
        "performance_report": performance_report_path
    }
    
    return best_genome, stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Debug-enhanced TRISOLARIS task runner with comprehensive logging and monitoring"
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
    
    # Debug parameters
    debug_group = parser.add_argument_group("Debug Parameters")
    debug_group.add_argument(
        "--debug-level",
        choices=["minimal", "normal", "verbose", "trace"],
        default="normal",
        help="Debug logging level (default: normal)"
    )
    debug_group.add_argument(
        "--save-all-genomes",
        action="store_true",
        help="Save all genomes, not just the best ones"
    )
    debug_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    args = parser.parse_args()

    try:
        # Initialize the task
        task_class = TASK_REGISTRY[args.task]
        task = task_class(template_path=args.template)
        logger.info(f"Initialized task: {task.get_name()} - {task.get_description()}")
    except KeyError:
        logger.error(f"Unknown task: {args.task}")
        return
    except Exception as e:
        logger.error(f"Error initializing task: {str(e)}")
        traceback.print_exc()
        return

    # Get task-specific parameters and override with command line arguments if provided
    task_params = task.get_evolution_params()
    
    population_size = args.pop_size if args.pop_size is not None else task_params.get("population_size", 20)
    num_generations = args.gens if args.gens is not None else task_params.get("num_generations", 10)
    mutation_rate = args.mutation_rate if args.mutation_rate is not None else task_params.get("mutation_rate", 0.1)
    crossover_rate = args.crossover_rate if args.crossover_rate is not None else task_params.get("crossover_rate", 0.7)
    
    # Print banner
    print("\n" + "="*60)
    print(f"TRISOLARIS DEBUG TASK RUNNER - {task.get_name()}")
    print("="*60)
    print(f"Task: {task.get_name()}")
    print(f"Description: {task.get_description()}")
    print(f"Population Size: {population_size}")
    print(f"Generations: {num_generations}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Ethics Level: {args.ethics_level}")
    print(f"Debug Level: {args.debug_level}")
    if args.use_islands:
        print(f"Island Model: Enabled ({args.islands} islands)")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run the evolution with debug logging
        best_genome, stats = run_evolution(
            task=task,
            output_dir=args.output_dir,
            num_generations=num_generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            ethics_level=args.ethics_level,
            resource_monitoring=args.resource_monitoring,
            use_git=args.use_git,
            use_islands=args.use_islands,
            islands=args.islands,
            migration_interval=args.migration_interval,
            diversity_threshold=args.diversity_threshold,
            debug_level=args.debug_level
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVOLUTION COMPLETE")
        print("="*60)
        print(f"Best Fitness: {stats['best_fitness']:.4f}")
        print(f"Total Runtime: {stats['duration_seconds']:.2f} seconds")
        print(f"Output Path: {stats['output_path']}")
        print(f"Performance Report: {stats['performance_report']}")
        print("="*60)
        
        # Print fitness progression
        print("\nFitness Progression:")
        print("Generation | Best Fitness | Avg Fitness | Time(s)")
        print("-" * 60)
        for gen_stats in stats["metrics"]:
            print(f"{gen_stats['generation']:10} | {gen_stats['best_fitness']:12.4f} | {gen_stats.get('avg_fitness', 0):11.4f} | {gen_stats['time_seconds']:6.2f}")
        
        print("\nEvolution completed successfully!")
        print(f"Run the evolved solution with: python {stats['output_path']}")
        
    except Exception as e:
        logger.error(f"Error during evolution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
