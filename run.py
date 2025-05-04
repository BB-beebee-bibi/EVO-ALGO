#!/usr/bin/env python
"""
Trisolaris Evolution Runner

A command-line interface to evolve code using the Trisolaris framework,
with robust safety guarantees, resource monitoring, and structured output.
"""

import argparse
import os
import logging
import time
import datetime
import sys
from typing import Optional, List, Dict, Any, Tuple

from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
from trisolaris.managers.resource import ResourceSteward
from trisolaris.managers.repository import GenomeRepository
from trisolaris.utils.paths import create_timestamped_output_dir, create_generation_dir

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

def setup_basic_safety_filter(output_dir: str) -> EthicalBoundaryEnforcer:
    """Create a basic ethical filter that prevents destructive operations."""
    enforcer = EthicalBoundaryEnforcer()
    
    # Safety-first boundaries
    enforcer.add_boundary("no_system_calls")
    enforcer.add_boundary("no_eval_exec")
    enforcer.add_boundary("no_network_access")
    
    # Allow limited file operations in the specified output directory only
    # Block complete filesystem access by default
    enforcer.add_boundary("no_file_operations")
    
    # Resource constraints
    enforcer.add_boundary("max_execution_time", max_execution_time=3.0)  # 3 seconds max
    enforcer.add_boundary("max_memory_usage", max_memory_usage=200)      # 200MB max
    
    # Allow controlled imports
    enforcer.add_boundary("no_imports", allowed_imports={
        'os', 'sys', 'time', 'random', 'math', 'json', 
        'datetime', 'collections', 're', 'logging'
    })
    
    return enforcer

def setup_full_ethics_filter(output_dir: str) -> EthicalBoundaryEnforcer:
    """Create a comprehensive ethical filter with all boundaries active."""
    enforcer = setup_basic_safety_filter(output_dir)
    
    # Add the Gurbani-inspired ethical boundaries
    enforcer.add_boundary("universal_equity")
    enforcer.add_boundary("truthful_communication")
    enforcer.add_boundary("humble_code")
    enforcer.add_boundary("service_oriented")
    enforcer.add_boundary("harmony_with_environment")
    
    return enforcer

def load_initial_genomes(input_dir: str) -> List[CodeGenome]:
    """Load initial genomes from Python files in the input directory."""
    genomes = []
    
    # Start with a directory-based genome
    try:
        genomes.append(CodeGenome.from_directory(input_dir))
        logger.info(f"Loaded directory genome from {input_dir}")
        return genomes
    except Exception as e:
        logger.warning(f"Could not load directory genome: {e}")
    
    # Fall back to individual Python files if directory loading failed
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        source = f.read()
                    genomes.append(CodeGenome.from_source(source))
                    logger.info(f"Loaded genome from {os.path.join(root, file)}")
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
    
    # If no genomes were loaded, create a random one
    if not genomes:
        logger.warning(f"No valid Python files found in {input_dir}. Creating a random genome.")
        genomes.append(CodeGenome())
    
    return genomes

def main():
    parser = argparse.ArgumentParser(
        description="Evolve code using the Trisolaris evolutionary framework"
    )
    parser.add_argument(
        "input_dir",
        help="Path to the folder with initial code (e.g., minja/)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Base directory to save evolved code (default: outputs)"
    )
    parser.add_argument(
        "--pop-size", "-p",
        type=int,
        default=20,
        help="Population size (default: 20)"
    )
    parser.add_argument(
        "--gens", "-g",
        type=int,
        default=10,
        help="Number of generations (default: 10)"
    )
    parser.add_argument(
        "--mutation-rate", "-m",
        type=float,
        default=0.1,
        help="Mutation rate (default: 0.1)"
    )
    parser.add_argument(
        "--crossover-rate", "-c",
        type=float,
        default=0.7,
        help="Crossover rate (default: 0.7)"
    )
    parser.add_argument(
        "--ethics-level", "-e",
        choices=["none", "basic", "full"],
        default="basic",
        help="Ethical filter level (none, basic, full) (default: basic)"
    )
    parser.add_argument(
        "--resource-monitoring",
        action="store_true",
        help="Enable resource monitoring and throttling"
    )
    parser.add_argument(
        "--min-available-memory",
        type=float,
        default=0.25,
        help="Minimum fraction of memory that should remain available (default: 0.25)"
    )
    parser.add_argument(
        "--min-available-cpu",
        type=float,
        default=0.25,
        help="Minimum fraction of CPU that should remain available (default: 0.25)"
    )
    parser.add_argument(
        "--use-git",
        action="store_true",
        help="Use Git for version control of solution history"
    )
    parser.add_argument(
        "--show-resource-report",
        action="store_true",
        help="Show resource usage report at the end"
    )
    args = parser.parse_args()

    # Create timestamped output directory
    run_dir = create_timestamped_output_dir(args.output_dir)
    logger.info(f"Created output directory: {run_dir}")
    
    # Initialize repository for genome storage
    repository = GenomeRepository(
        base_dir=args.output_dir,
        run_id=os.path.basename(run_dir),
        use_git=args.use_git
    )
    
    # Initialize resource monitoring if enabled
    resource_monitor = None
    if args.resource_monitoring:
        resource_monitor = ResourceSteward(
            min_available_memory=args.min_available_memory,
            min_available_cpu=args.min_available_cpu
        )
        logger.info("Resource monitoring enabled")
        logger.info(resource_monitor.get_system_info())
    
    # 1) Load initial genomes
    initial_genomes = load_initial_genomes(args.input_dir)
    if not initial_genomes:
        logger.error(f"Could not load any valid code from {args.input_dir}")
        return
    
    # 2) Setup ethical filter based on chosen level
    ethical_filter = None
    if args.ethics_level == "basic":
        ethical_filter = setup_basic_safety_filter(run_dir)
    elif args.ethics_level == "full":
        ethical_filter = setup_full_ethics_filter(run_dir)
    
    # 3) Build the fitness evaluator
    evaluator = FitnessEvaluator(ethical_filter=ethical_filter)
    
    # 4) Configure weights for objectives if ethics is enabled
    if args.ethics_level == "full":
        evaluator.set_weights(
            alignment=0.6,      # Stronger weight on ethical alignment
            functionality=0.25, # Still important but less than alignment
            efficiency=0.15     # Least important
        )
    else:
        evaluator.set_weights(
            alignment=0.2,      # Less focus on alignment
            functionality=0.5,  # Functionality is primary concern 
            efficiency=0.3      # Efficiency is secondary concern
        )
    
    # 5) Setup the evolution engine with resource monitor and repository
    engine = EvolutionEngine(
        population_size=args.pop_size,
        evaluator=evaluator,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        genome_class=CodeGenome,
        resource_monitor=resource_monitor,
        repository=repository
    )
    
    # Add initial genomes to the population
    engine.population = initial_genomes
    # Pad with random genomes if needed
    while len(engine.population) < args.pop_size:
        engine.population.append(CodeGenome())
    
    # 6) Run the evolution
    logger.info(f"Starting evolution with population size {args.pop_size} for {args.gens} generations")
    logger.info(f"Ethics level: {args.ethics_level}")
    
    start_time = time.time()
    
    # Track generations
    for gen in range(args.gens):
        gen_start_time = time.time()
        
        # Check if resource monitor allows us to proceed
        if resource_monitor and not resource_monitor.can_proceed():
            logger.warning("Resource constraints exceeded. Waiting for resources...")
            resource_monitor.wait_for_resources(timeout=60)  # Wait up to 60 seconds
        
        # Create generation directory
        gen_dir = create_generation_dir(run_dir, gen)
        
        # Evaluate the current generation
        engine.evaluate_population()
        
        # Save the best solution of this generation
        best_of_gen = engine.get_best_solution()
        if best_of_gen:
            # Save to repository
            repository.store_solution(
                genome=best_of_gen,
                fitness=engine.best_fitness,
                generation=gen,
                metadata={"generation_time": time.time() - gen_start_time}
            )
            
            # Also save to generation directory
            with open(os.path.join(gen_dir, "best.py"), "w", encoding="utf-8") as f:
                f.write(best_of_gen.to_source())
        
        # Log progress
        gen_time = time.time() - gen_start_time
        logger.info(f"Generation {gen} complete in {gen_time:.2f}s. Best fitness: {engine.best_fitness:.4f}")
        
        # Create the next generation (unless it's the last generation)
        if gen < args.gens - 1:
            # Apply throttling if resource monitor recommends it
            if resource_monitor and resource_monitor.get_throttle_level() > 0:
                params = resource_monitor.get_throttling_parameters()
                effective_pop_size = max(5, int(args.pop_size * params['population_scale_factor']))
                logger.info(f"Throttling: Reducing population size to {effective_pop_size} for next generation")
                
                # Select fewer parents but maintain diversity
                parents = engine.select_parents()[:effective_pop_size]
                offspring = engine.create_offspring(parents)
                engine.population = engine.select_survivors(offspring)[:effective_pop_size]
            else:
                # Normal operation
                parents = engine.select_parents()
                offspring = engine.create_offspring(parents)
                engine.population = engine.select_survivors(offspring)
    
    # 7) Save the final best solution
    best = engine.get_best_solution()
    if best:
        # Save to repository
        repository.store_best_solution(
            genome=best,
            fitness=engine.best_fitness,
            generation=args.gens - 1
        )
        
        # Save to run directory
        best_path = os.path.join(run_dir, "best.py")
        with open(best_path, "w", encoding="utf-8") as f:
            f.write(best.to_source())
        
        logger.info(f"Best solution saved to {best_path}")
        logger.info(f"Best fitness: {engine.best_fitness:.4f}")
    else:
        logger.warning("No valid solution found")
    
    # Generate a report
    total_time = time.time() - start_time
    logger.info(f"Evolution complete in {total_time:.2f}s")
    
    # Show resource report if requested
    if args.show_resource_report and resource_monitor:
        print("\n" + "="*50)
        print(resource_monitor.generate_report())
        print("="*50)
    
    # Save the repository summary
    repository_summary = repository.get_summary()
    logger.info(f"Repository summary: {repository_summary}")

if __name__ == "__main__":
    main()
