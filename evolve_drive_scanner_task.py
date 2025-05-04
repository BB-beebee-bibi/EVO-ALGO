#!/usr/bin/env python3
"""
Evolve Drive Scanner (Task-Based Version)

A script that uses the task-based TRISOLARIS framework to evolve a drive scanner program.
This is a wrapper around the original evolve_drive_scanner.py that uses the new
TaskInterface architecture.
"""

import os
import sys
import argparse
import datetime
import time

# Import the TRISOLARIS task components
from trisolaris.tasks import DriveScannerTask
from trisolaris.evaluation import EthicalBoundaryEnforcer
from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.managers.resource import ResourceSteward
from trisolaris.managers.repository import GenomeRepository
from trisolaris.utils.paths import create_timestamped_output_dir, create_generation_dir

def main():
    # Parse command line arguments - keep the same interface as evolve_drive_scanner.py
    parser = argparse.ArgumentParser(
        description="Evolve a drive scanner program using the task-based TRISOLARIS framework"
    )
    parser.add_argument(
        "--template",
        default="drive_scanner.py",
        help="Path to the template drive scanner program (default: drive_scanner.py)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/drive_scanner_evolution",
        help="Directory to save evolved code (default: outputs/drive_scanner_evolution)"
    )
    parser.add_argument(
        "--pop-size",
        "-p",
        type=int,
        default=10,
        help="Population size (default: 10)"
    )
    parser.add_argument(
        "--gens",
        "-g",
        type=int,
        default=5,
        help="Number of generations (default: 5)"
    )
    parser.add_argument(
        "--mutation-rate",
        "-m",
        type=float,
        default=0.1,
        help="Mutation rate (default: 0.1)"
    )
    parser.add_argument(
        "--crossover-rate",
        "-c",
        type=float,
        default=0.7,
        help="Crossover rate (default: 0.7)"
    )
    parser.add_argument(
        "--ethics-level",
        "-e",
        choices=["none", "basic", "full"],
        default="basic",
        help="Ethical filter level (none, basic, full) (default: basic)"
    )
    parser.add_argument(
        "--resource-monitoring",
        action="store_true",
        help="Enable resource monitoring and throttling"
    )
    args = parser.parse_args()

    # Check if the template file exists
    if not os.path.exists(args.template):
        print(f"Error: Template file {args.template} not found.")
        print("Please create this file first or specify a different template with --template.")
        return

    # Initialize the drive scanner task
    task = DriveScannerTask(template_path=args.template)
    print(f"Initialized task: {task.get_name()} - {task.get_description()}")

    # Create output directory
    run_dir = create_timestamped_output_dir(args.output_dir)
    print(f"Created output directory: {run_dir}")
    
    # Initialize repository for genome storage
    repository = GenomeRepository(
        base_dir=args.output_dir,
        run_id=os.path.basename(run_dir),
        use_git=True
    )
    
    # Initialize resource monitoring if enabled
    resource_monitor = None
    if args.resource_monitoring:
        resource_monitor = ResourceSteward(
            min_available_memory=0.25,
            min_available_cpu=0.25
        )
        print("Resource monitoring enabled")
        print(resource_monitor.get_system_info())
    
    # Setup ethical filter based on chosen level
    ethical_filter = None
    if args.ethics_level != "none":
        ethical_filter = EthicalBoundaryEnforcer()
        
        # Add task-specific boundaries based on ethics level
        required_boundaries = task.get_required_boundaries()
        
        if args.ethics_level == "basic":
            for boundary_name, params in required_boundaries.items():
                ethical_filter.add_boundary(boundary_name, **params)
        
        elif args.ethics_level == "full":
            # Add all required boundaries
            for boundary_name, params in required_boundaries.items():
                ethical_filter.add_boundary(boundary_name, **params)
            
            # Add additional Gurbani-inspired boundaries
            ethical_filter.add_boundary("universal_equity")
            ethical_filter.add_boundary("truthful_communication")
            ethical_filter.add_boundary("humble_code")
            ethical_filter.add_boundary("service_oriented")
            ethical_filter.add_boundary("harmony_with_environment")
        
        # Add import restrictions based on task's allowed imports
        allowed_imports = task.get_allowed_imports()
        ethical_filter.add_boundary("no_imports", allowed_imports=set(allowed_imports))
    
    # Set up the fitness evaluator (using the task's evaluate_fitness method)
    from trisolaris.evaluation import FitnessEvaluator
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
    
    # Set up the evolution engine
    engine = EvolutionEngine(
        population_size=args.pop_size,
        evaluator=evaluator,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        genome_class=CodeGenome,
        resource_monitor=resource_monitor,
        repository=repository
    )
    
    # Create the initial population
    try:
        # Get the template code from the task
        template_code = task.get_template()
        
        # Create the first genome from the template
        base_genome = CodeGenome.from_source(template_code)
        engine.population = [base_genome]
        
        # Create variants with different mutation rates to ensure diversity
        for i in range(1, args.pop_size):
            # Clone the base genome
            genome = base_genome.clone()
            
            # Apply mutations with increasing intensity
            mutation_intensity = 0.05 * (i / (args.pop_size - 1) * 5)
            genome.mutate(mutation_rate=mutation_intensity)
            
            engine.population.append(genome)
            
    except Exception as e:
        print(f"Error creating initial population: {e}")
        return
    
    # Run the evolution
    print(f"Starting evolution with population size {args.pop_size} for {args.gens} generations")
    print(f"Ethics level: {args.ethics_level}")
    
    start_time = datetime.datetime.now()
    
    # Track generations
    best_fitness = 0.0
    best_genome = None
    
    for gen in range(args.gens):
        gen_start_time = datetime.datetime.now()
        
        # Check if resource monitor allows us to proceed
        if resource_monitor and not resource_monitor.can_proceed():
            print("Resource constraints exceeded. Waiting for resources...")
            resource_monitor.wait_for_resources(timeout=60)  # Wait up to 60 seconds
        
        # Create generation directory
        gen_dir = create_generation_dir(run_dir, gen)
        
        # Evaluate the current generation
        engine.evaluate_population()
        
        # Save the best solution of this generation
        best_of_gen = engine.get_best_solution()
        if best_of_gen:
            gen_fitness = engine.best_fitness
            
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
            
            # Also save to generation directory with post-processing
            best_gen_path = os.path.join(gen_dir, "best.py")
            with open(best_gen_path, "w", encoding="utf-8") as f:
                # Apply task-specific post-processing
                post_processed = task.post_process(best_of_gen.to_source())
                f.write(post_processed)
            
            # Make it executable
            os.chmod(best_gen_path, 0o755)
        
        # Log progress
        gen_time = (datetime.datetime.now() - gen_start_time).total_seconds()
        print(f"Generation {gen} complete in {gen_time:.2f}s. Best fitness: {engine.best_fitness:.4f}")
        
        # Create the next generation (unless it's the last generation)
        if gen < args.gens - 1:
            # Apply throttling if resource monitor recommends it
            if resource_monitor and resource_monitor.get_throttle_level() > 0:
                params = resource_monitor.get_throttling_parameters()
                effective_pop_size = max(5, int(args.pop_size * params['population_scale_factor']))
                print(f"Throttling: Reducing population size to {effective_pop_size} for next generation")
                
                # Select fewer parents but maintain diversity
                parents = engine.select_parents()[:effective_pop_size]
                offspring = engine.create_offspring(parents)
                engine.population = engine.select_survivors(offspring)[:effective_pop_size]
            else:
                # Normal operation
                parents = engine.select_parents()
                offspring = engine.create_offspring(parents)
                engine.population = engine.select_survivors(offspring)
    
    # Save the final best solution
    if best_genome:
        # Save to repository
        repository.store_best_solution(
            genome=best_genome,
            fitness=best_fitness,
            generation=args.gens - 1
        )
        
        # Save to run directory with post-processing
        best_path = os.path.join(run_dir, f"best_{task.get_name()}.py")
        with open(best_path, "w", encoding="utf-8") as f:
            # Apply task-specific post-processing
            post_processed = task.post_process(best_genome.to_source())
            f.write(post_processed)
        
        # Make it executable
        os.chmod(best_path, 0o755)
        
        print(f"Best solution saved to {best_path}")
        print(f"Best fitness: {best_fitness:.4f}")
        
        # Copy to a standard location for easy access
        import shutil
        final_path = os.path.join(os.path.dirname(args.output_dir), f"evolved_{task.get_name()}.py")
        shutil.copy2(best_path, final_path)
        os.chmod(final_path, 0o755)
        print(f"Best solution also copied to {final_path}")
    else:
        print("No valid solution found")
    
    # Generate a report
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Evolution complete in {total_time:.2f}s")
    
    # Save the repository summary
    repository_summary = repository.get_summary()
    print(f"Repository summary: {repository_summary}")

if __name__ == "__main__":
    main()
