#!/usr/bin/env python3
"""
Drive Scanner Task Evolution Example

This example shows how to use the TRISOLARIS task-based architecture
to evolve a drive scanner program.
"""

import os
import sys
import time
import argparse

# Add parent directory to path to allow imports from trisolaris
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trisolaris.tasks import DriveScannerTask
from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
from trisolaris.utils.paths import create_timestamped_output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Example of evolving a drive scanner with TRISOLARIS task interface"
    )
    parser.add_argument(
        "--template", 
        default="../drive_scanner.py",
        help="Path to template drive scanner file (default: ../drive_scanner.py)"
    )
    parser.add_argument(
        "--pop-size", 
        type=int, 
        default=5,
        help="Population size (default: 5)"
    )
    parser.add_argument(
        "--generations", 
        type=int, 
        default=3,
        help="Number of generations (default: 3)"
    )
    args = parser.parse_args()

    print("Drive Scanner Task Evolution Example")
    print("====================================")
    
    # Initialize the task
    task = DriveScannerTask(template_path=args.template)
    print(f"Task: {task.get_name()} - {task.get_description()}")
    
    # Get task-specific evolution parameters
    params = task.get_evolution_params()
    print(f"Recommended evolution parameters: {params}")
    
    # Create output directory
    output_dir = create_timestamped_output_dir("example_output")
    print(f"Output directory: {output_dir}")
    
    # Set up ethical filter with task-specific boundaries
    ethical_filter = EthicalBoundaryEnforcer()
    for boundary_name, boundary_params in task.get_required_boundaries().items():
        ethical_filter.add_boundary(boundary_name, **boundary_params)
    
    # Set up fitness evaluator
    evaluator = FitnessEvaluator(ethical_filter=ethical_filter)
    evaluator.set_weights(**task.get_fitness_weights())
    
    # Add task-specific fitness evaluation
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
    
    # Set up evolution engine
    engine = EvolutionEngine(
        population_size=args.pop_size,
        evaluator=evaluator,
        mutation_rate=params.get("mutation_rate", 0.1),
        crossover_rate=params.get("crossover_rate", 0.7),
        genome_class=CodeGenome
    )
    
    # Create initial population
    template_code = task.get_template()
    base_genome = CodeGenome.from_source(template_code)
    
    population = [base_genome]
    for i in range(1, args.pop_size):
        # Create variants with different mutation intensities
        genome = base_genome.clone()
        mutation_intensity = 0.05 * (i / (args.pop_size - 1) * 5)
        genome.mutate(mutation_rate=mutation_intensity)
        population.append(genome)
    
    engine.population = population
    
    # Run evolution
    print(f"\nStarting evolution with {args.pop_size} individuals for {args.generations} generations")
    
    for gen in range(args.generations):
        print(f"\nGeneration {gen}")
        print("-" * 20)
        
        # Evaluate current generation
        start_time = time.time()
        engine.evaluate_population()
        
        # Get best solution
        best = engine.get_best_solution()
        best_fitness = engine.best_fitness
        
        # Log progress
        eval_time = time.time() - start_time
        print(f"Evaluation completed in {eval_time:.2f}s")
        print(f"Best fitness: {best_fitness:.4f}")
        
        # Save best solution
        best_path = os.path.join(output_dir, f"best_gen_{gen}.py")
        post_processed = task.post_process(best.to_source())
        
        with open(best_path, "w", encoding="utf-8") as f:
            f.write(post_processed)
        
        os.chmod(best_path, 0o755)
        print(f"Best solution saved to {best_path}")
        
        # Create next generation (unless it's the last generation)
        if gen < args.generations - 1:
            parents = engine.select_parents()
            offspring = engine.create_offspring(parents)
            engine.population = engine.select_survivors(offspring)
    
    # Save final best solution
    final_path = os.path.join(output_dir, "best_drive_scanner.py")
    post_processed = task.post_process(best.to_source())
    
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(post_processed)
    
    os.chmod(final_path, 0o755)
    print(f"\nFinal solution saved to {final_path}")
    print(f"Final fitness: {best_fitness:.4f}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
