#!/usr/bin/env python3
"""
Performance demonstration script for the TRISOLARIS framework.

This script demonstrates the performance improvements in the evolution process
by running a simple task with different optimization settings.
"""

import time
import argparse
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("performance_demo.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import trisolaris
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import CodeGenome
from trisolaris.evaluation.fitness import FitnessEvaluator
from trisolaris.tasks.network_scanner import NetworkScannerTask

def run_evolution_with_settings(
    population_size=50,
    generations=10,
    parallel=True,
    caching=True,
    early_stopping=True,
    resource_aware=True,
    max_workers=None
):
    """
    Run evolution with the specified settings.
    
    Args:
        population_size: Size of the population
        generations: Number of generations to run
        parallel: Whether to use parallel evaluation
        caching: Whether to use caching
        early_stopping: Whether to use early stopping
        resource_aware: Whether to use resource-aware scheduling
        max_workers: Number of worker processes for parallel evaluation
        
    Returns:
        Tuple of (best_solution, execution_stats)
    """
    logger.info(f"Running evolution with settings:")
    logger.info(f"  Population size: {population_size}")
    logger.info(f"  Generations: {generations}")
    logger.info(f"  Parallel evaluation: {parallel}")
    logger.info(f"  Caching: {caching}")
    logger.info(f"  Early stopping: {early_stopping}")
    logger.info(f"  Resource-aware: {resource_aware}")
    
    # Create task and evaluator
    task = NetworkScannerTask()
    evaluator = FitnessEvaluator(use_caching=caching)
    
    # Add test cases manually since NetworkScannerTask doesn't have get_test_cases method
    # Simple test cases for network scanning functionality
    evaluator.add_test_case(
        input_data={"target": "127.0.0.1", "ports": [80, 443]},
        expected_output={"open_ports": [], "scan_time": 0.1},
        weight=1.0,
        name="Local scan test"
    )
    evaluator.add_test_case(
        input_data={"target": "localhost", "ports": [22, 80]},
        expected_output={"open_ports": [], "scan_time": 0.1},
        weight=1.0,
        name="Localhost scan test"
    )
    
    # Create the evolution engine
    engine = EvolutionEngine(
        population_size=population_size,
        evaluator=evaluator,
        mutation_rate=0.1,
        crossover_rate=0.7,
        genome_class=CodeGenome,
        parallel_evaluation=parallel,
        max_workers=max_workers,
        use_caching=caching,
        early_stopping=early_stopping,
        early_stopping_generations=3,
        early_stopping_threshold=0.01,
        resource_aware=resource_aware
    )
    
    # Create initial population
    template_code = task.get_template()
    base_genome = CodeGenome.from_source(template_code)
    
    # Create population with variants
    population = [base_genome]
    for _ in range(population_size - 1):
        genome = base_genome.clone()
        genome.mutate()
        population.append(genome)
    
    engine.population = population
    
    # Run evolution and measure time
    start_time = time.time()
    best_solution = engine.evolve(generations=generations)
    elapsed_time = time.time() - start_time
    
    # Get execution stats
    stats = engine.get_execution_stats()
    stats['elapsed_time'] = elapsed_time
    stats['time_per_generation'] = elapsed_time / max(1, stats['generations'])
    
    logger.info(f"Evolution completed in {elapsed_time:.2f}s")
    logger.info(f"Best fitness: {engine.best_fitness:.4f}")
    logger.info(f"Time per generation: {stats['time_per_generation']:.2f}s")
    
    return best_solution, stats

def run_comparison():
    """Run a comparison of different optimization settings."""
    # Define configurations to test
    configs = [
        {"name": "Baseline", "parallel": False, "caching": False, "early_stopping": False, "resource_aware": False},
        {"name": "All Optimizations", "parallel": True, "caching": True, "early_stopping": True, "resource_aware": True}
    ]
    
    results = {}
    
    # Run each configuration
    for config in configs:
        logger.info(f"\n\n=== Running {config['name']} configuration ===\n")
        
        _, stats = run_evolution_with_settings(
            parallel=config["parallel"],
            caching=config["caching"],
            early_stopping=config["early_stopping"],
            resource_aware=config["resource_aware"]
        )
        
        results[config["name"]] = stats
    
    # Print comparison
    logger.info("\n\n=== Performance Comparison ===\n")
    
    baseline_time = results["Baseline"]["elapsed_time"]
    for name, stats in results.items():
        speedup = (baseline_time - stats["elapsed_time"]) / baseline_time * 100 if name != "Baseline" else 0
        logger.info(f"{name}:")
        logger.info(f"  Total time: {stats['elapsed_time']:.2f}s")
        logger.info(f"  Time per generation: {stats['time_per_generation']:.2f}s")
        if name != "Baseline":
            logger.info(f"  Speedup: {speedup:.1f}%")
        logger.info("")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Demonstrate performance improvements in TRISOLARIS")
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--mode", choices=["compare", "optimized", "baseline"], default="compare",
                      help="Mode to run (compare, optimized, or baseline)")
    args = parser.parse_args()
    
    if args.mode == "compare":
        run_comparison()
    elif args.mode == "optimized":
        run_evolution_with_settings(
            population_size=args.population,
            generations=args.generations,
            parallel=True,
            caching=True,
            early_stopping=True,
            resource_aware=True
        )
    elif args.mode == "baseline":
        run_evolution_with_settings(
            population_size=args.population,
            generations=args.generations,
            parallel=False,
            caching=False,
            early_stopping=False,
            resource_aware=False
        )

if __name__ == "__main__":
    main()