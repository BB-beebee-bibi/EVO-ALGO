#!/usr/bin/env python3
"""
Benchmark script for the TRISOLARIS evolution process.

This script measures the performance of the evolution process with different optimization settings.
It compares the performance of the original implementation with the optimized implementation.
"""

import time
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import CodeGenome
from trisolaris.evaluation.fitness import FitnessEvaluator
from trisolaris.tasks.network_scanner import NetworkScannerTask

def run_benchmark(
    population_size=50,
    generations=10,
    parallel=True,
    caching=True,
    early_stopping=True,
    max_workers=None
):
    """
    Run a benchmark of the evolution process with the specified settings.
    
    Args:
        population_size: Size of the population
        generations: Number of generations to run
        parallel: Whether to use parallel evaluation
        caching: Whether to use caching
        early_stopping: Whether to use early stopping
        max_workers: Number of worker processes for parallel evaluation
        
    Returns:
        Dictionary with benchmark results
    """
    # Create task and evaluator
    task = NetworkScannerTask()
    evaluator = FitnessEvaluator(use_caching=caching)
    
    # Add test cases from the task
    for test_case in task.get_test_cases():
        evaluator.add_test_case(**test_case)
    
    # Create the evolution engine
    engine = EvolutionEngine(
        population_size=population_size,
        evaluator=evaluator,
        mutation_rate=0.1,
        crossover_rate=0.7,
        genome_class=CodeGenome,
        parallel_evaluation=parallel,
        max_workers=max_workers or max(1, multiprocessing.cpu_count() - 1),
        use_caching=caching,
        early_stopping=early_stopping,
        early_stopping_generations=3,
        early_stopping_threshold=0.01
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
    
    # Measure time
    start_time = time.time()
    
    # Run evolution
    engine.evolve(generations=generations)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Get stats
    stats = engine.get_execution_stats()
    stats['elapsed_time'] = elapsed_time
    stats['time_per_generation'] = elapsed_time / max(1, stats['generations'])
    
    return stats

def run_all_benchmarks(output_dir="benchmark_results"):
    """
    Run all benchmark configurations and save the results.
    
    Args:
        output_dir: Directory to save benchmark results
    
    Returns:
        Dictionary with all benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define benchmark configurations
    configs = [
        {"name": "Baseline", "parallel": False, "caching": False, "early_stopping": False},
        {"name": "Parallel Only", "parallel": True, "caching": False, "early_stopping": False},
        {"name": "Caching Only", "parallel": False, "caching": True, "early_stopping": False},
        {"name": "Early Stopping Only", "parallel": False, "caching": False, "early_stopping": True},
        {"name": "Parallel + Caching", "parallel": True, "caching": True, "early_stopping": False},
        {"name": "Fully Optimized", "parallel": True, "caching": True, "early_stopping": True},
    ]
    
    results = {}
    
    # Run benchmarks
    for config in configs:
        print(f"Running benchmark: {config['name']}...")
        result = run_benchmark(
            parallel=config["parallel"],
            caching=config["caching"],
            early_stopping=config["early_stopping"]
        )
        results[config["name"]] = result
        print(f"  Elapsed time: {result['elapsed_time']:.2f}s")
        print(f"  Time per generation: {result['time_per_generation']:.2f}s")
    
    # Save results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_plots(results, output_dir)
    
    return results

def generate_plots(results, output_dir):
    """
    Generate plots from benchmark results.
    
    Args:
        results: Dictionary with benchmark results
        output_dir: Directory to save plots
    """
    # Extract data for plotting
    names = list(results.keys())
    elapsed_times = [results[name]["elapsed_time"] for name in names]
    times_per_gen = [results[name]["time_per_generation"] for name in names]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot total elapsed time
    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x, elapsed_times, width, label='Total Time')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Total Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    # Plot time per generation
    ax2.bar(x, times_per_gen, width, label='Time per Generation')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time per Generation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    
    # Add speedup percentages
    baseline_time = results["Baseline"]["elapsed_time"]
    for i, name in enumerate(names):
        if name != "Baseline":
            speedup = (baseline_time - results[name]["elapsed_time"]) / baseline_time * 100
            ax1.text(i, elapsed_times[i] + 0.1, f"{speedup:.1f}%", ha='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_results.png"))
    
    # Create a second figure for cache statistics
    cache_configs = [name for name in names if "Caching" in name or name == "Fully Optimized"]
    if cache_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract cache hit rates if available
        cache_hits = []
        cache_misses = []
        cache_sizes = []
        
        for name in cache_configs:
            if "cache_hits" in results[name]:
                cache_hits.append(results[name]["cache_hits"])
                cache_misses.append(results[name]["cache_misses"])
                total = results[name]["cache_hits"] + results[name]["cache_misses"]
                cache_sizes.append(results[name].get("cache_size", 0))
        
        if cache_hits:
            # Plot cache statistics
            x = np.arange(len(cache_configs))
            width = 0.35
            
            ax.bar(x - width/2, cache_hits, width, label='Cache Hits')
            ax.bar(x + width/2, cache_misses, width, label='Cache Misses')
            
            ax.set_ylabel('Count')
            ax.set_title('Cache Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(cache_configs)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cache_performance.png"))

def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark the TRISOLARIS evolution process")
    parser.add_argument("--output", default="benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    args = parser.parse_args()
    
    print(f"Running benchmarks with population={args.population}, generations={args.generations}")
    results = run_all_benchmarks(args.output)
    
    # Print summary
    print("\nBenchmark Summary:")
    baseline_time = results["Baseline"]["elapsed_time"]
    for name, result in results.items():
        speedup = (baseline_time - result["elapsed_time"]) / baseline_time * 100 if name != "Baseline" else 0
        print(f"{name}: {result['elapsed_time']:.2f}s ({speedup:.1f}% faster than baseline)")
    
    print(f"\nResults saved to {args.output}/")

if __name__ == "__main__":
    main()