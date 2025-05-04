#!/usr/bin/env python3
"""
Evolve Network Scanner

A script that uses the TRISOLARIS framework to evolve a network scanner program
optimized for discovering devices on local networks, especially Nest thermostats.
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure the trisolaris package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the TRISOLARIS task runner and the network scanner task
from trisolaris.task_runner import TaskRunner
from trisolaris.tasks.network_scanner import NetworkScannerTask

def main():
    """Main function to run the network scanner evolution."""
    parser = argparse.ArgumentParser(description="Evolve a network scanner using TRISOLARIS")

    # Add arguments for evolution parameters
    parser.add_argument("--gens", type=int, default=15,
                        help="Number of generations to evolve (default: 15)")
    parser.add_argument("--pop-size", type=int, default=40,
                        help="Population size for each generation (default: 40)")
    parser.add_argument("--mutation-rate", type=float, default=0.2,
                        help="Mutation rate (default: 0.2)")
    parser.add_argument("--crossover-rate", type=float, default=0.7,
                        help="Crossover rate (default: 0.7)")
    parser.add_argument("--template", type=str, default="network_scanner.py",
                        help="Path to template network scanner script")
    parser.add_argument("--resource-monitoring", action="store_true",
                        help="Enable resource monitoring")
    parser.add_argument("--show-resource-report", action="store_true",
                        help="Show resource usage report after evolution")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (defaults to timestamped directory)")
    parser.add_argument("--use-islands", action="store_true",
                        help="Use island model for evolution")
    parser.add_argument("--islands", type=int, default=4,
                        help="Number of islands for island model (default: 4)")
    parser.add_argument("--migration-interval", type=int, default=3,
                        help="Migration interval for island model (default: 3)")

    args = parser.parse_args()

    # Create output directory if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/network_scanner_evolution/run_{timestamp}"

    # Create the task instance with focus on finding Nest thermostats
    task = NetworkScannerTask(template_path=args.template)

    print(f"\n{'='*50}")
    print(f"TRISOLARIS Network Scanner Evolution")
    print(f"{'='*50}")
    print(f"Starting evolution with the following parameters:")
    print(f"- Generations: {args.gens}")
    print(f"- Population size: {args.pop_size}")
    print(f"- Mutation rate: {args.mutation_rate}")
    print(f"- Crossover rate: {args.crossover_rate}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Resource monitoring: {'Enabled' if args.resource_monitoring else 'Disabled'}")
    print(f"- Island model: {'Enabled' if args.use_islands else 'Disabled'}")
    if args.use_islands:
        print(f"  - Islands: {args.islands}")
        print(f"  - Migration interval: {args.migration_interval}")
    print(f"{'='*50}\n")

    # Create the task runner
    runner = TaskRunner(
        task=task,
        output_dir=args.output_dir,
        num_generations=args.gens,
        population_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        resource_monitoring=args.resource_monitoring,
        show_resource_report=args.show_resource_report,
        use_islands=args.use_islands,
        islands=args.islands,
        migration_interval=args.migration_interval
    )

    print("Starting evolution process...")
    print("This may take some time depending on the number of generations and population size.")
    print("Progress will be logged to trisolaris_evolution.log")

    # Run the evolution
    best_genome, stats = runner.run()

    # Save the best evolved network scanner
    output_path = os.path.join(args.output_dir, "evolved_network_scanner.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(best_genome.to_source())

    # Make the evolved scanner executable
    os.chmod(output_path, 0o755)

    # Create a symlink to the evolved scanner for easy access
    symlink_path = "outputs/evolved_network_scanner.py"
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(os.path.abspath(output_path), symlink_path)

    print(f"\n{'='*50}")
    print(f"Evolution complete!")
    print(f"{'='*50}")
    print(f"Best genome fitness: {best_genome.fitness:.4f}")
    print(f"Total run time: {stats['duration_seconds']:.2f} seconds")
    print(f"Total generations: {stats['generations']}")
    print(f"\nFiles:")
    print(f"- Best genome saved to: {output_path}")
    print(f"- Symlink created at: {symlink_path}")
    print(f"\nTo run the evolved network scanner:")
    print(f"  ./outputs/evolved_network_scanner.py")
    print(f"{'='*50}")

    # Show detailed fitness metrics
    print("\nFitness Progression:")
    print("Generation | Fitness | Time(s)")
    print("-" * 40)
    for i, gen_stats in enumerate(stats.get("metrics", [])):
        print(f"{i:10} | {gen_stats.get('best_fitness', 0):.4f} | {gen_stats.get('time_seconds', 0):.2f}")

if __name__ == "__main__":
    main()
