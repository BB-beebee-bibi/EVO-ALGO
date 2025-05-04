#!/usr/bin/env python3
"""
Network Scanner Task Example

This example demonstrates how to use the TRISOLARIS framework
to evolve a network scanner program.
"""

import os
import sys
import argparse

# Ensure the trisolaris package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trisolaris.task_runner import TaskRunner
from trisolaris.tasks.network_scanner import NetworkScannerTask

def main():
    """Run the network scanner evolution example."""
    parser = argparse.ArgumentParser(description="TRISOLARIS Network Scanner Example")
    
    parser.add_argument("--gens", type=int, default=5, 
                        help="Number of generations to evolve (default: 5)")
    parser.add_argument("--pop-size", type=int, default=10, 
                        help="Population size for each generation (default: 10)")
    parser.add_argument("--resource-monitoring", action="store_true",
                        help="Enable resource monitoring")
    parser.add_argument("--show-resource-report", action="store_true",
                        help="Show resource usage report after evolution")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRISOLARIS Network Scanner Evolution Example")
    print("=" * 80)
    print(f"Running evolution with {args.gens} generations and population size {args.pop_size}")
    print("=" * 80)
    
    # Create the network scanner task
    task = NetworkScannerTask()
    
    # Create the task runner
    runner = TaskRunner(
        task=task,
        output_dir="outputs/network_scanner_example",
        num_generations=args.gens,
        population_size=args.pop_size,
        resource_monitoring=args.resource_monitoring,
        show_resource_report=args.show_resource_report
    )
    
    # Run the evolution
    best_genome, stats = runner.run()
    
    # Save the best evolved program
    output_path = "outputs/network_scanner_example/evolved_network_scanner.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(best_genome.source_code)
    
    # Make it executable
    os.chmod(output_path, 0o755)
    
    print("\nEvolution complete!")
    print(f"Best genome fitness: {best_genome.fitness:.4f}")
    print(f"Best genome saved to: {output_path}")
    
    print("\nTo run the evolved network scanner:")
    print(f"  {output_path}")

if __name__ == "__main__":
    main() 