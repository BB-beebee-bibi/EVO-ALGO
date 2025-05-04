#!/usr/bin/env python3
"""
Evolve Network Scanner

A script that uses the TRISOLARIS framework to evolve a network scanner program.
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure the trisolaris package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the TRISOLARIS task runner
from trisolaris.task_runner import TaskRunner
from trisolaris.tasks.network_scanner import NetworkScannerTask

def main():
    """Main function to run the network scanner evolution."""
    parser = argparse.ArgumentParser(description="Evolve a network scanner using TRISOLARIS")
    
    # Add arguments for evolution parameters
    parser.add_argument("--gens", type=int, default=10, 
                        help="Number of generations to evolve")
    parser.add_argument("--pop-size", type=int, default=20, 
                        help="Population size for each generation")
    parser.add_argument("--mutation-rate", type=float, default=0.1, 
                        help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, 
                        help="Crossover rate")
    parser.add_argument("--template", type=str, default="network_scanner.py",
                        help="Path to template network scanner script")
    parser.add_argument("--resource-monitoring", action="store_true",
                        help="Enable resource monitoring")
    parser.add_argument("--show-resource-report", action="store_true",
                        help="Show resource usage report after evolution")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (defaults to timestamped directory)")
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/network_scanner_evolution/run_{timestamp}"
    
    # Create the task instance
    task = NetworkScannerTask(template_path=args.template)
    
    # Create the task runner
    runner = TaskRunner(
        task=task,
        output_dir=args.output_dir,
        num_generations=args.gens,
        population_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        resource_monitoring=args.resource_monitoring,
        show_resource_report=args.show_resource_report
    )
    
    # Run the evolution
    best_genome, stats = runner.run()
    
    # Save the best evolved network scanner
    output_path = os.path.join(args.output_dir, "evolved_network_scanner.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(best_genome.source_code)
    
    # Make the evolved scanner executable
    os.chmod(output_path, 0o755)
    
    # Create a symlink to the evolved scanner for easy access
    symlink_path = "outputs/evolved_network_scanner.py"
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(os.path.abspath(output_path), symlink_path)
    
    print(f"\nEvolution complete!")
    print(f"Best genome fitness: {best_genome.fitness:.4f}")
    print(f"Best genome saved to: {output_path}")
    print(f"Symlink created at: {symlink_path}")
    print(f"\nTo run the evolved network scanner:")
    print(f"  ./outputs/evolved_network_scanner.py")

if __name__ == "__main__":
    main() 