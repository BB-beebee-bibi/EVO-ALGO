#!/usr/bin/env python3
"""
Evolve Bluetooth Scanner

A script that uses the TRISOLARIS framework to evolve a Bluetooth IoT device scanner
with security vulnerability detection capabilities, focusing on LATCH door locks.
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure the trisolaris package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the TRISOLARIS task runner
from trisolaris.task_runner import TaskRunner
from trisolaris.tasks.bluetooth_scanner import BluetoothScannerTask

def main():
    """Main function to run the Bluetooth scanner evolution."""
    parser = argparse.ArgumentParser(description="Evolve a Bluetooth IoT scanner using TRISOLARIS")
    
    # Add arguments for evolution parameters
    parser.add_argument("--gens", type=int, default=100, 
                        help="Number of generations to evolve (default: 100)")
    parser.add_argument("--pop-size", type=int, default=1000, 
                        help="Population size for each generation (default: 1000)")
    parser.add_argument("--mutation-rate", type=float, default=0.1, 
                        help="Mutation rate (default: 0.1)")
    parser.add_argument("--crossover-rate", type=float, default=0.7, 
                        help="Crossover rate (default: 0.7)")
    parser.add_argument("--template", type=str, default="bluetooth_scanner.py",
                        help="Path to template Bluetooth scanner script (default: bluetooth_scanner.py)")
    parser.add_argument("--resource-monitoring", action="store_true",
                        help="Enable resource monitoring")
    parser.add_argument("--show-resource-report", action="store_true",
                        help="Show resource usage report after evolution")
    parser.add_argument("--use-islands", action="store_true",
                        help="Use island model for evolution (recommended for large populations)")
    parser.add_argument("--islands", type=int, default=10,
                        help="Number of islands when using island model (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (defaults to timestamped directory)")
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/bluetooth_scanner_evolution/run_{timestamp}"
    
    print("=" * 80)
    print(f"Starting Bluetooth Scanner Evolution")
    print(f"Generations: {args.gens}")
    print(f"Population Size: {args.pop_size}")
    if args.use_islands:
        print(f"Island Model: Enabled with {args.islands} islands")
    print(f"Template: {args.template}")
    print("=" * 80)
    
    # Create the task instance
    task = BluetoothScannerTask(template_path=args.template)
    
    # Create the task runner
    runner = TaskRunner(
        task=task,
        output_dir=args.output_dir,
        num_generations=args.gens,
        population_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        resource_monitoring=args.resource_monitoring,
        use_islands=args.use_islands,
        islands=args.islands,
        show_resource_report=args.show_resource_report
    )
    
    # Run the evolution
    print(f"Starting evolution - this may take a while with the specified parameters...")
    best_genome, stats = runner.run()
    
    # Save the best evolved Bluetooth scanner
    output_path = os.path.join(args.output_dir, "evolved_bluetooth_scanner.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(best_genome.source_code)
    
    # Make the evolved scanner executable
    os.chmod(output_path, 0o755)
    
    # Create a symlink to the evolved scanner for easy access
    symlink_path = "outputs/evolved_bluetooth_scanner.py"
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(os.path.abspath(output_path), symlink_path)
    
    print(f"\nEvolution complete!")
    print(f"Best genome fitness: {best_genome.fitness:.4f}")
    print(f"Best genome saved to: {output_path}")
    print(f"Symlink created at: {symlink_path}")
    print(f"\nTo run the evolved Bluetooth scanner:")
    print(f"  ./outputs/evolved_bluetooth_scanner.py")
    
    # Print summary of stats
    print("\nEvolution Statistics:")
    print(f"Total generations: {stats.get('generations', args.gens)}")
    print(f"Total duration: {stats.get('duration_seconds', 0):.2f} seconds")
    
    # Remind about required libraries
    print("\nREMINDER: To use the Bluetooth scanner, you'll need to install the required libraries:")
    print("  pip install pybluez gattlib")
    print("On Linux, you may also need:")
    print("  sudo apt-get install bluetooth libbluetooth-dev")

if __name__ == "__main__":
    main() 