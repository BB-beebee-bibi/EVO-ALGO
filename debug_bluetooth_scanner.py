#!/usr/bin/env python3
"""
Debug Bluetooth Scanner Evolution

A script that uses the debug-enhanced TRISOLARIS framework to evolve a bluetooth scanner
program with comprehensive logging and performance monitoring.
"""

import os
import sys
import argparse
import datetime

# Ensure the trisolaris package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the debug-enhanced bluetooth scanner evolution."""
    parser = argparse.ArgumentParser(
        description="Debug and evolve a bluetooth scanner using TRISOLARIS with enhanced monitoring"
    )

    # Add arguments for evolution parameters
    parser.add_argument("--gens", type=int, default=5,
                        help="Number of generations to evolve (default: 5)")
    parser.add_argument("--pop-size", type=int, default=20,
                        help="Population size for each generation (default: 20)")
    parser.add_argument("--mutation-rate", type=float, default=0.2,
                        help="Mutation rate (default: 0.2)")
    parser.add_argument("--crossover-rate", type=float, default=0.7,
                        help="Crossover rate (default: 0.7)")
    parser.add_argument("--template", type=str, default="bluetooth_scanner.py",
                        help="Path to template bluetooth scanner script")
    parser.add_argument("--resource-monitoring", action="store_true",
                        help="Enable resource monitoring")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (defaults to timestamped directory)")
    parser.add_argument("--use-islands", action="store_true",
                        help="Use island model for evolution")
    parser.add_argument("--islands", type=int, default=3,
                        help="Number of islands for island model (default: 3)")
    parser.add_argument("--debug-level", choices=["minimal", "normal", "verbose", "trace"],
                        default="verbose", help="Debug logging level (default: verbose)")
    parser.add_argument("--ethics-level", choices=["none", "basic", "full"],
                        default="basic", help="Ethical filter level (default: basic)")

    args = parser.parse_args()

    # Create output directory if not specified
    if not args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/bluetooth_scanner_debug_{timestamp}"

    # Print banner
    print("\n" + "="*70)
    print("TRISOLARIS DEBUG BLUETOOTH SCANNER EVOLUTION")
    print("="*70)
    print("This script runs the bluetooth scanner evolution task with enhanced debugging")
    print("and comprehensive performance monitoring.")
    print("\nParameters:")
    print(f"- Generations: {args.gens}")
    print(f"- Population size: {args.pop_size}")
    print(f"- Mutation rate: {args.mutation_rate}")
    print(f"- Crossover rate: {args.crossover_rate}")
    print(f"- Debug level: {args.debug_level}")
    print(f"- Ethics level: {args.ethics_level}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Resource monitoring: {'Enabled' if args.resource_monitoring else 'Disabled'}")
    print(f"- Island model: {'Enabled' if args.use_islands else 'Disabled'}")
    if args.use_islands:
        print(f"  - Islands: {args.islands}")
    print("="*70 + "\n")

    # Build the command to run the debug task runner
    cmd = [
        "python3", "trisolaris_debug_runner.py",
        "bluetooth_scanner",
        f"--gens={args.gens}",
        f"--pop-size={args.pop_size}",
        f"--mutation-rate={args.mutation_rate}",
        f"--crossover-rate={args.crossover_rate}",
        f"--output-dir={args.output_dir}",
        f"--debug-level={args.debug_level}",
        f"--ethics-level={args.ethics_level}"
    ]
    
    # Add optional arguments
    if args.template:
        cmd.append(f"--template={args.template}")
    if args.resource_monitoring:
        cmd.append("--resource-monitoring")
    if args.use_islands:
        cmd.append("--use-islands")
        cmd.append(f"--islands={args.islands}")

    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    print("\nStarting evolution process...")
    
    # Execute the command
    import subprocess
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
