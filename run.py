#!/usr/bin/env python
"""
Trisolaris Evolution Runner

A simple CLI to evolve code using the Trisolaris framework, with basic safety guarantees.
"""

import argparse
import os
import logging
import platform
from typing import Optional, List
from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
import datetime
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_basic_safety_filter(output_dir: str) -> EthicalBoundaryEnforcer:
    """Create a basic ethical filter that prevents destructive operations."""
    enforcer = EthicalBoundaryEnforcer()
    
    # Safety-first boundaries
    enforcer.add_boundary("no_system_calls")
    enforcer.add_boundary("no_eval_exec")
    
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

def setup_usb_scan_filter(output_dir: str) -> EthicalBoundaryEnforcer:
    """Create a safety filter that allows USB drive operations but prevents other destructive operations."""
    enforcer = EthicalBoundaryEnforcer()
    
    # Safety-first boundaries
    enforcer.add_boundary("no_eval_exec")
    
    # Allow specific system calls for USB scanning (we need os.popen to run wmic)
    # enforcer.add_boundary("no_system_calls")  # Commented out to allow wmic calls
    
    # Resource constraints
    enforcer.add_boundary("max_execution_time", max_execution_time=5.0)  # 5 seconds max for USB ops
    enforcer.add_boundary("max_memory_usage", max_memory_usage=200)      # 200MB max
    
    # Allow controlled imports specifically needed for USB scanning
    enforcer.add_boundary("no_imports", allowed_imports={
        'os', 'sys', 'time', 'random', 'math', 'json', 
        'datetime', 'collections', 're', 'logging', 'platform'
    })
    
    # We need to allow file operations for USB scanning
    # enforcer.add_boundary("no_file_operations")  # Commented out to allow file operations
    
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

def load_initial_genomes(input_dir: str, task: str = "general") -> List[CodeGenome]:
    """Load initial genomes from Python files in the input directory."""
    genomes = []
    
    # Try to find a template that matches the task type
    template_path = os.path.join(input_dir, "guidance", f"{task}_template.py")
    if os.path.exists(template_path):
        logger.info(f"Using {task} template from {template_path}")
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                source = f.read()
            genomes.append(CodeGenome.from_source(source))
            return genomes
        except Exception as e:
            logger.warning(f"Could not load {task} template: {e}")
    
    # If no specific template found, use a general template if available
    general_template_path = os.path.join(input_dir, "guidance", "general_template.py")
    if os.path.exists(general_template_path):
        logger.info(f"Using general template from {general_template_path}")
        try:
            with open(general_template_path, 'r', encoding='utf-8') as f:
                source = f.read()
            genomes.append(CodeGenome.from_source(source))
            return genomes
        except Exception as e:
            logger.warning(f"Could not load general template: {e}")
    
    # If no template found, start with a directory-based genome
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

def add_usb_scan_test_cases(evaluator: FitnessEvaluator):
    """Add test cases specifically for evaluating USB scanning functionality."""
    
    # Test case 1: Finding USB drives
    evaluator.add_test_case(
        input_data={},  # No input needed
        expected_output=["E"],  # Expect at least drive E to be found
        weight=0.3,
        name="find_usb_drives",
        custom_validator=lambda actual, expected: (
            isinstance(actual, list) and any(drive in expected for drive in actual)
        ) if isinstance(actual, list) else False
    )
    
    # Test case 2: Get drive info for drive E
    evaluator.add_test_case(
        input_data={"drive_letter": "E"},
        expected_output={"path": "E"},  # Just check that the path is returned correctly
        weight=0.3,
        name="get_drive_info",
        custom_validator=lambda actual, expected: (
            isinstance(actual, dict) and actual.get("path") == expected["path"]
        ) if isinstance(actual, dict) else False
    )
    
    # Test case 3: Scan directory structure
    evaluator.add_test_case(
        input_data={"dir_path": "E:"},
        expected_output={"type": "directory"},  # Just check that it returns directory info
        weight=0.4,
        name="scan_directory",
        custom_validator=lambda actual, expected: (
            isinstance(actual, dict) and actual.get("type") == expected["type"]
        ) if isinstance(actual, dict) else False
    )

def add_bluetooth_test_cases(evaluator: FitnessEvaluator):
    """Add test cases specifically for evaluating Bluetooth functionality."""
    
    # Test case 1: Scan for nearby devices
    evaluator.add_test_case(
        input_data={},  # No input needed
        expected_output=["name", "address", "signal_strength"],  # Check for these fields
        weight=0.4,
        name="scan_devices",
        custom_validator=lambda actual, expected: (
            isinstance(actual, list) and 
            all(isinstance(item, dict) for item in actual) and
            all(key in item for item in actual for key in expected)
        )
    )
    
    # Test case 2: Get device info
    evaluator.add_test_case(
        input_data={"address": "00:11:22:33:44:55"},  # Example address
        expected_output={
            "name": str,
            "type": str,
            "signal_strength": float
        },
        weight=0.3,
        name="get_device_info",
        custom_validator=lambda actual, expected: (
            isinstance(actual, dict) and 
            all(isinstance(actual[key], expected[key]) for key in expected)
        )
    )
    
    # Test case 3: Check device type detection
    evaluator.add_test_case(
        input_data={"address": "00:11:22:33:44:55", "type": "phone"},
        expected_output={"type": "phone"},
        weight=0.3,
        name="detect_device_type",
        custom_validator=lambda actual, expected: (
            isinstance(actual, dict) and 
            actual.get("type") == expected["type"]
        )
    )

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
        default="evolved_output",
        help="Base directory for all evolution trials"
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
        choices=["none", "basic", "full", "usb"],
        default="basic",
        help="Ethical filter level (none, basic, full, usb) (default: basic)"
    )
    parser.add_argument(
        "--save-all-generations",
        action="store_true",
        help="Save best solution from each generation"
    )
    parser.add_argument(
        "--task",
        default="general",
        help="The type of task to evolve (e.g., bluetooth_scanner, usb_scanner, etc.)"
    )
    args = parser.parse_args()

    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean up any old trial directories that might be incomplete
    for trial_dir in [d for d in os.listdir(args.output_dir) if d.startswith("trial_")]:
        trial_path = os.path.join(args.output_dir, trial_dir)
        if not os.path.exists(os.path.join(trial_path, "completed")):
            logger.info(f"Cleaning up incomplete trial: {trial_dir}")
            shutil.rmtree(trial_path)
    
    # Create a unique trial directory for this run
    trial_count = len([d for d in os.listdir(args.output_dir) if d.startswith("trial_")]) + 1
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    trial_dir = os.path.join(args.output_dir, f"trial_{trial_count:04d}_{current_time}")
    os.makedirs(trial_dir, exist_ok=True)
    logger.info(f"Starting trial {trial_count} in directory: {trial_dir}")
    
    # 1) Load initial genomes
    initial_genomes = load_initial_genomes(args.input_dir, args.task)
    if not initial_genomes:
        logger.error(f"Could not load any valid code from {args.input_dir}")
        return
    
    # 2) Setup ethical filter based on chosen level
    ethical_filter = None
    if args.ethics_level == "basic":
        ethical_filter = setup_basic_safety_filter(args.output_dir)
    elif args.ethics_level == "full":
        ethical_filter = setup_full_ethics_filter(args.output_dir)
    elif args.ethics_level == "usb":
        ethical_filter = setup_usb_scan_filter(args.output_dir)
    
    # 3) Build the fitness evaluator
    evaluator = FitnessEvaluator(ethical_filter=ethical_filter)
    
    # 4) Add task-specific test cases if needed
    if args.task == "usb_scanner":
        add_usb_scan_test_cases(evaluator)
        logger.info("Added USB scanner test cases")
    elif args.task == "bluetooth_scanner":
        add_bluetooth_test_cases(evaluator)
        logger.info("Added Bluetooth scanner test cases")
    
    # 5) Configure weights for objectives if ethics is enabled
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
    
    # 6) Setup the evolution engine
    engine = EvolutionEngine(
        population_size=args.pop_size,
        evaluator=evaluator,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        genome_class=CodeGenome,
    )
    
    # Add initial genomes to the population
    engine.population = initial_genomes
    # Pad with random genomes if needed
    while len(engine.population) < args.pop_size:
        engine.population.append(CodeGenome())
    
    # 7) Run the evolution
    logger.info(f"Starting evolution with population size {args.pop_size} for {args.gens} generations")
    logger.info(f"Ethics level: {args.ethics_level}")
    logger.info(f"Task: {args.task}")
    
    # Track generations
    for gen in range(args.gens):
        # Evaluate the current generation
        engine.evaluate_population()
        
        # Create generation directory
        gen_dir = os.path.join(trial_dir, f"generation_{gen:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Evaluate the current generation
        engine.evaluate_population()
        
        # Save all solutions in this generation
        solutions = sorted(engine.population, key=lambda x: x.fitness, reverse=True)
        for i, solution in enumerate(solutions[:3], 1):  # Save top 3 solutions
            solution_path = os.path.join(gen_dir, f"solution_{i:02d}_fitness_{solution.fitness:.4f}.py")
            with open(solution_path, "w", encoding="utf-8") as f:
                f.write(solution.to_source())
        
        # Create the next generation (unless it's the last generation)
        if gen < args.gens - 1:
            parents = engine.select_parents()
            offspring = engine.create_offspring(parents)
            engine.population = engine.select_survivors(offspring)
    
    # 8) Save the final best solution
    best = engine.get_best_solution()
    if best:
        # Mark this trial as completed
        with open(os.path.join(trial_dir, "completed"), "w") as f:
            f.write(f"Trial completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Save the final generation's top solutions
        gen_dir = os.path.join(trial_dir, f"generation_{args.gens-1:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Save the top 3 solutions from the final generation
        solutions = sorted(engine.population, key=lambda x: x.fitness, reverse=True)
        for i, solution in enumerate(solutions[:3], 1):
            solution_dir = os.path.join(trial_dir, f"generation_{gen}")
            os.makedirs(solution_dir, exist_ok=True)
            solution_path = os.path.join(solution_dir, f"solution_{i}.py")
            with open(solution_path, "w", encoding="utf-8") as f:
                f.write(solution.to_source())
        logger.info(f"Best solution saved to {trial_dir}")
        logger.info(f"Best fitness: {engine.best_fitness:.4f}")
    else:
        logger.warning("No valid solution found")

if __name__ == "__main__":
    main()
    main() 