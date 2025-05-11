#!/usr/bin/env python
"""
Enhanced Progremon System

This is the main module for the enhanced Progremon system, an evolutionary code
generation platform that uses the Trisolaris evolution engine to evolve program
solutions based on user requirements.

This enhanced version addresses integration issues, improves error handling,
enhances ethical boundary enforcement, adds specialized task functionality,
and improves session management and output organization.
"""

import os
import sys
import json
import time
import random
import argparse
import logging
import traceback
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path

# Import utilities
from utils import Colors, print_color, print_banner, setup_logging, generate_unique_id

# Import core modules
try:
    from trisolaris.core import EvolutionEngine, CodeGenome
    from trisolaris.evaluation import FitnessEvaluator
    from trisolaris.evaluation.boundary_enforcer import EthicalBoundaryEnforcer

    # Use the fixed version of adaptive tweaker or fall back to original
    try:
        from adaptive_tweaker_fix import AdaptiveTweaker, EvolutionMetrics
    except ImportError:
        # Fall back to original if fix isn't available
        try:
            from adaptive_tweaker import AdaptiveTweaker, EvolutionMetrics
        except ImportError:
            print("Error: AdaptiveTweaker module not found")
            sys.exit(1)
            
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Make sure the Trisolaris package is installed or in your PYTHONPATH")
    sys.exit(1)

# Import local components
from task_template_loader import TaskTemplateLoader
from evolution_session import EvolutionSession


class ProgemonTrainer:
    """
    Main interface for the enhanced Progremon system.
    
    This class handles user interaction, request parsing, and orchestrates
    the evolution process. It integrates all components of the system and
    provides a consistent interface for different ways of using the system.
    """
    
    def __init__(self):
        """Initialize the Progremon interface with enhanced components"""
        self.settings = {
            "output_dir": "evolved_output",
            "pop_size": 50,
            "gens": 25,
            "mutation_rate": 0.15,
            "crossover_rate": 0.7,
            "ethics_level": "basic",
            "save_all_generations": True,
            "input_dir": "guidance",
            "task": "general"
        }
        
        self.template_loader = TaskTemplateLoader()
        
        # Initialize components
        self.adaptive_tweaker = AdaptiveTweaker(self.settings)
        self.session = None
        
        # Set up improved logging
        self._setup_logging()
        
        # Welcome messages for interactive mode
        self.welcome_messages = [
            "Hi Trainer! What code would you like to evolve today?",
            "Welcome to the Progremon evolution lab! What shall we create?",
            "Ready to evolve some powerful code? What's your challenge?",
            "Greetings, code trainer! What program would you like to breed today?",
            "Welcome! What kind of code species are we evolving in this session?"
        ]
    
    def _setup_logging(self):
        """Set up logging for the enhanced system."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"progremon_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logging(log_file)
        self.logger.info("Progremon enhanced system initialized")
    
    def show_welcome(self):
        """Display welcome message and ASCII banner."""
        print_banner()
        welcome = random.choice(self.welcome_messages)
        print_color(welcome, Colors.CYAN, bold=True)
    
    def parse_request(self, description: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a natural language request into structured settings.
        
        Args:
            description: Natural language description of the task
            task_type: Optional explicit task type
            
        Returns:
            Dictionary of settings for the evolution process
        """
        self.logger.info(f"Parsing request: {description[:50]}...")
        
        # Detect task type from description if not explicitly provided
        if not task_type:
            task_type = self.template_loader.get_task_type_from_description(description)
            self.logger.debug(f"Detected task type: {task_type}")
        
        # Start with default settings
        parsed_settings = self.settings.copy()
        
        # Update with task-specific settings
        parsed_settings["description"] = description
        parsed_settings["task"] = task_type
        
        # Adjust settings based on task type
        if task_type == "bluetooth_scan":
            parsed_settings["pop_size"] = 40  # Smaller population for more focused task
            parsed_settings["gens"] = 20
            parsed_settings["ethics_level"] = "strict"  # Higher ethics for device access
        
        self.logger.info(f"Request parsed with task type: {task_type}")
        return parsed_settings
    
    def _configure_ethical_boundaries(self, enforcer: EthicalBoundaryEnforcer, settings: Dict[str, Any]):
        """
        Configure ethical boundaries with progressive scaling.
        
        Args:
            enforcer: The ethical boundary enforcer instance
            settings: Evolution settings
        """
        ethics_level = settings.get("ethics_level", "basic")
        task_type = settings.get("task", "general")
        
        # Base configuration
        enforcer.set_level(ethics_level)
        
        # Task-specific ethical configurations
        if task_type == "bluetooth_scan":
            enforcer.add_restriction("no_unlimited_scanning")
            enforcer.add_restriction("respect_device_privacy")
            enforcer.add_restriction("conserve_battery")
            enforcer.set_resource_limit("scan_duration", 30)  # Max 30 seconds scan
        
        # Progressive ethics that scale with solution quality
        enforcer.enable_progressive_ethics(True)
        
        self.logger.info(f"Configured ethical boundaries: level={ethics_level}, task={task_type}")
    
    def run_evolution(self, settings: Dict[str, Any]) -> bool:
        """
        Run evolution with enhanced error handling and component integration.
        
        Args:
            settings: Dictionary of evolution settings
            
        Returns:
            True if evolution was successful, False otherwise
        """
        try:
            # Create a new evolution session with unique ID
            self.session = EvolutionSession(
                base_dir=settings["output_dir"],
                task_type=settings["task"]
            )
            self.logger.info(f"Starting evolution session {self.session.session_id}")
            
            # Update session with settings
            self.session.update_settings(settings)
            
            # Initialize components with proper integration
            evaluator = FitnessEvaluator()
            enforcer = EthicalBoundaryEnforcer()
            
            # Configure ethical boundaries with progressive scaling
            self._configure_ethical_boundaries(enforcer, settings)
            
            # Initialize evolution engine
            engine = EvolutionEngine(
                population_size=settings["pop_size"],
                evaluator=evaluator,
                genome_class=CodeGenome,
                mutation_rate=settings["mutation_rate"],
                crossover_rate=settings["crossover_rate"],
                elitism_ratio=0.1
            )
            
            # Set up ethical boundaries
            evaluator.ethical_filter = enforcer
            
            # Load task template if available
            template_code = self.template_loader.load_template(settings["task"])
            template_info = {"template_code": template_code} if template_code else {}
            
            # Initialize population with proper template integration
            print_color("\nInitializing population...", Colors.BLUE)
            engine.initialize_population(
                size=settings["pop_size"],
                task_description=settings["description"],
                task_type=settings["task"],
                **template_info
            )
            
            # Initialize adaptive tweaker for this evolution
            self.adaptive_tweaker.reset()
            self.adaptive_tweaker.set_initial_params(settings)
            
            # Initialize metrics collection
            metrics = EvolutionMetrics()
            
            # Evolution loop with proper metrics collection and parameter adjustment
            print_color(f"\nEvolving for {settings['gens']} generations...", Colors.BLUE)
            for gen in range(settings["gens"]):
                try:
                    # Display progress
                    progress = (gen + 1) / settings["gens"] * 100
                    print_color(f"Generation {gen + 1}/{settings['gens']} ({progress:.1f}%)", Colors.YELLOW)
                    
                    # Evaluate population
                    engine.evaluate_population()
                    
                    # Collect metrics
                    best_fitness = engine.get_best_fitness()
                    avg_fitness = engine.get_average_fitness()
                    metrics.record_generation(gen, best_fitness, avg_fitness)
                    
                    # Record generation in session
                    self.session.record_generation(
                        generation_number=gen,
                        best_fitness=best_fitness,
                        avg_fitness=avg_fitness,
                        population_size=engine.population_size
                    )
                    
                    # Display progress metrics
                    print_color(f"  Best Fitness: {best_fitness:.4f}", Colors.GREEN)
                    print_color(f"  Avg Fitness: {avg_fitness:.4f}", Colors.GREEN)
                    
                    # Adjust parameters using adaptive tweaker
                    params = self.adaptive_tweaker.adjust_parameters(
                        generation=gen,
                        best_fitness=best_fitness,
                        avg_fitness=avg_fitness,
                        diversity=engine.calculate_diversity()
                    )
                    
                    # Apply parameter adjustments
                    engine.mutation_rate = params.get("mutation_rate", engine.mutation_rate)
                    engine.crossover_rate = params.get("crossover_rate", engine.crossover_rate)
                    
                    # Save best solution of this generation if enabled
                    if settings.get("save_all_generations", False):
                        best_code = engine.get_best_solution().code
                        self.session.save_solution(
                            code=best_code,
                            generation=gen,
                            fitness=best_fitness,
                            is_best=False
                        )
                    
                    # Create next generation
                    engine.create_next_generation()
                    
                    # Create checkpoint periodically
                    if gen % 5 == 0 and gen > 0:
                        self.session.create_checkpoint()
                    
                except Exception as gen_error:
                    self.logger.error(f"Error in generation {gen}: {str(gen_error)}")
                    print_color(f"Warning: Error in generation {gen}: {str(gen_error)}", Colors.RED)
                    print_color("Attempting to continue with next generation...", Colors.YELLOW)
                    continue
            
            # Get best solution
            best_solution = engine.get_best_solution()
            best_code = best_solution.code
            best_fitness = best_solution.fitness
            
            # Save the final best solution
            solution_path = self.session.save_solution(
                code=best_code,
                generation=settings["gens"] - 1,
                fitness=best_fitness,
                is_best=True
            )
            
            # Finalize session
            self.session.finalize()
            
            # Display success message
            print_color("\nEvolution completed successfully!", Colors.GREEN, bold=True)
            print_color(f"Best solution saved to: {solution_path}", Colors.CYAN)
            print_color(f"Final fitness: {best_fitness:.4f}", Colors.CYAN)
            
            return True
            
        except Exception as e:
            # Enhanced error handling
            self.logger.error(f"Error during evolution: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            print_color("\nError during evolution process:", Colors.RED, bold=True)
            print_color(str(e), Colors.RED)
            print_color("\nSee logs for details.", Colors.YELLOW)
            
            # Attempt to save partial results if session exists
            if self.session:
                try:
                    self.session.finalize()
                    print_color("Partial results have been saved.", Colors.YELLOW)
                except Exception as save_error:
                    self.logger.error(f"Error saving partial results: {str(save_error)}")
            
            return False
    
    def run_interactive(self):
        """Run the system in interactive mode with user prompts."""
        self.show_welcome()
        
        # Get task description
        print_color("\nDescribe the task you want to evolve code for:", Colors.GREEN)
        description = input("> ")
        
        if not description.strip():
            print_color("No task description provided. Exiting.", Colors.RED)
            return False
        
        # Detect task type
        task_type = self.template_loader.get_task_type_from_description(description)
        print_color(f"\nDetected task type: {task_type}", Colors.BLUE)
        
        # Allow user to override task type
        print_color("Is this correct? (Y/n, or enter a different task type)", Colors.GREEN)
        task_response = input("> ").strip().lower()
        
        if task_response and task_response != "y" and task_response != "yes":
            if task_response in ["n", "no"]:
                print_color("\nAvailable task types:", Colors.BLUE)
                for task in ["general", "bluetooth_scan", "usb_scan", "web", "game"]:
                    print_color(f"  - {task}", Colors.CYAN)
                print_color("\nEnter task type:", Colors.GREEN)
                task_type = input("> ").strip().lower()
            else:
                task_type = task_response
        
        # Parse request
        settings = self.parse_request(description, task_type)
        
        # Allow customizing settings
        print_color("\nWould you like to customize evolution settings? (y/N)", Colors.GREEN)
        customize = input("> ").strip().lower()
        
        if customize in ["y", "yes"]:
            print_color("\nEnter settings (press Enter to keep default):", Colors.BLUE)
            
            # Population size
            print_color(f"Population size [{settings['pop_size']}]:", Colors.GREEN)
            pop_input = input("> ").strip()
            if pop_input:
                try:
                    settings["pop_size"] = int(pop_input)
                except ValueError:
                    print_color("Invalid input, using default.", Colors.YELLOW)
            
            # Generations
            print_color(f"Number of generations [{settings['gens']}]:", Colors.GREEN)
            gen_input = input("> ").strip()
            if gen_input:
                try:
                    settings["gens"] = int(gen_input)
                except ValueError:
                    print_color("Invalid input, using default.", Colors.YELLOW)
            
            # Mutation rate
            print_color(f"Mutation rate [{settings['mutation_rate']}]:", Colors.GREEN)
            mut_input = input("> ").strip()
            if mut_input:
                try:
                    settings["mutation_rate"] = float(mut_input)
                except ValueError:
                    print_color("Invalid input, using default.", Colors.YELLOW)
            
            # Output directory
            print_color(f"Output directory [{settings['output_dir']}]:", Colors.GREEN)
            dir_input = input("> ").strip()
            if dir_input:
                settings["output_dir"] = dir_input
        
        # Confirm and run evolution
        print_color("\nReady to start evolution with these settings:", Colors.BLUE)
        for key, value in settings.items():
            if key != "description":  # Skip printing the full description
                print_color(f"  {key}: {value}", Colors.CYAN)
        
        print_color("\nStart evolution? (Y/n)", Colors.GREEN)
        confirm = input("> ").strip().lower()
        
        if confirm in ["n", "no"]:
            print_color("Evolution cancelled.", Colors.YELLOW)
            return False
        
        # Run evolution
        return self.run_evolution(settings)
    
    def run_command_line(self, args):
        """
        Run the system in command-line mode with provided arguments.
        
        Args:
            args: Command-line arguments
            
        Returns:
            True if evolution was successful, False otherwise
        """
        # Parse description from arguments
        if not args.description:
            print_color("Error: No task description provided.", Colors.RED)
            return False
        
        # Parse settings from arguments
        settings = self.parse_request(args.description, args.task)
        
        # Override settings from command-line arguments
        if args.output_dir:
            settings["output_dir"] = args.output_dir
        
        if args.pop_size:
            settings["pop_size"] = args.pop_size
        
        if args.generations:
            settings["gens"] = args.generations
        
        if args.mutation_rate:
            settings["mutation_rate"] = args.mutation_rate
        
        if args.crossover_rate:
            settings["crossover_rate"] = args.crossover_rate
        
        if args.ethics_level:
            settings["ethics_level"] = args.ethics_level
        
        # Display settings
        print_banner()
        print_color("Starting evolution with settings:", Colors.BLUE)
        for key, value in settings.items():
            if key != "description":  # Skip printing the full description
                print_color(f"  {key}: {value}", Colors.CYAN)
        
        # Run evolution
        return self.run_evolution(settings)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Progremon: Evolution-based Code Generation')
    
    parser.add_argument('--description', type=str, help='Natural language description of the coding task')
    parser.add_argument('--task', type=str, default='general', help='Task type (general, bluetooth_scan, etc.)')
    parser.add_argument('--output-dir', type=str, help='Directory for evolution output')
    parser.add_argument('--pop-size', type=int, help='Population size')
    parser.add_argument('--generations', type=int, help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, help='Crossover rate')
    parser.add_argument('--ethics-level', type=str, choices=['basic', 'strict'], help='Ethical boundary level')
    
    return parser.parse_args()


def main():
    """Main entry point for the enhanced Progremon system."""
    args = parse_arguments()
    
    # Create the Progremon trainer
    trainer = ProgemonTrainer()
    
    # Run in appropriate mode
    if args.description:
        # Command-line mode
        success = trainer.run_command_line(args)
    else:
        # Interactive mode
        success = trainer.run_interactive()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()