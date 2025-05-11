#!/usr/bin/env python
"""
Progremon: A conversational interface for the Trisolaris evolution system.
Gotta evolve 'em all! 🚀
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

# Core modules
from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator
from trisolaris.evaluation.boundary_enforcer import EthicalBoundaryEnforcer
from adaptive_tweaker import AdaptiveTweaker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI escape codes for colorful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def format(cls, text: str, color: str, bold: bool = False) -> str:
        """Format text with color and optional bold"""
        return f"{color}{cls.BOLD if bold else ''}{text}{cls.END}"

def print_color(text: str, color: str, bold: bool = False, end: str = '\n') -> None:
    """Print colored text to console"""
    print(Colors.format(text, color, bold), end=end)

def print_banner() -> None:
    """Print a stylish ASCII art banner for Progremon"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗ ██████╗  ██████╗  ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ███╗   ██╗   ║
    ║   ██╔══██╗██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗██╔════╝████╗ ████║██╔═══██╗████╗  ██║   ║
    ║   ██████╔╝██████╔╝██║   ██║██║  ███╗██████╔╝█████╗  ██╔████╔██║██║   ██║██╔██╗ ██║   ║
    ║   ██╔═══╝ ██╔══██╗██║   ██║██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║██║╚██╗██║   ║
    ║   ██║     ██║  ██║╚██████╔╝╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║ ╚████║   ║
    ║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ║
    ║                                                               ║
    ║          Gotta evolve 'em all! Evolution Runner v1.0          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print_color(banner, Colors.CYAN)

class ProgemonTrainer:
    """Interactive session manager for evolving code"""
    
    def __init__(self):
        """Initialize the Progremon interface"""
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
        
        self.adaptive_tweaker = AdaptiveTweaker(self.settings)
        self.colors = Colors()
        
        self.welcome_messages = [
            "Hi Trainer! What code would you like to evolve today?",
            "Welcome to the Progremon evolution lab! What shall we create?",
            "Ready to evolve some powerful code? What's your challenge?",
            "Greetings, code trainer! What program would you like to breed today?",
            "Welcome! What kind of code species are we evolving in this session?"
        ]
    
    def welcome(self):
        """Print a random welcome message"""
        print_color(random.choice(self.welcome_messages), Colors.GREEN, bold=True)
    
    def show_welcome(self):
        """Show a friendly welcome message"""
        print_color("\nWelcome to Progremon! I'm here to help you evolve any program you can imagine.", Colors.GREEN, bold=True)
        print_color("Just tell me what you'd like to create, and I'll help you evolve it!", Colors.GREEN)
    
    def get_user_request(self) -> str:
        """Get the user's request for what to evolve"""
        print_color("\nWhat would you like to create?", Colors.CYAN)
        return input("> ").strip()
    
    def parse_request(self, request: str) -> Dict[str, Any]:
        """
        Parse the user's natural language request and return configuration settings.
        Uses simple keyword matching for task detection.
        """
        config = self.settings.copy()
        request_lower = request.lower()
        
        if "bluetooth" in request_lower:
            config.update({
                "task": "bluetooth_scan",
                "pop_size": 50,
                "gens": 30,
                "mutation_rate": 0.25,
                "crossover_rate": 0.8,
                "max_execution_time": 10.0,
                "max_memory_usage": 500,
                "allowed_libraries": ["bluetooth"],
                "update_interval": 0.20,  # Update every 0.20 seconds
                "output_format": "table"
            })
            print_color("Detected Bluetooth scanner request. Configuring for Bluetooth scanning.", Colors.BLUE)
            print_color("Setting up for real-time scanning with table output format.", Colors.BLUE)
        else:
            config.update({
                "task": "general",
                "pop_size": 50,
                "gens": 25,
                "mutation_rate": 0.15,
                "crossover_rate": 0.7
            })
            print_color("Using default general-purpose evolution settings.", Colors.BLUE)
        
        return config
    
    def process_request(self, request: str) -> Dict[str, Any]:
        """Process the user's natural language request and configure evolution"""
        print_color("\nProcessing your request...", Colors.BLUE)
        config = self.parse_request(request)
        config["description"] = f"A program that {request}"
        return config
    
    def configure_evolution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration settings and return the final configuration"""
        final_settings = self.settings.copy()
        final_settings.update(config)
        
        print_color("\n⚙️ EVOLUTION SETTINGS ⚙️", Colors.BOLD)
        for key, value in final_settings.items():
            if key in ["task", "description"]:
                print_color(f"  {key}: {value}", Colors.GREEN, bold=True)
            else:
                print_color(f"  {key}: {value}", Colors.BLUE)
        
        print_color("\nWould you like to customize any of these settings? (y/n)", Colors.BLUE)
        if input("> ").lower().startswith("y"):
            self._customize_settings(final_settings)
        
        return final_settings
    
    def _customize_settings(self, settings: Dict[str, Any]):
        """Let user customize specific settings"""
        print_color("Enter new values (or press Enter to keep current value):", Colors.CYAN)
        
        # Task description
        print_color(f"description [{settings['description']}]: ", Colors.GREEN, bold=True, end='')
        desc_value = input().strip()
        if desc_value:
            settings["description"] = desc_value
        
        # Integer settings
        for key in ["pop_size", "gens"]:
            while True:
                try:
                    value = input(f"{key} [{settings[key]}]: ").strip()
                    if value:
                        settings[key] = int(value)
                    break
                except ValueError:
                    print_color("Please enter a valid integer.", Colors.FAIL)
        
        # Float settings
        for key in ["mutation_rate", "crossover_rate"]:
            while True:
                try:
                    value = input(f"{key} [{settings[key]}]: ").strip()
                    if value:
                        settings[key] = float(value)
                    break
                except ValueError:
                    print_color("Please enter a valid number between 0 and 1.", Colors.FAIL)
        
        # String settings
        for key in ["ethics_level"]:
            value = input(f"{key} [{settings[key]}]: ").strip()
            if value:
                settings[key] = value

    def run_evolution(self, settings: Dict[str, Any]) -> bool:
        """Run the evolution process with the given settings"""
        try:
            # Set up output directory
            output_dir = Path(settings["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize evolution components
            evaluator = FitnessEvaluator()
            enforcer = EthicalBoundaryEnforcer()
            
            # Configure ethical boundaries
            if settings["task"] == "bluetooth_scan":
                enforcer.add_boundary("max_execution_time", max_execution_time=settings["max_execution_time"])
                enforcer.add_boundary("max_memory_usage", max_memory_usage=settings["max_memory_usage"])
                enforcer.add_boundary("allowed_imports", allowed_imports=settings["allowed_libraries"])
                enforcer.add_boundary("no_eval_exec")
            
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
            
            # Initialize population with task description
            engine.initialize_population(
                size=settings["pop_size"],
                task_description=settings["description"],
                task_type=settings["task"]
            )
            

            
            # Run evolution
            print_color("\n🚀 Starting evolution process...", Colors.GREEN)
            for gen in range(1, settings["gens"] + 1):
                print_color(f"\nGeneration {gen}/{settings['gens']}", Colors.BOLD)
                
                # Evaluate current population
                fitness_scores = engine.evaluate_population()
                if not fitness_scores:
                    print_color("No valid solutions found in this generation.", Colors.WARNING)
                    continue
                
                # Calculate statistics
                avg_fitness = sum(f for f in fitness_scores if f != float('-inf')) / len(fitness_scores)
                best_fitness = max(fitness_scores)
                
                print_color(f"Average fitness: {avg_fitness:.2f}", Colors.BLUE)
                print_color(f"Best fitness: {best_fitness:.2f}", Colors.GREEN)
                
                # Save current generation if requested
                if settings["save_all_generations"]:
                    gen_dir = output_dir / f"generation_{gen}"
                    gen_dir.mkdir(exist_ok=True)
                    
                    # Save best solution
                    best_solution = engine.get_best_solution()
                    with open(gen_dir / "best.py", "w") as f:
                        f.write(best_solution.to_source())
                
                # Apply adaptive tweaking
                self.adaptive_tweaker.update_parameters(
                    avg_fitness,
                    best_fitness
                )
                
                # Generate next generation
                engine.generate_next_generation()
            
            # Save final best solution
            final_dir = output_dir / "final_solution"
            final_dir.mkdir(exist_ok=True)
            
            best_solution = engine.get_best_solution()
            with open(final_dir / "best.py", "w") as f:
                f.write(best_solution.to_source())
            
            print_color("\nEvolution complete! Best solution saved.", Colors.GREEN)
            return True
            
        except Exception as e:
            logger.error(f"Error during evolution: {str(e)}")
            print_color(f"\n❌ Error during evolution: {str(e)}", Colors.FAIL)
            return False

    def main(self) -> None:
        """Main entry point for Progremon"""
        try:
            print_banner()
            self.welcome()
            
            # Get user request
            request = self.get_user_request()
            if not request:
                print_color("No request provided. Exiting.", Colors.WARNING)
                return
            
            # Process and configure evolution
            config = self.process_request(request)
            settings = self.configure_evolution(config)
            
            # Get user confirmation
            print_color("\n🚀 Ready to evolve your program! 🚀", Colors.GREEN)
            print_color(settings["description"], Colors.GREEN, bold=True)
            print_color("\nPress Enter to start or 'q' to quit.", Colors.BLUE)
            
            if input("> ").lower().startswith("q"):
                print_color("\nGoodbye! Come back when you're ready to evolve more code!", Colors.GREEN)
                return
            
            # Run evolution
            if self.run_evolution(settings):
                print_color("\nEvolution complete! Check the output directory for results.", Colors.GREEN)
            
        except KeyboardInterrupt:
            print_color("\n\nEvolution interrupted by user. Goodbye!", Colors.YELLOW)
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
            print_color(f"\n❌ Fatal error: {str(e)}", Colors.FAIL)

if __name__ == "__main__":
    trainer = ProgemonTrainer()
    trainer.main()

