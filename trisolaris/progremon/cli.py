"""
Command-line interface for Progrémon UX component.
"""

import sys
import time
from typing import Optional, List, Dict, Any
from .config import ProgrémonConfig
from .core import ProgrémonCore

class ProgrémonCLI:
    """Command-line interface for Progrémon."""
    
    ASCII_LOGO = r"""
    ____            ____                  
   / __ \___  ____ / __ \___  ____  ____ 
  / /_/ / _ \/ __  / /_/ / _ \/ __ \/ __ \
 / ____/  __/ /_/ / ____/  __/ / / / / / /
/_/   \___/\__,_/_/    \___/_/ /_/_/ /_/ 
    """
    
    WELCOME_MESSAGE = """
    Welcome to Progrémon!
    Gotta evolve 'em all!
    
    Type 'help' for available commands.
    """
    
    def __init__(self, config: Optional[ProgrémonConfig] = None):
        """Initialize the CLI interface."""
        self.config = config or ProgrémonConfig()
        self.core = ProgrémonCore(self.config)
        self.running = False
    
    def display_logo(self) -> None:
        """Display the Progrémon ASCII art logo."""
        if self.config.show_ascii_art:
            print(self.ASCII_LOGO)
    
    def display_welcome(self) -> None:
        """Display the welcome message."""
        if self.config.show_welcome_message:
            print(self.WELCOME_MESSAGE)
    
    def start(self) -> None:
        """Start the Progrémon CLI interface."""
        self.running = True
        self.display_logo()
        self.display_welcome()
        
        while self.running:
            try:
                command = input("\nProgrémon> ").strip()
                self.process_command(command)
            except KeyboardInterrupt:
                print("\nExiting Progrémon...")
                self.running = False
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def process_command(self, command: str) -> None:
        """Process user commands."""
        if not command:
            return
            
        cmd = command.lower().split()
        action = cmd[0]
        
        if action == "help":
            self.show_help()
        elif action == "start":
            # Handle natural language start command
            target_description = " ".join(cmd[1:]) if len(cmd) > 1 else None
            if not target_description:
                print("Please provide a description of what you want to evolve.")
                print("Example: start evolve a program that sorts files by date")
                return
            self.start_evolution(target_description)
        elif action == "status":
            self._show_status()
        elif action == "stop":
            self.stop_evolution()
        elif action == "exit":
            self.running = False
        else:
            print(f"Unknown command: {action}")
            print("Type 'help' for available commands.")
    
    def show_help(self) -> None:
        """Display available commands."""
        help_text = """
Available commands:
  help    - Show this help message
  start   - Start a new evolution session
           Example: start evolve a program that sorts files by date
  status  - Show current evolution status
  stop    - Stop the current evolution
  exit    - Exit Progrémon
        """
        print(help_text)
    
    def start_evolution(self, target_description: str) -> None:
        """Start a new evolution session."""
        try:
            print(f"\nStarting evolution session for: {target_description}")
            
            # Create evolution config with default parameters
            evolution_config = {
                'evolution': {
                    'population_size': self.config.initial_population_size,
                    'mutation_rate': self.config.mutation_rate,
                    'crossover_rate': self.config.crossover_rate,
                    'selection_pressure': self.config.selection_pressure,
                    'elitism_ratio': self.config.elitism_ratio
                }
            }
            
            # Start the evolution
            self.core.start_evolution_session(target_description, evolution_config)
            print("Evolution started successfully!")
        except Exception as e:
            print(f"Error starting evolution: {str(e)}")
    
    def _show_status(self) -> None:
        """Show current evolution status."""
        status = self.core.get_evolution_status()
        
        if status["status"] == "not_started":
            print("Evolution status: Not started")
            return
        
        # Format elapsed time
        elapsed = int(status["elapsed_time"])
        minutes = elapsed // 60
        seconds = elapsed % 60
        
        # Print status header
        print("\nEvolution Status:")
        print(f"Status: {status['status'].upper()}")
        print(f"Target: {status['target_description']}")
        print(f"Time: {minutes}m {seconds}s")
        print(f"Generation: {status['generation']}")
        print(f"Population Size: {status['population_size']}")
        print(f"Best Fitness: {status['best_fitness']:.2%}")
    
    def stop_evolution(self) -> None:
        """Stop the current evolution session."""
        self.core.stop_evolution_session()
        print("Evolution stopped.") 