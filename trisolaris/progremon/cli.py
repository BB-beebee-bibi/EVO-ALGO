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
    
    DEFAULT_PARAMS = {
        'population_size': 50,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'max_generations': 100,
        'selection_pressure': 0.7,
        'elitism_ratio': 0.1
    }
    
    def __init__(self, config: Optional[ProgrémonConfig] = None):
        """Initialize the CLI interface."""
        self.config = config or ProgrémonConfig()
        self.core = ProgrémonCore(self.config)
        self.running = False
        self.status_update_interval = 2.0  # seconds
        self.last_status_update = 0
    
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
                self._update_status_if_needed()
                command = input("\nProgrémon> ").strip()
                self.process_command(command)
            except KeyboardInterrupt:
                print("\nExiting Progrémon...")
                self.running = False
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _update_status_if_needed(self) -> None:
        """Update status display if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_status_update >= self.status_update_interval:
            self._show_status(force=True)
            self.last_status_update = current_time
    
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
        elif action == "show":
            if len(cmd) < 2:
                print("Please specify a solution ID to show.")
                print("Example: show sol_1")
                return
            self.show_solution(cmd[1])
        elif action == "list":
            self.list_solutions()
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
  show    - Show details of a specific solution
           Example: show sol_1
  list    - List all solutions with their fitness scores
  exit    - Exit Progrémon
        """
        print(help_text)
    
    def _setup_evolution_parameters(self) -> Dict[str, Any]:
        """Interactively set up evolution parameters."""
        params = self.DEFAULT_PARAMS.copy()
        
        print("\nEvolution Parameters Setup")
        print("-------------------------")
        print("Press Enter to keep default values, or enter new values.")
        
        for param, default in params.items():
            while True:
                try:
                    value = input(f"{param} [{default}]: ").strip()
                    if not value:
                        break
                    
                    # Convert to appropriate type
                    if isinstance(default, int):
                        value = int(value)
                    elif isinstance(default, float):
                        value = float(value)
                    
                    # Validate value
                    if param in ['mutation_rate', 'crossover_rate', 'selection_pressure', 'elitism_ratio']:
                        if not 0 <= value <= 1:
                            print("Value must be between 0 and 1")
                            continue
                    elif param in ['population_size', 'max_generations']:
                        if value < 1:
                            print("Value must be greater than 0")
                            continue
                    
                    params[param] = value
                    break
                except ValueError:
                    print("Invalid value. Please try again.")
        
        return params
    
    def start_evolution(self, target_description: str) -> None:
        """Start a new evolution session."""
        try:
            print(f"\nStarting evolution session for: {target_description}")
            
            # Set up parameters interactively
            params = self._setup_evolution_parameters()
            
            # Update configuration
            self.config.initial_population_size = params['population_size']
            self.config.max_generations = params['max_generations']
            self.config.mutation_rate = params['mutation_rate']
            self.config.crossover_rate = params['crossover_rate']
            
            # Create evolution config
            evolution_config = {
                'evolution': {
                    'population_size': params['population_size'],
                    'max_generations': params['max_generations'],
                    'mutation_rate': params['mutation_rate'],
                    'crossover_rate': params['crossover_rate'],
                    'selection_pressure': params['selection_pressure'],
                    'elitism_ratio': params['elitism_ratio']
                },
                'target_description': target_description
            }
            
            # Start the evolution
            self.core.start_evolution_session(target_description, evolution_config)
            print("Evolution started successfully!")
        except Exception as e:
            print(f"Error starting evolution: {str(e)}")
    
    def _show_status(self, force: bool = False) -> None:
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
        
        # Show recent solutions if any
        if status["solutions"]:
            print("\nRecent Solutions:")
            for solution in status["solutions"]:
                print(f"  {solution['id']}: Fitness {solution['fitness']:.2%}")
    
    def show_solution(self, solution_id: str) -> None:
        """Show details of a specific solution."""
        status = self.core.get_evolution_status()
        if not status["solutions"]:
            print("No solutions available.")
            return
            
        # Find the solution
        solution = None
        for sol in status["solutions"]:
            if sol["id"] == solution_id:
                solution = sol
                break
                
        if not solution:
            print(f"Solution {solution_id} not found.")
            return
            
        print(f"\nSolution: {solution['id']}")
        print(f"Fitness: {solution['fitness']:.2%}")
        print("\nCode:")
        print("-" * 40)
        print(solution["code"])
        print("-" * 40)
    
    def list_solutions(self) -> None:
        """List all solutions with their fitness scores."""
        status = self.core.get_evolution_status()
        if not status["solutions"]:
            print("No solutions available.")
            return
            
        print("\nAll Solutions:")
        print("-" * 40)
        for solution in status["solutions"]:
            print(f"{solution['id']}: Fitness {solution['fitness']:.2%}")
        print("-" * 40)
    
    def stop_evolution(self) -> None:
        """Stop the current evolution."""
        try:
            print("Stopping evolution...")
            self.core.stop_evolution_session()
            print("Evolution stopped successfully!")
        except Exception as e:
            print(f"Error stopping evolution: {str(e)}") 