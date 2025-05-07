#!/usr/bin/env python
"""
Progremon: A conversational interface for the Trisolaris evolution system.
Like Pokemon, but for evolving code!
"""

import os
import sys
import json
import argparse
import subprocess
import random
from typing import Dict, Any, List, Optional

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

def print_color(text, color, end='\n'):
    """Print colored text to console"""
    print(f"{color}{text}{Colors.END}", end=end)

def print_banner():
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
        """Initialize the trainer with default settings"""
        self.settings = {
            "output_dir": "evolved_output",
            "pop_size": 20,
            "gens": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "ethics_level": "basic",
            "save_all_generations": False,
            "input_dir": "minja",
            "task": "general"
        }
        
        self.welcome_messages = [
            "Hi Trainer! What code would you like to evolve today?",
            "Welcome to the Progremon evolution lab! What shall we create?",
            "Ready to evolve some powerful code? What's your challenge?",
            "Greetings, code trainer! What program would you like to breed today?",
            "Welcome! What kind of code species are we evolving in this session?"
        ]
        
        self.task_presets = {
            "usb_scanner": {
                "description": "A program that scans a USB drive and lists its contents",
                "input_dir": "minja",
                "ethics_level": "usb",
                "task": "usb_scanner"
            },
            "calculator": {
                "description": "A simple calculator program that handles basic operations",
                "input_dir": "minja",
                "ethics_level": "basic",
                "task": "general"
            }
        }
        
        self.available_tasks = ["general", "usb_scanner"]
    
    def welcome(self):
        """Print a random welcome message"""
        print_color(random.choice(self.welcome_messages), Colors.GREEN + Colors.BOLD)
    
    def show_presets(self):
        """Display the available task presets"""
        print_color("\nAvailable task presets:", Colors.BOLD)
        for name, preset in self.task_presets.items():
            print(f"  {Colors.CYAN}{name}{Colors.END}: {preset['description']}")
        print()
    
    def get_user_request(self) -> str:
        """Get the user's request for what to evolve"""
        return input("> ").strip()
    
    def parse_request(self, request: str) -> Dict[str, Any]:
        """
        Parse the user's natural language request and return configuration settings.
        In a full implementation, this would call an LLM API to extract task parameters.
        
        For now, we'll use simple keyword matching.
        """
        request_lower = request.lower()
        
        # Try to match against presets first
        for name, preset in self.task_presets.items():
            if name in request_lower:
                print_color(f"Using the '{name}' preset configuration.", Colors.BLUE)
                return preset
        
        # Handle USB scanner request specifically (since that's our current focus)
        if any(term in request_lower for term in ["usb", "drive", "scan", "storage"]):
            print_color("Detected a USB scanner request.", Colors.BLUE)
            return self.task_presets["usb_scanner"]
        
        # Default to a general purpose evolution
        print_color("Using default general-purpose evolution settings.", Colors.WARNING)
        return {
            "description": "A general-purpose program based on your request",
            "input_dir": "minja",
            "ethics_level": "basic",
            "task": "general"
        }
    
    def select_task(self) -> str:
        """Explicitly prompt the user to select a task"""
        print_color("\n📋 TASK SELECTION 📋", Colors.BOLD + Colors.HEADER)
        print_color("Please select the type of program you want to evolve:", Colors.CYAN)
        
        # Show available tasks with numbers
        for i, task in enumerate(self.available_tasks, 1):
            description = next((preset["description"] for name, preset in self.task_presets.items() 
                              if preset["task"] == task), f"A {task} program")
            print(f"  {i}. {Colors.CYAN}{task}{Colors.END}: {description}")
        
        # Get user selection
        while True:
            try:
                choice = input(f"\nSelect a task [1-{len(self.available_tasks)}]: ").strip()
                
                # Check if they entered a number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_tasks):
                        selected_task = self.available_tasks[idx]
                        print_color(f"You selected: {selected_task}", Colors.GREEN)
                        
                        # Find the preset for this task if it exists
                        for name, preset in self.task_presets.items():
                            if preset["task"] == selected_task:
                                print_color(f"Using the '{name}' preset configuration.", Colors.BLUE)
                                return preset
                        
                        # If no preset exists, create basic config with this task
                        return {
                            "description": f"A {selected_task} program",
                            "input_dir": "minja",
                            "ethics_level": "basic" if selected_task == "general" else "usb",
                            "task": selected_task
                        }
                
                # Check if they entered the task name directly
                elif choice in self.available_tasks:
                    selected_task = choice
                    print_color(f"You selected: {selected_task}", Colors.GREEN)
                    
                    # Find the preset for this task if it exists
                    for name, preset in self.task_presets.items():
                        if preset["task"] == selected_task:
                            print_color(f"Using the '{name}' preset configuration.", Colors.BLUE)
                            return preset
                    
                    # If no preset exists, create basic config with this task
                    return {
                        "description": f"A {selected_task} program",
                        "input_dir": "minja",
                        "ethics_level": "basic" if selected_task == "general" else "usb",
                        "task": selected_task
                    }
                
                print_color("Invalid selection. Please try again.", Colors.FAIL)
            except ValueError:
                print_color("Please enter a valid number.", Colors.FAIL)
    
    def configure_evolution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration settings and return the final configuration"""
        # Merge the parsed config with default settings
        final_settings = self.settings.copy()
        final_settings.update(config)
        
        # Allow user to customize settings
        print_color("\n⚙️ EVOLUTION SETTINGS ⚙️", Colors.BOLD)
        for key, value in final_settings.items():
            if key == "task":
                print(f"  {Colors.BOLD}{key}{Colors.END}: {Colors.GREEN}{value}{Colors.END}")
            else:
                print(f"  {key}: {value}")
        
        print_color("\nWould you like to customize any of these settings? (y/n)", Colors.BLUE)
        if input("> ").lower().startswith("y"):
            self._customize_settings(final_settings)
        
        return final_settings
    
    def _customize_settings(self, settings: Dict[str, Any]):
        """Let user customize specific settings"""
        print_color("Enter new values (or press Enter to keep current value):", Colors.CYAN)
        
        # Task selection (first, most important)
        print_color(f"task [{settings['task']}]: ", Colors.BOLD + Colors.GREEN, end='')
        task_value = input().strip()
        if task_value:
            if task_value in self.available_tasks:
                settings["task"] = task_value
                # Update ethics level based on task
                if task_value == "usb_scanner" and settings["ethics_level"] == "basic":
                    print_color("Setting ethics_level to 'usb' for USB scanner task", Colors.BLUE)
                    settings["ethics_level"] = "usb"
            else:
                print_color(f"Invalid task: {task_value}. Using {settings['task']}", Colors.FAIL)
        
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
        for key in ["ethics_level", "input_dir", "output_dir"]:
            value = input(f"{key} [{settings[key]}]: ").strip()
            if value:
                settings[key] = value
        
        # Boolean settings
        value = input(f"save_all_generations [{settings['save_all_generations']}]: ").strip()
        if value.lower() in ["true", "yes", "y", "1"]:
            settings["save_all_generations"] = True
        elif value.lower() in ["false", "no", "n", "0"]:
            settings["save_all_generations"] = False
    
    def run_evolution(self, settings: Dict[str, Any]):
        """Run the evolution process with the given settings"""
        # Build the command
        cmd = [
            "python", "run.py",
            settings["input_dir"],
            "--output-dir", settings["output_dir"],
            "--pop-size", str(settings["pop_size"]),
            "--gens", str(settings["gens"]),
            "--mutation-rate", str(settings["mutation_rate"]),
            "--crossover-rate", str(settings["crossover_rate"]),
            "--ethics-level", settings["ethics_level"],
            "--task", settings["task"]
        ]
        
        if settings["save_all_generations"]:
            cmd.append("--save-all-generations")
        
        # Show command
        print_color("\nRunning evolution with command:", Colors.BLUE)
        print(" ".join(cmd))
        print_color("\nEvolution in progress... This might take a while!\n", Colors.WARNING)
        
        # Run the command
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream and colorize the output
            for line in iter(process.stdout.readline, ''):
                if "ERROR" in line or "error" in line.lower():
                    print_color(line.strip(), Colors.FAIL)
                elif "WARNING" in line or "warning" in line.lower():
                    print_color(line.strip(), Colors.WARNING)
                elif "Generation" in line or "fitness" in line:
                    print_color(line.strip(), Colors.GREEN)
                else:
                    print(line.strip())
                
                if process.poll() is not None:
                    break
            
            process.wait()
            
            if process.returncode != 0:
                print_color(f"\nEvolution process failed with code {process.returncode}", Colors.FAIL)
                return False
            
            print_color("\nEvolution completed successfully!", Colors.GREEN + Colors.BOLD)
            return True
            
        except Exception as e:
            print_color(f"\nError running evolution: {str(e)}", Colors.FAIL)
            return False
    
    def display_results(self, settings: Dict[str, Any]):
        """Display the results of the evolution"""
        best_file = os.path.join(settings["output_dir"], "best.py")
        
        if os.path.exists(best_file):
            print_color("\nBest evolved solution:", Colors.BOLD)
            
            # Display the evolved code
            with open(best_file, 'r') as f:
                code = f.read()
            
            print_color("\n```python", Colors.CYAN)
            print(code)
            print_color("```\n", Colors.CYAN)
            
            # Ask if user wants to test the solution
            print_color("Would you like to test this solution? (y/n)", Colors.BLUE)
            if input("> ").lower().startswith("y"):
                self.test_solution(best_file)
        else:
            print_color("\nNo solution found. Check the logs for details.", Colors.FAIL)
    
    def test_solution(self, solution_file: str):
        """Test the evolved solution"""
        print_color("\nTesting the evolved solution...", Colors.BLUE)
        
        try:
            result = subprocess.run(
                ["python", solution_file],
                capture_output=True, 
                text=True
            )
            
            print_color("\nOutput:", Colors.BOLD)
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print_color("Errors:", Colors.FAIL)
                print(result.stderr)
            
            print_color("\nTest completed with exit code: " + str(result.returncode), 
                      Colors.GREEN if result.returncode == 0 else Colors.FAIL)
            
        except Exception as e:
            print_color(f"Error testing solution: {str(e)}", Colors.FAIL)

def main():
    """Main entry point for the Progremon CLI"""
    print_banner()
    
    trainer = ProgemonTrainer()
    trainer.welcome()
    trainer.show_presets()
    
    # Get user's request
    request = trainer.get_user_request()
    
    # Try to parse the request to determine initial task settings
    initial_config = trainer.parse_request(request)
    
    # Explicit task selection step
    final_config = trainer.select_task()
    
    # Keep any non-task parameters from the initial request
    for key, value in initial_config.items():
        if key != "task" and key != "ethics_level" and key not in final_config:
            final_config[key] = value
    
    # Configure the evolution
    settings = trainer.configure_evolution(final_config)
    
    # Final confirmation of the task
    print_color(f"\n🚀 Ready to evolve a {Colors.BOLD}{settings['task']}{Colors.END} program! 🚀", Colors.GREEN)
    print_color("Press Enter to start or 'q' to quit.", Colors.BLUE)
    if input("> ").lower().startswith("q"):
        print_color("Evolution cancelled. Goodbye!", Colors.CYAN)
        return
    
    # Run the evolution
    if trainer.run_evolution(settings):
        # Display results
        trainer.display_results(settings)

if __name__ == "__main__":
    main() 