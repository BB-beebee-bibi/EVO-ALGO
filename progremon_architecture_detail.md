# Progremon Implementation Details

This document provides detailed implementation guidance for rewriting `progremon.py`. It follows the high-level architecture outlined in `project_summary.md` and provides specific code structure, method signatures, and implementation details.

## Import Structure

The proper import structure should be:

```python
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
import time
import datetime
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path

# Core modules
from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator
from trisolaris.evaluation.boundary_enforcer import EthicalBoundaryEnforcer

# Local imports
from adaptive_tweaker import AdaptiveTweaker, EvolutionMetrics
```

## Colors and Display Utilities

The Colors class needs to be accessible to the AdaptiveTweaker. Either:

1. Export Colors from progremon.py to a shared utilities module, or
2. Import the print_color function into adaptive_tweaker.py 

```python
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
```

## Task Templates

Create a TaskTemplateLoader class to manage task-specific templates:

```python
class TaskTemplateLoader:
    """Loads and configures task-specific code templates."""
    
    def __init__(self, templates_dir: str = "guidance"):
        self.templates_dir = templates_dir
        self.templates = {
            "bluetooth_scan": "bluetooth_scanner_template.py",
            "general": None  # General tasks don't use a specific template
        }
    
    def load_template(self, task_type: str) -> Optional[str]:
        """Load template code for a specific task type."""
        if task_type not in self.templates or not self.templates[task_type]:
            return None
            
        template_file = Path(self.templates_dir) / self.templates[task_type]
        if not template_file.exists():
            logging.warning(f"Template file {template_file} not found")
            return None
            
        with open(template_file, 'r') as f:
            return f.read()
    
    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """Get task-specific configuration settings."""
        if task_type == "bluetooth_scan":
            return {
                "pop_size": 50,
                "gens": 30,
                "mutation_rate": 0.25,
                "crossover_rate": 0.8,
                "max_execution_time": 10.0,
                "max_memory_usage": 500,
                "allowed_libraries": ["bluetooth"],
                "update_interval": 0.20,
                "output_format": "table"
            }
        else:
            return {
                "pop_size": 50,
                "gens": 25,
                "mutation_rate": 0.15,
                "crossover_rate": 0.7
            }
```

## Session Management

Add proper session management to track evolution runs:

```python
class EvolutionSession:
    """Manages a single evolution session with unique ID and output directory."""
    
    def __init__(self, base_dir: str = "evolved_output", task_type: str = "general"):
        self.session_id = f"trial_{random.randint(1, 9999):04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = Path(base_dir) / self.session_id
        self.task_type = task_type
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.best_fitness = float('-inf')
        self.generations_completed = 0
        
        # Create session directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save session metadata to file."""
        metadata = {
            "session_id": self.session_id,
            "task_type": self.task_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "best_fitness": self.best_fitness,
            "generations_completed": self.generations_completed
        }
        
        with open(self.base_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_generation_dir(self, generation: int) -> Path:
        """Get directory for a specific generation."""
        gen_dir = self.base_dir / f"generation_{generation:03d}"
        gen_dir.mkdir(exist_ok=True)
        return gen_dir
    
    def update_stats(self, generations_completed: int, best_fitness: float) -> None:
        """Update session statistics."""
        self.generations_completed = generations_completed
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
        self._save_metadata()
    
    def complete(self) -> None:
        """Mark session as complete."""
        self.end_time = datetime.datetime.now()
        self._save_metadata()
    
    def save_best_solution(self, source_code: str) -> Path:
        """Save best solution to the session directory."""
        solution_dir = self.base_dir / "best_solution"
        solution_dir.mkdir(exist_ok=True)
        
        solution_file = solution_dir / "best.py"
        with open(solution_file, "w") as f:
            f.write(source_code)
            
        return solution_file
```

## ProgemonTrainer Enhancements

The ProgemonTrainer class should be enhanced with the following methods:

```python
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
        self.template_loader = TaskTemplateLoader()
        self.session = None
        self.colors = Colors()
        
        # Set up improved logging
        self._setup_logging()
        
        self.welcome_messages = [
            "Hi Trainer! What code would you like to evolve today?",
            "Welcome to the Progremon evolution lab! What shall we create?",
            "Ready to evolve some powerful code? What's your challenge?",
            "Greetings, code trainer! What program would you like to breed today?",
            "Welcome! What kind of code species are we evolving in this session?"
        ]
    
    def _setup_logging(self) -> None:
        """Set up enhanced logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"progremon_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, console_handler]
        )
```

## Enhanced Request Parsing

Improve the request parsing to better detect task type and extract parameters:

```python
def parse_request(self, request: str) -> Dict[str, Any]:
    """
    Parse the user's natural language request and return configuration settings.
    Uses keyword matching and parameter inference.
    """
    config = self.settings.copy()
    request_lower = request.lower()
    
    # Detect task type
    if any(keyword in request_lower for keyword in ["bluetooth", "ble", "bt device"]):
        task_type = "bluetooth_scan"
        print_color("Detected Bluetooth scanner request. Configuring for Bluetooth scanning.", Colors.BLUE)
    else:
        task_type = "general"
        print_color("Using default general-purpose evolution settings.", Colors.BLUE)
    
    # Load task-specific configuration
    task_config = self.template_loader.get_task_config(task_type)
    config.update(task_config)
    config["task"] = task_type
    
    # Extract update interval if specified
    if "update" in request_lower and "second" in request_lower:
        try:
            # Try to find update interval pattern (e.g., "update every 0.5 seconds")
            import re
            update_match = re.search(r"update (?:every\s+)?(\d+\.?\d*)\s*seconds?", request_lower)
            if update_match:
                config["update_interval"] = float(update_match.group(1))
                print_color(f"Setting update interval to {config['update_interval']} seconds", Colors.BLUE)
        except Exception as e:
            logging.warning(f"Failed to parse update interval: {e}")
    
    # Extract output format if specified
    if "table" in request_lower:
        config["output_format"] = "table"
        print_color("Setting output format to table", Colors.BLUE)
    elif "json" in request_lower:
        config["output_format"] = "json"
        print_color("Setting output format to JSON", Colors.BLUE)
    
    return config
```

## Run Evolution Method

The `run_evolution` method should be enhanced with proper error handling and integration:

```python
def run_evolution(self, settings: Dict[str, Any]) -> bool:
    """Run the evolution process with the given settings"""
    try:
        # Create a new evolution session
        self.session = EvolutionSession(
            base_dir=settings["output_dir"],
            task_type=settings["task"]
        )
        logging.info(f"Starting evolution session {self.session.session_id}")
        
        # Initialize evolution components
        evaluator = FitnessEvaluator()
        enforcer = EthicalBoundaryEnforcer()
        
        # Configure ethical boundaries based on task type
        self._configure_ethical_boundaries(enforcer, settings)
        
        # Initialize evolution engine with parameters
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
        
        # Initialize population with task description and template
        engine.initialize_population(
            size=settings["pop_size"],
            task_description=settings["description"],
            task_type=settings["task"],
            **template_info
        )
        
        # Run evolution
        print_color("\n🚀 Starting evolution process...", Colors.GREEN)
        population_metrics = []
        
        for gen in range(1, settings["gens"] + 1):
            print_color(f"\nGeneration {gen}/{settings['gens']}", Colors.BOLD)
            gen_start_time = time.time()
            
            try:
                # Evaluate current population
                fitness_scores = engine.evaluate_population()
                
                if not fitness_scores or all(f == float('-inf') for f in fitness_scores):
                    print_color("No valid solutions found in this generation.", Colors.WARNING)
                    logging.warning(f"Generation {gen}: No valid solutions")
                    continue
                
                # Calculate statistics
                valid_scores = [f for f in fitness_scores if f != float('-inf')]
                avg_fitness = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                best_fitness = max(valid_scores) if valid_scores else float('-inf')
                best_idx = fitness_scores.index(best_fitness)
                
                # Collect population metrics
                population = engine.get_population()
                population_metrics.append({
                    "generation": gen,
                    "avg_fitness": avg_fitness,
                    "best_fitness": best_fitness,
                    "valid_solutions": len(valid_scores),
                    "total_solutions": len(fitness_scores)
                })
                
                print_color(f"Average fitness: {avg_fitness:.2f}", Colors.BLUE)
                print_color(f"Best fitness: {best_fitness:.2f}", Colors.GREEN)
                print_color(f"Valid solutions: {len(valid_scores)}/{len(fitness_scores)}", Colors.BLUE)
                
                # Save current generation if requested
                if settings["save_all_generations"]:
                    gen_dir = self.session.get_generation_dir(gen)
                    
                    # Save best solution
                    best_solution = engine.get_best_solution()
                    with open(gen_dir / "best.py", "w") as f:
                        f.write(best_solution.to_source())
                    
                    # Save generation metrics
                    with open(gen_dir / "metrics.json", "w") as f:
                        json.dump(population_metrics[-1], f, indent=2)
                
                # Update session stats
                self.session.update_stats(gen, best_fitness)
                
                # Apply adaptive tweaking using the proper method
                self.adaptive_tweaker.record_metrics(
                    population=population,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fitness
                )
                
                # Get adjusted parameters
                new_params = self.adaptive_tweaker.adjust_parameters()
                
                # Apply parameter changes to the engine
                if engine.mutation_rate != new_params["mutation_rate"]:
                    print_color(
                        f"Adjusting mutation rate: {engine.mutation_rate:.3f} -> {new_params['mutation_rate']:.3f}",
                        Colors.YELLOW
                    )
                    engine.mutation_rate = new_params["mutation_rate"]
                
                # Generate next generation
                engine.generate_next_generation()
                
                # Log generation completion
                gen_time = time.time() - gen_start_time
                logging.info(f"Generation {gen} completed in {gen_time:.2f}s. Best fitness: {best_fitness:.2f}")
                
            except Exception as e:
                logging.error(f"Error in generation {gen}: {str(e)}")
                logging.error(traceback.format_exc())
                print_color(f"⚠️ Error in generation {gen}: {str(e)}", Colors.WARNING)
                print_color("Attempting to continue with next generation...", Colors.WARNING)
        
        # Save final best solution
        best_solution = engine.get_best_solution()
        solution_path = self.session.save_best_solution(best_solution.to_source())
        
        # Complete the session
        self.session.complete()
        
        print_color(f"\nEvolution complete! Best solution saved to {solution_path}", Colors.GREEN)
        print_color(f"Session ID: {self.session.session_id}", Colors.BLUE)
        return True
        
    except Exception as e:
        logging.error(f"Critical error during evolution: {str(e)}")
        logging.error(traceback.format_exc())
        print_color(f"\n❌ Critical error during evolution: {str(e)}", Colors.FAIL)
        
        if self.session:
            print_color(f"Session ID: {self.session.session_id}", Colors.BLUE)
            print_color(f"Check logs for details.", Colors.BLUE)
            self.session.complete()
        
        return False
```

## Ethical Boundary Configuration

Add a dedicated method for configuring ethical boundaries:

```python
def _configure_ethical_boundaries(self, enforcer: EthicalBoundaryEnforcer, settings: Dict[str, Any]) -> None:
    """Configure ethical boundaries based on task type and settings."""
    # Common boundaries for all tasks
    enforcer.add_boundary("no_eval_exec")
    enforcer.add_boundary("no_destructive_operations")
    
    # Task-specific boundaries
    if settings["task"] == "bluetooth_scan":
        enforcer.add_boundary(
            "max_execution_time", 
            max_execution_time=settings.get("max_execution_time", 10.0)
        )
        enforcer.add_boundary(
            "max_memory_usage", 
            max_memory_usage=settings.get("max_memory_usage", 500)
        )
        enforcer.add_boundary(
            "allowed_imports", 
            allowed_imports=settings.get("allowed_libraries", ["bluetooth"])
        )
        # Add bluetooth-specific boundaries
        enforcer.add_boundary(
            "no_continuous_scanning", 
            max_scan_time=settings.get("max_scan_time", 30.0)
        )
        enforcer.add_boundary(
            "privacy_respecting",
            requires_user_consent=True
        )
```

## Modifications to Main Method

The main method should be enhanced with better error handling:

```python
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
        
        user_input = input("> ").lower()
        if user_input.startswith("q"):
            print_color("\nGoodbye! Come back when you're ready to evolve more code!", Colors.GREEN)
            return
        
        # Run evolution
        success = self.run_evolution(settings)
        
        if success:
            print_color("\nEvolution complete! Check the output directory for results.", Colors.GREEN)
            if self.session:
                best_dir = self.session.base_dir / "best_solution"
                print_color(f"Best solution saved to: {best_dir}", Colors.GREEN)
        else:
            print_color("\nEvolution process encountered errors. Check logs for details.", Colors.WARNING)
        
    except KeyboardInterrupt:
        print_color("\n\nEvolution interrupted by user. Goodbye!", Colors.YELLOW)
    except Exception as e:
        logging.error(f"Fatal error in main loop: {str(e)}")
        logging.error(traceback.format_exc())
        print_color(f"\n❌ Fatal error: {str(e)}", Colors.FAIL)
```

## AdaptiveTweaker Modifications

The following modifications should be made to the adaptive_tweaker.py file to fix the integration issues:

```python
# Import color utilities
from progremon import Colors, print_color

class AdaptiveTweaker:
    # ... existing code ...
    
    def update_parameters(self, avg_fitness: float, best_fitness: float) -> None:
        """
        Legacy method for compatibility with the original Progremon.
        This method exists as a bridge to the record_metrics method.
        """
        # Create a mock population with the provided fitness values
        mock_population = [
            type('MockGenome', (), {'fitness': avg_fitness}),
            type('MockGenome', (), {'fitness': best_fitness})
        ]
        
        # Call the actual method
        self.record_metrics(mock_population, best_fitness, avg_fitness)
```

## Modifications for Bluetooth Integration

Add specific utilities for Bluetooth integration:

```python
def _load_bluetooth_template(self) -> str:
    """Load the Bluetooth scanner template with correct imports."""
    template = self.template_loader.load_template("bluetooth_scan")
    if not template:
        # Fallback template if none found
        template = """
import bluetooth
import time
from typing import List, Dict, Any

def scan_bluetooth_devices() -> List[Dict[str, Any]]:
    """
    Scan for nearby Bluetooth devices and return their information.
    
    Returns:
        List of dictionaries containing device information
    """
    devices = []
    try:
        nearby_devices = bluetooth.discover_devices(
            duration=8,
            lookup_names=True,
            lookup_class=True,
            device_id=-1
        )
        
        for addr, name, device_class in nearby_devices:
            device_info = {
                "address": addr,
                "name": name if name else "Unknown",
                "class": device_class,
                "signal_strength": 0,  # Not available without additional libraries
                "timestamp": time.time()
            }
            devices.append(device_info)
    except Exception as e:
        return [{"error": str(e)}]
    return devices

def format_as_table(devices: List[Dict[str, Any]]) -> str:
    """Format devices as an ASCII table."""
    if not devices:
        return "No devices found"
        
    # Check if we have an error
    if len(devices) == 1 and "error" in devices[0]:
        return f"Error: {devices[0]['error']}"
        
    # Format as table
    header = "| {:<17} | {:<20} | {:<8} |".format("Address", "Name", "Class")
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    for device in devices:
        lines.append("| {:<17} | {:<20} | {:<8} |".format(
            device["address"],
            device["name"][:20],
            device["class"]
        ))
    
    lines.append(separator)
    return "\\n".join(lines)

def main():
    """Main function to demonstrate Bluetooth scanning."""
    devices = scan_bluetooth_devices()
    print(format_as_table(devices))
    
if __name__ == "__main__":
    main()
"""
    return template
```

## Comprehensive Implementation Notes

For a complete and robust implementation of the rewritten progremon.py, consider these additional notes:

1. **Error Handling Strategy**
   - Use specific exception types for different error categories
   - Log all exceptions with proper stack traces
   - Provide meaningful error messages to users

2. **Parameter Validation**
   - Validate all user-provided parameters before use
   - Enforce range constraints on numerical parameters
   - Implement fallbacks for invalid values

3. **Code Generation Enhancement**
   - Use template injection for task-specific code
   - Add context-aware commenting in generated code
   - Implement proper indentation and code style

4. **Testing Strategy**
   - Mock the evolution engine for unit testing
   - Create test fixtures for common tasks
   - Implement property-based testing for parameter tweaking

5. **Performance Considerations**
   - Use lazy loading for templates and components
   - Implement caching for repetitive operations
   - Add progress indicators for long-running operations

These implementation details provide a comprehensive guide for rewriting the progremon.py file to address all identified issues and create a robust implementation of the Bluetooth scanning functionality.