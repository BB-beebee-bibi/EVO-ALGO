#!/usr/bin/env python
"""
Progremon: A modular framework for evolutionary code generation.
Gotta evolve 'em all! 🚀

This module serves as the main interface to the Trisolaris evolution system,
providing a user-friendly way to evolve code for various tasks including
Bluetooth scanning and other programmable tasks.
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
from trisolaris.core import EvolutionEngine, CodeGenome as TriCodeGenome
from trisolaris.evaluation import FitnessEvaluator as TriFitnessEvaluator
from trisolaris.evaluation.boundary_enforcer import EthicalBoundaryEnforcer as TriEthicalBoundaryEnforcer

# Ensure adaptive_tweaker_fix.py is imported instead of adaptive_tweaker.py
try:
    # Try to import the fixed version first
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adaptive_tweaker_fix import AdaptiveTweaker
    print("Using fixed adaptive tweaker")
except ImportError:
    # Fall back to regular import if fixed version isn't available
    from adaptive_tweaker import AdaptiveTweaker
    print("Using standard adaptive tweaker")

# Set up root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI escape codes for colorful output
class Colors:
    """Color codes for terminal output styling."""
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
    """Print colored text to console."""
    print(Colors.format(text, color, bold), end=end)

def print_banner() -> None:
    """Print a stylish ASCII art banner for Progremon."""
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


# Enhanced CodeGenome with compatibility layer
class CodeGenome(TriCodeGenome):
    """
    Enhanced CodeGenome that provides compatibility with boundary enforcer.
    
    This class extends the Trisolaris CodeGenome with a 'code' property
    to ensure compatibility with boundary enforcement mechanisms that expect
    this attribute to be available for inspection.
    """
    
    @property
    def code(self) -> str:
        """
        Property that returns source code for compatibility with boundary enforcer.
        
        Returns:
            The source code as a string
        """
        return self.to_source()
    
    def __init__(self, ast_tree=None, source_code=None):
        """
        Initialize the enhanced CodeGenome with the same parameters as TriCodeGenome.
        
        Args:
            ast_tree: Optional AST representation of the code
            source_code: Optional source code string
        """
        super().__init__(ast_tree=ast_tree, source_code=source_code)


# TieredEthicalEnforcer for progressive ethical enforcement
class TieredEthicalEnforcer:
    """
    Ethical enforcer with tiered enforcement levels based on solution fitness.
    
    This class implements a progressive approach to ethical enforcement, where
    different levels of ethical constraints can be applied based on solution
    fitness relative to a baseline.
    
    Ethics tiers:
    - Tier 0: No ethics checks (for solutions below baseline)
    - Tier 1: Basic safety (for solutions above baseline)
    - Tier 2: More constraints (for solutions significantly above baseline)
    - Tier 3: Full constraints (for solutions far above baseline)
    """
    
    def __init__(self):
        """Initialize the tiered ethical enforcer."""
        self.boundaries = {}
        self.logger = logging.getLogger(__name__ + ".TieredEthics")
        
        # Define the ethics tiers (which boundaries apply at each tier)
        self.ethics_tiers = [
            # Tier 0: No ethics checks (for solutions below baseline)
            [],
            # Tier 1: Basic safety (for solutions above baseline)
            ["no_eval_exec", "no_destructive_operations"],
            # Tier 2: More constraints (for solutions >10% above baseline)
            ["no_eval_exec", "no_destructive_operations", "allowed_imports"],
            # Tier 3: Full constraints (for solutions >25% above baseline)
            ["no_eval_exec", "no_destructive_operations", "allowed_imports",
             "no_continuous_scanning", "privacy_respecting", "max_execution_time"]
        ]
        
        # Track statistics on which boundaries fail most often
        self.failure_stats = {}
        for tier in range(1, 4):  # Stats for tiers 1-3
            for boundary in self.ethics_tiers[tier]:
                self.failure_stats[boundary] = 0
        
    def add_boundary(self, name: str, **params) -> None:
        """
        Add a new ethical boundary with validation.
        
        Args:
            name: Name of the boundary
            **params: Additional parameters for the boundary
        """
        self.boundaries[name] = params
        self.logger.debug(f"Added boundary: {name} with params {params}")
        
    def check(self, genome, active_boundaries: List[str] = None) -> bool:
        """
        Check if a solution violates active ethical boundaries.
        
        Args:
            genome: The genome to check
            active_boundaries: List of boundary names to check, or None for all
            
        Returns:
            True if all active boundaries pass, False otherwise
        """
        try:
            # Get the code as a string
            code = genome.code if hasattr(genome, 'code') else (
                   genome.to_source() if hasattr(genome, 'to_source') else None)
            
            if code is None:
                self.logger.error("Cannot check genome: no code or to_source() method found")
                return False
            
            # If no specific boundaries are provided, check them all
            if active_boundaries is None:
                active_boundaries = list(self.boundaries.keys())
                
            # Check each active boundary
            for boundary_name in active_boundaries:
                if boundary_name not in self.boundaries:
                    self.logger.debug(f"Boundary '{boundary_name}' not defined, skipping")
                    continue
                    
                params = self.boundaries[boundary_name]
                
                if not self._check_specific_boundary(code, boundary_name, params):
                    self.logger.info(f"Failed boundary check: {boundary_name}")
                    # Update failure statistics
                    if boundary_name in self.failure_stats:
                        self.failure_stats[boundary_name] += 1
                    return False
                    
            return True
        except Exception as e:
            self.logger.error(f"Error in boundary enforcement: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _check_specific_boundary(self, code: str, boundary_name: str, params: Dict[str, Any]) -> bool:
        """
        Check a specific boundary for a given code.
        
        Args:
            code: Source code as string
            boundary_name: Name of boundary to check
            params: Parameters for the boundary
            
        Returns:
            True if boundary check passes, False otherwise
        """
        if boundary_name == "no_eval_exec":
            return not self._contains_eval_or_exec(code)
                
        elif boundary_name == "no_destructive_operations":
            return not self._contains_destructive_operations(code)
            
        elif boundary_name == "allowed_imports":
            allowed = params.get("allowed_imports", [])
            unauthorized_imports = self._find_unauthorized_imports(code, allowed)
            if unauthorized_imports:
                self.logger.info(f"Unauthorized imports detected: {unauthorized_imports}")
                return False
            return True
            
        elif boundary_name == "max_execution_time":
            return not self._contains_infinite_loops(code)
            
        elif boundary_name == "max_memory_usage":
            return not self._contains_excessive_memory_usage(code)
            
        elif boundary_name == "no_continuous_scanning":
            return not self._contains_continuous_scanning(code)
            
        elif boundary_name == "privacy_respecting":
            return not self._contains_privacy_violations(code)
            
        else:
            # Unknown boundary type, log warning and default to pass
            self.logger.warning(f"Unknown boundary type: {boundary_name}")
            return True
    
    def _contains_eval_or_exec(self, code: str) -> bool:
        """Check if code contains eval() or exec() calls."""
        import re
        pattern = r'\b(eval|exec)\s*\('
        return bool(re.search(pattern, code))
    
    def _find_unauthorized_imports(self, code: str, allowed_imports: List[str]) -> List[str]:
        """Find imports that are not in the allowed list."""
        import re
        import_pattern = r'import\s+([a-zA-Z0-9_.]+)|from\s+([a-zA-Z0-9_.]+)\s+import'
        matches = re.finditer(import_pattern, code)
        
        unauthorized = []
        for match in matches:
            module = match.group(1) or match.group(2)
            base_module = module.split('.')[0]
            if (base_module not in allowed_imports and
                base_module not in ['typing', 'os', 'sys', 'time']):
                unauthorized.append(base_module)
        return unauthorized
    
    def _contains_infinite_loops(self, code: str) -> bool:
        """Check for potential infinite loops (heuristic)."""
        import re
        # Look for while loops without clear exit conditions
        while_true_pattern = r'while\s+True:|while\s+1:|for\s+.*\s+in\s+iter\('
        has_break = 'break' in code
        # Check for while True without break
        return bool(re.search(while_true_pattern, code)) and not has_break
    
    def _contains_excessive_memory_usage(self, code: str) -> bool:
        """Check for patterns indicating excessive memory usage."""
        import re
        patterns = [
            r'\[\s*[a-zA-Z0-9_\.]+\s*for\s+.*\s+in\s+range\s*\(\s*\d{7,}\s*\)\s*\]',  # Large list comprehensions
            r'array\.array\s*\(\s*[\'"][a-zA-Z]\s*[\'"],\s*\[\s*.*\s*\]\s*\*\s*\d{6,}\s*\)'  # Large arrays
        ]
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def _contains_destructive_operations(self, code: str) -> bool:
        """Check for potentially destructive operations."""
        import re
        patterns = [
            r'os\s*\.\s*system\s*\(',
            r'subprocess\s*\.\s*call\s*\(',
            r'shutil\s*\.\s*rmtree\s*\(',
            r'os\s*\.\s*remove\s*\(',
            r'\.delete\s*\('
        ]
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def _contains_continuous_scanning(self, code: str) -> bool:
        """Check for patterns indicating continuous bluetooth scanning."""
        import re
        patterns = [
            r'while\s+True.*bluetooth\.discover_devices',
            r'for\s+.*\s+in\s+range\s*\(\s*\d{3,}\s*\).*bluetooth'
        ]
        for pattern in patterns:
            if re.search(pattern, code):
                return True
        return False
        
    def _contains_privacy_violations(self, code: str) -> bool:
        """Check for potential privacy violations."""
        import re
        patterns = [
            r'\.mac_address',
            r'\.getmac',
            r'\bmac\s*=',
            r'\.address',
        ]
        privacy_context = ('bluetooth' in code.lower() and
                          any(re.search(pattern, code.lower()) for pattern in patterns))
        return privacy_context
        
    def get_failure_statistics(self) -> Dict[str, int]:
        """
        Get statistics on which boundaries fail most often.
        
        Returns:
            Dictionary mapping boundary names to failure counts
        """
        return self.failure_stats


# Progressive Fitness Evaluator for adaptive ethics enforcement
class ProgressiveFitnessEvaluator(TriFitnessEvaluator):
    """
    Enhanced fitness evaluator that applies progressive ethical boundaries
    based on solution fitness relative to baseline.
    
    This class implements a flexible approach to ethics enforcement where
    lower-fitness solutions face fewer ethical constraints initially,
    allowing for more exploration. As solutions improve in fitness,
    more ethical constraints are progressively applied.
    """
    
    def __init__(self, task: str = "general", description: str = None):
        """
        Initialize the progressive fitness evaluator.
        
        Args:
            task: The type of task being evolved
            description: Description of the task
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".ProgressiveFitnessEvaluator")
        self.task = task
        self.description = description or "Generic task"
        
        # Ethics enforcer is the new TieredEthicalEnforcer
        self.ethical_enforcer = TieredEthicalEnforcer()
        
        # Track fitness baseline for determining which ethical tier to apply
        self.fitness_baseline = float('-inf')  # Initially no baseline
        self.fitness_history = []
        self.generation_count = 0
        
        # Define thresholds for progressively applying ethics tiers
        self.tier_thresholds = [
            0.0,    # Tier 0: Below or at baseline
            0.05,   # Tier 1: >5% above baseline
            0.15,   # Tier 2: >15% above baseline
            0.30    # Tier 3: >30% above baseline
        ]
        
    def initialize_baseline(self, initial_population: List[Any]) -> None:
        """
        Initialize fitness baseline from an initial population.
        
        Args:
            initial_population: Initial population of genomes
        """
        try:
            # Evaluate each genome with tier 0 ethics (no constraints)
            fitness_scores = []
            for genome in initial_population:
                try:
                    score = self.evaluate_fitness(genome, ethics_tier=0)
                    if score != float('-inf'):
                        fitness_scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error evaluating genome for baseline: {e}")
                    continue
                    
            # Set baseline as the mean of valid scores
            if fitness_scores:
                self.fitness_baseline = sum(fitness_scores) / len(fitness_scores)
                self.logger.info(f"Initialized fitness baseline: {self.fitness_baseline}")
            else:
                self.fitness_baseline = 0.0
                self.logger.warning("Could not initialize fitness baseline, defaulting to 0.0")
        except Exception as e:
            self.logger.error(f"Failed to initialize baseline: {e}")
            self.logger.error(traceback.format_exc())
            self.fitness_baseline = 0.0
    
    def update_fitness_metrics(self, population: List[Any], current_gen: int = None) -> None:
        """
        Update fitness metrics based on the current population.
        
        Args:
            population: Current population of genomes
            current_gen: Current generation number
        """
        if current_gen is not None:
            self.generation_count = current_gen
            
        try:
            # Calculate valid fitness scores
            valid_scores = [g.fitness for g in population
                           if hasattr(g, 'fitness') and g.fitness != float('-inf')]
                           
            if valid_scores:
                avg_fitness = sum(valid_scores) / len(valid_scores)
                best_fitness = max(valid_scores)
                
                # Update fitness history
                self.fitness_history.append({
                    'generation': self.generation_count,
                    'avg_fitness': avg_fitness,
                    'best_fitness': best_fitness,
                    'baseline': self.fitness_baseline
                })
                
                # Update baseline periodically using a moving window
                if len(self.fitness_history) >= 3:  # Use last 3 generations
                    # Update baseline every 5 generations
                    if self.generation_count % 5 == 0:
                        recent_avg = [h['avg_fitness'] for h in self.fitness_history[-3:]]
                        new_baseline = sum(recent_avg) / len(recent_avg)
                        
                        # Only update if significantly different
                        if abs((new_baseline - self.fitness_baseline) / max(1, self.fitness_baseline)) > 0.1:
                            old_baseline = self.fitness_baseline
                            self.fitness_baseline = new_baseline
                            self.logger.info(f"Updated fitness baseline: {old_baseline:.2f} -> {new_baseline:.2f}")
                
                self.logger.debug(f"Gen {self.generation_count}: Avg={avg_fitness:.2f}, Best={best_fitness:.2f}, Baseline={self.fitness_baseline:.2f}")
        except Exception as e:
            self.logger.error(f"Error updating fitness metrics: {e}")
    
    def evaluate_fitness(self, genome: Any, ethics_tier: int = None) -> float:
        """
        Evaluate a genome's fitness with the appropriate ethics tier.
        
        Args:
            genome: The genome to evaluate
            ethics_tier: Explicit ethics tier to apply, or None for automatic
            
        Returns:
            Fitness score, or -inf if solution is invalid
        """
        # Get the code as a string
        code = genome.code if hasattr(genome, 'code') else (
               genome.to_source() if hasattr(genome, 'to_source') else None)
        
        if code is None:
            self.logger.error("Cannot evaluate genome: no code or to_source() method found")
            return float('-inf')
        
        # Initialize fitness with safety check
        fitness = 0.0
        
        try:
            # Determine which ethics tier to apply
            tier_to_apply = ethics_tier
            if tier_to_apply is None:
                # If genome already has a fitness score, use it to determine tier
                if hasattr(genome, 'fitness') and genome.fitness != float('-inf'):
                    # Calculate relative improvement over baseline
                    if self.fitness_baseline > 0:
                        relative_improvement = (genome.fitness - self.fitness_baseline) / self.fitness_baseline
                        tier_to_apply = self._get_tier_for_improvement(relative_improvement)
                    else:
                        tier_to_apply = 0
                else:
                    # No fitness yet, start with tier 0
                    tier_to_apply = 0
            
            # Get active boundaries for this tier
            if tier_to_apply == 0:
                active_boundaries = []  # No ethics checks for tier 0
            else:
                active_boundaries = self.ethical_enforcer.ethics_tiers[tier_to_apply]
                
            # Check ethical boundaries if applicable
            if active_boundaries:
                boundaries_passed = self.ethical_enforcer.check(genome, active_boundaries)
                if not boundaries_passed:
                    self.logger.info(f"Solution failed ethical checks at tier {tier_to_apply}")
                    return float('-inf')  # Invalid solution
            
            # Calculate actual fitness
            if self.task == "bluetooth_scan":
                fitness = self._evaluate_bluetooth_scanner(code)
            elif self.task == "usb_scan":
                fitness = self._evaluate_usb_scanner(code)
            else:
                # General task evaluation
                fitness = self._evaluate_general_code(code)
                
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error in fitness evaluation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return float('-inf')
    
    def _get_tier_for_improvement(self, relative_improvement: float) -> int:
        """
        Determine which ethics tier should apply based on relative improvement.
        
        Args:
            relative_improvement: Relative improvement over baseline
            
        Returns:
            Ethics tier (0-3)
        """
        for tier, threshold in enumerate(self.tier_thresholds):
            if relative_improvement <= threshold:
                return min(tier, len(self.tier_thresholds)-1)
        return len(self.tier_thresholds) - 1
    
    def _evaluate_bluetooth_scanner(self, code: str) -> float:
        """
        Evaluate a bluetooth scanning solution.
        
        Args:
            code: Source code as string
            
        Returns:
            Fitness score
        """
        fitness = 0.0
        
        # Check for basic functionality patterns
        if 'bluetooth.discover_devices' in code:
            fitness += 5.0
            
        if 'try:' in code and 'except' in code:
            fitness += 3.0  # Error handling
            
        # Check for flexible interface and parameters
        if 'def scan_bluetooth_devices(' in code and ')' in code.split('def scan_bluetooth_devices(')[1].split(')')[0]:
            fitness += 2.0  # Parameterized function
            
        # Check for documentation
        if '"""' in code or "'''" in code:
            fitness += 1.0
            
        # Check for device filtering
        if 'filter' in code or 'if device' in code or 'if addr' in code:
            fitness += 2.0
            
        # Check for output formatting
        if 'json' in code.lower():
            fitness += 1.0
        if 'print' in code and 'device' in code:
            fitness += 1.0
            
        # Check for retry logic
        if 'retry' in code.lower() or ('for' in code and 'attempt' in code.lower()):
            fitness += 3.0
            
        # Penalize overly complex solutions
        if len(code.splitlines()) > 100:
            fitness -= 2.0
            
        # Add bonuses for advanced features
        if 'signal_strength' in code or 'rssi' in code:
            fitness += 2.0
        if 'sleep' in code and ('time.sleep' in code or 'import time' in code):
            fitness += 1.0
        
        return max(0.1, fitness)  # Ensure minimum positive fitness
        
    def _evaluate_usb_scanner(self, code: str) -> float:
        """
        Evaluate a USB scanning solution.
        
        Args:
            code: Source code as string
            
        Returns:
            Fitness score
        """
        fitness = 0.0
        
        # Check for USB-specific modules
        if 'import usb' in code or 'from usb' in code:
            fitness += 5.0
            
        if 'try:' in code and 'except' in code:
            fitness += 3.0  # Error handling
            
        # Check for scanning logic
        if 'for' in code and ('device' in code or 'dev' in code):
            fitness += 2.0
            
        # Check for device information extraction
        if 'idVendor' in code or 'idProduct' in code:
            fitness += 2.0
            
        # Check for documentation
        if '"""' in code or "'''" in code:
            fitness += 1.0
            
        # Check for output formatting
        if 'json' in code.lower():
            fitness += 1.0
        if 'print' in code and 'device' in code:
            fitness += 1.0
        
        return max(0.1, fitness)  # Ensure minimum positive fitness
    
    def _evaluate_general_code(self, code: str) -> float:
        """
        Evaluate a general programming solution.
        
        Args:
            code: Source code as string
            
        Returns:
            Fitness score
        """
        fitness = 0.0
        
        # Basic structure checks
        if 'def' in code and 'return' in code:
            fitness += 3.0  # Has functions with return values
            
        if 'class' in code:
            fitness += 2.0  # Object-oriented design
            
        if 'try:' in code and 'except' in code:
            fitness += 3.0  # Error handling
            
        # Check for documentation
        docstring_count = code.count('"""') + code.count("'''")
        if docstring_count >= 2:
            fitness += docstring_count / 2  # One point per full docstring
            
        # Check for type hints
        type_hints = 0
        for line in code.splitlines():
            if '->' in line and ':' in line:
                type_hints += 1
        fitness += min(3.0, type_hints * 0.5)  # Up to 3 points for type hints
        
        # Check for imports (modular design)
        import_count = 0
        for line in code.splitlines():
            if line.startswith('import ') or line.startswith('from '):
                import_count += 1
        fitness += min(2.0, import_count * 0.5)  # Up to 2 points for imports
        
        # Check for testing
        if 'test' in code.lower() or 'assert' in code:
            fitness += 2.0
            
        # Penalize overly complex code
        if len(code.splitlines()) > 200:
            fitness -= 3.0
            
        return max(0.1, fitness)  # Ensure minimum positive fitness


class TaskTemplateLoader:
    """Loads and configures task-specific code templates."""
    
    def __init__(self, templates_dir: str = "guidance"):
        """
        Initialize the template loader.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir
        self.templates = {
            "bluetooth_scan": "bluetooth_scanner_template.py",
            "usb_scan": "usb_scanner_template.py",
            "general": None  # General tasks don't use a specific template
        }
        self._template_cache = {}
    
    def load_template(self, task_type: str) -> Optional[str]:
        """
        Load template code for a specific task type.
        
        Args:
            task_type: Type of task to load template for
            
        Returns:
            String containing template code or None if no template exists
        """
        # Return from cache if available
        if task_type in self._template_cache:
            return self._template_cache[task_type]
            
        # Check if we have a template mapping for this task
        if task_type not in self.templates or not self.templates[task_type]:
            return None
            
        # Attempt to load the template file
        template_file = Path(self.templates_dir) / self.templates[task_type]
        if not template_file.exists():
            logging.warning(f"Template file {template_file} not found")
            return None
            
        # Load and cache the template
        try:
            with open(template_file, 'r') as f:
                template_code = f.read()
                self._template_cache[task_type] = template_code
                return template_code
        except Exception as e:
            logging.error(f"Error loading template {template_file}: {e}")
            return None
    
    def _get_bluetooth_scan_fallback(self) -> str:
        """Returns the bluetooth scan fallback template."""
        return '''import bluetooth

def scan_bluetooth_devices():
    """Scan for nearby Bluetooth devices and return their information."""
    devices = []
    try:
        nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True, lookup_class=True)
        for addr, name, device_class in nearby_devices:
            devices.append({"address": addr, "name": name or "Unknown", "class": device_class})
    except Exception as e:
        return [{"error": str(e)}]
    return devices

def main():
    print("Scanning for devices...")
    devices = scan_bluetooth_devices()
    print(f"Found {len(devices)} devices")
    for device in devices:
        print(device)

if __name__ == "__main__":
    main()'''

    def get_fallback_template(self, task_type: str) -> Optional[str]:
        """
        Get a fallback template when the primary template is unavailable.
        
        Args:
            task_type: Type of task to get fallback template for
            
        Returns:
            String containing fallback template code or None
        """
        # Define fallback mappings based on task type
        if task_type == "bluetooth_scan":
            return self._get_bluetooth_scan_fallback()
        return None


class EvolutionSession:
    """Manages a single evolution session with output organization."""
    
    def __init__(self, base_dir: str = "evolved_output", task_type: str = "general"):
        """
        Initialize a new evolution session.
        
        Args:
            base_dir: Base directory for output files
            task_type: Type of task being evolved
        """
        self.base_dir = Path(base_dir)
        self.task_type = task_type
        self.session_id = f"trial_{random.randint(1, 9999):04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.base_dir / self.session_id
        self.best_dir = self.base_dir / "best_solution"
        
        # Create directory structure
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.best_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize session data
        self.stats = {
            "start_time": datetime.datetime.now().isoformat(),
            "generations_completed": 0,
            "best_fitness": float("-inf"),
            "task_type": task_type
        }
        
        # Save initial session info
        self._save_session_info()
    
    def get_generation_dir(self, gen_num: int) -> Path:
        """
        Get directory for a specific generation.
        
        Args:
            gen_num: Generation number
            
        Returns:
            Path object for the generation directory
        """
        gen_dir = self.output_dir / f"generation_{gen_num:03d}"
        gen_dir.mkdir(exist_ok=True)
        return gen_dir
    
    def update_stats(self, gen_num: int, best_fitness: float):
        """
        Update session statistics.
        
        Args:
            gen_num: Current generation number
            best_fitness: Best fitness achieved in this generation
        """
        self.stats["generations_completed"] = gen_num
        if best_fitness > self.stats["best_fitness"]:
            self.stats["best_fitness"] = best_fitness
            self.stats["best_generation"] = gen_num
        
        # Save current stats
        self._save_session_info()
    
    def _save_session_info(self):
        """Save current session information to a JSON file."""
        with open(self.output_dir / "session_info.json", "w") as f:
            json.dump(self.stats, f, indent=2)
    
    def save_best_solution(self, solution: CodeGenome):
        """
        Save the best solution from the evolution.
        
        Args:
            solution: The best CodeGenome from evolution
        """
        # Save to session directory
        with open(self.output_dir / "best_solution.py", "w") as f:
            f.write(solution.to_source())
        
        # Save to best solutions directory with task type and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.best_dir / f"{self.task_type}_{timestamp}.py", "w") as f:
            f.write(solution.to_source())
        
        # Save metadata
        with open(self.best_dir / f"{self.task_type}_{timestamp}_meta.json", "w") as f:
            json.dump({
                "task_type": self.task_type,
                "session_id": self.session_id,
                "fitness": solution.fitness,
                "created_at": timestamp
            }, f, indent=2)


class ProgemonTrainer:
    """Main class for the Progremon evolution system."""
    
    def __init__(self):
        """Initialize the Progremon trainer."""
        self.logger = logging.getLogger(__name__ + ".ProgemonTrainer")
        
        # Initialize components
        self.template_loader = TaskTemplateLoader()
        
        # Set default settings
        self.settings = {
            "pop_size": 10,
            "gens": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "task": "general",
            "description": "Write a generic function",
            "output_dir": "evolved_output",
            "save_all_generations": True,
            "ethics_level": "standard"
        }
        
        # Initialize adaptive tweaker with default settings
        self.adaptive_tweaker = AdaptiveTweaker({
            "mutation_rate": self.settings["mutation_rate"],
            "crossover_rate": self.settings["crossover_rate"]
        })
        
        # Initialize session
        self.session = None
    
    def process_request(self, request_text: str) -> Dict[str, Any]:
        """
        Process a natural language request and extract task information.
        
        Args:
            request_text: User's natural language request
            
        Returns:
            Dictionary with extracted task parameters
        """
        # Initialize config with defaults
        config = {
            "description": request_text,
            "task": "general"  # Default task type
        }
        
        # Detect task type from keywords
        request_lower = request_text.lower()
        
        # Check for bluetooth scanning task
        if any(keyword in request_lower for keyword in ["bluetooth", "bt", "wireless device", "bluetooth device"]):
            config["task"] = "bluetooth_scan"
            print_color("Detected task: Bluetooth Device Scanning", Colors.GREEN, bold=True)
        
        # Check for USB scanning task
        elif any(keyword in request_lower for keyword in ["usb", "drive", "storage", "disk", "memory stick"]):
            config["task"] = "usb_scan"
            print_color("Detected task: USB Drive Scanning", Colors.GREEN, bold=True)
        else:
            print_color("Detected task: General Code Evolution", Colors.GREEN, bold=True)
        
        # Extract additional parameters
        if "signal strength" in request_lower:
            config["include_signal_strength"] = True
        
        if "update every" in request_lower:
            try:
                update_text = request_lower.split("update every")[1].split("seconds")[0].strip()
                update_seconds = float(update_text)
                config["update_interval"] = update_seconds
                print_color(f"  Setting update interval: {update_seconds} seconds", Colors.BLUE)
            except (ValueError, IndexError):
                config["update_interval"] = 1.0  # Default update interval
        
        # Extract output format preference
        if "table" in request_lower:
            config["output_format"] = "table"
            print_color("  Output format: Table view", Colors.BLUE)
        elif "json" in request_lower:
            config["output_format"] = "json"
            print_color("  Output format: JSON", Colors.BLUE)
        
        return config
    
    def configure_evolution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration settings and return the final configuration.
        
        Args:
            config: Dictionary with configuration settings
            
        Returns:
            Dictionary with final configuration
        """
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
    
    def _customize_settings(self, settings: Dict[str, Any]) -> None:
        """
        Let user customize specific settings.
        
        Args:
            settings: Dictionary with settings to customize
        """
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
    
    def _configure_ethical_boundaries(self, enforcer: TieredEthicalEnforcer, settings: Dict[str, Any]) -> None:
        """
        Configure ethical boundaries based on task type and settings.
        
        Args:
            enforcer: The ethical boundary enforcer instance
            settings: Dictionary containing configuration settings
            
        This method sets up appropriate ethical boundaries for different task types,
        with special configurations for bluetooth scanning tasks that respect:
        - Execution time limits
        - Memory usage constraints
        - Allowed import restrictions
        - Scanning time limitations
        - Privacy concerns
        """
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
    
    def run_evolution(self, settings: Dict[str, Any]) -> bool:
        """
        Run the evolution process with the given settings.
        
        Args:
            settings: Dictionary containing all evolution parameters
            
        Returns:
            bool: True if evolution completed successfully, False otherwise
            
        This method handles the entire evolution process including:
        - Creating a new evolution session
        - Setting up ethical boundaries
        - Initializing the evolution engine
        - Running the generational loop
        - Saving outputs at each step
        - Applying adaptive parameter tweaking
        - Comprehensive error handling
        """
        try:
            # Create a new evolution session
            self.session = EvolutionSession(
                base_dir=settings["output_dir"],
                task_type=settings["task"]
            )
            self.logger.info(f"Starting evolution session {self.session.session_id}")
            
            # Initialize evolution components with proper logging
            self.logger.info("Initializing progressive fitness evaluator")
            evaluator = ProgressiveFitnessEvaluator(task=settings["task"], description=settings["description"])
            
            # Configure ethical boundaries based on task type
            self._configure_ethical_boundaries(evaluator.ethical_enforcer, settings)
            
            # Load task template if available
            template_code = self.template_loader.load_template(settings["task"])
            if not template_code and settings["task"] != "general":
                template_code = self.template_loader.get_fallback_template(settings["task"])
                if template_code:
                    self.logger.info(f"Using fallback template for {settings['task']}")
            
            # Prepare source code with task information as comments
            if not template_code:
                # Create a minimal template with task information
                template_code = f"""# Task: {settings['task']}
# Description: {settings['description']}
# Generated by Progremon

def solve_{settings['task'].replace('-', '_')}():
    \"\"\"
    {settings['description']}
    \"\"\"
    # TODO: Implement task-specific functionality
    pass
"""
            else:
                # Enhance existing template with task information
                template_code = f"""# Task: {settings['task']}
# Description: {settings['description']}
# Generated by Progremon

{template_code}
"""
            
            # Create a CodeGenome factory function that uses our enhanced CodeGenome wrapper
            def genome_factory():
                try:
                    return CodeGenome(source_code=template_code)
                except Exception as e:
                    self.logger.error(f"Error creating genome: {str(e)}")
                    # Fallback to a minimal valid template if there's an error
                    minimal_template = "def main():\n    return True\n\nif __name__ == '__main__':\n    main()"
                    return CodeGenome(source_code=minimal_template)
            
            try:
                # Initialize evolution engine with parameters
                self.logger.info("Initializing evolution engine")
                engine = EvolutionEngine(
                    population_size=settings["pop_size"],
                    evaluator=evaluator,
                    genome_class=genome_factory,  # Use factory instead of class
                    mutation_rate=settings["mutation_rate"],
                    crossover_rate=settings["crossover_rate"],
                    elitism_ratio=0.1
                )
            except Exception as e:
                self.logger.error(f"Error initializing evolution engine: {str(e)}")
                self.logger.error(traceback.format_exc())
                print_color(f"❌ Error initializing evolution engine: {str(e)}", Colors.FAIL)
                return False
            
            # Progressive evaluator already has the ethical enforcer integrated
            
            # Initialize population
            self.logger.info(f"Initializing population for task: {settings['task']}")
            engine.initialize_population()
            
            # Initialize fitness baseline with initial population
            self.logger.info("Initializing fitness baseline for progressive ethics")
            try:
                evaluator.initialize_baseline(engine.population)
            except Exception as e:
                self.logger.warning(f"Error initializing fitness baseline: {e}")
            
            # Run evolution
            print_color("\n🚀 Starting evolution process...", Colors.GREEN)
            population_metrics = []
            
            for gen in range(1, settings["gens"] + 1):
                print_color(f"\nGeneration {gen}/{settings['gens']} [{gen*100//settings['gens']}%]", Colors.BOLD)
                gen_start_time = time.time()
                
                try:
                    # Evaluate current population with timeout protection
                    self.logger.info(f"Evaluating population for generation {gen}")
                    try:
                        # Set a timeout for evaluation in case it gets stuck
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Population evaluation timed out")
                        
                        # Register timeout handler if on Unix-like systems
                        if hasattr(signal, 'SIGALRM'):
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(30)  # 30 second timeout
                            
                        fitness_scores = engine.evaluate_population()
                        
                        # Cancel alarm if set
                        if hasattr(signal, 'SIGALRM'):
                            signal.alarm(0)
                    except TimeoutError as te:
                        self.logger.error(f"Evaluation timed out: {str(te)}")
                        print_color("⏱️ Evaluation timed out, proceeding with available results", Colors.WARNING)
                        fitness_scores = engine.get_fitness_scores() or []
                    except Exception as e:
                        self.logger.error(f"Error in population evaluation: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        print_color(f"⚠️ Error in evaluation: {str(e)}", Colors.WARNING)
                        fitness_scores = []
                    
                    if not fitness_scores or all(f == float('-inf') for f in fitness_scores):
                        print_color("No valid solutions found in this generation.", Colors.WARNING)
                        self.logger.warning(f"Generation {gen}: No valid solutions")
                        # Create at least one valid solution to prevent stalling
                        # Check if fitness_scores is None or empty
                        if fitness_scores is None or len(fitness_scores) == 0 or gen > settings["gens"] // 2:
                            self.logger.info("Adding seed solution to prevent stalling")
                            try:
                                # Create a fallback solution
                                fallback = self._create_fallback_solution(settings["task"])
                                # Add fallback directly to the population list
                                engine.population.append(fallback)
                                # Re-evaluate population
                                fitness_scores = engine.evaluate_population()
                            except Exception as e:
                                self.logger.error(f"Error creating fallback solution: {str(e)}")
                        continue
                    
                    # Calculate statistics
                    valid_scores = [f for f in fitness_scores if f != float('-inf')]
                    avg_fitness = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                    best_fitness = max(valid_scores) if valid_scores else float('-inf')
                    best_idx = fitness_scores.index(best_fitness) if valid_scores else -1
                    
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
                    
                    # Apply adaptive tweaking with enhanced robustness
                    try:
                        # Wrap adaptive tweaking in its own try-except block
                        self.logger.info("Applying adaptive parameter tweaking")
                        
                        # Update metrics in adaptive tweaker
                        if hasattr(self.adaptive_tweaker, 'record_metrics'):
                            self.adaptive_tweaker.record_metrics(
                                population=population,
                                best_fitness=best_fitness,
                                avg_fitness=avg_fitness
                            )
                        else:
                            self.adaptive_tweaker.update_parameters(
                                avg_fitness=avg_fitness,
                                best_fitness=best_fitness
                            )
                        
                        # Get adjusted parameters with validation
                        new_params = self.adaptive_tweaker.adjust_parameters()
                        
                        # Validate parameters before applying them
                        if not isinstance(new_params, dict):
                            raise ValueError("Adaptive tweaker returned invalid parameters (not a dictionary)")
                            
                        # Apply parameter changes to the engine with bounds checking
                        if 'mutation_rate' in new_params:
                            # Ensure mutation rate is within valid range
                            new_mutation_rate = max(0.01, min(0.5, new_params["mutation_rate"]))
                            
                            if engine.mutation_rate != new_mutation_rate:
                                print_color(
                                    f"Adjusting mutation rate: {engine.mutation_rate:.3f} -> {new_mutation_rate:.3f}",
                                    Colors.YELLOW
                                )
                                engine.mutation_rate = new_mutation_rate
                        
                        # Apply crossover rate changes if present
                        if 'crossover_rate' in new_params:
                            # Ensure crossover rate is within valid range
                            new_crossover_rate = max(0.1, min(0.9, new_params["crossover_rate"]))
                            
                            if engine.crossover_rate != new_crossover_rate:
                                print_color(
                                    f"Adjusting crossover rate: {engine.crossover_rate:.3f} -> {new_crossover_rate:.3f}",
                                    Colors.YELLOW
                                )
                                engine.crossover_rate = new_crossover_rate
                    except Exception as e:
                        self.logger.error(f"Error in adaptive parameter tweaking: {e}")
                        self.logger.error(traceback.format_exc())
                        print_color("Error adjusting parameters, continuing with current settings", Colors.WARNING)
                    
                    # Generate next generation
                    engine.generate_next_generation()
                    
                    # Log generation completion
                    gen_time = time.time() - gen_start_time
                    self.logger.info(f"Generation {gen} completed in {gen_time:.2f}s. Best fitness: {best_fitness:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in generation {gen}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    print_color(f"⚠️ Error in generation {gen}: {str(e)}", Colors.WARNING)
                    print_color("Attempting to continue with next generation...", Colors.WARNING)
            
            # Evolution complete - save best solution
            best_solution = engine.get_best_solution()
            if best_solution:
                self.session.save_best_solution(best_solution)
                
                print_color("\n✨ Evolution complete! Best solution:", Colors.GREEN, bold=True)
                print_color(f"Fitness score: {best_solution.fitness:.2f}", Colors.GREEN)
                print_color(f"Solution saved to: {self.session.output_dir / 'best_solution.py'}", Colors.CYAN)
                
                # Display solution summary
                with open(self.session.output_dir / "best_solution.py", "r") as f:
                    code = f.read()
                    lines = code.split("\n")
                    first_10_lines = "\n".join(lines[:10])
                    print_color("\nSolution preview:", Colors.BOLD)
                    print_color(first_10_lines + "\n...", Colors.CYAN)
                
                # Show evolution metrics
                print_color("\nEvolution metrics:", Colors.BOLD)
                if population_metrics:
                    initial_fitness = population_metrics[0]["avg_fitness"]
                    final_fitness = population_metrics[-1]["avg_fitness"]
                    improvement = ((final_fitness - initial_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0
                    print_color(f"Initial average fitness: {initial_fitness:.2f}", Colors.BLUE)
                    print_color(f"Final average fitness: {final_fitness:.2f}", Colors.BLUE)
                    print_color(f"Improvement: {improvement:.2f}%", Colors.GREEN)
                
                return True
            else:
                print_color("\n❌ Evolution failed: No valid solution found", Colors.FAIL)
                return False
                
        except Exception as e:
            self.logger.error(f"Error in evolution process: {str(e)}")
            self.logger.error(traceback.format_exc())
            print_color(f"❌ Evolution error: {str(e)}", Colors.FAIL)
            return False

    def main(self):
        """Run the main interactive command loop."""
        print_banner()
        print_color("\nWelcome to Progremon! What would you like to create?", Colors.GREEN)
        
        # Get user request
        request = input("> ")
        
        # Process request
        config = self.process_request(request)
        settings = self.configure_evolution(config)
        
        # Run evolution
        if self.run_evolution(settings):
            print_color("\nWould you like to run the evolved code? (y/n)", Colors.GREEN)
            if input("> ").lower().startswith("y"):
                best_solution_path = self.session.output_dir / "best_solution.py"
                try:
                    print_color("\nRunning evolved code:", Colors.BLUE)
                    print_color("-" * 50, Colors.BLUE)
                    
                    # Run the code in a subprocess to avoid affecting the main process
                    import subprocess
                    result = subprocess.run([sys.executable, best_solution_path], 
                                           capture_output=True, text=True)
                    
                    # Print the output
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print_color(result.stderr, Colors.FAIL)
                    
                    print_color("-" * 50, Colors.BLUE)
                except Exception as e:
                    print_color(f"Error running evolved code: {str(e)}", Colors.FAIL)
                    self.logger.error(f"Error running evolved code: {str(e)}")
                    self.logger.error(traceback.format_exc())
        
        print_color("\nThanks for using Progremon! Gotta evolve 'em all! 🚀", Colors.GREEN)
        
    def _create_fallback_solution(self, task_type: str) -> CodeGenome:
        """Create a fallback solution when evolution is stalling."""
        if task_type == "bluetooth_scan":
            fallback_code = """
import bluetooth
import time

def scan_bluetooth_devices(scan_duration=8, lookup_names=True):
    # Scan for nearby Bluetooth devices and return their information.
    print("Starting Bluetooth scan...")
    devices = []
    try:
        nearby_devices = bluetooth.discover_devices(
            duration=scan_duration,
            lookup_names=lookup_names,
            flush_cache=True,
            lookup_class=True
        )
        
        for addr, name, device_class in nearby_devices:
            device_info = {
                "address": addr,
                "name": name or "Unknown",
                "class": device_class
            }
            devices.append(device_info)
            print(f"Found: {name or 'Unknown'} ({addr})")
            
    except Exception as e:
        print(f"Error scanning: {e}")
        return [{"error": str(e)}]
        
    print(f"Scan complete. Found {len(devices)} devices.")
    return devices

def main():
    # Run a Bluetooth scan and display results.
    print("Bluetooth Scanner")
    print("-" * 40)
    
    results = scan_bluetooth_devices()
    
    if not results:
        print("No devices found.")
        return
        
    print("\\nDevices found:")
    for i, device in enumerate(results, 1):
        if "error" in device:
            print(f"Error: {device['error']}")
        else:
            print(f"{i}. {device['name']} - {device['address']}")
    
    print("\\nScan complete!")

if __name__ == "__main__":
    main()
"""
            return CodeGenome(source_code=fallback_code)
        else:
            # Generic fallback
            return CodeGenome(source_code=f"def main():\\n    print('Hello from {task_type} solution!')\\n    return True\\n\\nif __name__ == '__main__':\\n    main()")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Progremon: Evolution-based code generator")
    parser.add_argument("--task", type=str, help="Specific task type (bluetooth_scan, usb_scan, general)")
    parser.add_argument("--desc", type=str, help="Task description")
    parser.add_argument("--pop", type=int, help="Population size")
    parser.add_argument("--gens", type=int, help="Number of generations")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = ProgemonTrainer()
    
    if args.interactive or (not args.task and not args.desc):
        # Interactive mode
        trainer.main()
    else:
        # CLI mode with arguments
        config = {}
        
        if args.task:
            config["task"] = args.task
        
        if args.desc:
            config["description"] = args.desc
        
        if args.pop:
            config["pop_size"] = args.pop
        
        if args.gens:
            config["gens"] = args.gens
        
        settings = trainer.configure_evolution(config)
        trainer.run_evolution(settings)

