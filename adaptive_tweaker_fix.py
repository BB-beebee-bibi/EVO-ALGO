"""
Adaptive parameter tweaking system for the Trisolaris evolution framework.
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np

# Import these for printing color output
try:
    from progremon import Colors, print_color
except ImportError:
    # Fallback if imports fail
    class Colors:
        """Simple color constants fallback."""
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        YELLOW = '\033[93m'
        FAIL = '\033[91m'
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    
    def print_color(text: str, color: str, bold: bool = False, end: str = '\n') -> None:
        """Fallback print colored text function."""
        print(f"{color}{Colors.BOLD if bold else ''}{text}{Colors.END}", end=end)

class EvolutionMetrics:
    """Tracks metrics for an evolutionary run."""
    def __init__(self):
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.generation_count = 0

    def record_generation(self, population: List, best_fitness: float, avg_fitness: float):
        """Record metrics for a generation."""
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Calculate diversity as the standard deviation of fitness scores
        fitness_scores = [p.fitness for p in population]
        self.diversity_history.append(np.std(fitness_scores))
        
        self.generation_count += 1

class AdaptiveTweaker:
    """Adaptively tweaks evolution parameters based on performance metrics."""
    
    def __init__(self, initial_settings: Dict[str, float]):
        """
        Initialize the adaptive tweaker with initial settings.
        
        Args:
            initial_settings: Dictionary containing initial parameter settings
        """
        self.settings = initial_settings.copy()
        self.metrics = EvolutionMetrics()
        self.history: List[Dict[str, float]] = []
        
        # Define thresholds for adjustments
        self.plateau_threshold = 3  # Number of generations to consider a plateau
        self.diversity_threshold = 0.1  # Minimum acceptable diversity
        self.max_mutation_rate = 0.3  # Upper limit for mutation rate
        self.min_mutation_rate = 0.05  # Lower limit for mutation rate
        
        # Create directory for metrics logging
        self.metrics_dir = "evolution_metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

    def record_metrics(self, population: List, best_fitness: float, avg_fitness: float):
        """
        Record current metrics and save to history.
        
        Args:
            population: The current population of genomes
            best_fitness: The best fitness in the population
            avg_fitness: The average fitness in the population
        """
        self.metrics.record_generation(population, best_fitness, avg_fitness)
        
        # Save metrics to file
        metrics_file = os.path.join(self.metrics_dir, f"metrics_gen_{self.metrics.generation_count}.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'generation': self.metrics.generation_count,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'diversity': self.metrics.diversity_history[-1],
                'settings': self.settings
            }, f, indent=2)

    def update_parameters(self, avg_fitness: float, best_fitness: float) -> None:
        """
        Legacy method for compatibility with the original Progremon.
        This method exists as a bridge to the record_metrics method.
        
        Args:
            avg_fitness: The average fitness in the population
            best_fitness: The best fitness in the population
        """
        # Create a mock population with the provided fitness values
        mock_population = [
            type('MockGenome', (), {'fitness': avg_fitness}),
            type('MockGenome', (), {'fitness': best_fitness})
        ]
        
        # Call the actual method
        self.record_metrics(mock_population, best_fitness, avg_fitness)

    def adjust_parameters(self) -> Dict[str, float]:
        """
        Adjust evolution parameters based on performance metrics.
        
        Returns:
            Dict containing updated parameter settings
        """
        if len(self.metrics.best_fitness_history) < self.plateau_threshold:
            return self.settings  # Not enough data yet

        # Check for fitness plateau
        recent_best = self.metrics.best_fitness_history[-self.plateau_threshold:]
        if max(recent_best) - min(recent_best) < 0.01:  # Small improvement threshold
            # Increase mutation rate if plateau detected
            self.settings['mutation_rate'] = min(
                self.settings['mutation_rate'] * 1.1,
                self.max_mutation_rate
            )
            try:
                print_color(f"Detected fitness plateau. Increasing mutation rate to {self.settings['mutation_rate']:.3f}", Colors.YELLOW)
            except (NameError, AttributeError):
                print(f"Detected fitness plateau. Increasing mutation rate to {self.settings['mutation_rate']:.3f}")

        # Check diversity
        current_diversity = self.metrics.diversity_history[-1]
        if current_diversity < self.diversity_threshold:
            # Increase mutation rate if diversity is too low
            self.settings['mutation_rate'] = min(
                self.settings['mutation_rate'] * 1.2,
                self.max_mutation_rate
            )
            try:
                print_color(f"Low diversity detected. Increasing mutation rate to {self.settings['mutation_rate']:.3f}", Colors.YELLOW)
            except (NameError, AttributeError):
                print(f"Low diversity detected. Increasing mutation rate to {self.settings['mutation_rate']:.3f}")

        # Check if mutation rate is too high
        if self.settings['mutation_rate'] > self.max_mutation_rate:
            self.settings['mutation_rate'] = self.max_mutation_rate

        # Check if mutation rate is too low
        if self.settings['mutation_rate'] < self.min_mutation_rate:
            self.settings['mutation_rate'] = self.min_mutation_rate

        return self.settings

    def get_current_settings(self) -> Dict[str, float]:
        """Get current settings."""
        return self.settings.copy()

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get complete metrics history."""
        return self.history