"""
Adaptive parameter tuning for the TRISOLARIS evolution engine.
Implements dynamic adjustment of mutation rates, selection pressure, and validation thresholds
based on population metrics and performance feedback.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import entropy

@dataclass
class EvolutionParameters:
    """Container for all tunable evolution parameters."""
    mutation_rate: float
    selection_pressure: float
    validation_threshold: float
    crossover_rate: float
    population_size: int
    elite_size: int

class AdaptiveParameterTuner:
    """Manages dynamic adjustment of evolution parameters based on population metrics."""
    
    def __init__(self, 
                 initial_params: EvolutionParameters,
                 target_diversity: float = 0.7,
                 learning_rate: float = 0.1):
        self.current_params = initial_params
        self.target_diversity = target_diversity
        self.learning_rate = learning_rate
        self.history: Dict[str, list] = {
            'diversity': [],
            'fitness': [],
            'mutation_rate': [],
            'selection_pressure': []
        }
    
    def calculate_population_diversity(self, population: list) -> float:
        """Calculate normalized Shannon entropy of the population."""
        # Convert population to feature vectors
        features = self._extract_features(population)
        # Calculate probability distribution
        probs = np.mean(features, axis=0)
        probs = probs / np.sum(probs)  # Normalize
        # Calculate Shannon entropy
        return entropy(probs)
    
    def _extract_features(self, population: list) -> np.ndarray:
        """Extract numerical features from population for diversity calculation."""
        # TODO: Implement feature extraction based on AST structure
        # For now, return random features for testing
        return np.random.rand(len(population), 10)
    
    def update_parameters(self, 
                         current_diversity: float,
                         avg_fitness: float,
                         stagnation_count: int) -> EvolutionParameters:
        """Update evolution parameters based on current metrics."""
        # Record metrics
        self.history['diversity'].append(current_diversity)
        self.history['fitness'].append(avg_fitness)
        
        # Adjust mutation rate based on diversity
        diversity_diff = self.target_diversity - current_diversity
        self.current_params.mutation_rate *= (1 + self.learning_rate * diversity_diff)
        self.current_params.mutation_rate = np.clip(
            self.current_params.mutation_rate, 0.1, 0.9
        )
        
        # Adjust selection pressure based on fitness progress
        if len(self.history['fitness']) > 1:
            fitness_diff = avg_fitness - self.history['fitness'][-2]
            self.current_params.selection_pressure *= (1 + self.learning_rate * fitness_diff)
            self.current_params.selection_pressure = np.clip(
                self.current_params.selection_pressure, 0.1, 0.9
            )
        
        # Adjust validation threshold based on stagnation
        if stagnation_count > 0:
            self.current_params.validation_threshold *= (1 - self.learning_rate * stagnation_count)
            self.current_params.validation_threshold = np.clip(
                self.current_params.validation_threshold, 0.1, 0.9
            )
        
        # Record parameter history
        self.history['mutation_rate'].append(self.current_params.mutation_rate)
        self.history['selection_pressure'].append(self.current_params.selection_pressure)
        
        return self.current_params
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for potential serialization."""
        return {
            'parameters': self.current_params.__dict__,
            'history': self.history,
            'target_diversity': self.target_diversity,
            'learning_rate': self.learning_rate
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self.current_params = EvolutionParameters(**state['parameters'])
        self.history = state['history']
        self.target_diversity = state['target_diversity']
        self.learning_rate = state['learning_rate']

    def get_mutation_rate(self) -> float:
        """Get the current mutation rate."""
        return self.current_params.mutation_rate

    def get_selection_pressure(self) -> float:
        """Get the current selection pressure."""
        return self.current_params.selection_pressure

    def get_validation_threshold(self) -> float:
        """Get the current validation threshold."""
        return self.current_params.validation_threshold

    def get_crossover_rate(self) -> float:
        """Get the current crossover rate."""
        return self.current_params.crossover_rate 