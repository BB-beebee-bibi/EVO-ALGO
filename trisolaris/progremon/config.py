"""
Configuration system for Progrémon UX component.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os

@dataclass
class ProgrémonConfig:
    """Configuration settings for Progrémon."""
    
    # Display settings
    show_ascii_art: bool = True
    show_welcome_message: bool = True
    show_evolution_progress: bool = True
    
    # Evolution settings
    initial_population_size: int = 10
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Session settings
    save_session_history: bool = True
    session_history_path: str = "progremon_history"
    
    # Integration settings
    evolution_engine: Optional[str] = None
    fitness_evaluator: Optional[str] = None
    code_genome: Optional[str] = None
    resource_scheduler: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProgrémonConfig':
        """Create a ProgrémonConfig instance from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'show_ascii_art': self.show_ascii_art,
            'show_welcome_message': self.show_welcome_message,
            'show_evolution_progress': self.show_evolution_progress,
            'initial_population_size': self.initial_population_size,
            'max_generations': self.max_generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'save_session_history': self.save_session_history,
            'session_history_path': self.session_history_path,
            'evolution_engine': self.evolution_engine,
            'fitness_evaluator': self.fitness_evaluator,
            'code_genome': self.code_genome,
            'resource_scheduler': self.resource_scheduler
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save the configuration to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProgrémonConfig':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict) 