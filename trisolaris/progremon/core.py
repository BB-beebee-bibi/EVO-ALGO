"""
Core functionality for Progrémon UX component.
"""

from typing import Optional, Dict, Any, List
from .config import ProgrémonConfig
import time
import threading
from trisolaris.core.engine import EvolutionEngine
from trisolaris.config import BaseConfig

class EvolutionSession:
    """Represents an active evolution session."""
    
    def __init__(self, target_description: str):
        self.target_description = target_description
        self.start_time = time.time()
        self.generation = 0
        self.best_fitness = 0.0
        self.population_size = 0
        self.is_running = False
        self.evolution_thread = None
        self.solutions: List[Dict[str, Any]] = []

class ProgrémonCore:
    """Core functionality for Progrémon."""
    
    def __init__(self, config: ProgrémonConfig):
        """Initialize the Progrémon core."""
        self.config = config
        self.evolution_engine = None
        self.current_session: Optional[EvolutionSession] = None
        self.initialize_components()
    
    def initialize_components(self) -> None:
        """Initialize connection to Trisolaris."""
        # Initialize evolution engine with default config
        self.evolution_engine = EvolutionEngine(
            config=None  # Will be updated when starting evolution
        )
    
    def start_evolution_session(self, target_description: str, evolution_config: Dict[str, Any]) -> None:
        """Start a new evolution session."""
        if self.current_session and self.current_session.is_running:
            raise RuntimeError("An evolution session is already running")
        
        self.current_session = EvolutionSession(target_description)
        self.current_session.is_running = True
        
        # Create Trisolaris config object
        trisolaris_config = BaseConfig()
        
        # Pass through the evolution configuration
        trisolaris_config.evolution = evolution_config['evolution']
        trisolaris_config.target_description = target_description
        
        # Update the evolution engine configuration
        self.evolution_engine.update_config(trisolaris_config)
        
        # Start evolution in a separate thread
        self.current_session.evolution_thread = threading.Thread(
            target=self._run_evolution,
            args=(target_description,)
        )
        self.current_session.evolution_thread.start()
    
    def _run_evolution(self, target_description: str) -> None:
        """Run the evolution process in a background thread."""
        try:
            # Start the evolution process
            self.evolution_engine.initialize_population()
            
            while self.current_session and self.current_session.is_running:
                # Run one generation
                self.evolution_engine.evolve(generations=1)
                
                # Update session state
                self.current_session.generation = self.evolution_engine.generation
                self.current_session.population_size = len(self.evolution_engine.population)
                self.current_session.best_fitness = self.evolution_engine.best_fitness
                
                # Get the best solution
                best_solution = self.evolution_engine.get_best_solution()
                if best_solution:
                    solution = {
                        "id": f"sol_{self.current_session.generation}",
                        "fitness": self.current_session.best_fitness,
                        "code": best_solution.get_code()
                    }
                    self.current_session.solutions.append(solution)
                
                # Check if we should stop
                if self.current_session.generation >= self.config.max_generations:
                    self.stop_evolution_session()
                    break
                    
        except Exception as e:
            print(f"Evolution error: {str(e)}")
            self.stop_evolution_session()
    
    def stop_evolution_session(self) -> None:
        """Stop the current evolution session."""
        if self.current_session:
            self.current_session.is_running = False
            if self.current_session.evolution_thread:
                self.current_session.evolution_thread.join(timeout=2.0)
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get the current status of the evolution process."""
        if not self.current_session:
            return {
                "status": "not_started",
                "generation": 0,
                "best_fitness": 0.0,
                "population_size": 0,
                "elapsed_time": 0,
                "target_description": ""
            }
        
        elapsed_time = time.time() - self.current_session.start_time
        
        # Get execution stats from the engine
        engine_stats = self.evolution_engine.get_execution_stats() if self.evolution_engine else {}
        
        return {
            "status": "running" if self.current_session.is_running else "stopped",
            "generation": self.current_session.generation,
            "best_fitness": self.current_session.best_fitness,
            "population_size": self.current_session.population_size,
            "elapsed_time": elapsed_time,
            "target_description": self.current_session.target_description,
            "solutions": self.current_session.solutions[-5:] if self.current_session.solutions else [],
            "engine_stats": engine_stats
        }
    
    def connect_to_trisolaris(self) -> None:
        """Connect to Trisolaris components."""
        # Components are already initialized in initialize_components()
        pass 