"""
Core functionality for Progrémon UX component.
"""

from typing import Optional, Dict, Any, List
from .config import ProgrémonConfig
import time
import threading
from trisolaris.core.engine import EvolutionEngine
from trisolaris.config import BaseConfig, EvolutionConfig, SandboxConfig, ResourceLimits, ResourceSchedulerConfig, EthicalBoundaryConfig
from trisolaris.environment.sandbox import SandboxedEnvironment
from trisolaris.evaluation import FitnessEvaluator, EthicalBoundaryEnforcer
from trisolaris.core.genome import CodeGenome
from trisolaris.config import get_config

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

class ProgrémonCore:
    """Core functionality for Progrémon."""
    
    def __init__(self, config: ProgrémonConfig):
        """Initialize the Progrémon core."""
        self.config = config
        self.evolution_engine = None
        self.current_session: Optional[EvolutionSession] = None
        
        # Initialize Trisolaris configuration
        self.trisolaris_config = get_config("progremon")
        self.initialize_components()
    
    def initialize_components(self) -> None:
        """Initialize connection to Trisolaris."""
        # Create base configuration
        base_config = BaseConfig(
            evolution=EvolutionConfig(
                population_size=self.config.initial_population_size,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                selection_pressure=self.config.selection_pressure,
                elitism_ratio=self.config.elitism_ratio,
                parallel_evaluation=True,
                max_workers=None,       # Will use CPU count - 1
                use_caching=True,
                early_stopping=True,
                early_stopping_generations=10,
                early_stopping_threshold=0.001,
                resource_aware=True
            ),
            sandbox=SandboxConfig(
                base_dir="sandbox",
                resource_limits=ResourceLimits(
                    max_cpu_percent=self.config.max_cpu_percent,
                    max_memory_percent=self.config.max_memory_percent,
                    max_execution_time=self.config.max_execution_time
                )
            ),
            resource_scheduler=ResourceSchedulerConfig(
                target_cpu_usage=70,
                target_memory_usage=70,
                min_cpu_available=20,
                min_memory_available=20,
                adaptive_batch_size=True,
                initial_batch_size=10
            ),
            ethical_boundaries=EthicalBoundaryConfig(
                use_post_evolution=True,
                boundaries={},
                allowed_imports={
                    'os', 'sys', 'time', 'random', 'math', 'json', 
                    'datetime', 'collections', 're', 'logging'
                }
            )
        )
        
        # Merge the newly constructed configuration with the existing one (if any)
        # BaseConfig does not expose an `update` method – instead we create a merged
        # configuration and re-assign the result so that downstream components
        # always see a valid `BaseConfig` instance.
        self.trisolaris_config = self.trisolaris_config.merge(base_config)
        
        # Create ethical filter
        ethical_filter = EthicalBoundaryEnforcer(config=self.trisolaris_config)
        ethical_filter.add_boundary("no_system_calls")
        ethical_filter.add_boundary("no_eval_exec")
        ethical_filter.add_boundary("no_network_access")
        ethical_filter.add_boundary("no_file_operations")
        ethical_filter.add_boundary("max_execution_time", max_execution_time=self.config.max_execution_time)
        ethical_filter.add_boundary("max_memory_usage", max_memory_usage=self.config.max_memory_percent)
        ethical_filter.add_boundary("no_imports", allowed_imports={
            'os', 'sys', 'time', 'random', 'math', 'json', 
            'datetime', 'collections', 're', 'logging'
        })
        
        # Create fitness evaluator
        evaluator = FitnessEvaluator(ethical_filter=ethical_filter)
        evaluator.set_weights(
            alignment=0.4,      # Focus on ethical alignment
            functionality=0.4,  # Equal focus on functionality
            efficiency=0.2      # Some focus on efficiency
        )
        
        # Initialize evolution engine with configuration and evaluator
        self.evolution_engine = EvolutionEngine(
            config=self.trisolaris_config,
            evaluator=evaluator,
            genome_class=CodeGenome
        )
    
    def start_evolution_session(self, target_description: str, evolution_config: Dict[str, Any]) -> None:
        """Start a new evolution session."""
        if self.current_session and self.current_session.is_running:
            raise RuntimeError("An evolution session is already running")
        
        # Create new session
        self.current_session = EvolutionSession(target_description)
        self.current_session.is_running = True
        
        # Create Trisolaris config objects
        evolution = EvolutionConfig(
            population_size=evolution_config['evolution']['population_size'],
            mutation_rate=evolution_config['evolution']['mutation_rate'],
            crossover_rate=evolution_config['evolution']['crossover_rate'],
            selection_pressure=evolution_config['evolution']['selection_pressure'],
            elitism_ratio=evolution_config['evolution']['elitism_ratio'],
            parallel_evaluation=True,
            max_workers=None,
            use_caching=True,
            early_stopping=True,
            early_stopping_generations=10,
            early_stopping_threshold=0.001,
            resource_aware=True
        )
        
        # Create base config
        trisolaris_config = BaseConfig(
            evolution=evolution,
            sandbox=SandboxConfig(
                base_dir="sandbox",
                resource_limits=ResourceLimits(
                    max_cpu_percent=self.config.max_cpu_percent,
                    max_memory_percent=self.config.max_memory_percent,
                    max_execution_time=self.config.max_execution_time
                )
            ),
            resource_scheduler=ResourceSchedulerConfig(
                target_cpu_usage=70,
                target_memory_usage=70,
                min_cpu_available=20,
                min_memory_available=20,
                adaptive_batch_size=True,
                initial_batch_size=10
            ),
            ethical_boundaries=EthicalBoundaryConfig(
                use_post_evolution=True,
                boundaries={},
                allowed_imports={
                    'os', 'sys', 'time', 'random', 'math', 'json', 
                    'datetime', 'collections', 're', 'logging'
                }
            )
        )
        
        # Merge requested evolution-specific overrides into the active config
        self.trisolaris_config = self.trisolaris_config.merge(trisolaris_config)
        
        # Update the evolution engine configuration
        self.evolution_engine.update_config(self.trisolaris_config)
        
        # Initialize population size in session
        self.current_session.population_size = evolution.population_size
        
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
            
            # Update initial state
            self.current_session.generation = 0
            self.current_session.population_size = len(self.evolution_engine.population)
            self.current_session.best_fitness = 0.0
            
            while self.current_session and self.current_session.is_running:
                # Run one generation
                self.evolution_engine.evolve(generations=1)
                
                # Update session state
                self.current_session.generation += 1
                self.current_session.population_size = len(self.evolution_engine.population)
                self.current_session.best_fitness = self.evolution_engine.best_fitness
                
                # Check if we should stop
                if self.current_session.generation >= self.config.max_generations:
                    self.stop_evolution_session()
                    break
                    
                # 2) Early-stopping triggered inside EvolutionEngine
                if (self.evolution_engine.early_stopping and
                    self.evolution_engine.generations_without_improvement >=
                        self.evolution_engine.early_stopping_generations):
                    self.stop_evolution_session()
                    break
                    
        except Exception as e:
            print(f"Evolution error: {str(e)}")
            self.stop_evolution_session()
    
    def stop_evolution_session(self) -> None:
        """Stop the current evolution session."""
        if self.current_session:
            self.current_session.is_running = False
            if (self.current_session.evolution_thread and
                    threading.current_thread() is not self.current_session.evolution_thread):
                # Only join if we are **not** calling from the same thread, otherwise
                # Python will raise `RuntimeError: cannot join current thread`.
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
        
        return {
            "status": "running" if self.current_session.is_running else "stopped",
            "generation": self.current_session.generation,
            "best_fitness": self.current_session.best_fitness,
            "population_size": self.current_session.population_size,
            "elapsed_time": elapsed_time,
            "target_description": self.current_session.target_description
        }
    
    def connect_to_trisolaris(self) -> None:
        """Connect to Trisolaris components."""
        # Components are already initialized in initialize_components()
        pass 