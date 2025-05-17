from typing import List, Any, Dict, Optional
import numpy as np
from ..core.meta_control.adaptive_parameters import AdaptiveParameterTuner, EvolutionParameters
from ..core.population.dual_population import DualPopulationManager
from ..tasks.text_file_sorting import TextFileSortingTask
import ast

class EvolutionaryEngine:
    """
    Main evolutionary engine that coordinates the evolution process.
    Integrates meta-control, population management, and task components.
    """
    def __init__(
        self,
        task: TextFileSortingTask,
        population_size: int = 100,
        max_generations: int = 100,
        initial_mutation_rate: float = 0.1,
        initial_selection_pressure: float = 0.5,
        initial_validation_threshold: float = 0.5,
        initial_crossover_rate: float = 0.7,
        elite_size: int = 1
    ):
        self.task = task
        self.max_generations = max_generations
        self.current_generation = 0
        
        # Create initial evolution parameters dataclass
        initial_params = EvolutionParameters(
            mutation_rate=initial_mutation_rate,
            selection_pressure=initial_selection_pressure,
            validation_threshold=initial_validation_threshold,
            crossover_rate=initial_crossover_rate,
            population_size=population_size,
            elite_size=elite_size
        )
        # Initialize meta-control components
        self.parameter_tuner = AdaptiveParameterTuner(
            initial_params=initial_params
        )
        
        # Initialize population management
        self.population_manager = DualPopulationManager(
            program_pop_size=population_size,
            operator_pop_size=5,  # Default to 5 operators for now
            elite_size=elite_size
        )
        # Set initial program population
        self.population_manager.program_population = task.initialize_population(population_size)
        # Operator population can be initialized in setup_operators
        
        # Evolution history
        self.history: Dict[str, List[float]] = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
        
        # Telemetry statistics
        self.stats = {
            'crossover_attempts': 0,
            'crossover_success': 0,
            'mutation_attempts': 0,
            'mutation_success': 0
        }

    def setup_operators(self):
        """Initialize the operator population with basic genetic operators."""
        # TODO: Implement operator initialization
        pass

    def evaluate_population(self) -> List[float]:
        """Evaluate the entire population using the task's fitness function."""
        return [self.task.evaluate_fitness(program) 
                for program in self.population_manager.program_population]

    def select_parents(self, fitness_scores: List[float]) -> List[Any]:
        """Select parents for reproduction using tournament selection."""
        selection_pressure = self.parameter_tuner.get_selection_pressure()
        tournament_size = max(2, int(len(fitness_scores) * selection_pressure))
        
        parents = []
        for _ in range(len(fitness_scores)):
            # Tournament selection
            tournament = np.random.choice(
                len(fitness_scores),
                size=tournament_size,
                replace=False
            )
            winner_idx = max(tournament, key=lambda i: fitness_scores[i])
            parents.append(self.population_manager.program_population[winner_idx])
        
        return parents

    def create_next_generation(self, parents: List[Any], fitness_scores: List[float]):
        """Create the next generation through selection, crossover, and mutation."""
        mutation_rate = self.parameter_tuner.get_mutation_rate()
        next_gen = []
        
        # Elitism: Keep the best individual
        best_idx = np.argmax(fitness_scores)
        next_gen.append(self.population_manager.program_population[best_idx])
        
        # Create rest of population through reproduction
        while len(next_gen) < len(self.population_manager.program_population):
            # Select two parents
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            if np.random.random() < mutation_rate:
                child = self.mutate(child)
            
            next_gen.append(child)
        
        self.population_manager.program_population = next_gen

    def crossover(self, parent1: Any, parent2: Any) -> Any:
        """Perform crossover between two parents."""
        self.stats['crossover_attempts'] += 1
        try:
            child1, child2 = ProgramAST.crossover(parent1, parent2)
            # Validate the children
            if (validate_ast(child1.ast_tree)[0] and 
                validate_ast(child2.ast_tree)[0] and
                ast.dump(child1.ast_tree) != ast.dump(parent1.ast_tree) and
                ast.dump(child1.ast_tree) != ast.dump(parent2.ast_tree) and
                ast.dump(child2.ast_tree) != ast.dump(parent1.ast_tree) and
                ast.dump(child2.ast_tree) != ast.dump(parent2.ast_tree)):
                self.stats['crossover_success'] += 1
                return child1  # Return first child for now
            return parent1  # Fallback to parent if validation fails
        except Exception:
            return parent1  # Fallback to parent if crossover fails

    def mutate(self, individual: Any) -> Any:
        """Apply mutation to an individual."""
        self.stats['mutation_attempts'] += 1
        try:
            mutated = individual.mutate(self.parameter_tuner.get_mutation_rate())
            if validate_ast(mutated.ast_tree)[0]:
                self.stats['mutation_success'] += 1
                return mutated
            return individual  # Fallback to original if validation fails
        except Exception:
            return individual  # Fallback to original if mutation fails

    def update_meta_parameters(self, fitness_scores: List[float]):
        """Update meta-control parameters based on population statistics."""
        diversity = self.population_manager._calculate_diversity(self.population_manager.program_population)
        avg_fitness = np.mean(fitness_scores)
        
        self.parameter_tuner.update_parameters(
            current_diversity=diversity,
            avg_fitness=avg_fitness,
            stagnation_count=0  # TODO: Implement stagnation detection
        )

    def run(self, max_generations: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Run the evolutionary process for the specified number of generations.
        Returns the evolution history.
        """
        if max_generations is not None:
            self.max_generations = max_generations
        
        self.setup_operators()
        
        while self.current_generation < self.max_generations:
            # Evaluate current population
            fitness_scores = self.evaluate_population()
            
            # Update history
            self.history['best_fitness'].append(max(fitness_scores))
            self.history['avg_fitness'].append(np.mean(fitness_scores))
            self.history['diversity'].append(
                self.population_manager._calculate_diversity(self.population_manager.program_population)
            )
            
            # Select parents
            parents = self.select_parents(fitness_scores)
            
            # Create next generation
            self.create_next_generation(parents, fitness_scores)
            
            # Update meta-control parameters
            self.update_meta_parameters(fitness_scores)
            
            # Print telemetry stats every 5 generations
            if self.current_generation % 5 == 0:
                print(f"\nGeneration {self.current_generation} Telemetry:")
                print(f"Crossover success rate: {self.stats['crossover_success']/self.stats['crossover_attempts']:.2%}")
                print(f"Mutation success rate: {self.stats['mutation_success']/self.stats['mutation_attempts']:.2%}")
            
            self.current_generation += 1
        
        return self.history 