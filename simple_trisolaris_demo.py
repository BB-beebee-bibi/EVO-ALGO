#!/usr/bin/env python3
"""
Simple TRISOLARIS Framework Demonstration

This script demonstrates the key improvements of the TRISOLARIS framework:
1. Improved code generation quality (syntax validation and repair)
2. Standardized parameters (unified configuration system)
3. Basic visualization capabilities

The script runs a Bluetooth scanner evolution task with reasonable parameters.
"""

import os
import sys
import time
import json
import logging
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_trisolaris_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple_trisolaris_demo")

# Import TRISOLARIS components
from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import SyntaxAwareCodeGenome
from trisolaris.core.syntax_validator import SyntaxValidator
from trisolaris.tasks.bluetooth_scanner import BluetoothScannerTask
from trisolaris.evaluation.fitness import FitnessEvaluator
from trisolaris.config import (
    BaseConfig, EvolutionConfig, SandboxConfig, ResourceLimits, 
    ResourceSchedulerConfig, EthicalBoundaryConfig, TaskConfig
)

class SimpleTrisolarisDemonstration:
    """
    Simple demonstration of the TRISOLARIS framework.
    
    This class runs a demonstration of the key features of the TRISOLARIS framework,
    focusing on improved code generation quality and standardized parameters.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 5,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        output_dir: str = "demo_output"
    ):
        """
        Initialize the demonstration.
        
        Args:
            population_size: Size of the population to evolve
            num_generations: Number of generations to run
            mutation_rate: Mutation rate for genetic operations
            crossover_rate: Crossover rate for genetic operations
            output_dir: Directory to save outputs
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.task = BluetoothScannerTask()
        
        # Create configuration
        self.config = self._create_config()
        
        # Metrics storage
        self.metrics = {
            'best_fitness_per_generation': [],
            'avg_fitness_per_generation': [],
            'syntax_errors_per_generation': [],
            'execution_time_per_generation': []
        }
        
        logger.info(f"Initialized Simple TRISOLARIS Demonstration with {population_size} individuals, {num_generations} generations")
    
    def _create_config(self) -> BaseConfig:
        """Create a configuration with key features enabled."""
        config = BaseConfig(
            evolution=EvolutionConfig(
                population_size=self.population_size,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                elitism_ratio=0.1,
                parallel_evaluation=False,  # Disable parallel evaluation for simplicity
                use_caching=True,
                early_stopping=True,
                early_stopping_generations=3,
                early_stopping_threshold=0.01,
                resource_aware=False,  # Disable resource-aware scheduling
                max_workers=1  # Use single worker
            ),
            task=TaskConfig(
                name="bluetooth_scanner",
                description="Bluetooth scanner task for demonstration",
                fitness_weights={
                    "functionality": 0.6,
                    "efficiency": 0.3,
                    "alignment": 0.1
                },
                allowed_imports=["os", "sys", "time", "random", "math", "json", 
                               "datetime", "collections", "re", "logging", "bluetooth"],
                evolution_params={
                    "population_size": self.population_size,
                    "num_generations": self.num_generations,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate
                }
            ),
            log_level="INFO",
            output_dir=str(self.output_dir),
            debug_mode=False
        )
        
        return config
    
    def run_demonstration(self):
        """Run the complete demonstration process."""
        try:
            # Start timing
            start_time = time.time()
            
            # Create a fitness evaluator for the task
            evaluator = FitnessEvaluator(weights=self.task.get_fitness_weights())
            
            # Initialize evolution engine with SyntaxAwareCodeGenome
            engine = EvolutionEngine(
                evaluator=evaluator,
                genome_class=SyntaxAwareCodeGenome,
                config=self.config,
                component_name="evolution_engine"
            )
            
            # Disable sandbox for simplicity
            engine.use_sandbox = False
            
            # Get template code from task
            template_code = self.task.get_template()
            
            # Initialize population with template
            self._initialize_population(engine, template_code)
            
            # Run evolution for specified number of generations
            for generation in range(self.num_generations):
                logger.info(f"Simple Evolution - Generation {generation+1}/{self.num_generations}")
                
                # Start generation timing
                gen_start_time = time.time()
                
                # Count syntax errors before evolution
                syntax_errors = self._count_syntax_errors(engine.population)
                self.metrics['syntax_errors_per_generation'].append(syntax_errors)
                logger.info(f"Syntax errors before evolution: {syntax_errors}/{len(engine.population)}")
                
                # Evolve one generation
                self._evolve_one_generation(engine, generation)
                
                # Save best individual from this generation
                best_idx = engine.fitness_scores.index(max(engine.fitness_scores))
                best_individual = engine.population[best_idx]
                best_code = best_individual.to_source()
                
                with open(self.output_dir / f"best_gen_{generation+1}.py", "w") as f:
                    f.write(best_code)
                
                # Record generation time
                gen_time = time.time() - gen_start_time
                self.metrics['execution_time_per_generation'].append(gen_time)
                logger.info(f"Generation {generation+1} completed in {gen_time:.2f} seconds")
            
            # Save the final best individual
            best_idx = engine.fitness_scores.index(max(engine.fitness_scores))
            best_individual = engine.population[best_idx]
            best_code = best_individual.to_source()
            
            with open(self.output_dir / "best.py", "w") as f:
                f.write(best_code)
            
            # Record total time
            total_time = time.time() - start_time
            logger.info(f"Demonstration completed in {total_time:.2f} seconds")
            
            # Generate visualizations
            self._generate_visualizations()
            
            # Save metrics
            self._save_metrics()
            
            # Print summary
            self._print_summary(engine, best_individual, total_time)
            
            return best_individual, best_code
            
        except Exception as e:
            logger.error(f"Error during demonstration: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _initialize_population(self, engine, template_code):
        """Initialize population with template code."""
        # Create genomes based on the template
        genomes = []
        for _ in range(engine.population_size):
            # Use the class method correctly
            genome = engine.genome_class.from_source(template_code)
            # Apply small mutations to create diversity
            genome.mutate(self.mutation_rate * 0.5)
            genomes.append(genome)
        
        # Set the population
        engine.population = genomes
        logger.info(f"Initialized population with {len(genomes)} individuals based on template")
    
    def _evolve_one_generation(self, engine, generation):
        """Evolve one generation step by step."""
        # 1. Evaluate current population
        engine.evaluate_population()
        
        # Store fitness metrics
        best_fitness = max(engine.fitness_scores)
        avg_fitness = sum(engine.fitness_scores) / len(engine.fitness_scores)
        self.metrics['best_fitness_per_generation'].append(best_fitness)
        self.metrics['avg_fitness_per_generation'].append(avg_fitness)
        
        # 2. Create next generation (if not the last generation)
        if generation < self.num_generations - 1:
            parents = engine.select_parents()
            offspring = engine.create_offspring(parents)
            engine.population = engine.select_survivors(offspring)
    
    def _count_syntax_errors(self, population):
        """Count the number of individuals with syntax errors."""
        syntax_errors = 0
        for individual in population:
            source_code = individual.to_source()
            is_valid, _, _ = SyntaxValidator.validate(source_code)
            if not is_valid:
                syntax_errors += 1
        return syntax_errors
    
    def _generate_visualizations(self):
        """Generate visualizations of the evolution process and results."""
        logger.info("Generating visualizations")
        
        # Create figure for fitness progression
        plt.figure(figsize=(12, 8))
        
        # Plot fitness progression
        plt.subplot(2, 2, 1)
        generations = list(range(1, self.num_generations + 1))
        plt.plot(generations, self.metrics['best_fitness_per_generation'], 'b-', label='Best Fitness')
        plt.plot(generations, self.metrics['avg_fitness_per_generation'], 'g-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Progression')
        plt.legend()
        plt.grid(True)
        
        # Plot syntax errors
        plt.subplot(2, 2, 2)
        plt.plot(generations, self.metrics['syntax_errors_per_generation'], 'r-', label='Syntax Errors')
        plt.xlabel('Generation')
        plt.ylabel('Number of Errors')
        plt.title('Syntax Errors per Generation')
        plt.grid(True)
        
        # Plot execution time
        plt.subplot(2, 2, 3)
        plt.plot(generations, self.metrics['execution_time_per_generation'], 'm-', label='Execution Time')
        plt.xlabel('Generation')
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time per Generation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evolution_metrics.png")
        logger.info(f"Visualizations saved to {self.output_dir / 'evolution_metrics.png'}")
    
    def _save_metrics(self):
        """Save collected metrics to a JSON file."""
        metrics_file = self.output_dir / "evolution_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
    
    def _print_summary(self, engine, best_individual, total_time):
        """Print a summary of the demonstration results."""
        print("\n" + "="*80)
        print("SIMPLE TRISOLARIS FRAMEWORK DEMONSTRATION SUMMARY")
        print("="*80)
        
        # Print evolution parameters
        print(f"\nEvolution Parameters:")
        print(f"  Population Size: {self.population_size}")
        print(f"  Generations: {self.num_generations}")
        print(f"  Mutation Rate: {self.mutation_rate}")
        print(f"  Crossover Rate: {self.crossover_rate}")
        
        # Print enhanced features status
        print(f"\nEnhanced Features:")
        print(f"  Improved Code Generation: Enabled (SyntaxAwareCodeGenome)")
        print(f"  Standardized Parameters: Enabled (Unified Configuration System)")
        
        # Print syntax error statistics
        syntax_errors = self.metrics['syntax_errors_per_generation']
        
        print(f"\nSyntax Error Statistics:")
        print(f"  Initial Syntax Errors: {syntax_errors[0]}/{self.population_size} ({syntax_errors[0]/self.population_size*100:.1f}%)")
        print(f"  Final Syntax Errors: {syntax_errors[-1]}/{self.population_size} ({syntax_errors[-1]/self.population_size*100:.1f}%)")
        
        # Print fitness statistics
        best_fitness = self.metrics['best_fitness_per_generation']
        avg_fitness = self.metrics['avg_fitness_per_generation']
        
        print(f"\nFitness Statistics:")
        print(f"  Initial Best Fitness: {best_fitness[0]:.4f}")
        print(f"  Final Best Fitness: {best_fitness[-1]:.4f}")
        print(f"  Improvement: {(best_fitness[-1] - best_fitness[0]):.4f} ({(best_fitness[-1]/best_fitness[0]-1)*100:.1f}%)")
        print(f"  Initial Avg Fitness: {avg_fitness[0]:.4f}")
        print(f"  Final Avg Fitness: {avg_fitness[-1]:.4f}")
        
        # Print performance statistics
        exec_times = self.metrics['execution_time_per_generation']
        
        print(f"\nPerformance Statistics:")
        print(f"  Total Execution Time: {total_time:.2f} seconds")
        print(f"  Average Generation Time: {sum(exec_times)/len(exec_times):.2f} seconds")
        
        # Print output location
        print(f"\nDetailed results saved to: {self.output_dir}")
        print(f"Best solution: {self.output_dir / 'best.py'}")
        print(f"Visualizations: {self.output_dir / 'evolution_metrics.png'}")
        print("="*80 + "\n")


def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate the TRISOLARIS framework")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--output-dir", type=str, default="demo_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create and run the demonstration
    demo = SimpleTrisolarisDemonstration(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        output_dir=args.output_dir
    )
    
    demo.run_demonstration()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError during demonstration execution: {str(e)}")
        import traceback
        traceback.print_exc()