#!/usr/bin/env python3
"""
Enhanced TRISOLARIS Framework Demonstration

This script demonstrates the enhanced TRISOLARIS framework with all improvements:
1. Improved code generation quality (syntax validation and repair)
2. Optimized performance (parallel processing, caching, resource-aware scheduling)
3. Standardized parameters (unified configuration system)
4. Enhanced visualization capabilities (interactive visualizations and dashboard)

The script runs a File Organization task evolution with large-scale parameters:
- Population size: 500
- Generations: 100
- Parallel evaluation, fitness caching, resource-aware scheduling, and early stopping
- Full metrics collection (fitness, diversity, syntax error rates)
- Interactive HTML dashboard and data export (CSV, JSON)
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_trisolaris_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_trisolaris_demo")

# Import TRISOLARIS components
from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import SyntaxAwareCodeGenome
from trisolaris.core.syntax_validator import SyntaxValidator
from trisolaris.tasks.file_organization import FileOrganizationTask
from trisolaris.evaluation.fitness import FitnessEvaluator
from trisolaris.managers.resource_scheduler import ResourceScheduler
from trisolaris.managers.diversity import DiversityGuardian
from trisolaris.config import (
    BaseConfig, EvolutionConfig, SandboxConfig, ResourceLimits,
    ResourceSchedulerConfig, EthicalBoundaryConfig, TaskConfig
)
import pandas as pd
import csv

# Import visualization components
try:
    from trisolaris.visualization import create_visualization_dashboard
    from trisolaris.visualization.dashboard import EvolutionDashboard
    from trisolaris.visualization import track_diversity, track_syntax_errors
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced visualization components not available. Falling back to basic visualizations.")
    VISUALIZATION_AVAILABLE = False

class EnhancedTrisolarisDemonstration:
    """
    Demonstration of the enhanced TRISOLARIS framework.
    
    This class runs a demonstration of all the enhanced features of the TRISOLARIS framework,
    including improved code generation quality, optimized performance, standardized parameters,
    and enhanced visualization capabilities.
    """
    
    def __init__(
        self,
        population_size: int = 500,
        num_generations: int = 100,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        output_dir: str = "demo_output",
        parallel_evaluation: bool = True,
        use_caching: bool = True,
        resource_aware: bool = True,
        interactive_visualization: bool = True,
        early_stopping: bool = True
    ):
        """
        Initialize the demonstration.
        
        Args:
            population_size: Size of the population to evolve
            num_generations: Number of generations to run
            mutation_rate: Mutation rate for genetic operations
            crossover_rate: Crossover rate for genetic operations
            output_dir: Directory to save outputs
            parallel_evaluation: Whether to use parallel evaluation
            use_caching: Whether to use fitness caching
            resource_aware: Whether to use resource-aware scheduling
            interactive_visualization: Whether to use interactive visualizations
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.output_dir = Path(output_dir)
        self.parallel_evaluation = parallel_evaluation
        self.use_caching = use_caching
        self.resource_aware = resource_aware
        self.interactive_visualization = interactive_visualization
        self.early_stopping = early_stopping
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.task = FileOrganizationTask(test_directory="./test_files")
        
        # Create configuration
        self.config = self._create_config()
        
        # Initialize resource scheduler if resource-aware mode is enabled
        self.resource_scheduler = None
        if self.resource_aware:
            self.resource_scheduler = ResourceScheduler(
                config=self.config,
                component_name="resource_scheduler"
            )
        
        # Metrics storage
        self.metrics = {
            'best_fitness_per_generation': [],
            'avg_fitness_per_generation': [],
            'syntax_errors_per_generation': [],
            'repairs_per_generation': [],
            'execution_time_per_generation': [],
            'resource_usage_per_generation': [],
            'diversity_per_generation': [],
            'population_diversity': [],
            'syntax_error_rates': []
        }
        
        # Initialize diversity tracker
        self.diversity_tracker = DiversityGuardian()
        
        logger.info(f"Initialized Enhanced TRISOLARIS Demonstration with {population_size} individuals, {num_generations} generations")
    
    def _create_config(self) -> BaseConfig:
        """Create a configuration with all enhanced features enabled."""
        config = BaseConfig(
            evolution=EvolutionConfig(
                population_size=self.population_size,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                elitism_ratio=0.1,
                parallel_evaluation=self.parallel_evaluation,
                use_caching=self.use_caching,
                early_stopping=self.early_stopping,
                early_stopping_generations=5,
                early_stopping_threshold=0.005,
                resource_aware=self.resource_aware,
                max_workers=None  # Auto-determine based on CPU count
            ),
            sandbox=SandboxConfig(
                base_dir=str(self.output_dir / "sandbox"),
                preserve_sandbox=True
            ),
            ethical_boundaries=EthicalBoundaryConfig(
                use_post_evolution=True,
                allowed_imports={"typing", "collections", "datetime", "math",
                               "random", "re", "time", "json", "sys", "os", "hashlib",
                               "mimetypes", "pathlib", "filecmp", "difflib", "shutil"}
            ),
            resource_scheduler=ResourceSchedulerConfig(
                target_cpu_usage=70.0,
                target_memory_usage=70.0,
                min_cpu_available=20.0,
                min_memory_available=20.0,
                check_interval=1.0,
                adaptive_batch_size=True,
                initial_batch_size=10
            ),
            task=TaskConfig(
                name="file_organization",
                description="File organization task for large-scale evolution test",
                fitness_weights={
                    "functionality": 0.6,
                    "efficiency": 0.3,
                    "alignment": 0.1
                },
                allowed_imports=["os", "sys", "time", "random", "math", "json",
                               "datetime", "collections", "re", "logging", "hashlib",
                               "mimetypes", "pathlib", "filecmp", "difflib", "shutil"],
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
            
            # Initialize evolution engine with SyntaxAwareCodeGenome and all enhancements
            engine = EvolutionEngine(
                evaluator=evaluator,
                genome_class=SyntaxAwareCodeGenome,
                resource_monitor=self.resource_scheduler,
                config=self.config,
                component_name="evolution_engine"
            )
            
            # Get template code from task
            template_code = self.task.get_template()
            
            # Initialize population with template
            self._initialize_population(engine, template_code)
            
            # Run evolution for specified number of generations
            for generation in range(self.num_generations):
                logger.info(f"Enhanced Evolution - Generation {generation+1}/{self.num_generations}")
                
                # Start generation timing
                gen_start_time = time.time()
                
                # Count syntax errors before evolution
                syntax_errors = self._count_syntax_errors(engine.population)
                self.metrics['syntax_errors_per_generation'].append(syntax_errors)
                syntax_error_rate = syntax_errors / len(engine.population)
                self.metrics['syntax_error_rates'].append(syntax_error_rate)
                logger.info(f"Syntax errors before evolution: {syntax_errors}/{len(engine.population)} ({syntax_error_rate:.2%})")
                
                # Track diversity
                diversity = self.diversity_tracker.measure_diversity(engine.population)
                self.metrics['diversity_per_generation'].append(diversity)
                logger.info(f"Population diversity: {diversity:.4f}")
                
                # Evolve one generation
                self._evolve_one_generation(engine, generation)
                
                # Count repairs in this generation
                repairs_this_gen = self._count_repairs_in_log()
                self.metrics['repairs_per_generation'].append(repairs_this_gen)
                logger.info(f"Repairs in generation {generation+1}: {repairs_this_gen}")
                
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
                
                # Record resource usage if available
                if engine.resource_scheduler:
                    resource_summary = engine.resource_scheduler.get_resource_summary()
                    self.metrics['resource_usage_per_generation'].append(resource_summary)
                    logger.info(f"Resource usage: CPU {resource_summary['avg_cpu_percent']:.1f}%, Memory {resource_summary['avg_memory_percent']:.1f}%")
            
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
            
            # Export data in different formats
            self._export_data()
            
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
    
    def _count_repairs_in_log(self):
        """Count the number of repairs in the log file."""
        repairs = 0
        try:
            with open("enhanced_trisolaris_demo.log", "r") as f:
                for line in f:
                    if "Repaired code with" in line:
                        repairs += 1
        except FileNotFoundError:
            pass
        return repairs
    
    def _generate_visualizations(self):
        """Generate visualizations of the evolution process and results."""
        logger.info("Generating visualizations")
        
        if VISUALIZATION_AVAILABLE and self.interactive_visualization:
            try:
                # Create a dashboard with all available visualizations
                dashboard_path = create_visualization_dashboard(
                    metrics=self.metrics,
                    output_dir=str(self.output_dir),
                    interactive=True
                )
                logger.info(f"Interactive dashboard created at {dashboard_path}")
                
                # Create specific visualizations
                dashboard = EvolutionDashboard(
                    metrics=self.metrics,
                    output_dir=str(self.output_dir),
                    interactive=True
                )
                
                # Fitness progression
                fitness_path = dashboard.visualize_fitness_progression(include_std_dev=True)
                logger.info(f"Fitness progression visualization saved to {fitness_path}")
                
                # Syntax error rates
                syntax_path = dashboard.visualize_syntax_errors()
                logger.info(f"Syntax error visualization saved to {syntax_path}")
                
                # Resource usage
                if 'resource_usage_per_generation' in self.metrics and self.metrics['resource_usage_per_generation']:
                    resource_path = dashboard.visualize_resource_usage()
                    logger.info(f"Resource usage visualization saved to {resource_path}")
                
            except Exception as e:
                logger.error(f"Error generating interactive visualizations: {str(e)}")
                self._generate_fallback_visualizations()
        else:
            # Fall back to matplotlib visualizations
            self._generate_fallback_visualizations()
    
    def _generate_fallback_visualizations(self):
        """Generate fallback visualizations using matplotlib."""
        import matplotlib.pyplot as plt
        
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
        
        # Plot repairs
        plt.subplot(2, 2, 3)
        plt.bar(generations, self.metrics['repairs_per_generation'], color='green')
        plt.xlabel('Generation')
        plt.ylabel('Number of Repairs')
        plt.title('Code Repairs per Generation')
        plt.grid(True)
        
        # Plot execution time
        plt.subplot(2, 2, 4)
        plt.plot(generations, self.metrics['execution_time_per_generation'], 'm-', label='Execution Time')
        plt.xlabel('Generation')
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time per Generation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evolution_metrics.png")
        logger.info(f"Basic visualizations saved to {self.output_dir / 'evolution_metrics.png'}")
    
    def _save_metrics(self):
        """Save collected metrics to a JSON file."""
        metrics_file = self.output_dir / "evolution_metrics.json"
        
        # Convert any non-serializable objects to strings or basic types
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if key == 'resource_usage_per_generation':
                # Convert resource usage dictionaries to serializable format
                serializable_metrics[key] = [
                    {k: float(v) if isinstance(v, (float, int)) else str(v) for k, v in usage.items()}
                    for usage in value
                ]
            else:
                serializable_metrics[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
    
    def _export_data(self):
        """Export evolution data in CSV and additional JSON formats."""
        # Create a DataFrame for easier data manipulation
        generations = list(range(1, self.num_generations + 1))
        
        # Basic metrics for CSV export
        data = {
            'Generation': generations,
            'Best_Fitness': self.metrics['best_fitness_per_generation'],
            'Avg_Fitness': self.metrics['avg_fitness_per_generation'],
            'Syntax_Errors': self.metrics['syntax_errors_per_generation'],
            'Syntax_Error_Rate': self.metrics['syntax_error_rates'],
            'Repairs': self.metrics['repairs_per_generation'],
            'Execution_Time': self.metrics['execution_time_per_generation'],
            'Diversity': self.metrics['diversity_per_generation']
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Export to CSV
        csv_path = self.output_dir / "evolution_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Evolution data exported to CSV: {csv_path}")
        
        # Export detailed JSON for each generation
        detailed_json_path = self.output_dir / "detailed_evolution_data.json"
        detailed_data = []
        
        for i, gen in enumerate(generations):
            gen_data = {
                'generation': gen,
                'best_fitness': self.metrics['best_fitness_per_generation'][i] if i < len(self.metrics['best_fitness_per_generation']) else None,
                'avg_fitness': self.metrics['avg_fitness_per_generation'][i] if i < len(self.metrics['avg_fitness_per_generation']) else None,
                'syntax_errors': self.metrics['syntax_errors_per_generation'][i] if i < len(self.metrics['syntax_errors_per_generation']) else None,
                'syntax_error_rate': self.metrics['syntax_error_rates'][i] if i < len(self.metrics['syntax_error_rates']) else None,
                'repairs': self.metrics['repairs_per_generation'][i] if i < len(self.metrics['repairs_per_generation']) else None,
                'execution_time': self.metrics['execution_time_per_generation'][i] if i < len(self.metrics['execution_time_per_generation']) else None,
                'diversity': self.metrics['diversity_per_generation'][i] if i < len(self.metrics['diversity_per_generation']) else None
            }
            
            # Add resource usage if available
            if 'resource_usage_per_generation' in self.metrics and i < len(self.metrics['resource_usage_per_generation']):
                gen_data['resource_usage'] = self.metrics['resource_usage_per_generation'][i]
            
            detailed_data.append(gen_data)
        
        with open(detailed_json_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        logger.info(f"Detailed evolution data exported to JSON: {detailed_json_path}")
    
    def _print_summary(self, engine, best_individual, total_time):
        """Print a summary of the demonstration results."""
        print("\n" + "="*80)
        print("ENHANCED TRISOLARIS FRAMEWORK LARGE-SCALE EVOLUTION TEST SUMMARY")
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
        print(f"  Parallel Processing: {'Enabled' if self.parallel_evaluation else 'Disabled'}")
        print(f"  Fitness Caching: {'Enabled' if self.use_caching else 'Disabled'}")
        print(f"  Resource-Aware Scheduling: {'Enabled' if self.resource_aware else 'Disabled'}")
        print(f"  Early Stopping: {'Enabled' if self.early_stopping else 'Disabled'}")
        print(f"  Interactive Visualizations: {'Enabled' if VISUALIZATION_AVAILABLE and self.interactive_visualization else 'Disabled'}")
        
        # Print syntax error statistics
        syntax_errors = self.metrics['syntax_errors_per_generation']
        repairs = self.metrics['repairs_per_generation']
        
        print(f"\nSyntax Error Statistics:")
        print(f"  Initial Syntax Errors: {syntax_errors[0]}/{self.population_size} ({syntax_errors[0]/self.population_size*100:.1f}%)")
        print(f"  Final Syntax Errors: {syntax_errors[-1]}/{self.population_size} ({syntax_errors[-1]/self.population_size*100:.1f}%)")
        print(f"  Total Repairs: {sum(repairs)}")
        print(f"  Average Repairs per Generation: {sum(repairs)/self.num_generations:.1f}")
        
        # Print fitness statistics
        best_fitness = self.metrics['best_fitness_per_generation']
        avg_fitness = self.metrics['avg_fitness_per_generation']
        
        print(f"\nFitness Statistics:")
        print(f"  Initial Best Fitness: {best_fitness[0]:.4f}")
        print(f"  Final Best Fitness: {best_fitness[-1]:.4f}")
        print(f"  Improvement: {(best_fitness[-1] - best_fitness[0]):.4f} ({(best_fitness[-1]/best_fitness[0]-1)*100:.1f}%)")
        print(f"  Initial Avg Fitness: {avg_fitness[0]:.4f}")
        print(f"  Final Avg Fitness: {avg_fitness[-1]:.4f}")
        
        # Print diversity statistics
        diversity = self.metrics['diversity_per_generation']
        print(f"\nDiversity Statistics:")
        print(f"  Initial Diversity: {diversity[0]:.4f}")
        print(f"  Final Diversity: {diversity[-1]:.4f}")
        print(f"  Change: {(diversity[-1] - diversity[0]):.4f}")
        
        # Print performance statistics
        exec_times = self.metrics['execution_time_per_generation']
        
        print(f"\nPerformance Statistics:")
        print(f"  Total Execution Time: {total_time:.2f} seconds")
        print(f"  Average Generation Time: {sum(exec_times)/len(exec_times):.2f} seconds")
        
        if engine.resource_scheduler:
            resource_summary = engine.resource_scheduler.get_resource_summary()
            print(f"  Average CPU Usage: {resource_summary['avg_cpu_percent']:.1f}%")
            print(f"  Average Memory Usage: {resource_summary['avg_memory_percent']:.1f}%")
            print(f"  Optimal Worker Count: {resource_summary['optimal_worker_count']}")
            print(f"  Final Batch Size: {resource_summary['current_batch_size']}")
        
        # Print output location
        print(f"\nDetailed results saved to: {self.output_dir}")
        print(f"Best solution: {self.output_dir / 'best.py'}")
        print(f"Visualizations: {self.output_dir}")
        print(f"Data exports: {self.output_dir / 'evolution_data.csv'} and {self.output_dir / 'detailed_evolution_data.json'}")
        print("="*80 + "\n")


def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="Run large-scale evolution test with the enhanced TRISOLARIS framework")
    parser.add_argument("--population", type=int, default=500, help="Population size")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--output-dir", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel evaluation")
    parser.add_argument("--no-caching", action="store_true", help="Disable fitness caching")
    parser.add_argument("--no-resource-aware", action="store_true", help="Disable resource-aware scheduling")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive visualizations")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    
    args = parser.parse_args()
    
    # Create and run the demonstration
    demo = EnhancedTrisolarisDemonstration(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        output_dir=args.output_dir,
        parallel_evaluation=not args.no_parallel,
        use_caching=not args.no_caching,
        resource_aware=not args.no_resource_aware,
        interactive_visualization=not args.no_interactive,
        early_stopping=not args.no_early_stopping
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