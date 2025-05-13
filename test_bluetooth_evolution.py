#!/usr/bin/env python3
"""
Bluetooth Evolution Test Script for TRISOLARIS

This script tests the complete TRISOLARIS architecture by evolving a Bluetooth scanner
program in a sandboxed environment, guided by evolutionary mathematics, and evaluated
using the LLM ethics service.

The test demonstrates:
1. How the mathematical foundation guides the evolution process
2. How the sandboxed environment provides security during evolution
3. How the post-evolution ethical evaluation system analyzes the evolved solution
"""

import os
import sys
import time
import json
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bluetooth_evolution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bluetooth_evolution_test")

# Import TRISOLARIS components
from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import CodeGenome
from trisolaris.core.evolutionary_math import (
    calculate_price_equation,
    calculate_fisher_theorem,
    calculate_selection_gradient,
    calculate_fitness_landscape
)
from trisolaris.environment.sandbox import SandboxedEnvironment
from trisolaris.tasks.bluetooth_scanner import BluetoothScannerTask
from trisolaris.evaluation.ethics_client import EthicsServiceClient, evaluate_ethics, parse_ethics_report
from trisolaris.evaluation.fitness import FitnessEvaluator
# Import visualization module
from trisolaris.visualization import create_dashboard

class BluetoothEvolutionTest:
    """
    Test harness for the Bluetooth scanner evolution process.
    
    This class orchestrates the complete evolution process, from setup to evaluation,
    while collecting metrics and visualizing results.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        verbose: bool = True,
        output_dir: str = "test_output",
        monitor_resources: bool = True
    ):
        """
        Initialize the test harness.
        
        Args:
            population_size: Size of the population to evolve
            num_generations: Number of generations to run
            mutation_rate: Mutation rate for genetic operations
            crossover_rate: Crossover rate for genetic operations
            verbose: Whether to output detailed logs
            output_dir: Directory to save outputs
            monitor_resources: Whether to monitor resource usage
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.monitor_resources = monitor_resources
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.task = BluetoothScannerTask()
        self.sandbox = None
        self.engine = None
        self.ethics_client = None
        
        # Metrics storage
        self.metrics = {
            'fitness_history': [],
            'resource_usage': [],
            'price_equation': [],
            'fisher_theorem': [],
            'selection_gradients': [],
            'fitness_landscape': [],
            'ethics_evaluations': []
        }
        
        logger.info(f"Initialized Bluetooth Evolution Test with {population_size} individuals, {num_generations} generations")
    
    def setup_environment(self):
        """Set up the sandboxed evolution environment."""
        logger.info("Setting up sandboxed environment")
        
        # Create sandbox with resource monitoring
        self.sandbox = SandboxedEnvironment(
            base_dir=str(self.output_dir / "sandbox"),
            max_cpu_percent=80.0,
            max_memory_percent=80.0,
            max_execution_time=120.0,
            check_interval=1.0,
            preserve_sandbox=True
        )
        
        # Initialize ethics client
        self.ethics_client = EthicsServiceClient()
        
        # Initialize evolution engine with task parameters
        evolution_params = self.task.get_evolution_params()
        
        # Only include parameters that are accepted by EvolutionEngine
        filtered_params = {
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }
        
        # Update with task parameters, but exclude 'num_generations'
        for key, value in evolution_params.items():
            if key != 'num_generations':
                filtered_params[key] = value
        
        # Create a fitness evaluator for the task
        from trisolaris.evaluation.fitness import FitnessEvaluator
        from trisolaris.evaluation.ethical_filter import EthicalBoundaryEnforcer
        
        # Create ethical filter with task boundaries
        ethical_filter = EthicalBoundaryEnforcer()
        for boundary_name, params in self.task.get_required_boundaries().items():
            ethical_filter.add_boundary(boundary_name, **params)
        
        # Create evaluator with task-specific weights
        evaluator = FitnessEvaluator(ethical_filter=ethical_filter,
                                    weights=self.task.get_fitness_weights())
        
        self.engine = EvolutionEngine(
            evaluator=evaluator,
            sandbox_dir=str(self.sandbox.base_dir) if self.sandbox else None,
            max_cpu_percent=self.sandbox.max_cpu_percent if self.sandbox else 80.0,
            max_memory_percent=self.sandbox.max_memory_percent if self.sandbox else 80.0,
            max_execution_time=self.sandbox.max_execution_time if self.sandbox else 120.0,
            use_sandbox=True if self.sandbox else False,
            **filtered_params
        )
        
        logger.info("Environment setup complete")
    
    def run_evolution(self):
        """Run the evolution process in the sandboxed environment."""
        logger.info("Starting evolution process")
        
        # Get template code from task
        template_code = self.task.get_template()
        
        # Initialize population
        # First, create a custom initialization method that uses the template
        def initialize_with_template(template_code):
            # Create genomes based on the template
            genomes = []
            for _ in range(self.engine.population_size):
                # Use the class method correctly
                genome = CodeGenome.from_source(template_code)
                # Apply small mutations to create diversity
                genome.mutate(self.mutation_rate * 0.5)
                genomes.append(genome)
            
            # Set the population
            self.engine.population = genomes
            logger.info(f"Initialized population with {len(genomes)} individuals based on template")
        
        # Initialize population with template
        initialize_with_template(template_code)
        
        # Run evolution for specified number of generations
        for generation in range(self.num_generations):
            logger.info(f"Generation {generation+1}/{self.num_generations}")
            
            # Evolve one generation step by step
            # 1. Evaluate current population
            self.engine.evaluate_population()
            
            # 2. Get current population and fitness values
            population = self.engine.population
            fitness_values = self.engine.fitness_scores
            
            # 3. Create next generation (if not the last generation)
            if generation < self.num_generations - 1:
                parents = self.engine.select_parents()
                offspring = self.engine.create_offspring(parents)
                self.engine.population = self.engine.select_survivors(offspring)
            
            # Collect metrics
            self._collect_metrics(generation, population, fitness_values)
            
            # Log progress
            best_idx = fitness_values.index(max(fitness_values))
            best_fitness = fitness_values[best_idx]
            avg_fitness = sum(fitness_values) / len(fitness_values)
            logger.info(f"Generation {generation+1}: Best fitness = {best_fitness:.4f}, Avg fitness = {avg_fitness:.4f}")
            
            # Save best individual from this generation
            best_individual = population[best_idx]
            best_code = best_individual.to_source()
            with open(self.output_dir / f"best_gen_{generation+1}.py", "w") as f:
                f.write(best_code)
        
        # Save the final best individual
        best_idx = fitness_values.index(max(fitness_values))
        best_individual = population[best_idx]
        best_code = best_individual.to_source()
        with open(self.output_dir / "best.py", "w") as f:
            f.write(best_code)
        
        logger.info("Evolution process complete")
        return best_individual, best_code
    
    def _collect_metrics(self, generation: int, population: List[Any], fitness_values: List[float]):
        """
        Collect metrics for the current generation.
        
        Args:
            generation: Current generation number
            population: List of individuals in the population
            fitness_values: List of fitness values for the population
        """
        # Store fitness history
        self.metrics['fitness_history'].append({
            'generation': generation,
            'best_fitness': max(fitness_values),
            'avg_fitness': sum(fitness_values) / len(fitness_values),
            'min_fitness': min(fitness_values),
            'std_dev': np.std(fitness_values)
        })
        
        # Store resource usage if monitoring is enabled
        if self.monitor_resources and self.sandbox:
            self.metrics['resource_usage'].append({
                'generation': generation,
                **self.sandbox.get_resource_usage_report()
            })
        
        # Calculate and store Price equation components
        # For simplicity, we'll use code length as the trait
        trait_values = [len(ind.to_source()) for ind in population]
        total_change, selection_component, transmission_component = calculate_price_equation(
            population, fitness_values, trait_values
        )
        self.metrics['price_equation'].append({
            'generation': generation,
            'total_change': total_change,
            'selection_component': selection_component,
            'transmission_component': transmission_component
        })
        
        # Calculate and store Fisher's theorem
        # Estimate additive genetic variance as 10% of fitness variance
        fitness_variance = np.var(fitness_values)
        additive_genetic_variance = 0.1 * fitness_variance
        rate_of_increase = calculate_fisher_theorem(
            population, fitness_values, additive_genetic_variance
        )
        self.metrics['fisher_theorem'].append({
            'generation': generation,
            'rate_of_increase': rate_of_increase,
            'additive_genetic_variance': additive_genetic_variance
        })
        
        # Calculate and store selection gradients
        # We'll use a simple fitness function for this demonstration
        def simple_fitness_function(individual):
            code = individual.to_source()
            # Simple metrics: code length and estimated complexity
            length = len(code)
            complexity = code.count('if') + code.count('for') + code.count('while')
            return 1.0 + complexity * 0.05 - length * 0.01
        
        gradients = calculate_selection_gradient(population, simple_fitness_function)
        self.metrics['selection_gradients'].append({
            'generation': generation,
            'gradients': gradients,
            'mean_gradient': np.mean([abs(g) for g in gradients])
        })
        
        # Calculate and store fitness landscape characteristics
        environment = {
            'mutation_rate': self.mutation_rate,
            'selection_pressure': 0.7
        }
        landscape_info = calculate_fitness_landscape(population, environment)
        self.metrics['fitness_landscape'].append({
            'generation': generation,
            **landscape_info
        })
    
    async def evaluate_ethics(self, best_code: str):
        """
        Evaluate the ethics of the evolved solution.
        
        Args:
            best_code: Source code of the best evolved individual
            
        Returns:
            Ethics evaluation report
        """
        logger.info("Performing ethical evaluation of evolved solution")
        
        try:
            # Check if ethics service is available
            if await self.ethics_client.check_health():
                # Use the async API directly
                ethics_report = await self.ethics_client.evaluate_ethics(best_code)
                gurbani_report = await self.ethics_client.get_gurbani_alignment(best_code)
                suggestions = await self.ethics_client.get_improvement_suggestions(best_code)
                
                # Combine reports
                full_report = {
                    **ethics_report,
                    'gurbani_alignment': gurbani_report,
                    'improvement_suggestions': suggestions
                }
                
                logger.info(f"Ethics evaluation complete with score: {ethics_report.get('overall_score', 'N/A')}")
                logger.info(f"Gurbani alignment score: {gurbani_report.get('score', 'N/A')}")
                
                # Store ethics evaluation
                self.metrics['ethics_evaluations'].append(full_report)
                
                return full_report
            else:
                logger.warning("Ethics service is not available, using fallback mock implementation")
                # Use the synchronous fallback implementation
                ethics_report = evaluate_ethics(best_code)
                parsed_report = parse_ethics_report(ethics_report)
                
                # Store ethics evaluation
                self.metrics['ethics_evaluations'].append(parsed_report)
                
                return parsed_report
                
        except Exception as e:
            logger.error(f"Error during ethics evaluation: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'passed': False,
                'concerns': [{'description': f"Evaluation error: {str(e)}"}]
            }
    
    def visualize_results(self):
        """Generate visualizations of the evolution process and results."""
        logger.info("Generating visualizations")
        
        # Use the enhanced visualization capabilities
        from trisolaris.visualization import (
            create_visualization_dashboard,
            visualize_ethics_evaluation,
            export_visualization_data
        )
        
        # Create interactive dashboard
        dashboard_path = create_visualization_dashboard(
            self.metrics,
            self.output_dir,
            interactive=True
        )
        logger.info(f"Dashboard created at {dashboard_path}")
        
        # Create individual visualizations
        if self.metrics['ethics_evaluations']:
            # Get the latest ethics evaluation
            ethics = self.metrics['ethics_evaluations'][-1]
            
            # Extract category scores if available
            if 'categories' in ethics:
                categories = list(ethics['categories'].keys())
                scores = [ethics['categories'][cat].get('score', 0) for cat in categories]
                
                # Use the visualization module
                ethics_path = visualize_ethics_evaluation(
                    categories,
                    scores,
                    str(self.output_dir / "ethics_evaluation.png")
                )
                logger.info(f"Ethics evaluation visualization saved to {ethics_path}")
        
        # Export visualization data to CSV and JSON formats
        exported_files = export_visualization_data(
            self.metrics,
            self.output_dir,
            formats=['csv', 'json']
        )
        logger.info(f"Exported visualization data: {exported_files}")
        
        logger.info(f"All visualizations saved to {self.output_dir}")
    
    def save_metrics(self):
        """Save collected metrics to a JSON file."""
        metrics_file = self.output_dir / "evolution_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Track additional metrics if not already present
        if 'syntax_errors' not in self.metrics:
            # Add placeholder syntax error metrics for demonstration
            from trisolaris.visualization import track_syntax_errors
            for gen in range(len(self.metrics.get('fitness_history', []))):
                # Simulate decreasing error rates over generations
                error_count = max(0, 10 - gen)
                repair_count = max(0, error_count - 1)
                track_syntax_errors(
                    generation=gen,
                    population_size=50,
                    error_count=error_count,
                    repair_success_count=repair_count,
                    metrics=self.metrics
                )
    
    def run_test(self):
        """Run the complete test process."""
        try:
            # Setup environment
            self.setup_environment()
            
            # Run evolution
            best_individual, best_code = self.run_evolution()
            
            # Evaluate ethics
            ethics_report = asyncio.run(self.evaluate_ethics(best_code))
            
            # Generate visualizations
            self.visualize_results()
            
            # Save metrics
            self.save_metrics()
            
            # Print summary
            self._print_summary(best_individual, ethics_report)
            
            return best_individual, ethics_report
            
        except Exception as e:
            logger.error(f"Error during test: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
        finally:
            # Clean up
            if self.sandbox:
                self.sandbox.cleanup()
    
    def _print_summary(self, best_individual, ethics_report):
        """Print a summary of the test results."""
        print("\n" + "="*80)
        print("BLUETOOTH EVOLUTION TEST SUMMARY")
        print("="*80)
        
        # Print evolution parameters
        print(f"\nEvolution Parameters:")
        print(f"  Population Size: {self.population_size}")
        print(f"  Generations: {self.num_generations}")
        print(f"  Mutation Rate: {self.mutation_rate}")
        print(f"  Crossover Rate: {self.crossover_rate}")
        
        # Print fitness progression
        if self.metrics['fitness_history']:
            first_gen = self.metrics['fitness_history'][0]
            last_gen = self.metrics['fitness_history'][-1]
            print(f"\nFitness Progression:")
            print(f"  Initial Best Fitness: {first_gen['best_fitness']:.4f}")
            print(f"  Final Best Fitness: {last_gen['best_fitness']:.4f}")
            print(f"  Improvement: {(last_gen['best_fitness'] - first_gen['best_fitness']):.4f} " +
                  f"({(last_gen['best_fitness'] / first_gen['best_fitness'] - 1) * 100:.1f}%)")
        
        # Print ethics evaluation summary
        if ethics_report:
            print(f"\nEthics Evaluation:")
            print(f"  Overall Score: {ethics_report.get('overall_score', 'N/A')}")
            print(f"  Passed: {ethics_report.get('passed', False)}")
            
            # Print concerns if any
            concerns = ethics_report.get('concerns', [])
            if concerns:
                print(f"  Concerns ({len(concerns)}):")
                for i, concern in enumerate(concerns[:3], 1):  # Show top 3 concerns
                    print(f"    {i}. {concern.get('description', 'Unknown concern')}")
                if len(concerns) > 3:
                    print(f"    ... and {len(concerns) - 3} more concerns")
            else:
                print("  No ethical concerns identified")
            
            # Print Gurbani alignment if available
            gurbani = ethics_report.get('gurbani_alignment', {})
            if isinstance(gurbani, dict) and 'score' in gurbani:
                print(f"  Gurbani Alignment Score: {gurbani['score']}")
        
        # Print resource usage summary
        if self.metrics['resource_usage']:
            max_cpu = max(m.get('cpu_percent', 0) for m in self.metrics['resource_usage'])
            max_memory = max(m.get('memory_percent', 0) for m in self.metrics['resource_usage'])
            print(f"\nResource Usage:")
            print(f"  Peak CPU Usage: {max_cpu:.1f}%")
            print(f"  Peak Memory Usage: {max_memory:.1f}%")
        
        # Print output location
        print(f"\nDetailed results saved to: {self.output_dir}")
        print("="*80 + "\n")


def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test the TRISOLARIS Bluetooth evolution system")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--output-dir", type=str, default="test_output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-monitor", action="store_true", help="Disable resource monitoring")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = BluetoothEvolutionTest(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        verbose=args.verbose,
        output_dir=args.output_dir,
        monitor_resources=not args.no_monitor
    )
    
    test.run_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        import traceback
        traceback.print_exc()