#!/usr/bin/env python3
"""
Syntax-Aware Evolution Test Script for TRISOLARIS

This script tests the improved code generation quality in the TRISOLARIS framework
by using the new syntax validation and repair mechanisms to reduce syntax errors
in evolved solutions.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("syntax_aware_evolution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("syntax_aware_evolution_test")

# Import TRISOLARIS components
from trisolaris.core.engine import EvolutionEngine
from trisolaris.core.genome import CodeGenome, SyntaxAwareCodeGenome
from trisolaris.core.syntax_validator import SyntaxValidator
from trisolaris.tasks.bluetooth_scanner import BluetoothScannerTask
from trisolaris.evaluation.fitness import FitnessEvaluator

class SyntaxAwareEvolutionTest:
    """
    Test harness for demonstrating syntax-aware evolution.
    
    This class compares the standard evolution process with the syntax-aware
    evolution process to show the improvement in code generation quality.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 5,
        mutation_rate: float = 0.2,  # Higher mutation rate to stress test
        crossover_rate: float = 0.7,
        output_dir: str = "test_output/syntax_aware"
    ):
        """
        Initialize the test harness.
        
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
        os.makedirs(self.output_dir / "standard", exist_ok=True)
        os.makedirs(self.output_dir / "syntax_aware", exist_ok=True)
        
        # Initialize components
        self.task = BluetoothScannerTask()
        
        # Metrics storage
        self.metrics = {
            'standard': {
                'syntax_errors_per_generation': [],
                'avg_fitness_per_generation': []
            },
            'syntax_aware': {
                'syntax_errors_per_generation': [],
                'repairs_per_generation': [],
                'avg_fitness_per_generation': []
            }
        }
        
        logger.info(f"Initialized Syntax-Aware Evolution Test with {population_size} individuals, {num_generations} generations")
    
    def run_standard_evolution(self):
        """Run evolution with standard CodeGenome class."""
        logger.info("Starting standard evolution process")
        
        # Create a fitness evaluator for the task
        evaluator = FitnessEvaluator(weights=self.task.get_fitness_weights())
        
        # Initialize evolution engine with standard CodeGenome
        engine = EvolutionEngine(
            population_size=self.population_size,
            evaluator=evaluator,
            genome_class=CodeGenome,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            use_sandbox=False  # Disable sandbox for simplicity
        )
        
        # Get template code from task
        template_code = self.task.get_template()
        
        # Initialize population with template
        self._initialize_population(engine, template_code)
        
        # Run evolution for specified number of generations
        for generation in range(self.num_generations):
            logger.info(f"Standard Evolution - Generation {generation+1}/{self.num_generations}")
            
            # Count syntax errors before evolution
            syntax_errors = self._count_syntax_errors(engine.population)
            self.metrics['standard']['syntax_errors_per_generation'].append(syntax_errors)
            logger.info(f"Syntax errors before evolution: {syntax_errors}/{len(engine.population)}")
            
            # Evolve one generation
            self._evolve_one_generation(engine, generation)
            
            # Save best individual from this generation
            best_idx = engine.fitness_scores.index(max(engine.fitness_scores))
            best_individual = engine.population[best_idx]
            best_code = best_individual.to_source()
            
            with open(self.output_dir / "standard" / f"best_gen_{generation+1}.py", "w") as f:
                f.write(best_code)
        
        # Save the final best individual
        best_idx = engine.fitness_scores.index(max(engine.fitness_scores))
        best_individual = engine.population[best_idx]
        best_code = best_individual.to_source()
        
        with open(self.output_dir / "standard" / "best.py", "w") as f:
            f.write(best_code)
        
        logger.info("Standard evolution process complete")
        return best_individual, best_code
    
    def run_syntax_aware_evolution(self):
        """Run evolution with syntax-aware CodeGenome class."""
        logger.info("Starting syntax-aware evolution process")
        
        # Create a fitness evaluator for the task
        evaluator = FitnessEvaluator(weights=self.task.get_fitness_weights())
        
        # Initialize evolution engine with SyntaxAwareCodeGenome
        engine = EvolutionEngine(
            population_size=self.population_size,
            evaluator=evaluator,
            genome_class=SyntaxAwareCodeGenome,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            use_sandbox=False  # Disable sandbox for simplicity
        )
        
        # Get template code from task
        template_code = self.task.get_template()
        
        # Initialize population with template
        self._initialize_population(engine, template_code)
        
        # Track repairs
        total_repairs = 0
        
        # Run evolution for specified number of generations
        for generation in range(self.num_generations):
            logger.info(f"Syntax-Aware Evolution - Generation {generation+1}/{self.num_generations}")
            
            # Count syntax errors before evolution
            syntax_errors = self._count_syntax_errors(engine.population)
            self.metrics['syntax_aware']['syntax_errors_per_generation'].append(syntax_errors)
            logger.info(f"Syntax errors before evolution: {syntax_errors}/{len(engine.population)}")
            
            # Evolve one generation
            self._evolve_one_generation(engine, generation)
            
            # Count repairs in this generation
            repairs_this_gen = self._count_repairs_in_log()
            total_repairs += repairs_this_gen
            self.metrics['syntax_aware']['repairs_per_generation'].append(repairs_this_gen)
            logger.info(f"Repairs in generation {generation+1}: {repairs_this_gen}")
            
            # Save best individual from this generation
            best_idx = engine.fitness_scores.index(max(engine.fitness_scores))
            best_individual = engine.population[best_idx]
            best_code = best_individual.to_source()
            
            with open(self.output_dir / "syntax_aware" / f"best_gen_{generation+1}.py", "w") as f:
                f.write(best_code)
        
        # Save the final best individual
        best_idx = engine.fitness_scores.index(max(engine.fitness_scores))
        best_individual = engine.population[best_idx]
        best_code = best_individual.to_source()
        
        with open(self.output_dir / "syntax_aware" / "best.py", "w") as f:
            f.write(best_code)
        
        logger.info(f"Syntax-aware evolution process complete. Total repairs: {total_repairs}")
        return best_individual, best_code
    
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
        
        # Store average fitness
        avg_fitness = sum(engine.fitness_scores) / len(engine.fitness_scores)
        if isinstance(engine.genome_class, SyntaxAwareCodeGenome):
            self.metrics['syntax_aware']['avg_fitness_per_generation'].append(avg_fitness)
        else:
            self.metrics['standard']['avg_fitness_per_generation'].append(avg_fitness)
        
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
            with open("syntax_aware_evolution.log", "r") as f:
                for line in f:
                    if "Repaired code with" in line:
                        repairs += 1
        except FileNotFoundError:
            pass
        return repairs
    
    def visualize_results(self):
        """Generate visualizations of the evolution process and results."""
        logger.info("Generating visualizations")
        
        try:
            # Use the enhanced visualization capabilities
            from trisolaris.visualization import syntax_errors
            from trisolaris.visualization.interactive import make_subplots
            import plotly.graph_objects as go
            
            # Create interactive comparison visualization
            generations = list(range(1, self.num_generations + 1))
            
            # Create comparison figure
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=('Syntax Errors per Generation', 'Average Fitness per Generation')
            )
            
            # Add syntax error traces
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=self.metrics['standard']['syntax_errors_per_generation'],
                    mode='lines+markers',
                    name='Standard Evolution',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=self.metrics['syntax_aware']['syntax_errors_per_generation'],
                    mode='lines+markers',
                    name='Syntax-Aware Evolution',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # Add fitness traces
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=self.metrics['standard']['avg_fitness_per_generation'],
                    mode='lines+markers',
                    name='Standard Evolution Fitness',
                    line=dict(color='red', width=2, dash='dot')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=self.metrics['syntax_aware']['avg_fitness_per_generation'],
                    mode='lines+markers',
                    name='Syntax-Aware Evolution Fitness',
                    line=dict(color='green', width=2, dash='dot')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text='Evolution Comparison: Standard vs. Syntax-Aware',
                template='plotly_white'
            )
            
            # Update axes
            fig.update_xaxes(title_text='Generation', row=1, col=1)
            fig.update_xaxes(title_text='Generation', row=2, col=1)
            fig.update_yaxes(title_text='Number of Syntax Errors', row=1, col=1)
            fig.update_yaxes(title_text='Average Fitness', row=2, col=1)
            
            # Save interactive visualization
            html_path = str(self.output_dir / "evolution_comparison.html")
            fig.write_html(html_path, include_plotlyjs='cdn')
            logger.info(f"Interactive comparison visualization saved to {html_path}")
            
            # Create repairs visualization
            repairs_fig = go.Figure()
            repairs_fig.add_trace(
                go.Bar(
                    x=generations,
                    y=self.metrics['syntax_aware']['repairs_per_generation'],
                    marker_color='rgba(0, 128, 0, 0.6)',
                    name='Repairs'
                )
            )
            
            repairs_fig.update_layout(
                title='Code Repairs per Generation',
                xaxis_title='Generation',
                yaxis_title='Number of Repairs',
                template='plotly_white'
            )
            
            repairs_html_path = str(self.output_dir / "code_repairs.html")
            repairs_fig.write_html(repairs_html_path, include_plotlyjs='cdn')
            logger.info(f"Interactive repairs visualization saved to {repairs_html_path}")
            
            # Export data to CSV
            import pandas as pd
            
            # Create comparison DataFrame
            comparison_data = {
                'Generation': generations,
                'Standard_Syntax_Errors': self.metrics['standard']['syntax_errors_per_generation'],
                'Syntax_Aware_Syntax_Errors': self.metrics['syntax_aware']['syntax_errors_per_generation'],
                'Standard_Avg_Fitness': self.metrics['standard']['avg_fitness_per_generation'],
                'Syntax_Aware_Avg_Fitness': self.metrics['syntax_aware']['avg_fitness_per_generation'],
                'Repairs': self.metrics['syntax_aware']['repairs_per_generation']
            }
            
            df = pd.DataFrame(comparison_data)
            csv_path = str(self.output_dir / "evolution_comparison.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Comparison data exported to {csv_path}")
            
        except ImportError:
            # Fall back to matplotlib if plotly is not available
            logger.warning("Interactive visualization libraries not available, falling back to static plots")
            
            # Create figure for syntax errors comparison
            plt.figure(figsize=(12, 8))
            
            # Plot syntax errors per generation
            plt.subplot(2, 1, 1)
            generations = list(range(1, self.num_generations + 1))
            plt.plot(generations, self.metrics['standard']['syntax_errors_per_generation'], 'r-', label='Standard Evolution')
            plt.plot(generations, self.metrics['syntax_aware']['syntax_errors_per_generation'], 'g-', label='Syntax-Aware Evolution')
            plt.xlabel('Generation')
            plt.ylabel('Number of Syntax Errors')
            plt.title('Syntax Errors per Generation')
            plt.legend()
            plt.grid(True)
            
            # Plot average fitness per generation
            plt.subplot(2, 1, 2)
            plt.plot(generations, self.metrics['standard']['avg_fitness_per_generation'], 'r-', label='Standard Evolution')
            plt.plot(generations, self.metrics['syntax_aware']['avg_fitness_per_generation'], 'g-', label='Syntax-Aware Evolution')
            plt.xlabel('Generation')
            plt.ylabel('Average Fitness')
            plt.title('Average Fitness per Generation')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "evolution_comparison.png")
            
            # Create figure for repairs
            plt.figure(figsize=(10, 6))
            plt.bar(generations, self.metrics['syntax_aware']['repairs_per_generation'])
            plt.xlabel('Generation')
            plt.ylabel('Number of Repairs')
            plt.title('Code Repairs per Generation')
            plt.grid(True)
            plt.savefig(self.output_dir / "code_repairs.png")
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def save_metrics(self):
        """Save collected metrics to a JSON file."""
        metrics_file = self.output_dir / "evolution_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        logger.info(f"Metrics saved to {metrics_file}")
    
    def run_test(self):
        """Run the complete test process."""
        try:
            # Run standard evolution
            standard_best, standard_code = self.run_standard_evolution()
            
            # Run syntax-aware evolution
            syntax_aware_best, syntax_aware_code = self.run_syntax_aware_evolution()
            
            # Generate visualizations
            self.visualize_results()
            
            # Save metrics
            self.save_metrics()
            
            # Print summary
            self._print_summary(standard_best, syntax_aware_best)
            
            return standard_best, syntax_aware_best
            
        except Exception as e:
            logger.error(f"Error during test: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _print_summary(self, standard_best, syntax_aware_best):
        """Print a summary of the test results."""
        print("\n" + "="*80)
        print("SYNTAX-AWARE EVOLUTION TEST SUMMARY")
        print("="*80)
        
        # Print evolution parameters
        print(f"\nEvolution Parameters:")
        print(f"  Population Size: {self.population_size}")
        print(f"  Generations: {self.num_generations}")
        print(f"  Mutation Rate: {self.mutation_rate}")
        print(f"  Crossover Rate: {self.crossover_rate}")
        
        # Print syntax error statistics
        std_errors = self.metrics['standard']['syntax_errors_per_generation']
        sa_errors = self.metrics['syntax_aware']['syntax_errors_per_generation']
        
        print(f"\nSyntax Error Statistics:")
        print(f"  Standard Evolution:")
        print(f"    Initial Syntax Errors: {std_errors[0]}/{self.population_size} ({std_errors[0]/self.population_size*100:.1f}%)")
        print(f"    Final Syntax Errors: {std_errors[-1]}/{self.population_size} ({std_errors[-1]/self.population_size*100:.1f}%)")
        
        print(f"  Syntax-Aware Evolution:")
        print(f"    Initial Syntax Errors: {sa_errors[0]}/{self.population_size} ({sa_errors[0]/self.population_size*100:.1f}%)")
        print(f"    Final Syntax Errors: {sa_errors[-1]}/{self.population_size} ({sa_errors[-1]/self.population_size*100:.1f}%)")
        
        # Print repair statistics
        repairs = self.metrics['syntax_aware']['repairs_per_generation']
        total_repairs = sum(repairs)
        print(f"\nRepair Statistics:")
        print(f"  Total Repairs: {total_repairs}")
        print(f"  Average Repairs per Generation: {total_repairs/self.num_generations:.1f}")
        
        # Print fitness statistics
        std_fitness = self.metrics['standard']['avg_fitness_per_generation']
        sa_fitness = self.metrics['syntax_aware']['avg_fitness_per_generation']
        
        print(f"\nFitness Statistics:")
        print(f"  Standard Evolution:")
        print(f"    Initial Avg Fitness: {std_fitness[0]:.4f}")
        print(f"    Final Avg Fitness: {std_fitness[-1]:.4f}")
        print(f"    Improvement: {(std_fitness[-1] - std_fitness[0]):.4f} ({(std_fitness[-1]/std_fitness[0]-1)*100:.1f}%)")
        
        print(f"  Syntax-Aware Evolution:")
        print(f"    Initial Avg Fitness: {sa_fitness[0]:.4f}")
        print(f"    Final Avg Fitness: {sa_fitness[-1]:.4f}")
        print(f"    Improvement: {(sa_fitness[-1] - sa_fitness[0]):.4f} ({(sa_fitness[-1]/sa_fitness[0]-1)*100:.1f}%)")
        
        # Print output location
        print(f"\nDetailed results saved to: {self.output_dir}")
        print("="*80 + "\n")


def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test the TRISOLARIS syntax-aware evolution system")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--output-dir", type=str, default="test_output/syntax_aware", help="Output directory")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = SyntaxAwareEvolutionTest(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        output_dir=args.output_dir
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