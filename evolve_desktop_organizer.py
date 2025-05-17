#!/usr/bin/env python3
"""
Script to evolve a desktop file organizer using TRISOLARIS.
"""
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

from trisolaris.tasks.desktop_organizer_task import DesktopOrganizerTask
from trisolaris.core import EvolutionEngine, CodeGenome
from trisolaris.evaluation import FitnessEvaluator
from trisolaris.config import BaseConfig, EvolutionConfig

def main():
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"desktop_organizer_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create task instance
    task = DesktopOrganizerTask()
    
    # Create fitness evaluator and add the task's fitness function
    evaluator = FitnessEvaluator()
    evaluator.set_weights(**task.get_fitness_weights())
    
    # Add a custom alignment measure using the task's fitness evaluator
    def evaluate_task_fitness(genome):
        if hasattr(genome, 'to_source'):
            code = genome.to_source()
        else:
            code = str(genome)
        
        fitness, _ = task.evaluate_fitness(code)
        return fitness
    
    evaluator.add_alignment_measure(
        evaluate_task_fitness,
        weight=1.0,
        name=f"{task.get_name()}_fitness"
    )
    
    # Create configuration
    evolution_params = task.get_evolution_params()
    config = BaseConfig(
        evolution=EvolutionConfig(
            population_size=evolution_params['population_size'],
            mutation_rate=evolution_params['mutation_rate'],
            crossover_rate=evolution_params['crossover_rate'],
            selection_pressure=1.0,
            elitism_ratio=0.1
        )
    )
    
    # Create evolution engine
    engine = EvolutionEngine(
        evaluator=evaluator,
        genome_class=CodeGenome,
        config=config
    )
    
    # Create initial population from template
    template = task.get_template()
    base_genome = CodeGenome.from_source(template)
    engine.population = [base_genome.clone() for _ in range(evolution_params['population_size'])]
    
    # Run evolution
    logging.info("Starting desktop organizer evolution...")
    engine.evolve(generations=evolution_params['num_generations'])
    
    # Save best solution
    best_solution = engine.get_best_solution()
    output_dir = Path("evolved_solutions")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "desktop_organizer.py"
    
    with open(output_file, 'w') as f:
        f.write(best_solution.to_source())
    
    logging.info(f"Evolution complete. Best solution saved to {output_file}")
    
    # Print fitness scores
    fitness, scores = task.evaluate_fitness(best_solution.to_source())
    logging.info(f"Final fitness: {fitness:.4f}")
    logging.info("Individual scores:")
    for criterion, score in scores.items():
        logging.info(f"  {criterion}: {score:.4f}")

if __name__ == "__main__":
    main() 